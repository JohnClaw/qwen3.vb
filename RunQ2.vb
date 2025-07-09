' runq.vb
' Конвертация C-кода для инференса модели Qwen-3 на Visual Basic .NET.
Imports System
Imports System.Collections.Generic
Imports System.Diagnostics
Imports System.IO
Imports System.Linq
Imports System.Numerics ' <-- ДОБАВЛЕНО ДЛЯ ИСПРАВЛЕНИЯ
Imports System.Text
Imports System.Runtime.InteropServices ' Для MemoryMarshal, хоть и не используется напрямую, полезно для контекста
Imports System.Runtime.CompilerServices ' Для Unsafe, если бы использовался, но здесь нет
Imports System.IO.MemoryMappedFiles ' Added for MemoryMappedFile


' ----------------------------------------------------------------------------
' Глобальные переменные и константы
' ----------------------------------------------------------------------------
Module Globals
    ' Глобальный размер группы для квантования весов
    Public GS As Integer = 0
End Module

' ----------------------------------------------------------------------------
' Классы для структуры Transformer'а
' ----------------------------------------------------------------------------

Public Class Config
    Public Property MagicNumber As UInteger
    Public Property Version As Integer
    Public Property Dimension As Integer
    Public Property HiddenDimension As Integer
    Public Property NumLayers As Integer
    Public Property NumHeads As Integer
    Public Property NumKVHeads As Integer
    Public Property VocabSize As Integer
    Public Property SeqLength As Integer
    Public Property HeadDimension As Integer
    Public Property SharedClassifier As Integer
    Public Property GroupSize As Integer

    Public ReadOnly Property IsSharedClassifier As Boolean
        Get
            Return SharedClassifier = 1
        End Get
    End Property
End Class

Public Class QuantizedTensor
    Public Property Q As SByte() ' Квантованные значения (SByte is Int8)
    Public Property S As Single() ' Коэффициенты масштабирования (Single is float32)

    Public Sub New(quantizedValues As SByte(), scales As Single())
        Me.Q = quantizedValues
        Me.S = scales
    End Sub
End Class

Public Class TransformerWeights
    Public Property RmsAttWeight As Single()
    Public Property RmsFfnWeight As Single()
    Public Property RmsFinalWeight As Single()
    Public Property Q_Ln_Weights As Single()
    Public Property K_Ln_Weights As Single()

    Public Property Q_Tokens As QuantizedTensor
    Public Property Wq As QuantizedTensor()
    Public Property Wk As QuantizedTensor()
    Public Property Wv As QuantizedTensor()
    Public Property Wo As QuantizedTensor()
    Public Property W1 As QuantizedTensor()
    Public Property W2 As QuantizedTensor()
    Public Property W3 As QuantizedTensor()
    Public Property Wcls As QuantizedTensor

    Public Property TokenEmbeddingTable As Single()

    Public Sub New(p As Config, memory As Byte())
        Dim offset As Integer = 0

        ' Использование приватных функций для чтения из массива памяти
        Me.RmsAttWeight = ReadFloats(memory, offset, p.NumLayers * p.Dimension)
        Me.RmsFfnWeight = ReadFloats(memory, offset, p.NumLayers * p.Dimension)
        Me.RmsFinalWeight = ReadFloats(memory, offset, p.Dimension)
        Me.Q_Ln_Weights = ReadFloats(memory, offset, p.NumLayers * p.HeadDimension)
        Me.K_Ln_Weights = ReadFloats(memory, offset, p.NumLayers * p.HeadDimension)

        Me.Q_Tokens = ReadQuantizedTensors(memory, offset, 1, p.VocabSize * p.Dimension)(0)
        Me.Wq = ReadQuantizedTensors(memory, offset, p.NumLayers, p.Dimension * (p.NumHeads * p.HeadDimension))
        Me.Wk = ReadQuantizedTensors(memory, offset, p.NumLayers, p.Dimension * (p.NumKVHeads * p.HeadDimension))
        Me.Wv = ReadQuantizedTensors(memory, offset, p.NumLayers, p.Dimension * (p.NumKVHeads * p.HeadDimension))
        Me.Wo = ReadQuantizedTensors(memory, offset, p.NumLayers, (p.NumHeads * p.HeadDimension) * p.Dimension)
        Me.W1 = ReadQuantizedTensors(memory, offset, p.NumLayers, p.Dimension * p.HiddenDimension)
        Me.W2 = ReadQuantizedTensors(memory, offset, p.NumLayers, p.HiddenDimension * p.Dimension)
        Me.W3 = ReadQuantizedTensors(memory, offset, p.NumLayers, p.Dimension * p.HiddenDimension)

        If p.IsSharedClassifier Then
            Me.Wcls = Me.Q_Tokens
        Else
            Me.Wcls = ReadQuantizedTensors(memory, offset, 1, p.Dimension * p.VocabSize)(0)
        End If

        Me.TokenEmbeddingTable = New Single(p.VocabSize * p.Dimension - 1) {}
        Dequantize(Me.Q_Tokens, Me.TokenEmbeddingTable)
    End Sub

    ' Вспомогательные методы для чтения из байтового массива
    Private Function ReadFloats(memory As Byte(), ByRef offset As Integer, count As Integer) As Single()
        Dim result(count - 1) As Single
        Buffer.BlockCopy(memory, offset, result, 0, count * 4)
        offset += count * 4
        Return result
    End Function

    Private Function ReadQuantizedTensors(memory As Byte(), ByRef offset As Integer, numTensors As Integer, tensorSize As Integer) As QuantizedTensor()
        Dim tensors(numTensors - 1) As QuantizedTensor
        For i As Integer = 0 To numTensors - 1
            Dim qArray(tensorSize - 1) As SByte
            Buffer.BlockCopy(memory, offset, qArray, 0, tensorSize)
            offset += tensorSize

            Dim scalesSize As Integer = If(Globals.GS > 0, tensorSize \ Globals.GS, 0)
            Dim sArray(scalesSize - 1) As Single
            Buffer.BlockCopy(memory, offset, sArray, 0, scalesSize * 4)
            offset += scalesSize * 4

            tensors(i) = New QuantizedTensor(qArray, sArray)
        Next
        Return tensors
    End Function

    Private Sub Dequantize(qx As QuantizedTensor, x As Single())
        ' Этот Dequantize используется только для TokenEmbeddingTable при загрузке.
        ' В forward pass используется NetFunctions.Dequantize, который работает со Span.
        ' Эта версия переписана для соответствия C# Program.cs Dequantize, работающей с 1D массивом.
        If Globals.GS = 0 Then Return

        Dim qSpan = qx.Q
        Dim sSpan = qx.S
        For i As Integer = 0 To x.Length - 1
            x(i) = qSpan(i) * sSpan(i \ Globals.GS)
        Next
    End Sub
End Class

Public Class RunState
    Public Property X As Single()
    Public Property Xb As Single()
    Public Property Xb2 As Single()
    Public Property Hb As Single()
    Public Property Hb2 As Single()
    Public Property Xq_q As SByte()
    Public Property Xq_s As Single()
    Public Property Hq_q As SByte()
    Public Property Hq_s As Single()
    Public Property Q As Single()
    Public Property Att As Single() ' Flattened 2D array
    Public Property Logits As Single()
    Public Property KeyCache As Single() ' Flattened 3D array
    Public Property ValueCache As Single() ' Flattened 3D array

    Public Sub New(p As Config)
        Dim allHeadsDim As Integer = p.NumHeads * p.HeadDimension
        Dim kvDim As Integer = p.NumKVHeads * p.HeadDimension

        Me.X = New Single(p.Dimension - 1) {}
        Me.Xb = New Single(allHeadsDim - 1) {}
        Me.Xb2 = New Single(p.Dimension - 1) {}
        Me.Hb = New Single(p.HiddenDimension - 1) {}
        Me.Hb2 = New Single(p.HiddenDimension - 1) {}

        Dim xq_s_size As Integer = If(Globals.GS > 0, allHeadsDim \ Globals.GS, 0)
        Me.Xq_q = New SByte(allHeadsDim - 1) {}
        Me.Xq_s = New Single(xq_s_size - 1) {}

        Dim hq_s_size As Integer = If(Globals.GS > 0, p.HiddenDimension \ Globals.GS, 0)
        Me.Hq_q = New SByte(p.HiddenDimension - 1) {}
        Me.Hq_s = New Single(hq_s_size - 1) {}

        Me.Q = New Single(allHeadsDim - 1) {}
        Me.Att = New Single(p.NumHeads * p.SeqLength - 1) {}
        Me.Logits = New Single(p.VocabSize - 1) {}
        Me.KeyCache = New Single(CLng(p.NumLayers) * p.SeqLength * kvDim - 1) {}
        Me.ValueCache = New Single(CLng(p.NumLayers) * p.SeqLength * kvDim - 1) {}
    End Sub
End Class

' ---------- КЛАСС ТРАНСФОРМЕРА ----------
Public Class Transformer
    Implements IDisposable

    Public Property Config As Config
    Private ReadOnly Weights As TransformerWeights
    Private ReadOnly State As RunState
    Private _mmf As MemoryMappedFile

    Public Sub New(checkpointPath As String, ctxLength As Integer)
        If Not File.Exists(checkpointPath) Then
            Throw New FileNotFoundException("Checkpoint file not found.", checkpointPath)
        End If

        _mmf = MemoryMappedFile.CreateFromFile(checkpointPath, FileMode.Open, Nothing, 0, MemoryMappedFileAccess.Read)
        Using accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read)
            ' Read the full 256-byte header
            Dim headerBytes(255) As Byte
            accessor.ReadArray(Of Byte)(0, headerBytes, 0, 256)

            ' Reconstruct Config manually from headerBytes
            Me.Config = New Config With {
                .MagicNumber = BitConverter.ToUInt32(headerBytes, 0),
                .Version = BitConverter.ToInt32(headerBytes, 4),
                .Dimension = BitConverter.ToInt32(headerBytes, 8),
                .HiddenDimension = BitConverter.ToInt32(headerBytes, 12),
                .NumLayers = BitConverter.ToInt32(headerBytes, 16),
                .NumHeads = BitConverter.ToInt32(headerBytes, 20),
                .NumKVHeads = BitConverter.ToInt32(headerBytes, 24),
                .VocabSize = BitConverter.ToInt32(headerBytes, 28),
                .SeqLength = BitConverter.ToInt32(headerBytes, 32),
                .HeadDimension = BitConverter.ToInt32(headerBytes, 36),
                .SharedClassifier = BitConverter.ToInt32(headerBytes, 40),
                .GroupSize = BitConverter.ToInt32(headerBytes, 44)
            }

            If Me.Config.MagicNumber <> &H616A6331 Then
                Throw New InvalidDataException($"File {checkpointPath} is not a qwen3.c checkpoint")
            End If
            If Me.Config.Version <> 1 Then
                Throw New InvalidDataException($"Checkpoint {checkpointPath} is version {Me.Config.Version}, expected 1")
            End If

            If ctxLength > 0 AndAlso ctxLength <= Me.Config.SeqLength Then
                Me.Config.SeqLength = ctxLength
            End If

            Globals.GS = Me.Config.GroupSize

            ' Чтение весов из MemoryMappedFile в байтовый массив
            Dim weightsMemorySize As Integer = CInt(accessor.Capacity - 256) ' 256 bytes for header
            Dim weightsMemoryBytes(weightsMemorySize - 1) As Byte
            accessor.ReadArray(Of Byte)(256, weightsMemoryBytes, 0, weightsMemorySize)

            Me.Weights = New TransformerWeights(Me.Config, weightsMemoryBytes)
        End Using

        Me.State = New RunState(Me.Config)
    End Sub

    Public Sub Dispose() Implements IDisposable.Dispose
        _mmf?.Dispose()
        _mmf = Nothing
        GC.SuppressFinalize(Me)
    End Sub

    Public Shared Sub Softmax(x As Single())
        If x Is Nothing OrElse x.Length = 0 Then Return

        Dim maxVal As Single = Single.NegativeInfinity
        For i As Integer = 0 To x.Length - 1
            If x(i) > maxVal Then maxVal = x(i)
        Next

        Dim sum As Single = 0
        For i As Integer = 0 To x.Length - 1
            x(i) = MathF.Exp(x(i) - maxVal)
            sum += x(i)
        Next

        If sum = 0 Then Return
        For i As Integer = 0 To x.Length - 1
            x(i) /= sum
        Next
    End Sub

    Private Shared Sub RmsNorm(o As Single(), x As Single(), weight As Single(), weightOffset As Integer, length As Integer)
        Dim ss As Double = 0
        For i As Integer = 0 To length - 1
            ss += CDbl(x(i)) * x(i)
        Next
        ss /= length
        ss = 1.0 / Math.Sqrt(ss + 1e-6)

        For j As Integer = 0 To length - 1
            o(j) = weight(weightOffset + j) * CSng(ss * x(j))
        Next
    End Sub

    Private Shared Sub Quantize(q As SByte(), s As Single(), x As Single(), length As Integer)
        If Globals.GS = 0 Then Return
        Dim numGroups As Integer = length \ Globals.GS
        Const qMax As Single = 127.0F

        For group As Integer = 0 To numGroups - 1
            Dim groupOffset As Integer = group * Globals.GS

            Dim wmax As Single = 0
            For i As Integer = 0 To Globals.GS - 1
                Dim absVal As Single = MathF.Abs(x(groupOffset + i))
                If absVal > wmax Then wmax = absVal
            Next

            Dim scale As Single = wmax / qMax
            If scale = 0.0F Then scale = 1.0F

            s(group) = scale

            Dim invScale As Single = 1.0F / scale
            For i As Integer = 0 To Globals.GS - 1
                Dim quant_value As Single = x(groupOffset + i) * invScale
                q(groupOffset + i) = CType(Math.Round(quant_value), SByte)
            Next
        Next
    End Sub

    Private Shared Sub Matmul(xout As Single(), xq_q As SByte(), xq_s As Single(), w As QuantizedTensor, n As Integer, d As Integer)
        Dim wQ = w.Q
        Dim wS = w.S

        ' Используем Parallel.For для распараллеливания, как в C#
        Parallel.For(0, d, Sub(i As Integer)
                                Dim val As Single = 0
                                Dim in_offset As Integer = i * n ' Начало строки в матрице весов W

                                For j As Integer = 0 To n - Globals.GS Step Globals.GS
                                    Dim ival As Integer = 0
                                    For k As Integer = 0 To Globals.GS - 1
                                        ival += CInt(xq_q(j + k)) * wQ(in_offset + j + k)
                                    Next
                                    val += ival * wS((in_offset + j) \ Globals.GS) * xq_s(j \ Globals.GS)
                                Next
                                xout(i) = val
                            End Sub)
    End Sub

    ' Перегрузка Matmul для записи в смещение массива xout
    Private Shared Sub Matmul(xout As Single(), xout_offset As Integer, xq_q As SByte(), xq_s As Single(), w As QuantizedTensor, n As Integer, d As Integer)
        Dim wQ = w.Q
        Dim wS = w.S

        Parallel.For(0, d, Sub(i As Integer)
                                Dim val As Single = 0
                                Dim in_offset As Integer = i * n

                                For j As Integer = 0 To n - Globals.GS Step Globals.GS
                                    Dim ival As Integer = 0
                                    For k As Integer = 0 To Globals.GS - 1
                                        ival += CInt(xq_q(j + k)) * wQ(in_offset + j + k)
                                    Next
                                    val += ival * wS((in_offset + j) \ Globals.GS) * xq_s(j \ Globals.GS)
                                Next
                                xout(xout_offset + i) = val
                            End Sub)
    End Sub

    Public Function Forward(token As Integer, pos As Integer) As Single()
        Dim p = Me.Config
        Dim w = Me.Weights
        Dim s = Me.State
        Dim x = s.X ' x является ссылкой на s.X

        Dim dimension As Integer = p.Dimension
        Dim kvDim As Integer = p.NumKVHeads * p.HeadDimension
        Dim kvMul As Integer = p.NumHeads \ p.NumKVHeads
        Dim hiddenDim As Integer = p.HiddenDimension
        Dim allHeadsDim As Integer = p.NumHeads * p.HeadDimension
        Dim headDim As Integer = p.HeadDimension

        ' Copy token embedding
        Array.Copy(w.TokenEmbeddingTable, token * dimension, x, 0, dimension)

        For l As Integer = 0 To p.NumLayers - 1
            Dim loff As Long = CLng(l) * p.SeqLength * kvDim

            ' RmsNorm for attention
            Dim xb_att(dimension - 1) As Single
            RmsNorm(xb_att, x, w.RmsAttWeight, l * dimension, dimension)

            ' Quantize input for attention
            Quantize(s.Xq_q, s.Xq_s, xb_att, dimension)

            ' Q, K, V projections
            Matmul(s.Q, s.Xq_q, s.Xq_s, w.Wq(l), dimension, allHeadsDim)
            Matmul(s.KeyCache, CInt(loff + CLng(pos) * kvDim), s.Xq_q, s.Xq_s, w.Wk(l), dimension, kvDim)
            Matmul(s.ValueCache, CInt(loff + CLng(pos) * kvDim), s.Xq_q, s.Xq_s, w.Wv(l), dimension, kvDim)

            ' Q, K Layernorm and RoPE
            Dim gq_weight(headDim - 1) As Single
            Array.Copy(w.Q_Ln_Weights, l * headDim, gq_weight, 0, headDim)
            Dim gk_weight(headDim - 1) As Single
            Array.Copy(w.K_Ln_Weights, l * headDim, gk_weight, 0, headDim)

            Dim current_k_slice_from_cache(kvDim - 1) As Single
            Array.Copy(s.KeyCache, CInt(loff + CLng(pos) * kvDim), current_k_slice_from_cache, 0, kvDim)

            For h As Integer = 0 To p.NumHeads - 1
                Dim q_head(headDim - 1) As Single
                Array.Copy(s.Q, h * headDim, q_head, 0, headDim)
                RmsNorm(q_head, q_head, gq_weight, 0, headDim)
                ApplyRoPE(q_head, pos, headDim)
                Array.Copy(q_head, 0, s.Q, h * headDim, headDim)
            Next

            For h As Integer = 0 To p.NumKVHeads - 1
                Dim k_head(headDim - 1) As Single
                Array.Copy(current_k_slice_from_cache, h * headDim, k_head, 0, headDim)
                RmsNorm(k_head, k_head, gk_weight, 0, headDim)
                ApplyRoPE(k_head, pos, headDim)
                Array.Copy(k_head, 0, current_k_slice_from_cache, h * headDim, headDim)
            Next
            Array.Copy(current_k_slice_from_cache, 0, s.KeyCache, CInt(loff + CLng(pos) * kvDim), kvDim)


            ' Multi-head Attention
            Array.Clear(s.Xb, 0, s.Xb.Length)
            Dim inv_sqrt_head_dim As Single = 1.0F / MathF.Sqrt(headDim)

            Parallel.For(0, p.NumHeads, Sub(h As Integer)
                                            Dim q_h_offset As Integer = h * headDim
                                            Dim att_h_offset As Integer = h * p.SeqLength
                                            Dim kv_head_idx As Integer = h \ kvMul

                                            For t As Integer = 0 To pos
                                                Dim k_offset As Integer = CInt(loff + CLng(t) * kvDim + CLng(kv_head_idx) * headDim)
                                                Dim score As Double = 0
                                                For i As Integer = 0 To headDim - 1
                                                    score += CDbl(s.Q(q_h_offset + i)) * s.KeyCache(k_offset + i)
                                                Next
                                                s.Att(att_h_offset + t) = CSng(score) * inv_sqrt_head_dim
                                            Next

                                            Dim att_slice(pos) As Single
                                            Array.Copy(s.Att, att_h_offset, att_slice, 0, pos + 1)
                                            Softmax(att_slice)
                                            Array.Copy(att_slice, 0, s.Att, att_h_offset, pos + 1)

                                            Dim xb_h_offset As Integer = h * headDim
                                            For t As Integer = 0 To pos
                                                Dim v_offset As Integer = CInt(loff + CLng(t) * kvDim + CLng(kv_head_idx) * headDim)
                                                Dim a As Single = s.Att(att_h_offset + t)
                                                For i As Integer = 0 To headDim - 1
                                                    s.Xb(xb_h_offset + i) += a * s.ValueCache(v_offset + i)
                                                Next
                                            Next
                                        End Sub)

            ' Output projection
            Quantize(s.Xq_q, s.Xq_s, s.Xb, allHeadsDim)
            Matmul(s.Xb2, s.Xq_q, s.Xq_s, w.Wo(l), allHeadsDim, dimension)
            For i As Integer = 0 To dimension - 1 : x(i) += s.Xb2(i) : Next

            ' FFN RMSNorm
            Dim xb_ffn(dimension - 1) As Single
            RmsNorm(xb_ffn, x, w.RmsFfnWeight, l * dimension, dimension)

            ' Quantize input for FFN
            Quantize(s.Xq_q, s.Xq_s, xb_ffn, dimension)

            ' FFN projections
            Matmul(s.Hb, s.Xq_q, s.Xq_s, w.W1(l), dimension, hiddenDim)
            Matmul(s.Hb2, s.Xq_q, s.Xq_s, w.W3(l), dimension, hiddenDim)

            ' SILU activation
            For i As Integer = 0 To hiddenDim - 1
                Dim val As Single = s.Hb(i)
                val *= 1.0F / (1.0F + MathF.Exp(-val))
                val *= s.Hb2(i)
                s.Hb(i) = val
            Next

            ' Quantize FFN output and project back
            Quantize(s.Hq_q, s.Hq_s, s.Hb, hiddenDim)
            Matmul(s.Xb, s.Hq_q, s.Hq_s, w.W2(l), hiddenDim, dimension)

            For i As Integer = 0 To dimension - 1 : x(i) += s.Xb(i) : Next
        Next

        ' Final RMSNorm
        RmsNorm(x, x, w.RmsFinalWeight, 0, dimension) ' Use offset 0 for final weight

        ' Classifier
        Quantize(s.Xq_q, s.Xq_s, x, dimension)
        Matmul(s.Logits, s.Xq_q, s.Xq_s, w.Wcls, dimension, p.VocabSize)

        Return s.Logits
    End Function

    ' Вспомогательная функция для RoPE
    Private Shared Sub ApplyRoPE(vec As Single(), pos As Integer, headDim As Integer)
        For j As Integer = 0 To headDim \ 2 - 1
            Dim freq As Single = MathF.Pow(1000000.0F, -CSng(j) / (headDim / 2.0F))
            Dim val As Single = pos * freq
            Dim sin_freq As Single = MathF.Sin(val)
            Dim cos_freq As Single = MathF.Cos(val)

            Dim real As Single = vec(j)
            Dim imag As Single = vec(j + headDim \ 2)

            vec(j) = real * cos_freq - imag * sin_freq
            vec(j + headDim \ 2) = real * sin_freq + imag * cos_freq
        Next
    End Sub

End Class


' ---------- КЛАСС ТОКЕНИЗАТОРА ----------
Public Class Tokenizer
    Public ReadOnly Property VocabSize As Integer
    Public ReadOnly Property BosTokenId As UInteger
    Public ReadOnly Property EosTokenId As UInteger
    Public ReadOnly Property PromptTemplate As String
    Public ReadOnly Property SystemPromptTemplate As String
    Private ReadOnly _vocab As String()
    Private ReadOnly _mergeScores As Single()
    Private ReadOnly _vocabDict As Dictionary(Of String, Integer)

    ' Специфичные токены Qwen для остановки генерации
    Public ReadOnly Property ImEndId As Integer

    Public Sub New(checkpointPath As String, configVocabSize As Integer, enableThinking As Boolean)
        Dim tokenizerPath = $"{checkpointPath}.tokenizer"
        If Not File.Exists(tokenizerPath) Then
            Throw New FileNotFoundException("Tokenizer file not found", tokenizerPath)
        End If

        Dim tempVocab As New List(Of String)()
        Dim tempScores As New List(Of Single)()

        Using reader As New BinaryReader(File.OpenRead(tokenizerPath))
            reader.ReadUInt32() ' max_token_length, not used
            Me.BosTokenId = reader.ReadUInt32()
            Me.EosTokenId = reader.ReadUInt32()

            While reader.BaseStream.Position < reader.BaseStream.Length
                Dim score As Single = 0
                If reader.BaseStream.Position + 4 <= reader.BaseStream.Length Then
                    score = reader.ReadSingle()
                Else
                    Exit While ' Reached end of stream before score
                End If

                Dim len As Integer = 0
                If reader.BaseStream.Position + 4 <= reader.BaseStream.Length Then
                    len = reader.ReadInt32()
                Else
                    Exit While ' Reached end of stream before length
                End If

                If len > 0 Then
                    If reader.BaseStream.Position + len > reader.BaseStream.Length Then
                        ' Not enough bytes for token, probably corrupted file or incomplete
                        Exit While
                    End If
                    Dim tokenBytes As Byte() = reader.ReadBytes(len)
                    tempVocab.Add(Encoding.UTF8.GetString(tokenBytes))
                Else
                    ' Токен может быть пустой строкой
                    tempVocab.Add("")
                End If
                tempScores.Add(score)
            End While
        End Using

        Me.VocabSize = tempVocab.Count
        Me._vocab = tempVocab.ToArray()
        Me._mergeScores = tempScores.ToArray()

        Me._vocabDict = New Dictionary(Of String, Integer)()
        For i As Integer = 0 To Me.VocabSize - 1
            If Not String.IsNullOrEmpty(Me._vocab(i)) Then
                Me._vocabDict(Me._vocab(i)) = i
            End If
        Next

        If Me.VocabSize <> configVocabSize Then
            Console.WriteLine($"Warning: vocab_size in config ({configVocabSize}) does not match the actual number of tokens in tokenizer ({Me.VocabSize}).")
        End If

        ' Загрузка шаблонов промптов без замены %s на {0},
        ' так как Chat метод будет обрабатывать %s напрямую.
        Me.PromptTemplate = LoadPromptTemplate(checkpointPath, withSystemPrompt:=False, enableThinking:=enableThinking)
        Me.SystemPromptTemplate = LoadPromptTemplate(checkpointPath, withSystemPrompt:=True, enableThinking:=enableThinking)

        ' Qwen specific stop token
        If _vocabDict.ContainsKey("<|im_end|>") Then
            Me.ImEndId = _vocabDict("<|im_end|>")
        Else
            Me.ImEndId = -1 ' Not found
        End If
    End Sub

    Private Function LoadPromptTemplate(checkpointPath As String, withSystemPrompt As Boolean, enableThinking As Boolean) As String
        Dim suffix As String = ""
        If withSystemPrompt Then
            suffix = If(enableThinking, ".template.with-system-and-thinking", ".template.with-system")
        Else
            suffix = If(enableThinking, ".template.with-thinking", ".template")
        End If

        Dim path As String = $"{checkpointPath}{suffix}"
        If Not File.Exists(path) Then
            ' Fallback to non-thinking template if thinking version not found
            If enableThinking Then
                If withSystemPrompt Then
                    path = $"{checkpointPath}.template.with-system"
                Else
                    path = $"{checkpointPath}.template"
                End If
                If Not File.Exists(path) Then
                    Throw New FileNotFoundException($"Could not load any prompt template for path: {checkpointPath}")
                End If
            Else
                Throw New FileNotFoundException($"Could not load prompt template: {path}")
            End If
        End If

        Return File.ReadAllText(path).Replace(vbCrLf, vbLf) ' Normalize line endings
    End Function

    Public Function Decode(token As Integer) As String
        Return If(token >= 0 AndAlso token < _vocab.Length, _vocab(token), "")
    End Function

    ' --- ИСПРАВЛЕНИЕ: Полностью переписанный метод Encode ---
  Public Function Encode(text As String) As Integer()
    Dim tokens As New List(Of Integer)()
    Dim utf8Bytes As Byte() = Encoding.UTF8.GetBytes(text)

    ' 1. Начальная токенизация: разбиваем входной текст на токены.
    ' Сначала ищем специальные токены, затем обрабатываем побайтово.
    Dim i As Integer = 0
    While i < utf8Bytes.Length
        ' Поиск специальных токенов (например, <|im_start|>)
        If utf8Bytes(i) = CByte(AscW("<"c)) Then
            Dim endTokenPos As Integer = -1
            ' Ищем закрывающую скобку в разумных пределах (например, 64 байта)
            For k As Integer = i + 1 To Math.Min(i + 63, utf8Bytes.Length - 1)
                If utf8Bytes(k) = CByte(AscW(">"c)) Then
                    endTokenPos = k
                    Exit For
                End If
            Next

            If endTokenPos <> -1 Then
                ' Извлекаем потенциальный специальный токен
                Dim specialTokenLength As Integer = endTokenPos - i + 1
                Dim specialTokenBytes(specialTokenLength - 1) As Byte
                Array.Copy(utf8Bytes, i, specialTokenBytes, 0, specialTokenLength)
                Dim specialTokenStr As String = Encoding.UTF8.GetString(specialTokenBytes)

                Dim id As Integer
                If _vocabDict.TryGetValue(specialTokenStr, id) Then
                    tokens.Add(id)
                    i += specialTokenLength ' Пропускаем весь специальный токен
                    Continue While ' Переходим к следующей итерации основного цикла
                End If
            End If
        End If

        ' Если специальный токен не найден, обрабатываем как один байт
        Dim singleByteTokenStr As String = Char.ConvertFromUtf32(utf8Bytes(i))
        Dim byteId As Integer
        If _vocabDict.TryGetValue(singleByteTokenStr, byteId) Then
            tokens.Add(byteId)
        Else
            ' В оригинальном C-коде нет токенов для всех 256 байт,
            ' поэтому здесь может быть ошибка, если байт не найден.
            ' Для Qwen-tokenizer это маловероятно, но лучше быть готовым.
            Console.Error.WriteLine($"Warning: character for byte {utf8Bytes(i)} not found in vocab, skipping.")
        End If
        i += 1
    End While

    ' 2. Цикл слияния BPE
    While True
        Dim bestScore As Single = Single.NegativeInfinity
        Dim bestIdx As Integer = -1
        Dim bestId As Integer = -1

        If tokens.Count < 2 Then
            Exit While
        End If

        For idx As Integer = 0 To tokens.Count - 2
            ' Формируем строку для объединенного токена
            Dim mergedStr As String = _vocab(tokens(idx)) & _vocab(tokens(idx + 1))
            
            ' ИСПОЛЬЗУЕМ ВРЕМЕННУЮ ПЕРЕМЕННУЮ `currentId`
            Dim currentId As Integer
            If _vocabDict.TryGetValue(mergedStr, currentId) Then
                ' Сравниваем счет текущей найденной пары
                If _mergeScores(currentId) > bestScore Then
                    bestScore = _mergeScores(currentId)
                    bestIdx = idx
                    bestId = currentId ' СОХРАНЯЕМ ID ЛУЧШЕЙ ПАРЫ
                End If
            End If
        Next

        ' Если не найдено пар для слияния, выходим
        If bestIdx = -1 Then
            Exit While
        End If

        ' Выполняем слияние
        tokens(bestIdx) = bestId
        tokens.RemoveAt(bestIdx + 1)
    End While

    Return tokens.ToArray()
End Function
End Class


' ---------- КЛАСС СЭМПЛЕРА ----------
Public Class Sampler
    Private ReadOnly _vocabSize As Integer
    Private ReadOnly _temperature As Single
    Private ReadOnly _topp As Single
    Private _rngState As ULong ' Uses ULong to match C# unsigned long
    Private ReadOnly _probIndex As ProbIndex()

    Private Structure ProbIndex
        Public Prob As Single
        Public Index As Integer
    End Structure

    Public Sub New(vocabSize As Integer, temperature As Single, topp As Single, rngSeed As ULong)
        Me._vocabSize = vocabSize
        Me._temperature = temperature
        Me._topp = topp
        Me._rngState = rngSeed
        Me._probIndex = New ProbIndex(vocabSize - 1) {}
    End Sub

    Private Function RandomU32() As UInteger
        _rngState = _rngState Xor (_rngState >> 12)
        _rngState = _rngState Xor (_rngState << 25)
        _rngState = _rngState Xor (_rngState >> 27)

        ' ИСПРАВЛЕНИЕ 2: Предыдущее исправление с Decimal вызывало ошибку, так как
        ' результат умножения двух ULong может превысить максимальное значение Decimal.
        ' Используем System.Numerics.BigInteger, который может обрабатывать числа
        ' произвольного размера и является правильным способом для выполнения
        ' 128-битной арифметики.

        ' Создаем BigInteger из наших 64-битных чисел.
        Dim biState As New BigInteger(_rngState)
        Dim biConst As New BigInteger(&H2545F4914F6CDD1DUL)

        ' Выполняем умножение. Результат будет 128-битным.
        Dim biProduct As BigInteger = biState * biConst

        ' Чтобы имитировать "обертывание" (wrap-around), мы берем только
        ' младшие 64 бита результата. Это достигается с помощью побитового "И"
        ' с маской, равной ULong.MaxValue.
        Dim biMask As New BigInteger(ULong.MaxValue)
        Dim wrappedProduct As ULong = CULng(biProduct And biMask)

        ' Теперь выполняем сдвиг вправо, как и предполагалось в алгоритме.
        Dim resultULong As ULong = wrappedProduct >> 32

        ' Преобразования в UInteger достаточно для получения нужного результата.
        Return CType(resultULong, UInteger)
    End Function

    Private Function RandomF32() As Single
        Return CType(RandomU32() >> 8, Single) / 16777216.0F ' 2^24
    End Function

    Private Function SampleArgmax(probabilities As Single()) As Integer
        Dim max_i As Integer = 0
        Dim max_p As Single = probabilities(0)
        For i As Integer = 1 To probabilities.Length - 1
            If probabilities(i) > max_p Then
                max_i = i
                max_p = probabilities(i)
            End If
        Next
        Return max_i
    End Function

    Private Function SampleMult(probabilities As Single(), coin As Single) As Integer
        Dim cdf As Single = 0
        For i As Integer = 0 To probabilities.Length - 1
            cdf += probabilities(i)
            If coin < cdf Then Return i
        Next
        Return probabilities.Length - 1
    End Function

    Private Function SampleTopp(probabilities As Single(), coin As Single) As Integer
        Dim probIndexLocal(probabilities.Length - 1) As ProbIndex
        Dim n0 As Integer = 0

        Dim cutoff As Single = (1.0F - _topp) / (Me._vocabSize - 1)
        For i As Integer = 0 To probabilities.Length - 1
            If probabilities(i) >= cutoff Then
                probIndexLocal(n0).Index = i
                probIndexLocal(n0).Prob = probabilities(i)
                n0 += 1
            End If
        Next

        Array.Sort(probIndexLocal, 0, n0, Comparer(Of ProbIndex).Create(Function(a, b) b.Prob.CompareTo(a.Prob)))

        Dim cdf_cumulative_sum As Single = 0 ' Cumulative sum for Top-P truncation
        Dim lastIdx As Integer = n0 - 1
        For i As Integer = 0 To n0 - 1
            cdf_cumulative_sum += probIndexLocal(i).Prob
            If cdf_cumulative_sum > _topp Then
                lastIdx = i
                Exit For
            End If
        Next

        Dim r_sample As Single = coin * cdf_cumulative_sum
        Dim cdf_current_sum As Single = 0
        For i As Integer = 0 To lastIdx
            cdf_current_sum += probIndexLocal(i).Prob
            If r_sample < cdf_current_sum Then Return probIndexLocal(i).Index
        Next
        Return probIndexLocal(lastIdx).Index
    End Function

    Public Function Sample(logits As Single()) As Integer
        Dim relevantLogits(Me._vocabSize - 1) As Single
        Array.Copy(logits, 0, relevantLogits, 0, Me._vocabSize)

        If _temperature = 0.0F Then
            Return SampleArgmax(relevantLogits)
        End If

        For q As Integer = 0 To relevantLogits.Length - 1
            relevantLogits(q) /= _temperature
        Next

        Transformer.Softmax(relevantLogits)

        Dim coin As Single = RandomF32()

        If _topp <= 0 OrElse _topp >= 1 Then
            Return SampleMult(relevantLogits, coin)
        Else
            Return SampleTopp(relevantLogits, coin)
        End If
    End Function
End Class


' ---------- ГЛАВНАЯ ПРОГРАММА И ЦИКЛЫ ГЕНЕРАЦИИ ----------
Module RunQ
    Sub Generate(transformer As Transformer, tokenizer As Tokenizer, sampler As Sampler, prompt As String)
        Dim promptTokens As Integer() = tokenizer.Encode(prompt)
        Dim numPromptTokens As Integer = promptTokens.Length
        If numPromptTokens < 1 Then
            Console.Error.WriteLine("Cannot encode prompt.")
            Return
        End If

        Dim token As Integer = promptTokens(0)
        Dim pos As Integer = 0
        Dim sw As Stopwatch = Stopwatch.StartNew()

        While pos < transformer.Config.SeqLength
            Dim logits As Single() = transformer.Forward(token, pos)
            Dim next_val As Integer ' Renamed from 'next' to avoid reserved keyword
            If pos < numPromptTokens - 1 Then
                next_val = promptTokens(pos + 1)
            Else
                next_val = sampler.Sample(logits)
            End If

            Console.Write(tokenizer.Decode(token))

            pos += 1
            token = next_val

            If pos >= numPromptTokens AndAlso (token = tokenizer.EosTokenId OrElse token = tokenizer.BosTokenId) Then
                Exit While
            End If
        End While
        sw.Stop()
        Console.WriteLine($" {vbCrLf}{vbCrLf}Achieved: {If(pos > 1, (pos - 1) / sw.Elapsed.TotalSeconds, 0):F2} tokens/sec")
    End Sub

    Sub Chat(transformer As Transformer, tokenizer As Tokenizer, sampler As Sampler, cliUserPrompt As String, systemPrompt As String)
        Dim pos As Integer = 0
        Dim userTurn As Boolean = True
        Dim promptTokens As Integer() = {}
        Dim userIdx As Integer = 0
        Dim token As Integer = 0
        Dim next_token As Integer = -1
        Dim stopTokens As New HashSet(Of Integer) From {
            CInt(tokenizer.BosTokenId),
            CInt(tokenizer.EosTokenId)
        }
        If tokenizer.ImEndId <> -1 Then
            stopTokens.Add(tokenizer.ImEndId)
        End If

        While True
            If pos >= transformer.Config.SeqLength Then
                Console.WriteLine(vbCrLf & "(context window full, clearing)" & vbCrLf)
                pos = 0
                userTurn = True
            End If

            If userTurn Then
                Dim userPrompt As String
                If Not String.IsNullOrEmpty(cliUserPrompt) AndAlso pos = 0 Then
                    userPrompt = cliUserPrompt
                    cliUserPrompt = Nothing
                Else
                    Try
                        Console.Write(vbCrLf & "> ")
                        userPrompt = Console.ReadLine()
                        If String.IsNullOrWhiteSpace(userPrompt) Then Exit While
                    Catch ex As Exception When TypeOf ex Is EndOfStreamException OrElse TypeOf ex Is OperationCanceledException
                        Console.WriteLine()
                        Exit While
                    End Try
                End If

                Dim renderedPrompt As String
                Dim template As String

                If pos = 0 AndAlso Not String.IsNullOrEmpty(systemPrompt) Then
                    template = tokenizer.SystemPromptTemplate
                    Dim placeholderIndex As Integer = template.IndexOf("%s")
                    If placeholderIndex <> -1 Then
                        template = template.Remove(placeholderIndex, 2).Insert(placeholderIndex, systemPrompt)
                    End If
                    renderedPrompt = template.Replace("%s", userPrompt)
                Else
                    template = tokenizer.PromptTemplate
                    renderedPrompt = template.Replace("%s", userPrompt)
                End If

                promptTokens = tokenizer.Encode(renderedPrompt)
                userIdx = 0
                userTurn = False
                Console.Write("< ")
            End If

            If userIdx < promptTokens.Length Then
                token = promptTokens(userIdx)
                userIdx += 1
            Else
                token = next_token
            End If

            If pos >= transformer.Config.SeqLength Then Exit While

            Dim logits As Single() = transformer.Forward(token, pos)
            next_token = sampler.Sample(logits)
            pos += 1

            If userIdx >= promptTokens.Length Then
                If stopTokens.Contains(next_token) Then
                    userTurn = True
                Else
                    Console.Write(tokenizer.Decode(next_token))
                    Console.Out.Flush()
                End If
            End If
        End While
    End Sub

    Sub Main()
        Console.OutputEncoding = Encoding.UTF8
        Dim argsList As List(Of String) = Environment.GetCommandLineArgs().Skip(1).ToList()

        Dim checkpointPath As String = ""
        Dim temperature As Single = 1.0F
        Dim topp As Single = 0.9F
        Dim seed As ULong = 0UL
        Dim ctxLength As Integer = 0
        Dim mode As String = "chat"
        Dim prompt As String = Nothing
        Dim systemPrompt As String = Nothing
        Dim enableThinking As Boolean = False

        If argsList.Count < 1 Then
            ErrorUsage()
            Return
        End If

        checkpointPath = argsList(0)

        Dim i As Integer = 1
        While i < argsList.Count
            Dim flag As String = argsList(i)
            If i + 1 >= argsList.Count OrElse Not flag.StartsWith("-") OrElse flag.Length <> 2 Then
                ErrorUsage()
                Return
            End If

            Dim value As String = argsList(i + 1)
            Select Case flag(1)
                Case "t" : temperature = Single.Parse(value, Globalization.CultureInfo.InvariantCulture)
                Case "p" : topp = Single.Parse(value, Globalization.CultureInfo.InvariantCulture)
                Case "s" : seed = ULong.Parse(value)
                Case "c" : ctxLength = Integer.Parse(value)
                Case "i" : prompt = value
                Case "m" : mode = value
                Case "y" : systemPrompt = value
                Case "r" : enableThinking = (Integer.Parse(value) = 1)
                Case Else : ErrorUsage() : Return
            End Select
            i += 2
        End While

        If seed = 0UL Then seed = CULng(DateTime.Now.Ticks)
        If temperature < 0 Then temperature = 0
        If topp < 0 OrElse topp > 1 Then topp = 0.9F

        Try
            Using transformer As New Transformer(checkpointPath, ctxLength)
                Dim tokenizer As New Tokenizer(checkpointPath, transformer.Config.VocabSize, enableThinking)

                Dim sampler As New Sampler(tokenizer.VocabSize, temperature, topp, seed)

                If String.IsNullOrEmpty(prompt) Then
                    Dim p = transformer.Config
                    Console.WriteLine($"hidden_size={p.Dimension}, intermediate_size={p.HiddenDimension}, num_hidden_layers={p.NumLayers}, " &
                                      $"num_attention_heads={p.NumHeads}, num_kv_heads={p.NumKVHeads}, head_dim={p.HeadDimension}, " &
                                      $"ctx_length={p.SeqLength}, vocab_size={tokenizer.VocabSize}, shared_classifier={p.SharedClassifier}, " &
                                      $"quantization_block_size={p.GroupSize}")
                End If

                If mode.Equals("generate", StringComparison.OrdinalIgnoreCase) Then
                    If String.IsNullOrEmpty(prompt) Then
                        Console.Error.WriteLine("Generate mode requires a prompt (-i).")
                        Environment.Exit(1)
                    End If
                    Generate(transformer, tokenizer, sampler, prompt)
                ElseIf mode.Equals("chat", StringComparison.OrdinalIgnoreCase) Then
                    Chat(transformer, tokenizer, sampler, prompt, systemPrompt)
                Else
                    Console.Error.WriteLine($"Unknown mode: {mode}")
                    ErrorUsage()
                End If
            End Using
        Catch ex As Exception
            Console.Error.WriteLine($"An error occurred: {ex.Message}")
            Console.Error.WriteLine(ex.StackTrace)
            Environment.Exit(1)
        End Try
    End Sub

    Sub ErrorUsage()
        Console.Error.WriteLine("Usage:   dotnet run -- <checkpoint> [options]")
        Console.Error.WriteLine("Example: dotnet run -- Qwen3-4B.bin -r 1")
        Console.Error.WriteLine("Options:")
        ' ИСПРАВЛЕНИЕ: Разбиваем строки, чтобы избежать ошибок парсера XML-литералов
        Console.Error.WriteLine("  -t " & "<float>" & "  temperature in [0,inf], default 1.0")
        Console.Error.WriteLine("  -p " & "<float>" & "  p value in top-p (nucleus) sampling in [0,1], default 0.9")
        Console.Error.WriteLine("  -s " & "<int>" & "    random seed, default time(NULL)")
        Console.Error.WriteLine("  -c " & "<int>" & "    context window size, 0 (default) = max_seq_len")
        Console.Error.WriteLine("  -m " & "<string>" & " mode: generate|chat, default: chat")
        Console.Error.WriteLine("  -i " & "<string>" & " input prompt")
        Console.Error.WriteLine("  -y " & "<string>" & " system prompt in chat mode, default is none")
        Console.Error.WriteLine("  -r " & "<int>" & "    reasoning mode, 0 (default) = no thinking, 1 = thinking")
        Environment.Exit(1)
    End Sub
End Module