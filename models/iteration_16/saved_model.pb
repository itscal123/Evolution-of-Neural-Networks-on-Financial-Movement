à×3
³
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¢ý0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0

rnn/peephole_lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_namernn/peephole_lstm_cell/kernel

1rnn/peephole_lstm_cell/kernel/Read/ReadVariableOpReadVariableOprnn/peephole_lstm_cell/kernel*
_output_shapes
:	*
dtype0
«
'rnn/peephole_lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *8
shared_name)'rnn/peephole_lstm_cell/recurrent_kernel
¤
;rnn/peephole_lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp'rnn/peephole_lstm_cell/recurrent_kernel*
_output_shapes
:	 *
dtype0

rnn/peephole_lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namernn/peephole_lstm_cell/bias

/rnn/peephole_lstm_cell/bias/Read/ReadVariableOpReadVariableOprnn/peephole_lstm_cell/bias*
_output_shapes	
:*
dtype0
¼
2rnn/peephole_lstm_cell/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42rnn/peephole_lstm_cell/input_gate_peephole_weights
µ
Frnn/peephole_lstm_cell/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2rnn/peephole_lstm_cell/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¾
3rnn/peephole_lstm_cell/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53rnn/peephole_lstm_cell/forget_gate_peephole_weights
·
Grnn/peephole_lstm_cell/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp3rnn/peephole_lstm_cell/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¾
3rnn/peephole_lstm_cell/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53rnn/peephole_lstm_cell/output_gate_peephole_weights
·
Grnn/peephole_lstm_cell/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp3rnn/peephole_lstm_cell/output_gate_peephole_weights*
_output_shapes
: *
dtype0

!rnn_1/peephole_lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *2
shared_name#!rnn_1/peephole_lstm_cell_1/kernel

5rnn_1/peephole_lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOp!rnn_1/peephole_lstm_cell_1/kernel*
_output_shapes
:	 *
dtype0
³
+rnn_1/peephole_lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *<
shared_name-+rnn_1/peephole_lstm_cell_1/recurrent_kernel
¬
?rnn_1/peephole_lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp+rnn_1/peephole_lstm_cell_1/recurrent_kernel*
_output_shapes
:	 *
dtype0

rnn_1/peephole_lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!rnn_1/peephole_lstm_cell_1/bias

3rnn_1/peephole_lstm_cell_1/bias/Read/ReadVariableOpReadVariableOprnn_1/peephole_lstm_cell_1/bias*
_output_shapes	
:*
dtype0
Ä
6rnn_1/peephole_lstm_cell_1/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights
½
Jrnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp6rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights*
_output_shapes
: *
dtype0
Æ
7rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights
¿
Krnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp7rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
Æ
7rnn_1/peephole_lstm_cell_1/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights
¿
Krnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp7rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

RMSprop/conv1d/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv1d/kernel/rms

-RMSprop/conv1d/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1d/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/conv1d/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/conv1d/bias/rms

+RMSprop/conv1d/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1d/bias/rms*
_output_shapes
:*
dtype0

RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameRMSprop/dense/kernel/rms

,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:*
dtype0
¯
)RMSprop/rnn/peephole_lstm_cell/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*:
shared_name+)RMSprop/rnn/peephole_lstm_cell/kernel/rms
¨
=RMSprop/rnn/peephole_lstm_cell/kernel/rms/Read/ReadVariableOpReadVariableOp)RMSprop/rnn/peephole_lstm_cell/kernel/rms*
_output_shapes
:	*
dtype0
Ã
3RMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *D
shared_name53RMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rms
¼
GRMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp3RMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
§
'RMSprop/rnn/peephole_lstm_cell/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'RMSprop/rnn/peephole_lstm_cell/bias/rms
 
;RMSprop/rnn/peephole_lstm_cell/bias/rms/Read/ReadVariableOpReadVariableOp'RMSprop/rnn/peephole_lstm_cell/bias/rms*
_output_shapes	
:*
dtype0
Ô
>RMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rms
Í
RRMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ö
?RMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?RMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rms
Ï
SRMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp?RMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ö
?RMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?RMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rms
Ï
SRMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp?RMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
·
-RMSprop/rnn_1/peephole_lstm_cell_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *>
shared_name/-RMSprop/rnn_1/peephole_lstm_cell_1/kernel/rms
°
ARMSprop/rnn_1/peephole_lstm_cell_1/kernel/rms/Read/ReadVariableOpReadVariableOp-RMSprop/rnn_1/peephole_lstm_cell_1/kernel/rms*
_output_shapes
:	 *
dtype0
Ë
7RMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *H
shared_name97RMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rms
Ä
KRMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp7RMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¯
+RMSprop/rnn_1/peephole_lstm_cell_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+RMSprop/rnn_1/peephole_lstm_cell_1/bias/rms
¨
?RMSprop/rnn_1/peephole_lstm_cell_1/bias/rms/Read/ReadVariableOpReadVariableOp+RMSprop/rnn_1/peephole_lstm_cell_1/bias/rms*
_output_shapes	
:*
dtype0
Ü
BRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *S
shared_nameDBRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rms
Õ
VRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOpBRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Þ
CRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rms
×
WRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOpCRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Þ
CRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rms
×
WRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOpCRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0

NoOpNoOp
£B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÞA
valueÔABÑA BÊA

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
regularization_losses
 trainable_variables
!	keras_api
h

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
ñ
(iter
	)decay
*learning_rate
+momentum
,rho	rmsr	rmss	"rmst	#rmsu	-rmsv	.rmsw	/rmsx	0rmsy	1rmsz	2rms{	3rms|	4rms}	5rms~	6rms
7rms
8rms
v
0
1
-2
.3
/4
05
16
27
38
49
510
611
712
813
"14
#15
 
v
0
1
-2
.3
/4
05
16
27
38
49
510
611
712
813
"14
#15
­
9non_trainable_variables
	variables

:layers
regularization_losses
;layer_metrics
<metrics
=layer_regularization_losses
	trainable_variables
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
>non_trainable_variables
?layer_metrics

@layers
	variables
regularization_losses
Ametrics
Blayer_regularization_losses
trainable_variables
 
 
 
­
Cnon_trainable_variables
Dlayer_metrics

Elayers
	variables
regularization_losses
Fmetrics
Glayer_regularization_losses
trainable_variables
ó
H
state_size

-kernel
.recurrent_kernel
/bias
0input_gate_peephole_weights
 1forget_gate_peephole_weights
 2output_gate_peephole_weights
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
 
*
-0
.1
/2
03
14
25
 
*
-0
.1
/2
03
14
25
¹
Mnon_trainable_variables
	variables

Nlayers
Olayer_metrics
regularization_losses

Pstates
Qmetrics
Rlayer_regularization_losses
trainable_variables
ó
S
state_size

3kernel
4recurrent_kernel
5bias
6input_gate_peephole_weights
 7forget_gate_peephole_weights
 8output_gate_peephole_weights
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
 
*
30
41
52
63
74
85
 
*
30
41
52
63
74
85
¹
Xnon_trainable_variables
	variables

Ylayers
Zlayer_metrics
regularization_losses

[states
\metrics
]layer_regularization_losses
 trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
^non_trainable_variables
_layer_metrics

`layers
$	variables
%regularization_losses
ametrics
blayer_regularization_losses
&trainable_variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErnn/peephole_lstm_cell/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'rnn/peephole_lstm_cell/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUErnn/peephole_lstm_cell/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2rnn/peephole_lstm_cell/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3rnn/peephole_lstm_cell/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3rnn/peephole_lstm_cell/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!rnn_1/peephole_lstm_cell_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+rnn_1/peephole_lstm_cell_1/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUErnn_1/peephole_lstm_cell_1/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 

c0
 
 
 
 
 
 
 
 
 
 
 
 
*
-0
.1
/2
03
14
25
 
*
-0
.1
/2
03
14
25
­
dnon_trainable_variables
elayer_metrics

flayers
I	variables
Jregularization_losses
gmetrics
hlayer_regularization_losses
Ktrainable_variables
 

0
 
 
 
 
 
*
30
41
52
63
74
85
 
*
30
41
52
63
74
85
­
inon_trainable_variables
jlayer_metrics

klayers
T	variables
Uregularization_losses
lmetrics
mlayer_regularization_losses
Vtrainable_variables
 

0
 
 
 
 
 
 
 
 
 
4
	ntotal
	ocount
p	variables
q	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

n0
o1

p	variables

VARIABLE_VALUERMSprop/conv1d/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/conv1d/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)RMSprop/rnn/peephole_lstm_cell/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3RMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'RMSprop/rnn/peephole_lstm_cell/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?RMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?RMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-RMSprop/rnn_1/peephole_lstm_cell_1/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE7RMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+RMSprop/rnn_1/peephole_lstm_cell_1/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEBRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv1d_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
¯
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_inputconv1d/kernelconv1d/biasrnn/peephole_lstm_cell/kernel'rnn/peephole_lstm_cell/recurrent_kernelrnn/peephole_lstm_cell/bias2rnn/peephole_lstm_cell/input_gate_peephole_weights3rnn/peephole_lstm_cell/forget_gate_peephole_weights3rnn/peephole_lstm_cell/output_gate_peephole_weights!rnn_1/peephole_lstm_cell_1/kernel+rnn_1/peephole_lstm_cell_1/recurrent_kernelrnn_1/peephole_lstm_cell_1/bias6rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights7rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights7rnn_1/peephole_lstm_cell_1/output_gate_peephole_weightsdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1667900
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp1rnn/peephole_lstm_cell/kernel/Read/ReadVariableOp;rnn/peephole_lstm_cell/recurrent_kernel/Read/ReadVariableOp/rnn/peephole_lstm_cell/bias/Read/ReadVariableOpFrnn/peephole_lstm_cell/input_gate_peephole_weights/Read/ReadVariableOpGrnn/peephole_lstm_cell/forget_gate_peephole_weights/Read/ReadVariableOpGrnn/peephole_lstm_cell/output_gate_peephole_weights/Read/ReadVariableOp5rnn_1/peephole_lstm_cell_1/kernel/Read/ReadVariableOp?rnn_1/peephole_lstm_cell_1/recurrent_kernel/Read/ReadVariableOp3rnn_1/peephole_lstm_cell_1/bias/Read/ReadVariableOpJrnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/Read/ReadVariableOpKrnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/Read/ReadVariableOpKrnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-RMSprop/conv1d/kernel/rms/Read/ReadVariableOp+RMSprop/conv1d/bias/rms/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp=RMSprop/rnn/peephole_lstm_cell/kernel/rms/Read/ReadVariableOpGRMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rms/Read/ReadVariableOp;RMSprop/rnn/peephole_lstm_cell/bias/rms/Read/ReadVariableOpRRMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rms/Read/ReadVariableOpSRMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rms/Read/ReadVariableOpSRMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rms/Read/ReadVariableOpARMSprop/rnn_1/peephole_lstm_cell_1/kernel/rms/Read/ReadVariableOpKRMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rms/Read/ReadVariableOp?RMSprop/rnn_1/peephole_lstm_cell_1/bias/rms/Read/ReadVariableOpVRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rms/Read/ReadVariableOpWRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rms/Read/ReadVariableOpWRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1670849

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasdense/kernel
dense/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhornn/peephole_lstm_cell/kernel'rnn/peephole_lstm_cell/recurrent_kernelrnn/peephole_lstm_cell/bias2rnn/peephole_lstm_cell/input_gate_peephole_weights3rnn/peephole_lstm_cell/forget_gate_peephole_weights3rnn/peephole_lstm_cell/output_gate_peephole_weights!rnn_1/peephole_lstm_cell_1/kernel+rnn_1/peephole_lstm_cell_1/recurrent_kernelrnn_1/peephole_lstm_cell_1/bias6rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights7rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights7rnn_1/peephole_lstm_cell_1/output_gate_peephole_weightstotalcountRMSprop/conv1d/kernel/rmsRMSprop/conv1d/bias/rmsRMSprop/dense/kernel/rmsRMSprop/dense/bias/rms)RMSprop/rnn/peephole_lstm_cell/kernel/rms3RMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rms'RMSprop/rnn/peephole_lstm_cell/bias/rms>RMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rms?RMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rms?RMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rms-RMSprop/rnn_1/peephole_lstm_cell_1/kernel/rms7RMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rms+RMSprop/rnn_1/peephole_lstm_cell_1/bias/rmsBRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rmsCRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rmsCRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rms*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1670976Ö­/

 
,__inference_sequential_layer_call_fn_1667773
conv1d_input
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:	 
	unknown_3:	
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:	 
	unknown_8:	 
	unknown_9:	

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16677012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv1d_input
r
ø
B__inference_rnn_1_layer_call_and_return_conditional_losses_1667101

inputsF
3peephole_lstm_cell_1_matmul_readvariableop_resource:	 H
5peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 C
4peephole_lstm_cell_1_biasadd_readvariableop_resource:	:
,peephole_lstm_cell_1_readvariableop_resource: <
.peephole_lstm_cell_1_readvariableop_1_resource: <
.peephole_lstm_cell_1_readvariableop_2_resource: 
identity¢+peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢*peephole_lstm_cell_1/MatMul/ReadVariableOp¢,peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢#peephole_lstm_cell_1/ReadVariableOp¢%peephole_lstm_cell_1/ReadVariableOp_1¢%peephole_lstm_cell_1/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2Í
*peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp3peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell_1/MatMul/ReadVariableOpÅ
peephole_lstm_cell_1/MatMulMatMulstrided_slice_2:output:02peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMulÓ
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp5peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02.
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpÁ
peephole_lstm_cell_1/MatMul_1MatMulzeros:output:04peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMul_1À
peephole_lstm_cell_1/addAddV2%peephole_lstm_cell_1/MatMul:product:0'peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/addÌ
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp4peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpÍ
peephole_lstm_cell_1/BiasAddBiasAddpeephole_lstm_cell_1/add:z:03peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/BiasAdd
$peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$peephole_lstm_cell_1/split/split_dim
peephole_lstm_cell_1/splitSplit-peephole_lstm_cell_1/split/split_dim:output:0%peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell_1/split³
#peephole_lstm_cell_1/ReadVariableOpReadVariableOp,peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell_1/ReadVariableOp¬
peephole_lstm_cell_1/mulMul+peephole_lstm_cell_1/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul¶
peephole_lstm_cell_1/add_1AddV2#peephole_lstm_cell_1/split:output:0peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_1
peephole_lstm_cell_1/SigmoidSigmoidpeephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Sigmoid¹
%peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp.peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_1²
peephole_lstm_cell_1/mul_1Mul-peephole_lstm_cell_1/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_1¸
peephole_lstm_cell_1/add_2AddV2#peephole_lstm_cell_1/split:output:1peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_2
peephole_lstm_cell_1/Sigmoid_1Sigmoidpeephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_1§
peephole_lstm_cell_1/mul_2Mul"peephole_lstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_2
peephole_lstm_cell_1/TanhTanh#peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh²
peephole_lstm_cell_1/mul_3Mul peephole_lstm_cell_1/Sigmoid:y:0peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_3³
peephole_lstm_cell_1/add_3AddV2peephole_lstm_cell_1/mul_2:z:0peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_3¹
%peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp.peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_2À
peephole_lstm_cell_1/mul_4Mul-peephole_lstm_cell_1/ReadVariableOp_2:value:0peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_4¸
peephole_lstm_cell_1/add_4AddV2#peephole_lstm_cell_1/split:output:3peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_4
peephole_lstm_cell_1/Sigmoid_2Sigmoidpeephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_2
peephole_lstm_cell_1/Tanh_1Tanhpeephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh_1¶
peephole_lstm_cell_1/mul_5Mul"peephole_lstm_cell_1/Sigmoid_2:y:0peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¨
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:03peephole_lstm_cell_1_matmul_readvariableop_resource5peephole_lstm_cell_1_matmul_1_readvariableop_resource4peephole_lstm_cell_1_biasadd_readvariableop_resource,peephole_lstm_cell_1_readvariableop_resource.peephole_lstm_cell_1_readvariableop_1_resource.peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1667000*
condR
while_cond_1666999*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ô
IdentityIdentitystrided_slice_3:output:0,^peephole_lstm_cell_1/BiasAdd/ReadVariableOp+^peephole_lstm_cell_1/MatMul/ReadVariableOp-^peephole_lstm_cell_1/MatMul_1/ReadVariableOp$^peephole_lstm_cell_1/ReadVariableOp&^peephole_lstm_cell_1/ReadVariableOp_1&^peephole_lstm_cell_1/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2Z
+peephole_lstm_cell_1/BiasAdd/ReadVariableOp+peephole_lstm_cell_1/BiasAdd/ReadVariableOp2X
*peephole_lstm_cell_1/MatMul/ReadVariableOp*peephole_lstm_cell_1/MatMul/ReadVariableOp2\
,peephole_lstm_cell_1/MatMul_1/ReadVariableOp,peephole_lstm_cell_1/MatMul_1/ReadVariableOp2J
#peephole_lstm_cell_1/ReadVariableOp#peephole_lstm_cell_1/ReadVariableOp2N
%peephole_lstm_cell_1/ReadVariableOp_1%peephole_lstm_cell_1/ReadVariableOp_12N
%peephole_lstm_cell_1/ReadVariableOp_2%peephole_lstm_cell_1/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


,__inference_sequential_layer_call_fn_1668782

inputs
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:	 
	unknown_3:	
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:	 
	unknown_8:	 
	unknown_9:	

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16677012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
!sequential_rnn_while_body_1664865:
6sequential_rnn_while_sequential_rnn_while_loop_counter@
<sequential_rnn_while_sequential_rnn_while_maximum_iterations$
 sequential_rnn_while_placeholder&
"sequential_rnn_while_placeholder_1&
"sequential_rnn_while_placeholder_2&
"sequential_rnn_while_placeholder_39
5sequential_rnn_while_sequential_rnn_strided_slice_1_0u
qsequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0:	]
Jsequential_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 X
Isequential_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	O
Asequential_rnn_while_peephole_lstm_cell_readvariableop_resource_0: Q
Csequential_rnn_while_peephole_lstm_cell_readvariableop_1_resource_0: Q
Csequential_rnn_while_peephole_lstm_cell_readvariableop_2_resource_0: !
sequential_rnn_while_identity#
sequential_rnn_while_identity_1#
sequential_rnn_while_identity_2#
sequential_rnn_while_identity_3#
sequential_rnn_while_identity_4#
sequential_rnn_while_identity_57
3sequential_rnn_while_sequential_rnn_strided_slice_1s
osequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensorY
Fsequential_rnn_while_peephole_lstm_cell_matmul_readvariableop_resource:	[
Hsequential_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 V
Gsequential_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource:	M
?sequential_rnn_while_peephole_lstm_cell_readvariableop_resource: O
Asequential_rnn_while_peephole_lstm_cell_readvariableop_1_resource: O
Asequential_rnn_while_peephole_lstm_cell_readvariableop_2_resource: ¢>sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢=sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp¢?sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢6sequential/rnn/while/peephole_lstm_cell/ReadVariableOp¢8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_1¢8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2á
Fsequential/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fsequential/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8sequential/rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqsequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensor_0 sequential_rnn_while_placeholderOsequential/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8sequential/rnn/while/TensorArrayV2Read/TensorListGetItem
=sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOpHsequential_rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02?
=sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp¥
.sequential/rnn/while/peephole_lstm_cell/MatMulMatMul?sequential/rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/rnn/while/peephole_lstm_cell/MatMul
?sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOpJsequential_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02A
?sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp
0sequential/rnn/while/peephole_lstm_cell/MatMul_1MatMul"sequential_rnn_while_placeholder_2Gsequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/rnn/while/peephole_lstm_cell/MatMul_1
+sequential/rnn/while/peephole_lstm_cell/addAddV28sequential/rnn/while/peephole_lstm_cell/MatMul:product:0:sequential/rnn/while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential/rnn/while/peephole_lstm_cell/add
>sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOpIsequential_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02@
>sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp
/sequential/rnn/while/peephole_lstm_cell/BiasAddBiasAdd/sequential/rnn/while/peephole_lstm_cell/add:z:0Fsequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/rnn/while/peephole_lstm_cell/BiasAdd´
7sequential/rnn/while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential/rnn/while/peephole_lstm_cell/split/split_dimß
-sequential/rnn/while/peephole_lstm_cell/splitSplit@sequential/rnn/while/peephole_lstm_cell/split/split_dim:output:08sequential/rnn/while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2/
-sequential/rnn/while/peephole_lstm_cell/splitî
6sequential/rnn/while/peephole_lstm_cell/ReadVariableOpReadVariableOpAsequential_rnn_while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype028
6sequential/rnn/while/peephole_lstm_cell/ReadVariableOp÷
+sequential/rnn/while/peephole_lstm_cell/mulMul>sequential/rnn/while/peephole_lstm_cell/ReadVariableOp:value:0"sequential_rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn/while/peephole_lstm_cell/mul
-sequential/rnn/while/peephole_lstm_cell/add_1AddV26sequential/rnn/while/peephole_lstm_cell/split:output:0/sequential/rnn/while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/add_1Ò
/sequential/rnn/while/peephole_lstm_cell/SigmoidSigmoid1sequential/rnn/while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/sequential/rnn/while/peephole_lstm_cell/Sigmoidô
8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOpCsequential_rnn_while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02:
8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_1ý
-sequential/rnn/while/peephole_lstm_cell/mul_1Mul@sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_1:value:0"sequential_rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/mul_1
-sequential/rnn/while/peephole_lstm_cell/add_2AddV26sequential/rnn/while/peephole_lstm_cell/split:output:11sequential/rnn/while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/add_2Ö
1sequential/rnn/while/peephole_lstm_cell/Sigmoid_1Sigmoid1sequential/rnn/while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn/while/peephole_lstm_cell/Sigmoid_1ò
-sequential/rnn/while/peephole_lstm_cell/mul_2Mul5sequential/rnn/while/peephole_lstm_cell/Sigmoid_1:y:0"sequential_rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/mul_2Î
,sequential/rnn/while/peephole_lstm_cell/TanhTanh6sequential/rnn/while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rnn/while/peephole_lstm_cell/Tanhþ
-sequential/rnn/while/peephole_lstm_cell/mul_3Mul3sequential/rnn/while/peephole_lstm_cell/Sigmoid:y:00sequential/rnn/while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/mul_3ÿ
-sequential/rnn/while/peephole_lstm_cell/add_3AddV21sequential/rnn/while/peephole_lstm_cell/mul_2:z:01sequential/rnn/while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/add_3ô
8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOpCsequential_rnn_while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02:
8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2
-sequential/rnn/while/peephole_lstm_cell/mul_4Mul@sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2:value:01sequential/rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/mul_4
-sequential/rnn/while/peephole_lstm_cell/add_4AddV26sequential/rnn/while/peephole_lstm_cell/split:output:31sequential/rnn/while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/add_4Ö
1sequential/rnn/while/peephole_lstm_cell/Sigmoid_2Sigmoid1sequential/rnn/while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn/while/peephole_lstm_cell/Sigmoid_2Í
.sequential/rnn/while/peephole_lstm_cell/Tanh_1Tanh1sequential/rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/rnn/while/peephole_lstm_cell/Tanh_1
-sequential/rnn/while/peephole_lstm_cell/mul_5Mul5sequential/rnn/while/peephole_lstm_cell/Sigmoid_2:y:02sequential/rnn/while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn/while/peephole_lstm_cell/mul_5±
9sequential/rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"sequential_rnn_while_placeholder_1 sequential_rnn_while_placeholder1sequential/rnn/while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02;
9sequential/rnn/while/TensorArrayV2Write/TensorListSetItemz
sequential/rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/rnn/while/add/y¥
sequential/rnn/while/addAddV2 sequential_rnn_while_placeholder#sequential/rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn/while/add~
sequential/rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/rnn/while/add_1/yÁ
sequential/rnn/while/add_1AddV26sequential_rnn_while_sequential_rnn_while_loop_counter%sequential/rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn/while/add_1ý
sequential/rnn/while/IdentityIdentitysequential/rnn/while/add_1:z:0?^sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp>^sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp@^sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp7^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp9^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_19^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
sequential/rnn/while/Identity
sequential/rnn/while/Identity_1Identity<sequential_rnn_while_sequential_rnn_while_maximum_iterations?^sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp>^sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp@^sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp7^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp9^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_19^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2!
sequential/rnn/while/Identity_1ÿ
sequential/rnn/while/Identity_2Identitysequential/rnn/while/add:z:0?^sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp>^sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp@^sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp7^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp9^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_19^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2!
sequential/rnn/while/Identity_2¬
sequential/rnn/while/Identity_3IdentityIsequential/rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp>^sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp@^sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp7^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp9^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_19^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2!
sequential/rnn/while/Identity_3¥
sequential/rnn/while/Identity_4Identity1sequential/rnn/while/peephole_lstm_cell/mul_5:z:0?^sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp>^sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp@^sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp7^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp9^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_19^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/rnn/while/Identity_4¥
sequential/rnn/while/Identity_5Identity1sequential/rnn/while/peephole_lstm_cell/add_3:z:0?^sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp>^sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp@^sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp7^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp9^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_19^sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/rnn/while/Identity_5"G
sequential_rnn_while_identity&sequential/rnn/while/Identity:output:0"K
sequential_rnn_while_identity_1(sequential/rnn/while/Identity_1:output:0"K
sequential_rnn_while_identity_2(sequential/rnn/while/Identity_2:output:0"K
sequential_rnn_while_identity_3(sequential/rnn/while/Identity_3:output:0"K
sequential_rnn_while_identity_4(sequential/rnn/while/Identity_4:output:0"K
sequential_rnn_while_identity_5(sequential/rnn/while/Identity_5:output:0"
Gsequential_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resourceIsequential_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0"
Hsequential_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resourceJsequential_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"
Fsequential_rnn_while_peephole_lstm_cell_matmul_readvariableop_resourceHsequential_rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0"
Asequential_rnn_while_peephole_lstm_cell_readvariableop_1_resourceCsequential_rnn_while_peephole_lstm_cell_readvariableop_1_resource_0"
Asequential_rnn_while_peephole_lstm_cell_readvariableop_2_resourceCsequential_rnn_while_peephole_lstm_cell_readvariableop_2_resource_0"
?sequential_rnn_while_peephole_lstm_cell_readvariableop_resourceAsequential_rnn_while_peephole_lstm_cell_readvariableop_resource_0"l
3sequential_rnn_while_sequential_rnn_strided_slice_15sequential_rnn_while_sequential_rnn_strided_slice_1_0"ä
osequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensorqsequential_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2
>sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp>sequential/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2~
=sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp=sequential/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2
?sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp?sequential/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp2p
6sequential/rnn/while/peephole_lstm_cell/ReadVariableOp6sequential/rnn/while/peephole_lstm_cell/ReadVariableOp2t
8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_18sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_12t
8sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_28sequential/rnn/while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

È
4__inference_peephole_lstm_cell_layer_call_fn_1670575

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity

identity_1

identity_2¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_16654222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
É
ÿ
'__inference_rnn_1_layer_call_fn_1670405

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16671012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ã'
¿
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1670619

inputs
states_0
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
ò

%__inference_signature_wrapper_1667900
conv1d_input
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:	 
	unknown_3:	
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:	 
	unknown_8:	 
	unknown_9:	

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_16651482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv1d_input
Ér
ú
B__inference_rnn_1_layer_call_and_return_conditional_losses_1669994
inputs_0F
3peephole_lstm_cell_1_matmul_readvariableop_resource:	 H
5peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 C
4peephole_lstm_cell_1_biasadd_readvariableop_resource:	:
,peephole_lstm_cell_1_readvariableop_resource: <
.peephole_lstm_cell_1_readvariableop_1_resource: <
.peephole_lstm_cell_1_readvariableop_2_resource: 
identity¢+peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢*peephole_lstm_cell_1/MatMul/ReadVariableOp¢,peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢#peephole_lstm_cell_1/ReadVariableOp¢%peephole_lstm_cell_1/ReadVariableOp_1¢%peephole_lstm_cell_1/ReadVariableOp_2¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2Í
*peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp3peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell_1/MatMul/ReadVariableOpÅ
peephole_lstm_cell_1/MatMulMatMulstrided_slice_2:output:02peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMulÓ
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp5peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02.
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpÁ
peephole_lstm_cell_1/MatMul_1MatMulzeros:output:04peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMul_1À
peephole_lstm_cell_1/addAddV2%peephole_lstm_cell_1/MatMul:product:0'peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/addÌ
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp4peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpÍ
peephole_lstm_cell_1/BiasAddBiasAddpeephole_lstm_cell_1/add:z:03peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/BiasAdd
$peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$peephole_lstm_cell_1/split/split_dim
peephole_lstm_cell_1/splitSplit-peephole_lstm_cell_1/split/split_dim:output:0%peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell_1/split³
#peephole_lstm_cell_1/ReadVariableOpReadVariableOp,peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell_1/ReadVariableOp¬
peephole_lstm_cell_1/mulMul+peephole_lstm_cell_1/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul¶
peephole_lstm_cell_1/add_1AddV2#peephole_lstm_cell_1/split:output:0peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_1
peephole_lstm_cell_1/SigmoidSigmoidpeephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Sigmoid¹
%peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp.peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_1²
peephole_lstm_cell_1/mul_1Mul-peephole_lstm_cell_1/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_1¸
peephole_lstm_cell_1/add_2AddV2#peephole_lstm_cell_1/split:output:1peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_2
peephole_lstm_cell_1/Sigmoid_1Sigmoidpeephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_1§
peephole_lstm_cell_1/mul_2Mul"peephole_lstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_2
peephole_lstm_cell_1/TanhTanh#peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh²
peephole_lstm_cell_1/mul_3Mul peephole_lstm_cell_1/Sigmoid:y:0peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_3³
peephole_lstm_cell_1/add_3AddV2peephole_lstm_cell_1/mul_2:z:0peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_3¹
%peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp.peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_2À
peephole_lstm_cell_1/mul_4Mul-peephole_lstm_cell_1/ReadVariableOp_2:value:0peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_4¸
peephole_lstm_cell_1/add_4AddV2#peephole_lstm_cell_1/split:output:3peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_4
peephole_lstm_cell_1/Sigmoid_2Sigmoidpeephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_2
peephole_lstm_cell_1/Tanh_1Tanhpeephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh_1¶
peephole_lstm_cell_1/mul_5Mul"peephole_lstm_cell_1/Sigmoid_2:y:0peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¨
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:03peephole_lstm_cell_1_matmul_readvariableop_resource5peephole_lstm_cell_1_matmul_1_readvariableop_resource4peephole_lstm_cell_1_biasadd_readvariableop_resource,peephole_lstm_cell_1_readvariableop_resource.peephole_lstm_cell_1_readvariableop_1_resource.peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1669893*
condR
while_cond_1669892*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1ô
IdentityIdentitystrided_slice_3:output:0,^peephole_lstm_cell_1/BiasAdd/ReadVariableOp+^peephole_lstm_cell_1/MatMul/ReadVariableOp-^peephole_lstm_cell_1/MatMul_1/ReadVariableOp$^peephole_lstm_cell_1/ReadVariableOp&^peephole_lstm_cell_1/ReadVariableOp_1&^peephole_lstm_cell_1/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2Z
+peephole_lstm_cell_1/BiasAdd/ReadVariableOp+peephole_lstm_cell_1/BiasAdd/ReadVariableOp2X
*peephole_lstm_cell_1/MatMul/ReadVariableOp*peephole_lstm_cell_1/MatMul/ReadVariableOp2\
,peephole_lstm_cell_1/MatMul_1/ReadVariableOp,peephole_lstm_cell_1/MatMul_1/ReadVariableOp2J
#peephole_lstm_cell_1/ReadVariableOp#peephole_lstm_cell_1/ReadVariableOp2N
%peephole_lstm_cell_1/ReadVariableOp_1%peephole_lstm_cell_1/ReadVariableOp_12N
%peephole_lstm_cell_1/ReadVariableOp_2%peephole_lstm_cell_1/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
r
ø
B__inference_rnn_1_layer_call_and_return_conditional_losses_1667376

inputsF
3peephole_lstm_cell_1_matmul_readvariableop_resource:	 H
5peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 C
4peephole_lstm_cell_1_biasadd_readvariableop_resource:	:
,peephole_lstm_cell_1_readvariableop_resource: <
.peephole_lstm_cell_1_readvariableop_1_resource: <
.peephole_lstm_cell_1_readvariableop_2_resource: 
identity¢+peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢*peephole_lstm_cell_1/MatMul/ReadVariableOp¢,peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢#peephole_lstm_cell_1/ReadVariableOp¢%peephole_lstm_cell_1/ReadVariableOp_1¢%peephole_lstm_cell_1/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2Í
*peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp3peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell_1/MatMul/ReadVariableOpÅ
peephole_lstm_cell_1/MatMulMatMulstrided_slice_2:output:02peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMulÓ
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp5peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02.
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpÁ
peephole_lstm_cell_1/MatMul_1MatMulzeros:output:04peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMul_1À
peephole_lstm_cell_1/addAddV2%peephole_lstm_cell_1/MatMul:product:0'peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/addÌ
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp4peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpÍ
peephole_lstm_cell_1/BiasAddBiasAddpeephole_lstm_cell_1/add:z:03peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/BiasAdd
$peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$peephole_lstm_cell_1/split/split_dim
peephole_lstm_cell_1/splitSplit-peephole_lstm_cell_1/split/split_dim:output:0%peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell_1/split³
#peephole_lstm_cell_1/ReadVariableOpReadVariableOp,peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell_1/ReadVariableOp¬
peephole_lstm_cell_1/mulMul+peephole_lstm_cell_1/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul¶
peephole_lstm_cell_1/add_1AddV2#peephole_lstm_cell_1/split:output:0peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_1
peephole_lstm_cell_1/SigmoidSigmoidpeephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Sigmoid¹
%peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp.peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_1²
peephole_lstm_cell_1/mul_1Mul-peephole_lstm_cell_1/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_1¸
peephole_lstm_cell_1/add_2AddV2#peephole_lstm_cell_1/split:output:1peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_2
peephole_lstm_cell_1/Sigmoid_1Sigmoidpeephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_1§
peephole_lstm_cell_1/mul_2Mul"peephole_lstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_2
peephole_lstm_cell_1/TanhTanh#peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh²
peephole_lstm_cell_1/mul_3Mul peephole_lstm_cell_1/Sigmoid:y:0peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_3³
peephole_lstm_cell_1/add_3AddV2peephole_lstm_cell_1/mul_2:z:0peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_3¹
%peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp.peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_2À
peephole_lstm_cell_1/mul_4Mul-peephole_lstm_cell_1/ReadVariableOp_2:value:0peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_4¸
peephole_lstm_cell_1/add_4AddV2#peephole_lstm_cell_1/split:output:3peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_4
peephole_lstm_cell_1/Sigmoid_2Sigmoidpeephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_2
peephole_lstm_cell_1/Tanh_1Tanhpeephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh_1¶
peephole_lstm_cell_1/mul_5Mul"peephole_lstm_cell_1/Sigmoid_2:y:0peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¨
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:03peephole_lstm_cell_1_matmul_readvariableop_resource5peephole_lstm_cell_1_matmul_1_readvariableop_resource4peephole_lstm_cell_1_biasadd_readvariableop_resource,peephole_lstm_cell_1_readvariableop_resource.peephole_lstm_cell_1_readvariableop_1_resource.peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1667275*
condR
while_cond_1667274*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ô
IdentityIdentitystrided_slice_3:output:0,^peephole_lstm_cell_1/BiasAdd/ReadVariableOp+^peephole_lstm_cell_1/MatMul/ReadVariableOp-^peephole_lstm_cell_1/MatMul_1/ReadVariableOp$^peephole_lstm_cell_1/ReadVariableOp&^peephole_lstm_cell_1/ReadVariableOp_1&^peephole_lstm_cell_1/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2Z
+peephole_lstm_cell_1/BiasAdd/ReadVariableOp+peephole_lstm_cell_1/BiasAdd/ReadVariableOp2X
*peephole_lstm_cell_1/MatMul/ReadVariableOp*peephole_lstm_cell_1/MatMul/ReadVariableOp2\
,peephole_lstm_cell_1/MatMul_1/ReadVariableOp,peephole_lstm_cell_1/MatMul_1/ReadVariableOp2J
#peephole_lstm_cell_1/ReadVariableOp#peephole_lstm_cell_1/ReadVariableOp2N
%peephole_lstm_cell_1/ReadVariableOp_1%peephole_lstm_cell_1/ReadVariableOp_12N
%peephole_lstm_cell_1/ReadVariableOp_2%peephole_lstm_cell_1/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_1670072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1670072___redundant_placeholder05
1while_while_cond_1670072___redundant_placeholder15
1while_while_cond_1670072___redundant_placeholder25
1while_while_cond_1670072___redundant_placeholder35
1while_while_cond_1670072___redundant_placeholder45
1while_while_cond_1670072___redundant_placeholder55
1while_while_cond_1670072___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


í
while_cond_1665517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1665517___redundant_placeholder05
1while_while_cond_1665517___redundant_placeholder15
1while_while_cond_1665517___redundant_placeholder25
1while_while_cond_1665517___redundant_placeholder35
1while_while_cond_1665517___redundant_placeholder45
1while_while_cond_1665517___redundant_placeholder55
1while_while_cond_1665517___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


í
while_cond_1669892
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1669892___redundant_placeholder05
1while_while_cond_1669892___redundant_placeholder15
1while_while_cond_1669892___redundant_placeholder25
1while_while_cond_1669892___redundant_placeholder35
1while_while_cond_1669892___redundant_placeholder45
1while_while_cond_1669892___redundant_placeholder55
1while_while_cond_1669892___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
á

'__inference_rnn_1_layer_call_fn_1670388
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16663562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
ÒZ
ï
 __inference__traced_save_1670849
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop<
8savev2_rnn_peephole_lstm_cell_kernel_read_readvariableopF
Bsavev2_rnn_peephole_lstm_cell_recurrent_kernel_read_readvariableop:
6savev2_rnn_peephole_lstm_cell_bias_read_readvariableopQ
Msavev2_rnn_peephole_lstm_cell_input_gate_peephole_weights_read_readvariableopR
Nsavev2_rnn_peephole_lstm_cell_forget_gate_peephole_weights_read_readvariableopR
Nsavev2_rnn_peephole_lstm_cell_output_gate_peephole_weights_read_readvariableop@
<savev2_rnn_1_peephole_lstm_cell_1_kernel_read_readvariableopJ
Fsavev2_rnn_1_peephole_lstm_cell_1_recurrent_kernel_read_readvariableop>
:savev2_rnn_1_peephole_lstm_cell_1_bias_read_readvariableopU
Qsavev2_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weights_read_readvariableopV
Rsavev2_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weights_read_readvariableopV
Rsavev2_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_rmsprop_conv1d_kernel_rms_read_readvariableop6
2savev2_rmsprop_conv1d_bias_rms_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableopH
Dsavev2_rmsprop_rnn_peephole_lstm_cell_kernel_rms_read_readvariableopR
Nsavev2_rmsprop_rnn_peephole_lstm_cell_recurrent_kernel_rms_read_readvariableopF
Bsavev2_rmsprop_rnn_peephole_lstm_cell_bias_rms_read_readvariableop]
Ysavev2_rmsprop_rnn_peephole_lstm_cell_input_gate_peephole_weights_rms_read_readvariableop^
Zsavev2_rmsprop_rnn_peephole_lstm_cell_forget_gate_peephole_weights_rms_read_readvariableop^
Zsavev2_rmsprop_rnn_peephole_lstm_cell_output_gate_peephole_weights_rms_read_readvariableopL
Hsavev2_rmsprop_rnn_1_peephole_lstm_cell_1_kernel_rms_read_readvariableopV
Rsavev2_rmsprop_rnn_1_peephole_lstm_cell_1_recurrent_kernel_rms_read_readvariableopJ
Fsavev2_rmsprop_rnn_1_peephole_lstm_cell_1_bias_rms_read_readvariableopa
]savev2_rmsprop_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weights_rms_read_readvariableopb
^savev2_rmsprop_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weights_rms_read_readvariableopb
^savev2_rmsprop_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weights_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÑ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*ã
valueÙBÖ(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesØ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÇ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop8savev2_rnn_peephole_lstm_cell_kernel_read_readvariableopBsavev2_rnn_peephole_lstm_cell_recurrent_kernel_read_readvariableop6savev2_rnn_peephole_lstm_cell_bias_read_readvariableopMsavev2_rnn_peephole_lstm_cell_input_gate_peephole_weights_read_readvariableopNsavev2_rnn_peephole_lstm_cell_forget_gate_peephole_weights_read_readvariableopNsavev2_rnn_peephole_lstm_cell_output_gate_peephole_weights_read_readvariableop<savev2_rnn_1_peephole_lstm_cell_1_kernel_read_readvariableopFsavev2_rnn_1_peephole_lstm_cell_1_recurrent_kernel_read_readvariableop:savev2_rnn_1_peephole_lstm_cell_1_bias_read_readvariableopQsavev2_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weights_read_readvariableopRsavev2_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weights_read_readvariableopRsavev2_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_rmsprop_conv1d_kernel_rms_read_readvariableop2savev2_rmsprop_conv1d_bias_rms_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableopDsavev2_rmsprop_rnn_peephole_lstm_cell_kernel_rms_read_readvariableopNsavev2_rmsprop_rnn_peephole_lstm_cell_recurrent_kernel_rms_read_readvariableopBsavev2_rmsprop_rnn_peephole_lstm_cell_bias_rms_read_readvariableopYsavev2_rmsprop_rnn_peephole_lstm_cell_input_gate_peephole_weights_rms_read_readvariableopZsavev2_rmsprop_rnn_peephole_lstm_cell_forget_gate_peephole_weights_rms_read_readvariableopZsavev2_rmsprop_rnn_peephole_lstm_cell_output_gate_peephole_weights_rms_read_readvariableopHsavev2_rmsprop_rnn_1_peephole_lstm_cell_1_kernel_rms_read_readvariableopRsavev2_rmsprop_rnn_1_peephole_lstm_cell_1_recurrent_kernel_rms_read_readvariableopFsavev2_rmsprop_rnn_1_peephole_lstm_cell_1_bias_rms_read_readvariableop]savev2_rmsprop_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weights_rms_read_readvariableop^savev2_rmsprop_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weights_rms_read_readvariableop^savev2_rmsprop_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*«
_input_shapes
: ::: :: : : : : :	:	 :: : : :	 :	 :: : : : : ::: ::	:	 :: : : :	 :	 :: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
:: 

_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: :%"!

_output_shapes
:	 :%#!

_output_shapes
:	 :!$

_output_shapes	
:: %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: :(

_output_shapes
: 
ôG

@__inference_rnn_layer_call_and_return_conditional_losses_1665598

inputs-
peephole_lstm_cell_1665499:	-
peephole_lstm_cell_1665501:	 )
peephole_lstm_cell_1665503:	(
peephole_lstm_cell_1665505: (
peephole_lstm_cell_1665507: (
peephole_lstm_cell_1665509: 
identity¢*peephole_lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
*peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0peephole_lstm_cell_1665499peephole_lstm_cell_1665501peephole_lstm_cell_1665503peephole_lstm_cell_1665505peephole_lstm_cell_1665507peephole_lstm_cell_1665509*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_16654222,
*peephole_lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0peephole_lstm_cell_1665499peephole_lstm_cell_1665501peephole_lstm_cell_1665503peephole_lstm_cell_1665505peephole_lstm_cell_1665507peephole_lstm_cell_1665509*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1665518*
condR
while_cond_1665517*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1¥
IdentityIdentitytranspose_1:y:0+^peephole_lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2X
*peephole_lstm_cell/StatefulPartitionedCall*peephole_lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
Í
G__inference_sequential_layer_call_and_return_conditional_losses_1667701

inputs$
conv1d_1667663:
conv1d_1667665:
rnn_1667669:	
rnn_1667671:	 
rnn_1667673:	
rnn_1667675: 
rnn_1667677: 
rnn_1667679:  
rnn_1_1667682:	  
rnn_1_1667684:	 
rnn_1_1667686:	
rnn_1_1667688: 
rnn_1_1667690: 
rnn_1_1667692: 
dense_1667695: 
dense_1667697:
identity¢conv1d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢rnn/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1667663conv1d_1667665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_16667082 
conv1d/StatefulPartitionedCallú
reshape/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_16667272
reshape/PartitionedCallÛ
rnn/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0rnn_1667669rnn_1667671rnn_1667673rnn_1667675rnn_1667677rnn_1667679*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16675902
rnn/StatefulPartitionedCallí
rnn_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0rnn_1_1667682rnn_1_1667684rnn_1_1667686rnn_1_1667688rnn_1_1667690rnn_1_1667692*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16673762
rnn_1/StatefulPartitionedCall«
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_1667695dense_1667697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16671252
dense/StatefulPartitionedCallù
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸d

while_body_1669285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_peephole_lstm_cell_matmul_readvariableop_resource_0:	N
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 I
:while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	@
2while_peephole_lstm_cell_readvariableop_resource_0: B
4while_peephole_lstm_cell_readvariableop_1_resource_0: B
4while_peephole_lstm_cell_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_peephole_lstm_cell_matmul_readvariableop_resource:	L
9while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 G
8while_peephole_lstm_cell_biasadd_readvariableop_resource:	>
0while_peephole_lstm_cell_readvariableop_resource: @
2while_peephole_lstm_cell_readvariableop_1_resource: @
2while_peephole_lstm_cell_readvariableop_2_resource: ¢/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢.while/peephole_lstm_cell/MatMul/ReadVariableOp¢0while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢'while/peephole_lstm_cell/ReadVariableOp¢)while/peephole_lstm_cell/ReadVariableOp_1¢)while/peephole_lstm_cell/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÛ
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOpé
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/peephole_lstm_cell/MatMulá
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpÒ
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell/MatMul_1Ð
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/peephole_lstm_cell/addÚ
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpÝ
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/peephole_lstm_cell/BiasAdd
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim£
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2 
while/peephole_lstm_cell/splitÁ
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp»
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/mulÆ
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_1¥
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell/SigmoidÇ
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1Á
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_1È
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_2©
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_1¶
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_2¡
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/TanhÂ
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_3Ã
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_3Ç
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2Ð
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_4È
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_4©
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_2 
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell/Tanh_1Æ
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_5æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ö
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1ø
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2¥
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

Õ
rnn_while_cond_1668020$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1=
9rnn_while_rnn_while_cond_1668020___redundant_placeholder0=
9rnn_while_rnn_while_cond_1668020___redundant_placeholder1=
9rnn_while_rnn_while_cond_1668020___redundant_placeholder2=
9rnn_while_rnn_while_cond_1668020___redundant_placeholder3=
9rnn_while_rnn_while_cond_1668020___redundant_placeholder4=
9rnn_while_rnn_while_cond_1668020___redundant_placeholder5=
9rnn_while_rnn_while_cond_1668020___redundant_placeholder6
rnn_while_identity

rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
÷
ÿ
%__inference_rnn_layer_call_fn_1669600
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16655982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¼

G__inference_sequential_layer_call_and_return_conditional_losses_1668304

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource:G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:H
5rnn_peephole_lstm_cell_matmul_readvariableop_resource:	J
7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource:	 E
6rnn_peephole_lstm_cell_biasadd_readvariableop_resource:	<
.rnn_peephole_lstm_cell_readvariableop_resource: >
0rnn_peephole_lstm_cell_readvariableop_1_resource: >
0rnn_peephole_lstm_cell_readvariableop_2_resource: L
9rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2rnn_1_peephole_lstm_cell_1_readvariableop_resource: B
4rnn_1_peephole_lstm_cell_1_readvariableop_1_resource: B
4rnn_1_peephole_lstm_cell_1_readvariableop_2_resource: 6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource:
identity¢)conv1d/conv1d/ExpandDims_1/ReadVariableOp¢0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp¢,rnn/peephole_lstm_cell/MatMul/ReadVariableOp¢.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp¢%rnn/peephole_lstm_cell/ReadVariableOp¢'rnn/peephole_lstm_cell/ReadVariableOp_1¢'rnn/peephole_lstm_cell/ReadVariableOp_2¢	rnn/while¢1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)rnn_1/peephole_lstm_cell_1/ReadVariableOp¢+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1¢+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2¢rnn_1/while
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDims/dim¯
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDimsÍ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimÓ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2´
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2
conv1d/conv1d/Reshape/shape¼
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ReshapeÞ
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/concat/axisØ
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat¹
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/Reshape_1µ
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape¨
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackµ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1¬
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ü
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice§
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeÙ
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!conv1d/squeeze_batch_dims/ReshapeÚ
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpí
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!conv1d/squeeze_batch_dims/BiasAdd§
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatæ
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#conv1d/squeeze_batch_dims/Reshape_1z
reshape/ShapeShape,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2È
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape±
reshape/ReshapeReshape,conv1d/squeeze_batch_dims/Reshape_1:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshape^
	rnn/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2ú
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	rnn/zerosh
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/mul/y
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lessn
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/packed/1
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/zeros_1}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm
rnn/transpose	Transposereshape/Reshape:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
rnn/TensorArrayV2/element_shapeÂ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ç
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
rnn/strided_slice_2Ó
,rnn/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp5rnn_peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,rnn/peephole_lstm_cell/MatMul/ReadVariableOpÏ
rnn/peephole_lstm_cell/MatMulMatMulrnn/strided_slice_2:output:04rnn/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/peephole_lstm_cell/MatMulÙ
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype020
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpË
rnn/peephole_lstm_cell/MatMul_1MatMulrnn/zeros:output:06rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
rnn/peephole_lstm_cell/MatMul_1È
rnn/peephole_lstm_cell/addAddV2'rnn/peephole_lstm_cell/MatMul:product:0)rnn/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/peephole_lstm_cell/addÒ
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6rnn_peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpÕ
rnn/peephole_lstm_cell/BiasAddBiasAddrnn/peephole_lstm_cell/add:z:05rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rnn/peephole_lstm_cell/BiasAdd
&rnn/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&rnn/peephole_lstm_cell/split/split_dim
rnn/peephole_lstm_cell/splitSplit/rnn/peephole_lstm_cell/split/split_dim:output:0'rnn/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rnn/peephole_lstm_cell/split¹
%rnn/peephole_lstm_cell/ReadVariableOpReadVariableOp.rnn_peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02'
%rnn/peephole_lstm_cell/ReadVariableOp¶
rnn/peephole_lstm_cell/mulMul-rnn/peephole_lstm_cell/ReadVariableOp:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul¾
rnn/peephole_lstm_cell/add_1AddV2%rnn/peephole_lstm_cell/split:output:0rnn/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_1
rnn/peephole_lstm_cell/SigmoidSigmoid rnn/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
rnn/peephole_lstm_cell/Sigmoid¿
'rnn/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_1¼
rnn/peephole_lstm_cell/mul_1Mul/rnn/peephole_lstm_cell/ReadVariableOp_1:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_1À
rnn/peephole_lstm_cell/add_2AddV2%rnn/peephole_lstm_cell/split:output:1 rnn/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_2£
 rnn/peephole_lstm_cell/Sigmoid_1Sigmoid rnn/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn/peephole_lstm_cell/Sigmoid_1±
rnn/peephole_lstm_cell/mul_2Mul$rnn/peephole_lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_2
rnn/peephole_lstm_cell/TanhTanh%rnn/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/Tanhº
rnn/peephole_lstm_cell/mul_3Mul"rnn/peephole_lstm_cell/Sigmoid:y:0rnn/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_3»
rnn/peephole_lstm_cell/add_3AddV2 rnn/peephole_lstm_cell/mul_2:z:0 rnn/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_3¿
'rnn/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_2È
rnn/peephole_lstm_cell/mul_4Mul/rnn/peephole_lstm_cell/ReadVariableOp_2:value:0 rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_4À
rnn/peephole_lstm_cell/add_4AddV2%rnn/peephole_lstm_cell/split:output:3 rnn/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_4£
 rnn/peephole_lstm_cell/Sigmoid_2Sigmoid rnn/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn/peephole_lstm_cell/Sigmoid_2
rnn/peephole_lstm_cell/Tanh_1Tanh rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/Tanh_1¾
rnn/peephole_lstm_cell/mul_5Mul$rnn/peephole_lstm_cell/Sigmoid_2:y:0!rnn/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_5
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2#
!rnn/TensorArrayV2_1/element_shapeÈ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counterä
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_peephole_lstm_cell_matmul_readvariableop_resource7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource6rnn_peephole_lstm_cell_biasadd_readvariableop_resource.rnn_peephole_lstm_cell_readvariableop_resource0rnn_peephole_lstm_cell_readvariableop_1_resource0rnn_peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*"
bodyR
rnn_while_body_1668021*"
condR
rnn_while_cond_1668020*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
	rnn/while½
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeø
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2²
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/permµ
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/transpose_1]
rnn_1/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:2
rnn_1/Shape
rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice/stack
rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_1
rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_2
rnn_1/strided_sliceStridedSlicernn_1/Shape:output:0"rnn_1/strided_slice/stack:output:0$rnn_1/strided_slice/stack_1:output:0$rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_sliceh
rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros/mul/y
rnn_1/zeros/mulMulrnn_1/strided_slice:output:0rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/mulk
rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros/Less/y
rnn_1/zeros/LessLessrnn_1/zeros/mul:z:0rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/Lessn
rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros/packed/1
rnn_1/zeros/packedPackrnn_1/strided_slice:output:0rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros/packedk
rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros/Const
rnn_1/zerosFillrnn_1/zeros/packed:output:0rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/zerosl
rnn_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros_1/mul/y
rnn_1/zeros_1/mulMulrnn_1/strided_slice:output:0rnn_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/mulo
rnn_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros_1/Less/y
rnn_1/zeros_1/LessLessrnn_1/zeros_1/mul:z:0rnn_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/Lessr
rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros_1/packed/1¡
rnn_1/zeros_1/packedPackrnn_1/strided_slice:output:0rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros_1/packedo
rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros_1/Const
rnn_1/zeros_1Fillrnn_1/zeros_1/packed:output:0rnn_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/zeros_1
rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose/perm
rnn_1/transpose	Transposernn/transpose_1:y:0rnn_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/transposea
rnn_1/Shape_1Shapernn_1/transpose:y:0*
T0*
_output_shapes
:2
rnn_1/Shape_1
rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_1/stack
rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_1
rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_2
rnn_1/strided_slice_1StridedSlicernn_1/Shape_1:output:0$rnn_1/strided_slice_1/stack:output:0&rnn_1/strided_slice_1/stack_1:output:0&rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_slice_1
!rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!rnn_1/TensorArrayV2/element_shapeÊ
rnn_1/TensorArrayV2TensorListReserve*rnn_1/TensorArrayV2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2Ë
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_1/transpose:y:0Drnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-rnn_1/TensorArrayUnstack/TensorListFromTensor
rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_2/stack
rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_1
rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_2 
rnn_1/strided_slice_2StridedSlicernn_1/transpose:y:0$rnn_1/strided_slice_2/stack:output:0&rnn_1/strided_slice_2/stack_1:output:0&rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rnn_1/strided_slice_2ß
0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp9rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype022
0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOpÝ
!rnn_1/peephole_lstm_cell_1/MatMulMatMulrnn_1/strided_slice_2:output:08rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!rnn_1/peephole_lstm_cell_1/MatMulå
2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp;rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype024
2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOpÙ
#rnn_1/peephole_lstm_cell_1/MatMul_1MatMulrnn_1/zeros:output:0:rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn_1/peephole_lstm_cell_1/MatMul_1Ø
rnn_1/peephole_lstm_cell_1/addAddV2+rnn_1/peephole_lstm_cell_1/MatMul:product:0-rnn_1/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rnn_1/peephole_lstm_cell_1/addÞ
1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"rnn_1/peephole_lstm_cell_1/BiasAddBiasAdd"rnn_1/peephole_lstm_cell_1/add:z:09rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"rnn_1/peephole_lstm_cell_1/BiasAdd
*rnn_1/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*rnn_1/peephole_lstm_cell_1/split/split_dim«
 rnn_1/peephole_lstm_cell_1/splitSplit3rnn_1/peephole_lstm_cell_1/split/split_dim:output:0+rnn_1/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 rnn_1/peephole_lstm_cell_1/splitÅ
)rnn_1/peephole_lstm_cell_1/ReadVariableOpReadVariableOp2rnn_1_peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02+
)rnn_1/peephole_lstm_cell_1/ReadVariableOpÄ
rnn_1/peephole_lstm_cell_1/mulMul1rnn_1/peephole_lstm_cell_1/ReadVariableOp:value:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
rnn_1/peephole_lstm_cell_1/mulÎ
 rnn_1/peephole_lstm_cell_1/add_1AddV2)rnn_1/peephole_lstm_cell_1/split:output:0"rnn_1/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_1«
"rnn_1/peephole_lstm_cell_1/SigmoidSigmoid$rnn_1/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn_1/peephole_lstm_cell_1/SigmoidË
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp4rnn_1_peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02-
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1Ê
 rnn_1/peephole_lstm_cell_1/mul_1Mul3rnn_1/peephole_lstm_cell_1/ReadVariableOp_1:value:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_1Ð
 rnn_1/peephole_lstm_cell_1/add_2AddV2)rnn_1/peephole_lstm_cell_1/split:output:1$rnn_1/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_2¯
$rnn_1/peephole_lstm_cell_1/Sigmoid_1Sigmoid$rnn_1/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn_1/peephole_lstm_cell_1/Sigmoid_1¿
 rnn_1/peephole_lstm_cell_1/mul_2Mul(rnn_1/peephole_lstm_cell_1/Sigmoid_1:y:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_2§
rnn_1/peephole_lstm_cell_1/TanhTanh)rnn_1/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rnn_1/peephole_lstm_cell_1/TanhÊ
 rnn_1/peephole_lstm_cell_1/mul_3Mul&rnn_1/peephole_lstm_cell_1/Sigmoid:y:0#rnn_1/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_3Ë
 rnn_1/peephole_lstm_cell_1/add_3AddV2$rnn_1/peephole_lstm_cell_1/mul_2:z:0$rnn_1/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_3Ë
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp4rnn_1_peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02-
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2Ø
 rnn_1/peephole_lstm_cell_1/mul_4Mul3rnn_1/peephole_lstm_cell_1/ReadVariableOp_2:value:0$rnn_1/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_4Ð
 rnn_1/peephole_lstm_cell_1/add_4AddV2)rnn_1/peephole_lstm_cell_1/split:output:3$rnn_1/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_4¯
$rnn_1/peephole_lstm_cell_1/Sigmoid_2Sigmoid$rnn_1/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn_1/peephole_lstm_cell_1/Sigmoid_2¦
!rnn_1/peephole_lstm_cell_1/Tanh_1Tanh$rnn_1/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rnn_1/peephole_lstm_cell_1/Tanh_1Î
 rnn_1/peephole_lstm_cell_1/mul_5Mul(rnn_1/peephole_lstm_cell_1/Sigmoid_2:y:0%rnn_1/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_5
#rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2%
#rnn_1/TensorArrayV2_1/element_shapeÐ
rnn_1/TensorArrayV2_1TensorListReserve,rnn_1/TensorArrayV2_1/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2_1Z

rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn_1/time
rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
rnn_1/while/maximum_iterationsv
rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/while/loop_counter
rnn_1/whileWhile!rnn_1/while/loop_counter:output:0'rnn_1/while/maximum_iterations:output:0rnn_1/time:output:0rnn_1/TensorArrayV2_1:handle:0rnn_1/zeros:output:0rnn_1/zeros_1:output:0rnn_1/strided_slice_1:output:0=rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:09rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource;rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource:rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource2rnn_1_peephole_lstm_cell_1_readvariableop_resource4rnn_1_peephole_lstm_cell_1_readvariableop_1_resource4rnn_1_peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*$
bodyR
rnn_1_while_body_1668197*$
condR
rnn_1_while_cond_1668196*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
rnn_1/whileÁ
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    28
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shape
(rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_1/while:output:3?rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02*
(rnn_1/TensorArrayV2Stack/TensorListStack
rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn_1/strided_slice_3/stack
rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_3/stack_1
rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_3/stack_2¾
rnn_1/strided_slice_3StridedSlice1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_1/strided_slice_3/stack:output:0&rnn_1/strided_slice_3/stack_1:output:0&rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rnn_1/strided_slice_3
rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose_1/perm½
rnn_1/transpose_1	Transpose1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0rnn_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/transpose_1
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulrnn_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddÐ
IdentityIdentitydense/BiasAdd:output:0*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp.^rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-^rnn/peephole_lstm_cell/MatMul/ReadVariableOp/^rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp&^rnn/peephole_lstm_cell/ReadVariableOp(^rnn/peephole_lstm_cell/ReadVariableOp_1(^rnn/peephole_lstm_cell/ReadVariableOp_2
^rnn/while2^rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp3^rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^rnn_1/peephole_lstm_cell_1/ReadVariableOp,^rnn_1/peephole_lstm_cell_1/ReadVariableOp_1,^rnn_1/peephole_lstm_cell_1/ReadVariableOp_2^rnn_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2^
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp2\
,rnn/peephole_lstm_cell/MatMul/ReadVariableOp,rnn/peephole_lstm_cell/MatMul/ReadVariableOp2`
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp2N
%rnn/peephole_lstm_cell/ReadVariableOp%rnn/peephole_lstm_cell/ReadVariableOp2R
'rnn/peephole_lstm_cell/ReadVariableOp_1'rnn/peephole_lstm_cell/ReadVariableOp_12R
'rnn/peephole_lstm_cell/ReadVariableOp_2'rnn/peephole_lstm_cell/ReadVariableOp_22
	rnn/while	rnn/while2f
1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)rnn_1/peephole_lstm_cell_1/ReadVariableOp)rnn_1/peephole_lstm_cell_1/ReadVariableOp2Z
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1+rnn_1/peephole_lstm_cell_1/ReadVariableOp_12Z
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2+rnn_1/peephole_lstm_cell_1/ReadVariableOp_22
rnn_1/whilernn_1/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
g
»
while_body_1670073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0N
;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 P
=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 K
<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	B
4while_peephole_lstm_cell_1_readvariableop_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_1_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorL
9while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2while_peephole_lstm_cell_1_readvariableop_resource: B
4while_peephole_lstm_cell_1_readvariableop_1_resource: B
4while_peephole_lstm_cell_1_readvariableop_2_resource: ¢1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)while/peephole_lstm_cell_1/ReadVariableOp¢+while/peephole_lstm_cell_1/ReadVariableOp_1¢+while/peephole_lstm_cell_1/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpï
!while/peephole_lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:08while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell_1/MatMulç
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype024
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpØ
#while/peephole_lstm_cell_1/MatMul_1MatMulwhile_placeholder_2:while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#while/peephole_lstm_cell_1/MatMul_1Ø
while/peephole_lstm_cell_1/addAddV2+while/peephole_lstm_cell_1/MatMul:product:0-while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/peephole_lstm_cell_1/addà
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"while/peephole_lstm_cell_1/BiasAddBiasAdd"while/peephole_lstm_cell_1/add:z:09while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"while/peephole_lstm_cell_1/BiasAdd
*while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*while/peephole_lstm_cell_1/split/split_dim«
 while/peephole_lstm_cell_1/splitSplit3while/peephole_lstm_cell_1/split/split_dim:output:0+while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 while/peephole_lstm_cell_1/splitÇ
)while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp4while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell_1/ReadVariableOpÁ
while/peephole_lstm_cell_1/mulMul1while/peephole_lstm_cell_1/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell_1/mulÎ
 while/peephole_lstm_cell_1/add_1AddV2)while/peephole_lstm_cell_1/split:output:0"while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_1«
"while/peephole_lstm_cell_1/SigmoidSigmoid$while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell_1/SigmoidÍ
+while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_1Ç
 while/peephole_lstm_cell_1/mul_1Mul3while/peephole_lstm_cell_1/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_1Ð
 while/peephole_lstm_cell_1/add_2AddV2)while/peephole_lstm_cell_1/split:output:1$while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_2¯
$while/peephole_lstm_cell_1/Sigmoid_1Sigmoid$while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_1¼
 while/peephole_lstm_cell_1/mul_2Mul(while/peephole_lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_2§
while/peephole_lstm_cell_1/TanhTanh)while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell_1/TanhÊ
 while/peephole_lstm_cell_1/mul_3Mul&while/peephole_lstm_cell_1/Sigmoid:y:0#while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_3Ë
 while/peephole_lstm_cell_1/add_3AddV2$while/peephole_lstm_cell_1/mul_2:z:0$while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_3Í
+while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_2Ø
 while/peephole_lstm_cell_1/mul_4Mul3while/peephole_lstm_cell_1/ReadVariableOp_2:value:0$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_4Ð
 while/peephole_lstm_cell_1/add_4AddV2)while/peephole_lstm_cell_1/split:output:3$while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_4¯
$while/peephole_lstm_cell_1/Sigmoid_2Sigmoid$while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_2¦
!while/peephole_lstm_cell_1/Tanh_1Tanh$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!while/peephole_lstm_cell_1/Tanh_1Î
 while/peephole_lstm_cell_1/mul_5Mul(while/peephole_lstm_cell_1/Sigmoid_2:y:0%while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_5è
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations2^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2±
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3¬
while/Identity_4Identity$while/peephole_lstm_cell_1/mul_5:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4¬
while/Identity_5Identity$while/peephole_lstm_cell_1/add_3:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"|
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"x
9while_peephole_lstm_cell_1_matmul_readvariableop_resource;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_1_resource6while_peephole_lstm_cell_1_readvariableop_1_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_2_resource6while_peephole_lstm_cell_1_readvariableop_2_resource_0"j
2while_peephole_lstm_cell_1_readvariableop_resource4while_peephole_lstm_cell_1_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2f
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0while/peephole_lstm_cell_1/MatMul/ReadVariableOp0while/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)while/peephole_lstm_cell_1/ReadVariableOp)while/peephole_lstm_cell_1/ReadVariableOp2Z
+while/peephole_lstm_cell_1/ReadVariableOp_1+while/peephole_lstm_cell_1/ReadVariableOp_12Z
+while/peephole_lstm_cell_1/ReadVariableOp_2+while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ér
ú
B__inference_rnn_1_layer_call_and_return_conditional_losses_1669814
inputs_0F
3peephole_lstm_cell_1_matmul_readvariableop_resource:	 H
5peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 C
4peephole_lstm_cell_1_biasadd_readvariableop_resource:	:
,peephole_lstm_cell_1_readvariableop_resource: <
.peephole_lstm_cell_1_readvariableop_1_resource: <
.peephole_lstm_cell_1_readvariableop_2_resource: 
identity¢+peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢*peephole_lstm_cell_1/MatMul/ReadVariableOp¢,peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢#peephole_lstm_cell_1/ReadVariableOp¢%peephole_lstm_cell_1/ReadVariableOp_1¢%peephole_lstm_cell_1/ReadVariableOp_2¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2Í
*peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp3peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell_1/MatMul/ReadVariableOpÅ
peephole_lstm_cell_1/MatMulMatMulstrided_slice_2:output:02peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMulÓ
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp5peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02.
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpÁ
peephole_lstm_cell_1/MatMul_1MatMulzeros:output:04peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMul_1À
peephole_lstm_cell_1/addAddV2%peephole_lstm_cell_1/MatMul:product:0'peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/addÌ
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp4peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpÍ
peephole_lstm_cell_1/BiasAddBiasAddpeephole_lstm_cell_1/add:z:03peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/BiasAdd
$peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$peephole_lstm_cell_1/split/split_dim
peephole_lstm_cell_1/splitSplit-peephole_lstm_cell_1/split/split_dim:output:0%peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell_1/split³
#peephole_lstm_cell_1/ReadVariableOpReadVariableOp,peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell_1/ReadVariableOp¬
peephole_lstm_cell_1/mulMul+peephole_lstm_cell_1/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul¶
peephole_lstm_cell_1/add_1AddV2#peephole_lstm_cell_1/split:output:0peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_1
peephole_lstm_cell_1/SigmoidSigmoidpeephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Sigmoid¹
%peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp.peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_1²
peephole_lstm_cell_1/mul_1Mul-peephole_lstm_cell_1/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_1¸
peephole_lstm_cell_1/add_2AddV2#peephole_lstm_cell_1/split:output:1peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_2
peephole_lstm_cell_1/Sigmoid_1Sigmoidpeephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_1§
peephole_lstm_cell_1/mul_2Mul"peephole_lstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_2
peephole_lstm_cell_1/TanhTanh#peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh²
peephole_lstm_cell_1/mul_3Mul peephole_lstm_cell_1/Sigmoid:y:0peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_3³
peephole_lstm_cell_1/add_3AddV2peephole_lstm_cell_1/mul_2:z:0peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_3¹
%peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp.peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_2À
peephole_lstm_cell_1/mul_4Mul-peephole_lstm_cell_1/ReadVariableOp_2:value:0peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_4¸
peephole_lstm_cell_1/add_4AddV2#peephole_lstm_cell_1/split:output:3peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_4
peephole_lstm_cell_1/Sigmoid_2Sigmoidpeephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_2
peephole_lstm_cell_1/Tanh_1Tanhpeephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh_1¶
peephole_lstm_cell_1/mul_5Mul"peephole_lstm_cell_1/Sigmoid_2:y:0peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¨
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:03peephole_lstm_cell_1_matmul_readvariableop_resource5peephole_lstm_cell_1_matmul_1_readvariableop_resource4peephole_lstm_cell_1_biasadd_readvariableop_resource,peephole_lstm_cell_1_readvariableop_resource.peephole_lstm_cell_1_readvariableop_1_resource.peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1669713*
condR
while_cond_1669712*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1ô
IdentityIdentitystrided_slice_3:output:0,^peephole_lstm_cell_1/BiasAdd/ReadVariableOp+^peephole_lstm_cell_1/MatMul/ReadVariableOp-^peephole_lstm_cell_1/MatMul_1/ReadVariableOp$^peephole_lstm_cell_1/ReadVariableOp&^peephole_lstm_cell_1/ReadVariableOp_1&^peephole_lstm_cell_1/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2Z
+peephole_lstm_cell_1/BiasAdd/ReadVariableOp+peephole_lstm_cell_1/BiasAdd/ReadVariableOp2X
*peephole_lstm_cell_1/MatMul/ReadVariableOp*peephole_lstm_cell_1/MatMul/ReadVariableOp2\
,peephole_lstm_cell_1/MatMul_1/ReadVariableOp,peephole_lstm_cell_1/MatMul_1/ReadVariableOp2J
#peephole_lstm_cell_1/ReadVariableOp#peephole_lstm_cell_1/ReadVariableOp2N
%peephole_lstm_cell_1/ReadVariableOp_1%peephole_lstm_cell_1/ReadVariableOp_12N
%peephole_lstm_cell_1/ReadVariableOp_2%peephole_lstm_cell_1/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
»'
½
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1665993

inputs

states
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Ï,
º	
while_body_1665518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
"while_peephole_lstm_cell_1665542_0:	5
"while_peephole_lstm_cell_1665544_0:	 1
"while_peephole_lstm_cell_1665546_0:	0
"while_peephole_lstm_cell_1665548_0: 0
"while_peephole_lstm_cell_1665550_0: 0
"while_peephole_lstm_cell_1665552_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
 while_peephole_lstm_cell_1665542:	3
 while_peephole_lstm_cell_1665544:	 /
 while_peephole_lstm_cell_1665546:	.
 while_peephole_lstm_cell_1665548: .
 while_peephole_lstm_cell_1665550: .
 while_peephole_lstm_cell_1665552: ¢0while/peephole_lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemþ
0while/peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3"while_peephole_lstm_cell_1665542_0"while_peephole_lstm_cell_1665544_0"while_peephole_lstm_cell_1665546_0"while_peephole_lstm_cell_1665548_0"while_peephole_lstm_cell_1665550_0"while_peephole_lstm_cell_1665552_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_166542222
0while/peephole_lstm_cell/StatefulPartitionedCallý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/peephole_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity¤
while/Identity_1Identitywhile_while_maximum_iterations1^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2À
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ð
while/Identity_4Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:11^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4Ð
while/Identity_5Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:21^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"F
 while_peephole_lstm_cell_1665542"while_peephole_lstm_cell_1665542_0"F
 while_peephole_lstm_cell_1665544"while_peephole_lstm_cell_1665544_0"F
 while_peephole_lstm_cell_1665546"while_peephole_lstm_cell_1665546_0"F
 while_peephole_lstm_cell_1665548"while_peephole_lstm_cell_1665548_0"F
 while_peephole_lstm_cell_1665550"while_peephole_lstm_cell_1665550_0"F
 while_peephole_lstm_cell_1665552"while_peephole_lstm_cell_1665552_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2d
0while/peephole_lstm_cell/StatefulPartitionedCall0while/peephole_lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

Ê
6__inference_peephole_lstm_cell_1_layer_call_fn_1670686

inputs
states_0
states_1
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity

identity_1

identity_2¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_16659932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
Î	
ó
B__inference_dense_layer_call_and_return_conditional_losses_1670432

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï
ô
#sequential_rnn_1_while_body_1665041>
:sequential_rnn_1_while_sequential_rnn_1_while_loop_counterD
@sequential_rnn_1_while_sequential_rnn_1_while_maximum_iterations&
"sequential_rnn_1_while_placeholder(
$sequential_rnn_1_while_placeholder_1(
$sequential_rnn_1_while_placeholder_2(
$sequential_rnn_1_while_placeholder_3=
9sequential_rnn_1_while_sequential_rnn_1_strided_slice_1_0y
usequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor_0_
Lsequential_rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 a
Nsequential_rnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 \
Msequential_rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	S
Esequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0: U
Gsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0: U
Gsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0: #
sequential_rnn_1_while_identity%
!sequential_rnn_1_while_identity_1%
!sequential_rnn_1_while_identity_2%
!sequential_rnn_1_while_identity_3%
!sequential_rnn_1_while_identity_4%
!sequential_rnn_1_while_identity_5;
7sequential_rnn_1_while_sequential_rnn_1_strided_slice_1w
ssequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor]
Jsequential_rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 _
Lsequential_rnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 Z
Ksequential_rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	Q
Csequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_resource: S
Esequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource: S
Esequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource: ¢Bsequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢Asequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢Csequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢:sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp¢<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1¢<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2å
Hsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2J
Hsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape¹
:sequential/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor_0"sequential_rnn_1_while_placeholderQsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02<
:sequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem
Asequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOpLsequential_rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02C
Asequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp³
2sequential/rnn_1/while/peephole_lstm_cell_1/MatMulMatMulAsequential/rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0Isequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/rnn_1/while/peephole_lstm_cell_1/MatMul
Csequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpNsequential_rnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02E
Csequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp
4sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1MatMul$sequential_rnn_1_while_placeholder_2Ksequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1
/sequential/rnn_1/while/peephole_lstm_cell_1/addAddV2<sequential/rnn_1/while/peephole_lstm_cell_1/MatMul:product:0>sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/rnn_1/while/peephole_lstm_cell_1/add
Bsequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpMsequential_rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02D
Bsequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp©
3sequential/rnn_1/while/peephole_lstm_cell_1/BiasAddBiasAdd3sequential/rnn_1/while/peephole_lstm_cell_1/add:z:0Jsequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd¼
;sequential/rnn_1/while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential/rnn_1/while/peephole_lstm_cell_1/split/split_dimï
1sequential/rnn_1/while/peephole_lstm_cell_1/splitSplitDsequential/rnn_1/while/peephole_lstm_cell_1/split/split_dim:output:0<sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split23
1sequential/rnn_1/while/peephole_lstm_cell_1/splitú
:sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOpReadVariableOpEsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype02<
:sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp
/sequential/rnn_1/while/peephole_lstm_cell_1/mulMulBsequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp:value:0$sequential_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/sequential/rnn_1/while/peephole_lstm_cell_1/mul
1sequential/rnn_1/while/peephole_lstm_cell_1/add_1AddV2:sequential/rnn_1/while/peephole_lstm_cell_1/split:output:03sequential/rnn_1/while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/add_1Þ
3sequential/rnn_1/while/peephole_lstm_cell_1/SigmoidSigmoid5sequential/rnn_1/while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 25
3sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid
<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOpGsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype02>
<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_1MulDsequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1:value:0$sequential_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_1
1sequential/rnn_1/while/peephole_lstm_cell_1/add_2AddV2:sequential/rnn_1/while/peephole_lstm_cell_1/split:output:15sequential/rnn_1/while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/add_2â
5sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid_1Sigmoid5sequential/rnn_1/while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 27
5sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid_1
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_2Mul9sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid_1:y:0$sequential_rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_2Ú
0sequential/rnn_1/while/peephole_lstm_cell_1/TanhTanh:sequential/rnn_1/while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/rnn_1/while/peephole_lstm_cell_1/Tanh
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_3Mul7sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid:y:04sequential/rnn_1/while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_3
1sequential/rnn_1/while/peephole_lstm_cell_1/add_3AddV25sequential/rnn_1/while/peephole_lstm_cell_1/mul_2:z:05sequential/rnn_1/while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/add_3
<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOpGsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype02>
<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_4MulDsequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2:value:05sequential/rnn_1/while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_4
1sequential/rnn_1/while/peephole_lstm_cell_1/add_4AddV2:sequential/rnn_1/while/peephole_lstm_cell_1/split:output:35sequential/rnn_1/while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/add_4â
5sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid_2Sigmoid5sequential/rnn_1/while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 27
5sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid_2Ù
2sequential/rnn_1/while/peephole_lstm_cell_1/Tanh_1Tanh5sequential/rnn_1/while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 24
2sequential/rnn_1/while/peephole_lstm_cell_1/Tanh_1
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_5Mul9sequential/rnn_1/while/peephole_lstm_cell_1/Sigmoid_2:y:06sequential/rnn_1/while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 23
1sequential/rnn_1/while/peephole_lstm_cell_1/mul_5½
;sequential/rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$sequential_rnn_1_while_placeholder_1"sequential_rnn_1_while_placeholder5sequential/rnn_1/while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype02=
;sequential/rnn_1/while/TensorArrayV2Write/TensorListSetItem~
sequential/rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/rnn_1/while/add/y­
sequential/rnn_1/while/addAddV2"sequential_rnn_1_while_placeholder%sequential/rnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/while/add
sequential/rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential/rnn_1/while/add_1/yË
sequential/rnn_1/while/add_1AddV2:sequential_rnn_1_while_sequential_rnn_1_while_loop_counter'sequential/rnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/while/add_1
sequential/rnn_1/while/IdentityIdentity sequential/rnn_1/while/add_1:z:0C^sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpB^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpD^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp;^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2!
sequential/rnn_1/while/Identity¿
!sequential/rnn_1/while/Identity_1Identity@sequential_rnn_1_while_sequential_rnn_1_while_maximum_iterationsC^sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpB^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpD^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp;^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2#
!sequential/rnn_1/while/Identity_1
!sequential/rnn_1/while/Identity_2Identitysequential/rnn_1/while/add:z:0C^sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpB^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpD^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp;^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2#
!sequential/rnn_1/while/Identity_2Ê
!sequential/rnn_1/while/Identity_3IdentityKsequential/rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0C^sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpB^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpD^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp;^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2#
!sequential/rnn_1/while/Identity_3Å
!sequential/rnn_1/while/Identity_4Identity5sequential/rnn_1/while/peephole_lstm_cell_1/mul_5:z:0C^sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpB^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpD^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp;^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/rnn_1/while/Identity_4Å
!sequential/rnn_1/while/Identity_5Identity5sequential/rnn_1/while/peephole_lstm_cell_1/add_3:z:0C^sequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpB^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpD^sequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp;^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1=^sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/rnn_1/while/Identity_5"K
sequential_rnn_1_while_identity(sequential/rnn_1/while/Identity:output:0"O
!sequential_rnn_1_while_identity_1*sequential/rnn_1/while/Identity_1:output:0"O
!sequential_rnn_1_while_identity_2*sequential/rnn_1/while/Identity_2:output:0"O
!sequential_rnn_1_while_identity_3*sequential/rnn_1/while/Identity_3:output:0"O
!sequential_rnn_1_while_identity_4*sequential/rnn_1/while/Identity_4:output:0"O
!sequential_rnn_1_while_identity_5*sequential/rnn_1/while/Identity_5:output:0"
Ksequential_rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resourceMsequential_rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"
Lsequential_rnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resourceNsequential_rnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"
Jsequential_rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resourceLsequential_rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"
Esequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resourceGsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0"
Esequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resourceGsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0"
Csequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_resourceEsequential_rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0"t
7sequential_rnn_1_while_sequential_rnn_1_strided_slice_19sequential_rnn_1_while_sequential_rnn_1_strided_slice_1_0"ì
ssequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensorusequential_rnn_1_while_tensorarrayv2read_tensorlistgetitem_sequential_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2
Bsequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpBsequential/rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2
Asequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpAsequential/rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp2
Csequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpCsequential/rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2x
:sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp:sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2|
<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12|
<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2<sequential/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Í
ý
%__inference_rnn_layer_call_fn_1669617

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16669082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î	
ó
B__inference_dense_layer_call_and_return_conditional_losses_1667125

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

`
D__inference_reshape_layer_call_and_return_conditional_losses_1668841

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»'
½
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1666180

inputs

states
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Îl
»
rnn_while_body_1668425$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0P
=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0:	R
?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 M
>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	D
6rnn_while_peephole_lstm_cell_readvariableop_resource_0: F
8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0: F
8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0: 
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorN
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resource:	P
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 K
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource:	B
4rnn_while_peephole_lstm_cell_readvariableop_resource: D
6rnn_while_peephole_lstm_cell_readvariableop_1_resource: D
6rnn_while_peephole_lstm_cell_readvariableop_2_resource: ¢3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp¢4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢+rnn/while/peephole_lstm_cell/ReadVariableOp¢-rnn/while/peephole_lstm_cell/ReadVariableOp_1¢-rnn/while/peephole_lstm_cell/ReadVariableOp_2Ë
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeë
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItemç
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype024
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpù
#rnn/while/peephole_lstm_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0:rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn/while/peephole_lstm_cell/MatMulí
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype026
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpâ
%rnn/while/peephole_lstm_cell/MatMul_1MatMulrnn_while_placeholder_2<rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/while/peephole_lstm_cell/MatMul_1à
 rnn/while/peephole_lstm_cell/addAddV2-rnn/while/peephole_lstm_cell/MatMul:product:0/rnn/while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 rnn/while/peephole_lstm_cell/addæ
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype025
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpí
$rnn/while/peephole_lstm_cell/BiasAddBiasAdd$rnn/while/peephole_lstm_cell/add:z:0;rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rnn/while/peephole_lstm_cell/BiasAdd
,rnn/while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,rnn/while/peephole_lstm_cell/split/split_dim³
"rnn/while/peephole_lstm_cell/splitSplit5rnn/while/peephole_lstm_cell/split/split_dim:output:0-rnn/while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2$
"rnn/while/peephole_lstm_cell/splitÍ
+rnn/while/peephole_lstm_cell/ReadVariableOpReadVariableOp6rnn_while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02-
+rnn/while/peephole_lstm_cell/ReadVariableOpË
 rnn/while/peephole_lstm_cell/mulMul3rnn/while/peephole_lstm_cell/ReadVariableOp:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn/while/peephole_lstm_cell/mulÖ
"rnn/while/peephole_lstm_cell/add_1AddV2+rnn/while/peephole_lstm_cell/split:output:0$rnn/while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_1±
$rnn/while/peephole_lstm_cell/SigmoidSigmoid&rnn/while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn/while/peephole_lstm_cell/SigmoidÓ
-rnn/while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_1Ñ
"rnn/while/peephole_lstm_cell/mul_1Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_1:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_1Ø
"rnn/while/peephole_lstm_cell/add_2AddV2+rnn/while/peephole_lstm_cell/split:output:1&rnn/while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_2µ
&rnn/while/peephole_lstm_cell/Sigmoid_1Sigmoid&rnn/while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn/while/peephole_lstm_cell/Sigmoid_1Æ
"rnn/while/peephole_lstm_cell/mul_2Mul*rnn/while/peephole_lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_2­
!rnn/while/peephole_lstm_cell/TanhTanh+rnn/while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rnn/while/peephole_lstm_cell/TanhÒ
"rnn/while/peephole_lstm_cell/mul_3Mul(rnn/while/peephole_lstm_cell/Sigmoid:y:0%rnn/while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_3Ó
"rnn/while/peephole_lstm_cell/add_3AddV2&rnn/while/peephole_lstm_cell/mul_2:z:0&rnn/while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_3Ó
-rnn/while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_2à
"rnn/while/peephole_lstm_cell/mul_4Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_2:value:0&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_4Ø
"rnn/while/peephole_lstm_cell/add_4AddV2+rnn/while/peephole_lstm_cell/split:output:3&rnn/while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_4µ
&rnn/while/peephole_lstm_cell/Sigmoid_2Sigmoid&rnn/while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn/while/peephole_lstm_cell/Sigmoid_2¬
#rnn/while/peephole_lstm_cell/Tanh_1Tanh&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#rnn/while/peephole_lstm_cell/Tanh_1Ö
"rnn/while/peephole_lstm_cell/mul_5Mul*rnn/while/peephole_lstm_cell/Sigmoid_2:y:0'rnn/while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_5ú
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder&rnn/while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1
rnn/while/IdentityIdentityrnn/while/add_1:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity±
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations4^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_1
rnn/while/Identity_2Identityrnn/while/add:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_2É
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_3Â
rnn/while/Identity_4Identity&rnn/while/peephole_lstm_cell/mul_5:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/while/Identity_4Â
rnn/while/Identity_5Identity&rnn/while/peephole_lstm_cell/add_3:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/while/Identity_5"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"~
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0"
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"|
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resource=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_1_resource8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_2_resource8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0"n
4rnn_while_peephole_lstm_cell_readvariableop_resource6rnn_while_peephole_lstm_cell_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"¸
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2j
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2h
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2l
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp2Z
+rnn/while/peephole_lstm_cell/ReadVariableOp+rnn/while/peephole_lstm_cell/ReadVariableOp2^
-rnn/while/peephole_lstm_cell/ReadVariableOp_1-rnn/while/peephole_lstm_cell/ReadVariableOp_12^
-rnn/while/peephole_lstm_cell/ReadVariableOp_2-rnn/while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


í
while_cond_1666275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1666275___redundant_placeholder05
1while_while_cond_1666275___redundant_placeholder15
1while_while_cond_1666275___redundant_placeholder25
1while_while_cond_1666275___redundant_placeholder35
1while_while_cond_1666275___redundant_placeholder45
1while_while_cond_1666275___redundant_placeholder55
1while_while_cond_1666275___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


í
while_cond_1668924
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1668924___redundant_placeholder05
1while_while_cond_1668924___redundant_placeholder15
1while_while_cond_1668924___redundant_placeholder25
1while_while_cond_1668924___redundant_placeholder35
1while_while_cond_1668924___redundant_placeholder45
1while_while_cond_1668924___redundant_placeholder55
1while_while_cond_1668924___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:

È
4__inference_peephole_lstm_cell_layer_call_fn_1670552

inputs
states_0
states_1
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity

identity_1

identity_2¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_16652352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
Ã'
¿
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1670663

inputs
states_0
states_11
matmul_readvariableop_resource:	 3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
¤H
«
B__inference_rnn_1_layer_call_and_return_conditional_losses_1666093

inputs/
peephole_lstm_cell_1_1665994:	 /
peephole_lstm_cell_1_1665996:	 +
peephole_lstm_cell_1_1665998:	*
peephole_lstm_cell_1_1666000: *
peephole_lstm_cell_1_1666002: *
peephole_lstm_cell_1_1666004: 
identity¢,peephole_lstm_cell_1/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2´
,peephole_lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0peephole_lstm_cell_1_1665994peephole_lstm_cell_1_1665996peephole_lstm_cell_1_1665998peephole_lstm_cell_1_1666000peephole_lstm_cell_1_1666002peephole_lstm_cell_1_1666004*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_16659932.
,peephole_lstm_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¬
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0peephole_lstm_cell_1_1665994peephole_lstm_cell_1_1665996peephole_lstm_cell_1_1665998peephole_lstm_cell_1_1666000peephole_lstm_cell_1_1666002peephole_lstm_cell_1_1666004*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1666013*
condR
while_cond_1666012*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1£
IdentityIdentitystrided_slice_3:output:0-^peephole_lstm_cell_1/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2\
,peephole_lstm_cell_1/StatefulPartitionedCall,peephole_lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_1669464
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1669464___redundant_placeholder05
1while_while_cond_1669464___redundant_placeholder15
1while_while_cond_1669464___redundant_placeholder25
1while_while_cond_1669464___redundant_placeholder35
1while_while_cond_1669464___redundant_placeholder45
1while_while_cond_1669464___redundant_placeholder55
1while_while_cond_1669464___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¥0
¸
C__inference_conv1d_layer_call_and_return_conditional_losses_1666708

inputsA
+conv1d_expanddims_1_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢"conv1d/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1f
conv1d/ShapeShapeconv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/Shape
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
conv1d/strided_slice/stack_1
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2
conv1d/Reshape/shape 
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ReshapeÂ
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d/Conv2D
conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/concat/values_1s
conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
conv1d/concat/axisµ
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/Reshape_1 
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack§
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2*
(squeeze_batch_dims/strided_slice/stack_1
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2Ò
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2"
 squeeze_batch_dims/Reshape/shape½
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
squeeze_batch_dims/ReshapeÅ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOpÑ
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
squeeze_batch_dims/BiasAdd
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
squeeze_batch_dims/concat/axisñ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concatÊ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
squeeze_batch_dims/Reshape_1Ò
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
p
Þ
@__inference_rnn_layer_call_and_return_conditional_losses_1669386

inputsD
1peephole_lstm_cell_matmul_readvariableop_resource:	F
3peephole_lstm_cell_matmul_1_readvariableop_resource:	 A
2peephole_lstm_cell_biasadd_readvariableop_resource:	8
*peephole_lstm_cell_readvariableop_resource: :
,peephole_lstm_cell_readvariableop_1_resource: :
,peephole_lstm_cell_readvariableop_2_resource: 
identity¢)peephole_lstm_cell/BiasAdd/ReadVariableOp¢(peephole_lstm_cell/MatMul/ReadVariableOp¢*peephole_lstm_cell/MatMul_1/ReadVariableOp¢!peephole_lstm_cell/ReadVariableOp¢#peephole_lstm_cell/ReadVariableOp_1¢#peephole_lstm_cell/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ç
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp¿
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMulÍ
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp»
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMul_1¸
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/addÆ
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOpÅ
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/BiasAdd
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell/split­
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp¦
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul®
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_1
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid³
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1¬
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_1°
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_2
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_1¡
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_2
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanhª
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_3«
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_3³
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2¸
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_4°
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_4
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_2
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanh_1®
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1669285*
condR
while_cond_1669284*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ã
IdentityIdentitytranspose_1:y:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á'
½
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1670485

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
p
Þ
@__inference_rnn_layer_call_and_return_conditional_losses_1667590

inputsD
1peephole_lstm_cell_matmul_readvariableop_resource:	F
3peephole_lstm_cell_matmul_1_readvariableop_resource:	 A
2peephole_lstm_cell_biasadd_readvariableop_resource:	8
*peephole_lstm_cell_readvariableop_resource: :
,peephole_lstm_cell_readvariableop_1_resource: :
,peephole_lstm_cell_readvariableop_2_resource: 
identity¢)peephole_lstm_cell/BiasAdd/ReadVariableOp¢(peephole_lstm_cell/MatMul/ReadVariableOp¢*peephole_lstm_cell/MatMul_1/ReadVariableOp¢!peephole_lstm_cell/ReadVariableOp¢#peephole_lstm_cell/ReadVariableOp_1¢#peephole_lstm_cell/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ç
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp¿
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMulÍ
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp»
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMul_1¸
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/addÆ
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOpÅ
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/BiasAdd
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell/split­
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp¦
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul®
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_1
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid³
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1¬
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_1°
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_2
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_1¡
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_2
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanhª
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_3«
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_3³
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2¸
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_4°
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_4
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_2
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanh_1®
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1667489*
condR
while_cond_1667488*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ã
IdentityIdentitytranspose_1:y:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1666999
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1666999___redundant_placeholder05
1while_while_cond_1666999___redundant_placeholder15
1while_while_cond_1666999___redundant_placeholder25
1while_while_cond_1666999___redundant_placeholder35
1while_while_cond_1666999___redundant_placeholder45
1while_while_cond_1666999___redundant_placeholder55
1while_while_cond_1666999___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¸d

while_body_1669105
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_peephole_lstm_cell_matmul_readvariableop_resource_0:	N
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 I
:while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	@
2while_peephole_lstm_cell_readvariableop_resource_0: B
4while_peephole_lstm_cell_readvariableop_1_resource_0: B
4while_peephole_lstm_cell_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_peephole_lstm_cell_matmul_readvariableop_resource:	L
9while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 G
8while_peephole_lstm_cell_biasadd_readvariableop_resource:	>
0while_peephole_lstm_cell_readvariableop_resource: @
2while_peephole_lstm_cell_readvariableop_1_resource: @
2while_peephole_lstm_cell_readvariableop_2_resource: ¢/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢.while/peephole_lstm_cell/MatMul/ReadVariableOp¢0while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢'while/peephole_lstm_cell/ReadVariableOp¢)while/peephole_lstm_cell/ReadVariableOp_1¢)while/peephole_lstm_cell/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÛ
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOpé
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/peephole_lstm_cell/MatMulá
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpÒ
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell/MatMul_1Ð
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/peephole_lstm_cell/addÚ
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpÝ
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/peephole_lstm_cell/BiasAdd
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim£
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2 
while/peephole_lstm_cell/splitÁ
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp»
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/mulÆ
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_1¥
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell/SigmoidÇ
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1Á
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_1È
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_2©
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_1¶
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_2¡
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/TanhÂ
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_3Ã
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_3Ç
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2Ð
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_4È
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_4©
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_2 
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell/Tanh_1Æ
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_5æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ö
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1ø
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2¥
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

 
,__inference_sequential_layer_call_fn_1667167
conv1d_input
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:	 
	unknown_3:	
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:	 
	unknown_8:	 
	unknown_9:	

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16671322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv1d_input


í
while_cond_1666012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1666012___redundant_placeholder05
1while_while_cond_1666012___redundant_placeholder15
1while_while_cond_1666012___redundant_placeholder25
1while_while_cond_1666012___redundant_placeholder35
1while_while_cond_1666012___redundant_placeholder45
1while_while_cond_1666012___redundant_placeholder55
1while_while_cond_1666012___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


í
while_cond_1665254
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1665254___redundant_placeholder05
1while_while_cond_1665254___redundant_placeholder15
1while_while_cond_1665254___redundant_placeholder25
1while_while_cond_1665254___redundant_placeholder35
1while_while_cond_1665254___redundant_placeholder45
1while_while_cond_1665254___redundant_placeholder55
1while_while_cond_1665254___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
r
ø
B__inference_rnn_1_layer_call_and_return_conditional_losses_1670354

inputsF
3peephole_lstm_cell_1_matmul_readvariableop_resource:	 H
5peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 C
4peephole_lstm_cell_1_biasadd_readvariableop_resource:	:
,peephole_lstm_cell_1_readvariableop_resource: <
.peephole_lstm_cell_1_readvariableop_1_resource: <
.peephole_lstm_cell_1_readvariableop_2_resource: 
identity¢+peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢*peephole_lstm_cell_1/MatMul/ReadVariableOp¢,peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢#peephole_lstm_cell_1/ReadVariableOp¢%peephole_lstm_cell_1/ReadVariableOp_1¢%peephole_lstm_cell_1/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2Í
*peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp3peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell_1/MatMul/ReadVariableOpÅ
peephole_lstm_cell_1/MatMulMatMulstrided_slice_2:output:02peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMulÓ
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp5peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02.
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpÁ
peephole_lstm_cell_1/MatMul_1MatMulzeros:output:04peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMul_1À
peephole_lstm_cell_1/addAddV2%peephole_lstm_cell_1/MatMul:product:0'peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/addÌ
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp4peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpÍ
peephole_lstm_cell_1/BiasAddBiasAddpeephole_lstm_cell_1/add:z:03peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/BiasAdd
$peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$peephole_lstm_cell_1/split/split_dim
peephole_lstm_cell_1/splitSplit-peephole_lstm_cell_1/split/split_dim:output:0%peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell_1/split³
#peephole_lstm_cell_1/ReadVariableOpReadVariableOp,peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell_1/ReadVariableOp¬
peephole_lstm_cell_1/mulMul+peephole_lstm_cell_1/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul¶
peephole_lstm_cell_1/add_1AddV2#peephole_lstm_cell_1/split:output:0peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_1
peephole_lstm_cell_1/SigmoidSigmoidpeephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Sigmoid¹
%peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp.peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_1²
peephole_lstm_cell_1/mul_1Mul-peephole_lstm_cell_1/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_1¸
peephole_lstm_cell_1/add_2AddV2#peephole_lstm_cell_1/split:output:1peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_2
peephole_lstm_cell_1/Sigmoid_1Sigmoidpeephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_1§
peephole_lstm_cell_1/mul_2Mul"peephole_lstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_2
peephole_lstm_cell_1/TanhTanh#peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh²
peephole_lstm_cell_1/mul_3Mul peephole_lstm_cell_1/Sigmoid:y:0peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_3³
peephole_lstm_cell_1/add_3AddV2peephole_lstm_cell_1/mul_2:z:0peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_3¹
%peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp.peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_2À
peephole_lstm_cell_1/mul_4Mul-peephole_lstm_cell_1/ReadVariableOp_2:value:0peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_4¸
peephole_lstm_cell_1/add_4AddV2#peephole_lstm_cell_1/split:output:3peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_4
peephole_lstm_cell_1/Sigmoid_2Sigmoidpeephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_2
peephole_lstm_cell_1/Tanh_1Tanhpeephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh_1¶
peephole_lstm_cell_1/mul_5Mul"peephole_lstm_cell_1/Sigmoid_2:y:0peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¨
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:03peephole_lstm_cell_1_matmul_readvariableop_resource5peephole_lstm_cell_1_matmul_1_readvariableop_resource4peephole_lstm_cell_1_biasadd_readvariableop_resource,peephole_lstm_cell_1_readvariableop_resource.peephole_lstm_cell_1_readvariableop_1_resource.peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1670253*
condR
while_cond_1670252*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ô
IdentityIdentitystrided_slice_3:output:0,^peephole_lstm_cell_1/BiasAdd/ReadVariableOp+^peephole_lstm_cell_1/MatMul/ReadVariableOp-^peephole_lstm_cell_1/MatMul_1/ReadVariableOp$^peephole_lstm_cell_1/ReadVariableOp&^peephole_lstm_cell_1/ReadVariableOp_1&^peephole_lstm_cell_1/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2Z
+peephole_lstm_cell_1/BiasAdd/ReadVariableOp+peephole_lstm_cell_1/BiasAdd/ReadVariableOp2X
*peephole_lstm_cell_1/MatMul/ReadVariableOp*peephole_lstm_cell_1/MatMul/ReadVariableOp2\
,peephole_lstm_cell_1/MatMul_1/ReadVariableOp,peephole_lstm_cell_1/MatMul_1/ReadVariableOp2J
#peephole_lstm_cell_1/ReadVariableOp#peephole_lstm_cell_1/ReadVariableOp2N
%peephole_lstm_cell_1/ReadVariableOp_1%peephole_lstm_cell_1/ReadVariableOp_12N
%peephole_lstm_cell_1/ReadVariableOp_2%peephole_lstm_cell_1/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Ê
6__inference_peephole_lstm_cell_1_layer_call_fn_1670709

inputs
states_0
states_1
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity

identity_1

identity_2¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_16661802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
g
»
while_body_1667275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0N
;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 P
=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 K
<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	B
4while_peephole_lstm_cell_1_readvariableop_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_1_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorL
9while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2while_peephole_lstm_cell_1_readvariableop_resource: B
4while_peephole_lstm_cell_1_readvariableop_1_resource: B
4while_peephole_lstm_cell_1_readvariableop_2_resource: ¢1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)while/peephole_lstm_cell_1/ReadVariableOp¢+while/peephole_lstm_cell_1/ReadVariableOp_1¢+while/peephole_lstm_cell_1/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpï
!while/peephole_lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:08while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell_1/MatMulç
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype024
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpØ
#while/peephole_lstm_cell_1/MatMul_1MatMulwhile_placeholder_2:while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#while/peephole_lstm_cell_1/MatMul_1Ø
while/peephole_lstm_cell_1/addAddV2+while/peephole_lstm_cell_1/MatMul:product:0-while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/peephole_lstm_cell_1/addà
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"while/peephole_lstm_cell_1/BiasAddBiasAdd"while/peephole_lstm_cell_1/add:z:09while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"while/peephole_lstm_cell_1/BiasAdd
*while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*while/peephole_lstm_cell_1/split/split_dim«
 while/peephole_lstm_cell_1/splitSplit3while/peephole_lstm_cell_1/split/split_dim:output:0+while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 while/peephole_lstm_cell_1/splitÇ
)while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp4while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell_1/ReadVariableOpÁ
while/peephole_lstm_cell_1/mulMul1while/peephole_lstm_cell_1/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell_1/mulÎ
 while/peephole_lstm_cell_1/add_1AddV2)while/peephole_lstm_cell_1/split:output:0"while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_1«
"while/peephole_lstm_cell_1/SigmoidSigmoid$while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell_1/SigmoidÍ
+while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_1Ç
 while/peephole_lstm_cell_1/mul_1Mul3while/peephole_lstm_cell_1/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_1Ð
 while/peephole_lstm_cell_1/add_2AddV2)while/peephole_lstm_cell_1/split:output:1$while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_2¯
$while/peephole_lstm_cell_1/Sigmoid_1Sigmoid$while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_1¼
 while/peephole_lstm_cell_1/mul_2Mul(while/peephole_lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_2§
while/peephole_lstm_cell_1/TanhTanh)while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell_1/TanhÊ
 while/peephole_lstm_cell_1/mul_3Mul&while/peephole_lstm_cell_1/Sigmoid:y:0#while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_3Ë
 while/peephole_lstm_cell_1/add_3AddV2$while/peephole_lstm_cell_1/mul_2:z:0$while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_3Í
+while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_2Ø
 while/peephole_lstm_cell_1/mul_4Mul3while/peephole_lstm_cell_1/ReadVariableOp_2:value:0$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_4Ð
 while/peephole_lstm_cell_1/add_4AddV2)while/peephole_lstm_cell_1/split:output:3$while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_4¯
$while/peephole_lstm_cell_1/Sigmoid_2Sigmoid$while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_2¦
!while/peephole_lstm_cell_1/Tanh_1Tanh$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!while/peephole_lstm_cell_1/Tanh_1Î
 while/peephole_lstm_cell_1/mul_5Mul(while/peephole_lstm_cell_1/Sigmoid_2:y:0%while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_5è
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations2^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2±
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3¬
while/Identity_4Identity$while/peephole_lstm_cell_1/mul_5:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4¬
while/Identity_5Identity$while/peephole_lstm_cell_1/add_3:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"|
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"x
9while_peephole_lstm_cell_1_matmul_readvariableop_resource;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_1_resource6while_peephole_lstm_cell_1_readvariableop_1_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_2_resource6while_peephole_lstm_cell_1_readvariableop_2_resource_0"j
2while_peephole_lstm_cell_1_readvariableop_resource4while_peephole_lstm_cell_1_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2f
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0while/peephole_lstm_cell_1/MatMul/ReadVariableOp0while/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)while/peephole_lstm_cell_1/ReadVariableOp)while/peephole_lstm_cell_1/ReadVariableOp2Z
+while/peephole_lstm_cell_1/ReadVariableOp_1+while/peephole_lstm_cell_1/ReadVariableOp_12Z
+while/peephole_lstm_cell_1/ReadVariableOp_2+while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Í
ý
%__inference_rnn_layer_call_fn_1669634

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16675902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
E
)__inference_reshape_layer_call_fn_1668846

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_16667272
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
p
Þ
@__inference_rnn_layer_call_and_return_conditional_losses_1666908

inputsD
1peephole_lstm_cell_matmul_readvariableop_resource:	F
3peephole_lstm_cell_matmul_1_readvariableop_resource:	 A
2peephole_lstm_cell_biasadd_readvariableop_resource:	8
*peephole_lstm_cell_readvariableop_resource: :
,peephole_lstm_cell_readvariableop_1_resource: :
,peephole_lstm_cell_readvariableop_2_resource: 
identity¢)peephole_lstm_cell/BiasAdd/ReadVariableOp¢(peephole_lstm_cell/MatMul/ReadVariableOp¢*peephole_lstm_cell/MatMul_1/ReadVariableOp¢!peephole_lstm_cell/ReadVariableOp¢#peephole_lstm_cell/ReadVariableOp_1¢#peephole_lstm_cell/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ç
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp¿
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMulÍ
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp»
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMul_1¸
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/addÆ
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOpÅ
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/BiasAdd
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell/split­
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp¦
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul®
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_1
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid³
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1¬
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_1°
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_2
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_1¡
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_2
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanhª
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_3«
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_3³
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2¸
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_4°
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_4
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_2
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanh_1®
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1666807*
condR
while_cond_1666806*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ã
IdentityIdentitytranspose_1:y:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï,
º	
while_body_1665255
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_05
"while_peephole_lstm_cell_1665279_0:	5
"while_peephole_lstm_cell_1665281_0:	 1
"while_peephole_lstm_cell_1665283_0:	0
"while_peephole_lstm_cell_1665285_0: 0
"while_peephole_lstm_cell_1665287_0: 0
"while_peephole_lstm_cell_1665289_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor3
 while_peephole_lstm_cell_1665279:	3
 while_peephole_lstm_cell_1665281:	 /
 while_peephole_lstm_cell_1665283:	.
 while_peephole_lstm_cell_1665285: .
 while_peephole_lstm_cell_1665287: .
 while_peephole_lstm_cell_1665289: ¢0while/peephole_lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemþ
0while/peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3"while_peephole_lstm_cell_1665279_0"while_peephole_lstm_cell_1665281_0"while_peephole_lstm_cell_1665283_0"while_peephole_lstm_cell_1665285_0"while_peephole_lstm_cell_1665287_0"while_peephole_lstm_cell_1665289_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_166523522
0while/peephole_lstm_cell/StatefulPartitionedCallý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/peephole_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity¤
while/Identity_1Identitywhile_while_maximum_iterations1^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2À
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ð
while/Identity_4Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:11^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4Ð
while/Identity_5Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:21^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"F
 while_peephole_lstm_cell_1665279"while_peephole_lstm_cell_1665279_0"F
 while_peephole_lstm_cell_1665281"while_peephole_lstm_cell_1665281_0"F
 while_peephole_lstm_cell_1665283"while_peephole_lstm_cell_1665283_0"F
 while_peephole_lstm_cell_1665285"while_peephole_lstm_cell_1665285_0"F
 while_peephole_lstm_cell_1665287"while_peephole_lstm_cell_1665287_0"F
 while_peephole_lstm_cell_1665289"while_peephole_lstm_cell_1665289_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2d
0while/peephole_lstm_cell/StatefulPartitionedCall0while/peephole_lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Á'
½
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1670529

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/1
÷
ÿ
%__inference_rnn_layer_call_fn_1669583
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16653352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¸d

while_body_1668925
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_peephole_lstm_cell_matmul_readvariableop_resource_0:	N
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 I
:while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	@
2while_peephole_lstm_cell_readvariableop_resource_0: B
4while_peephole_lstm_cell_readvariableop_1_resource_0: B
4while_peephole_lstm_cell_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_peephole_lstm_cell_matmul_readvariableop_resource:	L
9while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 G
8while_peephole_lstm_cell_biasadd_readvariableop_resource:	>
0while_peephole_lstm_cell_readvariableop_resource: @
2while_peephole_lstm_cell_readvariableop_1_resource: @
2while_peephole_lstm_cell_readvariableop_2_resource: ¢/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢.while/peephole_lstm_cell/MatMul/ReadVariableOp¢0while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢'while/peephole_lstm_cell/ReadVariableOp¢)while/peephole_lstm_cell/ReadVariableOp_1¢)while/peephole_lstm_cell/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÛ
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOpé
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/peephole_lstm_cell/MatMulá
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpÒ
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell/MatMul_1Ð
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/peephole_lstm_cell/addÚ
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpÝ
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/peephole_lstm_cell/BiasAdd
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim£
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2 
while/peephole_lstm_cell/splitÁ
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp»
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/mulÆ
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_1¥
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell/SigmoidÇ
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1Á
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_1È
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_2©
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_1¶
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_2¡
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/TanhÂ
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_3Ã
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_3Ç
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2Ð
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_4È
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_4©
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_2 
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell/Tanh_1Æ
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_5æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ö
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1ø
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2¥
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ñ
§
#sequential_rnn_1_while_cond_1665040>
:sequential_rnn_1_while_sequential_rnn_1_while_loop_counterD
@sequential_rnn_1_while_sequential_rnn_1_while_maximum_iterations&
"sequential_rnn_1_while_placeholder(
$sequential_rnn_1_while_placeholder_1(
$sequential_rnn_1_while_placeholder_2(
$sequential_rnn_1_while_placeholder_3@
<sequential_rnn_1_while_less_sequential_rnn_1_strided_slice_1W
Ssequential_rnn_1_while_sequential_rnn_1_while_cond_1665040___redundant_placeholder0W
Ssequential_rnn_1_while_sequential_rnn_1_while_cond_1665040___redundant_placeholder1W
Ssequential_rnn_1_while_sequential_rnn_1_while_cond_1665040___redundant_placeholder2W
Ssequential_rnn_1_while_sequential_rnn_1_while_cond_1665040___redundant_placeholder3W
Ssequential_rnn_1_while_sequential_rnn_1_while_cond_1665040___redundant_placeholder4W
Ssequential_rnn_1_while_sequential_rnn_1_while_cond_1665040___redundant_placeholder5W
Ssequential_rnn_1_while_sequential_rnn_1_while_cond_1665040___redundant_placeholder6#
sequential_rnn_1_while_identity
Å
sequential/rnn_1/while/LessLess"sequential_rnn_1_while_placeholder<sequential_rnn_1_while_less_sequential_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
sequential/rnn_1/while/Less
sequential/rnn_1/while/IdentityIdentitysequential/rnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2!
sequential/rnn_1/while/Identity"K
sequential_rnn_1_while_identity(sequential/rnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
â
Ó
G__inference_sequential_layer_call_and_return_conditional_losses_1667855
conv1d_input$
conv1d_1667817:
conv1d_1667819:
rnn_1667823:	
rnn_1667825:	 
rnn_1667827:	
rnn_1667829: 
rnn_1667831: 
rnn_1667833:  
rnn_1_1667836:	  
rnn_1_1667838:	 
rnn_1_1667840:	
rnn_1_1667842: 
rnn_1_1667844: 
rnn_1_1667846: 
dense_1667849: 
dense_1667851:
identity¢conv1d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢rnn/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_1667817conv1d_1667819*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_16667082 
conv1d/StatefulPartitionedCallú
reshape/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_16667272
reshape/PartitionedCallÛ
rnn/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0rnn_1667823rnn_1667825rnn_1667827rnn_1667829rnn_1667831rnn_1667833*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16675902
rnn/StatefulPartitionedCallí
rnn_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0rnn_1_1667836rnn_1_1667838rnn_1_1667840rnn_1_1667842rnn_1_1667844rnn_1_1667846*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16673762
rnn_1/StatefulPartitionedCall«
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_1667849dense_1667851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16671252
dense/StatefulPartitionedCallù
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv1d_input
Ïp
à
@__inference_rnn_layer_call_and_return_conditional_losses_1669026
inputs_0D
1peephole_lstm_cell_matmul_readvariableop_resource:	F
3peephole_lstm_cell_matmul_1_readvariableop_resource:	 A
2peephole_lstm_cell_biasadd_readvariableop_resource:	8
*peephole_lstm_cell_readvariableop_resource: :
,peephole_lstm_cell_readvariableop_1_resource: :
,peephole_lstm_cell_readvariableop_2_resource: 
identity¢)peephole_lstm_cell/BiasAdd/ReadVariableOp¢(peephole_lstm_cell/MatMul/ReadVariableOp¢*peephole_lstm_cell/MatMul_1/ReadVariableOp¢!peephole_lstm_cell/ReadVariableOp¢#peephole_lstm_cell/ReadVariableOp_1¢#peephole_lstm_cell/ReadVariableOp_2¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ç
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp¿
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMulÍ
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp»
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMul_1¸
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/addÆ
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOpÅ
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/BiasAdd
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell/split­
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp¦
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul®
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_1
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid³
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1¬
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_1°
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_2
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_1¡
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_2
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanhª
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_3«
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_3³
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2¸
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_4°
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_4
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_2
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanh_1®
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1668925*
condR
while_cond_1668924*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1ì
IdentityIdentitytranspose_1:y:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¸d

while_body_1667489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_peephole_lstm_cell_matmul_readvariableop_resource_0:	N
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 I
:while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	@
2while_peephole_lstm_cell_readvariableop_resource_0: B
4while_peephole_lstm_cell_readvariableop_1_resource_0: B
4while_peephole_lstm_cell_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_peephole_lstm_cell_matmul_readvariableop_resource:	L
9while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 G
8while_peephole_lstm_cell_biasadd_readvariableop_resource:	>
0while_peephole_lstm_cell_readvariableop_resource: @
2while_peephole_lstm_cell_readvariableop_1_resource: @
2while_peephole_lstm_cell_readvariableop_2_resource: ¢/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢.while/peephole_lstm_cell/MatMul/ReadVariableOp¢0while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢'while/peephole_lstm_cell/ReadVariableOp¢)while/peephole_lstm_cell/ReadVariableOp_1¢)while/peephole_lstm_cell/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÛ
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOpé
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/peephole_lstm_cell/MatMulá
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpÒ
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell/MatMul_1Ð
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/peephole_lstm_cell/addÚ
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpÝ
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/peephole_lstm_cell/BiasAdd
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim£
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2 
while/peephole_lstm_cell/splitÁ
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp»
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/mulÆ
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_1¥
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell/SigmoidÇ
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1Á
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_1È
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_2©
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_1¶
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_2¡
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/TanhÂ
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_3Ã
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_3Ç
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2Ð
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_4È
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_4©
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_2 
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell/Tanh_1Æ
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_5æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ö
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1ø
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2¥
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
À

(__inference_conv1d_layer_call_fn_1668828

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_16667082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹'
»
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1665235

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
¤H
«
B__inference_rnn_1_layer_call_and_return_conditional_losses_1666356

inputs/
peephole_lstm_cell_1_1666257:	 /
peephole_lstm_cell_1_1666259:	 +
peephole_lstm_cell_1_1666261:	*
peephole_lstm_cell_1_1666263: *
peephole_lstm_cell_1_1666265: *
peephole_lstm_cell_1_1666267: 
identity¢,peephole_lstm_cell_1/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2´
,peephole_lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0peephole_lstm_cell_1_1666257peephole_lstm_cell_1_1666259peephole_lstm_cell_1_1666261peephole_lstm_cell_1_1666263peephole_lstm_cell_1_1666265peephole_lstm_cell_1_1666267*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_16661802.
,peephole_lstm_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¬
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0peephole_lstm_cell_1_1666257peephole_lstm_cell_1_1666259peephole_lstm_cell_1_1666261peephole_lstm_cell_1_1666263peephole_lstm_cell_1_1666265peephole_lstm_cell_1_1666267*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1666276*
condR
while_cond_1666275*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1£
IdentityIdentitystrided_slice_3:output:0-^peephole_lstm_cell_1/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2\
,peephole_lstm_cell_1/StatefulPartitionedCall,peephole_lstm_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±s
±
rnn_1_while_body_1668601(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3'
#rnn_1_while_rnn_1_strided_slice_1_0c
_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0T
Arnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 V
Crnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 Q
Brnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	H
:rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0: J
<rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0: J
<rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
rnn_1_while_identity
rnn_1_while_identity_1
rnn_1_while_identity_2
rnn_1_while_identity_3
rnn_1_while_identity_4
rnn_1_while_identity_5%
!rnn_1_while_rnn_1_strided_slice_1a
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensorR
?rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 T
Arnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 O
@rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	F
8rnn_1_while_peephole_lstm_cell_1_readvariableop_resource: H
:rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource: H
:rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource: ¢7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp¢1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1¢1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2Ï
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2?
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape÷
/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0rnn_1_while_placeholderFrnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype021
/rnn_1/while/TensorArrayV2Read/TensorListGetItemó
6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOpArnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype028
6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp
'rnn_1/while/peephole_lstm_cell_1/MatMulMatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0>rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn_1/while/peephole_lstm_cell_1/MatMulù
8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpCrnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02:
8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpð
)rnn_1/while/peephole_lstm_cell_1/MatMul_1MatMulrnn_1_while_placeholder_2@rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn_1/while/peephole_lstm_cell_1/MatMul_1ð
$rnn_1/while/peephole_lstm_cell_1/addAddV21rnn_1/while/peephole_lstm_cell_1/MatMul:product:03rnn_1/while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rnn_1/while/peephole_lstm_cell_1/addò
7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpBrnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype029
7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpý
(rnn_1/while/peephole_lstm_cell_1/BiasAddBiasAdd(rnn_1/while/peephole_lstm_cell_1/add:z:0?rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn_1/while/peephole_lstm_cell_1/BiasAdd¦
0rnn_1/while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0rnn_1/while/peephole_lstm_cell_1/split/split_dimÃ
&rnn_1/while/peephole_lstm_cell_1/splitSplit9rnn_1/while/peephole_lstm_cell_1/split/split_dim:output:01rnn_1/while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&rnn_1/while/peephole_lstm_cell_1/splitÙ
/rnn_1/while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp:rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype021
/rnn_1/while/peephole_lstm_cell_1/ReadVariableOpÙ
$rnn_1/while/peephole_lstm_cell_1/mulMul7rnn_1/while/peephole_lstm_cell_1/ReadVariableOp:value:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn_1/while/peephole_lstm_cell_1/mulæ
&rnn_1/while/peephole_lstm_cell_1/add_1AddV2/rnn_1/while/peephole_lstm_cell_1/split:output:0(rnn_1/while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_1½
(rnn_1/while/peephole_lstm_cell_1/SigmoidSigmoid*rnn_1/while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(rnn_1/while/peephole_lstm_cell_1/Sigmoidß
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp<rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype023
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1ß
&rnn_1/while/peephole_lstm_cell_1/mul_1Mul9rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1:value:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_1è
&rnn_1/while/peephole_lstm_cell_1/add_2AddV2/rnn_1/while/peephole_lstm_cell_1/split:output:1*rnn_1/while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_2Á
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_1Sigmoid*rnn_1/while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_1Ô
&rnn_1/while/peephole_lstm_cell_1/mul_2Mul.rnn_1/while/peephole_lstm_cell_1/Sigmoid_1:y:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_2¹
%rnn_1/while/peephole_lstm_cell_1/TanhTanh/rnn_1/while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%rnn_1/while/peephole_lstm_cell_1/Tanhâ
&rnn_1/while/peephole_lstm_cell_1/mul_3Mul,rnn_1/while/peephole_lstm_cell_1/Sigmoid:y:0)rnn_1/while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_3ã
&rnn_1/while/peephole_lstm_cell_1/add_3AddV2*rnn_1/while/peephole_lstm_cell_1/mul_2:z:0*rnn_1/while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_3ß
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp<rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype023
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2ð
&rnn_1/while/peephole_lstm_cell_1/mul_4Mul9rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2:value:0*rnn_1/while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_4è
&rnn_1/while/peephole_lstm_cell_1/add_4AddV2/rnn_1/while/peephole_lstm_cell_1/split:output:3*rnn_1/while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_4Á
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_2Sigmoid*rnn_1/while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_2¸
'rnn_1/while/peephole_lstm_cell_1/Tanh_1Tanh*rnn_1/while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'rnn_1/while/peephole_lstm_cell_1/Tanh_1æ
&rnn_1/while/peephole_lstm_cell_1/mul_5Mul.rnn_1/while/peephole_lstm_cell_1/Sigmoid_2:y:0+rnn_1/while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_5
0rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_1_while_placeholder_1rnn_1_while_placeholder*rnn_1/while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype022
0rnn_1/while/TensorArrayV2Write/TensorListSetItemh
rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add/y
rnn_1/while/addAddV2rnn_1_while_placeholderrnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/addl
rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add_1/y
rnn_1/while/add_1AddV2$rnn_1_while_rnn_1_while_loop_counterrnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/add_1¸
rnn_1/while/IdentityIdentityrnn_1/while/add_1:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/IdentityÑ
rnn_1/while/Identity_1Identity*rnn_1_while_rnn_1_while_maximum_iterations8^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/Identity_1º
rnn_1/while/Identity_2Identityrnn_1/while/add:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/Identity_2ç
rnn_1/while/Identity_3Identity@rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/Identity_3â
rnn_1/while/Identity_4Identity*rnn_1/while/peephole_lstm_cell_1/mul_5:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/while/Identity_4â
rnn_1/while/Identity_5Identity*rnn_1/while/peephole_lstm_cell_1/add_3:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/while/Identity_5"5
rnn_1_while_identityrnn_1/while/Identity:output:0"9
rnn_1_while_identity_1rnn_1/while/Identity_1:output:0"9
rnn_1_while_identity_2rnn_1/while/Identity_2:output:0"9
rnn_1_while_identity_3rnn_1/while/Identity_3:output:0"9
rnn_1_while_identity_4rnn_1/while/Identity_4:output:0"9
rnn_1_while_identity_5rnn_1/while/Identity_5:output:0"
@rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resourceBrnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"
Arnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resourceCrnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"
?rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resourceArnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"z
:rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource<rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0"z
:rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource<rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0"v
8rnn_1_while_peephole_lstm_cell_1_readvariableop_resource:rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0"H
!rnn_1_while_rnn_1_strided_slice_1#rnn_1_while_rnn_1_strided_slice_1_0"À
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2r
7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2p
6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp2t
8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2b
/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2f
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_11rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12f
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_21rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
¥0
¸
C__inference_conv1d_layer_call_and_return_conditional_losses_1668819

inputsA
+conv1d_expanddims_1_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢"conv1d/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1f
conv1d/ShapeShapeconv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/Shape
conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
conv1d/strided_slice/stack
conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
conv1d/strided_slice/stack_1
conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
conv1d/strided_slice/stack_2
conv1d/strided_sliceStridedSliceconv1d/Shape:output:0#conv1d/strided_slice/stack:output:0%conv1d/strided_slice/stack_1:output:0%conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/strided_slice
conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2
conv1d/Reshape/shape 
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ReshapeÂ
conv1d/Conv2DConv2Dconv1d/Reshape:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d/Conv2D
conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/concat/values_1s
conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
conv1d/concat/axisµ
conv1d/concatConcatV2conv1d/strided_slice:output:0conv1d/concat/values_1:output:0conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/concat
conv1d/Reshape_1Reshapeconv1d/Conv2D:output:0conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/Reshape_1 
conv1d/SqueezeSqueezeconv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze{
squeeze_batch_dims/ShapeShapeconv1d/Squeeze:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack§
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2*
(squeeze_batch_dims/strided_slice/stack_1
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2Ò
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2"
 squeeze_batch_dims/Reshape/shape½
squeeze_batch_dims/ReshapeReshapeconv1d/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
squeeze_batch_dims/ReshapeÅ
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOpÑ
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
squeeze_batch_dims/BiasAdd
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2$
"squeeze_batch_dims/concat/values_1
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
squeeze_batch_dims/concat/axisñ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concatÊ
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
squeeze_batch_dims/Reshape_1Ò
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0#^conv1d/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1669712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1669712___redundant_placeholder05
1while_while_cond_1669712___redundant_placeholder15
1while_while_cond_1669712___redundant_placeholder25
1while_while_cond_1669712___redundant_placeholder35
1while_while_cond_1669712___redundant_placeholder45
1while_while_cond_1669712___redundant_placeholder55
1while_while_cond_1669712___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
©-
Ô	
while_body_1666013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
$while_peephole_lstm_cell_1_1666037_0:	 7
$while_peephole_lstm_cell_1_1666039_0:	 3
$while_peephole_lstm_cell_1_1666041_0:	2
$while_peephole_lstm_cell_1_1666043_0: 2
$while_peephole_lstm_cell_1_1666045_0: 2
$while_peephole_lstm_cell_1_1666047_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
"while_peephole_lstm_cell_1_1666037:	 5
"while_peephole_lstm_cell_1_1666039:	 1
"while_peephole_lstm_cell_1_1666041:	0
"while_peephole_lstm_cell_1_1666043: 0
"while_peephole_lstm_cell_1_1666045: 0
"while_peephole_lstm_cell_1_1666047: ¢2while/peephole_lstm_cell_1/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
2while/peephole_lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3$while_peephole_lstm_cell_1_1666037_0$while_peephole_lstm_cell_1_1666039_0$while_peephole_lstm_cell_1_1666041_0$while_peephole_lstm_cell_1_1666043_0$while_peephole_lstm_cell_1_1666045_0$while_peephole_lstm_cell_1_1666047_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_166599324
2while/peephole_lstm_cell_1/StatefulPartitionedCallÿ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder;while/peephole_lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:03^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity¦
while/Identity_1Identitywhile_while_maximum_iterations3^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:03^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Â
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:03^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ô
while/Identity_4Identity;while/peephole_lstm_cell_1/StatefulPartitionedCall:output:13^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4Ô
while/Identity_5Identity;while/peephole_lstm_cell_1/StatefulPartitionedCall:output:23^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"J
"while_peephole_lstm_cell_1_1666037$while_peephole_lstm_cell_1_1666037_0"J
"while_peephole_lstm_cell_1_1666039$while_peephole_lstm_cell_1_1666039_0"J
"while_peephole_lstm_cell_1_1666041$while_peephole_lstm_cell_1_1666041_0"J
"while_peephole_lstm_cell_1_1666043$while_peephole_lstm_cell_1_1666043_0"J
"while_peephole_lstm_cell_1_1666045$while_peephole_lstm_cell_1_1666045_0"J
"while_peephole_lstm_cell_1_1666047$while_peephole_lstm_cell_1_1666047_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2while/peephole_lstm_cell_1/StatefulPartitionedCall2while/peephole_lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 

Õ
rnn_while_cond_1668424$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1=
9rnn_while_rnn_while_cond_1668424___redundant_placeholder0=
9rnn_while_rnn_while_cond_1668424___redundant_placeholder1=
9rnn_while_rnn_while_cond_1668424___redundant_placeholder2=
9rnn_while_rnn_while_cond_1668424___redundant_placeholder3=
9rnn_while_rnn_while_cond_1668424___redundant_placeholder4=
9rnn_while_rnn_while_cond_1668424___redundant_placeholder5=
9rnn_while_rnn_while_cond_1668424___redundant_placeholder6
rnn_while_identity

rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¸d

while_body_1669465
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_peephole_lstm_cell_matmul_readvariableop_resource_0:	N
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 I
:while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	@
2while_peephole_lstm_cell_readvariableop_resource_0: B
4while_peephole_lstm_cell_readvariableop_1_resource_0: B
4while_peephole_lstm_cell_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_peephole_lstm_cell_matmul_readvariableop_resource:	L
9while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 G
8while_peephole_lstm_cell_biasadd_readvariableop_resource:	>
0while_peephole_lstm_cell_readvariableop_resource: @
2while_peephole_lstm_cell_readvariableop_1_resource: @
2while_peephole_lstm_cell_readvariableop_2_resource: ¢/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢.while/peephole_lstm_cell/MatMul/ReadVariableOp¢0while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢'while/peephole_lstm_cell/ReadVariableOp¢)while/peephole_lstm_cell/ReadVariableOp_1¢)while/peephole_lstm_cell/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÛ
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOpé
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/peephole_lstm_cell/MatMulá
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpÒ
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell/MatMul_1Ð
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/peephole_lstm_cell/addÚ
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpÝ
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/peephole_lstm_cell/BiasAdd
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim£
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2 
while/peephole_lstm_cell/splitÁ
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp»
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/mulÆ
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_1¥
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell/SigmoidÇ
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1Á
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_1È
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_2©
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_1¶
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_2¡
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/TanhÂ
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_3Ã
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_3Ç
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2Ð
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_4È
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_4©
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_2 
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell/Tanh_1Æ
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_5æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ö
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1ø
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2¥
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
g
»
while_body_1670253
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0N
;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 P
=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 K
<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	B
4while_peephole_lstm_cell_1_readvariableop_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_1_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorL
9while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2while_peephole_lstm_cell_1_readvariableop_resource: B
4while_peephole_lstm_cell_1_readvariableop_1_resource: B
4while_peephole_lstm_cell_1_readvariableop_2_resource: ¢1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)while/peephole_lstm_cell_1/ReadVariableOp¢+while/peephole_lstm_cell_1/ReadVariableOp_1¢+while/peephole_lstm_cell_1/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpï
!while/peephole_lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:08while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell_1/MatMulç
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype024
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpØ
#while/peephole_lstm_cell_1/MatMul_1MatMulwhile_placeholder_2:while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#while/peephole_lstm_cell_1/MatMul_1Ø
while/peephole_lstm_cell_1/addAddV2+while/peephole_lstm_cell_1/MatMul:product:0-while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/peephole_lstm_cell_1/addà
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"while/peephole_lstm_cell_1/BiasAddBiasAdd"while/peephole_lstm_cell_1/add:z:09while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"while/peephole_lstm_cell_1/BiasAdd
*while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*while/peephole_lstm_cell_1/split/split_dim«
 while/peephole_lstm_cell_1/splitSplit3while/peephole_lstm_cell_1/split/split_dim:output:0+while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 while/peephole_lstm_cell_1/splitÇ
)while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp4while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell_1/ReadVariableOpÁ
while/peephole_lstm_cell_1/mulMul1while/peephole_lstm_cell_1/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell_1/mulÎ
 while/peephole_lstm_cell_1/add_1AddV2)while/peephole_lstm_cell_1/split:output:0"while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_1«
"while/peephole_lstm_cell_1/SigmoidSigmoid$while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell_1/SigmoidÍ
+while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_1Ç
 while/peephole_lstm_cell_1/mul_1Mul3while/peephole_lstm_cell_1/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_1Ð
 while/peephole_lstm_cell_1/add_2AddV2)while/peephole_lstm_cell_1/split:output:1$while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_2¯
$while/peephole_lstm_cell_1/Sigmoid_1Sigmoid$while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_1¼
 while/peephole_lstm_cell_1/mul_2Mul(while/peephole_lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_2§
while/peephole_lstm_cell_1/TanhTanh)while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell_1/TanhÊ
 while/peephole_lstm_cell_1/mul_3Mul&while/peephole_lstm_cell_1/Sigmoid:y:0#while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_3Ë
 while/peephole_lstm_cell_1/add_3AddV2$while/peephole_lstm_cell_1/mul_2:z:0$while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_3Í
+while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_2Ø
 while/peephole_lstm_cell_1/mul_4Mul3while/peephole_lstm_cell_1/ReadVariableOp_2:value:0$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_4Ð
 while/peephole_lstm_cell_1/add_4AddV2)while/peephole_lstm_cell_1/split:output:3$while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_4¯
$while/peephole_lstm_cell_1/Sigmoid_2Sigmoid$while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_2¦
!while/peephole_lstm_cell_1/Tanh_1Tanh$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!while/peephole_lstm_cell_1/Tanh_1Î
 while/peephole_lstm_cell_1/mul_5Mul(while/peephole_lstm_cell_1/Sigmoid_2:y:0%while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_5è
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations2^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2±
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3¬
while/Identity_4Identity$while/peephole_lstm_cell_1/mul_5:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4¬
while/Identity_5Identity$while/peephole_lstm_cell_1/add_3:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"|
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"x
9while_peephole_lstm_cell_1_matmul_readvariableop_resource;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_1_resource6while_peephole_lstm_cell_1_readvariableop_1_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_2_resource6while_peephole_lstm_cell_1_readvariableop_2_resource_0"j
2while_peephole_lstm_cell_1_readvariableop_resource4while_peephole_lstm_cell_1_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2f
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0while/peephole_lstm_cell_1/MatMul/ReadVariableOp0while/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)while/peephole_lstm_cell_1/ReadVariableOp)while/peephole_lstm_cell_1/ReadVariableOp2Z
+while/peephole_lstm_cell_1/ReadVariableOp_1+while/peephole_lstm_cell_1/ReadVariableOp_12Z
+while/peephole_lstm_cell_1/ReadVariableOp_2+while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
ñû
ñ
"__inference__wrapped_model_1665148
conv1d_inputS
=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource:R
Dsequential_conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:S
@sequential_rnn_peephole_lstm_cell_matmul_readvariableop_resource:	U
Bsequential_rnn_peephole_lstm_cell_matmul_1_readvariableop_resource:	 P
Asequential_rnn_peephole_lstm_cell_biasadd_readvariableop_resource:	G
9sequential_rnn_peephole_lstm_cell_readvariableop_resource: I
;sequential_rnn_peephole_lstm_cell_readvariableop_1_resource: I
;sequential_rnn_peephole_lstm_cell_readvariableop_2_resource: W
Dsequential_rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource:	 Y
Fsequential_rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 T
Esequential_rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource:	K
=sequential_rnn_1_peephole_lstm_cell_1_readvariableop_resource: M
?sequential_rnn_1_peephole_lstm_cell_1_readvariableop_1_resource: M
?sequential_rnn_1_peephole_lstm_cell_1_readvariableop_2_resource: A
/sequential_dense_matmul_readvariableop_resource: >
0sequential_dense_biasadd_readvariableop_resource:
identity¢4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp¢;sequential/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢8sequential/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp¢7sequential/rnn/peephole_lstm_cell/MatMul/ReadVariableOp¢9sequential/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp¢0sequential/rnn/peephole_lstm_cell/ReadVariableOp¢2sequential/rnn/peephole_lstm_cell/ReadVariableOp_1¢2sequential/rnn/peephole_lstm_cell/ReadVariableOp_2¢sequential/rnn/while¢<sequential/rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢;sequential/rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp¢=sequential/rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢4sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp¢6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_1¢6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_2¢sequential/rnn_1/while
'sequential/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2)
'sequential/conv1d/conv1d/ExpandDims/dimÖ
#sequential/conv1d/conv1d/ExpandDims
ExpandDimsconv1d_input0sequential/conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2%
#sequential/conv1d/conv1d/ExpandDimsî
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype026
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp
)sequential/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/conv1d/conv1d/ExpandDims_1/dimÿ
%sequential/conv1d/conv1d/ExpandDims_1
ExpandDims<sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2'
%sequential/conv1d/conv1d/ExpandDims_1
sequential/conv1d/conv1d/ShapeShape,sequential/conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2 
sequential/conv1d/conv1d/Shape¦
,sequential/conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential/conv1d/conv1d/strided_slice/stack³
.sequential/conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ20
.sequential/conv1d/conv1d/strided_slice/stack_1ª
.sequential/conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/conv1d/conv1d/strided_slice/stack_2ö
&sequential/conv1d/conv1d/strided_sliceStridedSlice'sequential/conv1d/conv1d/Shape:output:05sequential/conv1d/conv1d/strided_slice/stack:output:07sequential/conv1d/conv1d/strided_slice/stack_1:output:07sequential/conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2(
&sequential/conv1d/conv1d/strided_slice©
&sequential/conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2(
&sequential/conv1d/conv1d/Reshape/shapeè
 sequential/conv1d/conv1d/ReshapeReshape,sequential/conv1d/conv1d/ExpandDims:output:0/sequential/conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential/conv1d/conv1d/Reshape
sequential/conv1d/conv1d/Conv2DConv2D)sequential/conv1d/conv1d/Reshape:output:0.sequential/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2!
sequential/conv1d/conv1d/Conv2D©
(sequential/conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2*
(sequential/conv1d/conv1d/concat/values_1
$sequential/conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$sequential/conv1d/conv1d/concat/axis
sequential/conv1d/conv1d/concatConcatV2/sequential/conv1d/conv1d/strided_slice:output:01sequential/conv1d/conv1d/concat/values_1:output:0-sequential/conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2!
sequential/conv1d/conv1d/concatå
"sequential/conv1d/conv1d/Reshape_1Reshape(sequential/conv1d/conv1d/Conv2D:output:0(sequential/conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2$
"sequential/conv1d/conv1d/Reshape_1Ö
 sequential/conv1d/conv1d/SqueezeSqueeze+sequential/conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2"
 sequential/conv1d/conv1d/Squeeze±
*sequential/conv1d/squeeze_batch_dims/ShapeShape)sequential/conv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2,
*sequential/conv1d/squeeze_batch_dims/Shape¾
8sequential/conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential/conv1d/squeeze_batch_dims/strided_slice/stackË
:sequential/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2<
:sequential/conv1d/squeeze_batch_dims/strided_slice/stack_1Â
:sequential/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/conv1d/squeeze_batch_dims/strided_slice/stack_2¾
2sequential/conv1d/squeeze_batch_dims/strided_sliceStridedSlice3sequential/conv1d/squeeze_batch_dims/Shape:output:0Asequential/conv1d/squeeze_batch_dims/strided_slice/stack:output:0Csequential/conv1d/squeeze_batch_dims/strided_slice/stack_1:output:0Csequential/conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask24
2sequential/conv1d/squeeze_batch_dims/strided_slice½
2sequential/conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      24
2sequential/conv1d/squeeze_batch_dims/Reshape/shape
,sequential/conv1d/squeeze_batch_dims/ReshapeReshape)sequential/conv1d/conv1d/Squeeze:output:0;sequential/conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential/conv1d/squeeze_batch_dims/Reshapeû
;sequential/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpDsequential_conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp
,sequential/conv1d/squeeze_batch_dims/BiasAddBiasAdd5sequential/conv1d/squeeze_batch_dims/Reshape:output:0Csequential/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential/conv1d/squeeze_batch_dims/BiasAdd½
4sequential/conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      26
4sequential/conv1d/squeeze_batch_dims/concat/values_1¯
0sequential/conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential/conv1d/squeeze_batch_dims/concat/axisË
+sequential/conv1d/squeeze_batch_dims/concatConcatV2;sequential/conv1d/squeeze_batch_dims/strided_slice:output:0=sequential/conv1d/squeeze_batch_dims/concat/values_1:output:09sequential/conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+sequential/conv1d/squeeze_batch_dims/concat
.sequential/conv1d/squeeze_batch_dims/Reshape_1Reshape5sequential/conv1d/squeeze_batch_dims/BiasAdd:output:04sequential/conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/conv1d/squeeze_batch_dims/Reshape_1
sequential/reshape/ShapeShape7sequential/conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/reshape/Shape
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stack
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2Ô
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_slice
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/1
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/2ÿ
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shapeÝ
sequential/reshape/ReshapeReshape7sequential/conv1d/squeeze_batch_dims/Reshape_1:output:0)sequential/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/reshape/Reshape
sequential/rnn/ShapeShape#sequential/reshape/Reshape:output:0*
T0*
_output_shapes
:2
sequential/rnn/Shape
"sequential/rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/rnn/strided_slice/stack
$sequential/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/rnn/strided_slice/stack_1
$sequential/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/rnn/strided_slice/stack_2¼
sequential/rnn/strided_sliceStridedSlicesequential/rnn/Shape:output:0+sequential/rnn/strided_slice/stack:output:0-sequential/rnn/strided_slice/stack_1:output:0-sequential/rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/rnn/strided_slicez
sequential/rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rnn/zeros/mul/y¨
sequential/rnn/zeros/mulMul%sequential/rnn/strided_slice:output:0#sequential/rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn/zeros/mul}
sequential/rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
sequential/rnn/zeros/Less/y£
sequential/rnn/zeros/LessLesssequential/rnn/zeros/mul:z:0$sequential/rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn/zeros/Less
sequential/rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rnn/zeros/packed/1¿
sequential/rnn/zeros/packedPack%sequential/rnn/strided_slice:output:0&sequential/rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/rnn/zeros/packed}
sequential/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/rnn/zeros/Const±
sequential/rnn/zerosFill$sequential/rnn/zeros/packed:output:0#sequential/rnn/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rnn/zeros~
sequential/rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rnn/zeros_1/mul/y®
sequential/rnn/zeros_1/mulMul%sequential/rnn/strided_slice:output:0%sequential/rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn/zeros_1/mul
sequential/rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
sequential/rnn/zeros_1/Less/y«
sequential/rnn/zeros_1/LessLesssequential/rnn/zeros_1/mul:z:0&sequential/rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn/zeros_1/Less
sequential/rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2!
sequential/rnn/zeros_1/packed/1Å
sequential/rnn/zeros_1/packedPack%sequential/rnn/strided_slice:output:0(sequential/rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/rnn/zeros_1/packed
sequential/rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/rnn/zeros_1/Const¹
sequential/rnn/zeros_1Fill&sequential/rnn/zeros_1/packed:output:0%sequential/rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rnn/zeros_1
sequential/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
sequential/rnn/transpose/permÄ
sequential/rnn/transpose	Transpose#sequential/reshape/Reshape:output:0&sequential/rnn/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rnn/transpose|
sequential/rnn/Shape_1Shapesequential/rnn/transpose:y:0*
T0*
_output_shapes
:2
sequential/rnn/Shape_1
$sequential/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/rnn/strided_slice_1/stack
&sequential/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn/strided_slice_1/stack_1
&sequential/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn/strided_slice_1/stack_2È
sequential/rnn/strided_slice_1StridedSlicesequential/rnn/Shape_1:output:0-sequential/rnn/strided_slice_1/stack:output:0/sequential/rnn/strided_slice_1/stack_1:output:0/sequential/rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential/rnn/strided_slice_1£
*sequential/rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*sequential/rnn/TensorArrayV2/element_shapeî
sequential/rnn/TensorArrayV2TensorListReserve3sequential/rnn/TensorArrayV2/element_shape:output:0'sequential/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/rnn/TensorArrayV2Ý
Dsequential/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
Dsequential/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape´
6sequential/rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/rnn/transpose:y:0Msequential/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6sequential/rnn/TensorArrayUnstack/TensorListFromTensor
$sequential/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/rnn/strided_slice_2/stack
&sequential/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn/strided_slice_2/stack_1
&sequential/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn/strided_slice_2/stack_2Ö
sequential/rnn/strided_slice_2StridedSlicesequential/rnn/transpose:y:0-sequential/rnn/strided_slice_2/stack:output:0/sequential/rnn/strided_slice_2/stack_1:output:0/sequential/rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
sequential/rnn/strided_slice_2ô
7sequential/rnn/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp@sequential_rnn_peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype029
7sequential/rnn/peephole_lstm_cell/MatMul/ReadVariableOpû
(sequential/rnn/peephole_lstm_cell/MatMulMatMul'sequential/rnn/strided_slice_2:output:0?sequential/rnn/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/rnn/peephole_lstm_cell/MatMulú
9sequential/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBsequential_rnn_peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02;
9sequential/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp÷
*sequential/rnn/peephole_lstm_cell/MatMul_1MatMulsequential/rnn/zeros:output:0Asequential/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/rnn/peephole_lstm_cell/MatMul_1ô
%sequential/rnn/peephole_lstm_cell/addAddV22sequential/rnn/peephole_lstm_cell/MatMul:product:04sequential/rnn/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential/rnn/peephole_lstm_cell/addó
8sequential/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAsequential_rnn_peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02:
8sequential/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp
)sequential/rnn/peephole_lstm_cell/BiasAddBiasAdd)sequential/rnn/peephole_lstm_cell/add:z:0@sequential/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/rnn/peephole_lstm_cell/BiasAdd¨
1sequential/rnn/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential/rnn/peephole_lstm_cell/split/split_dimÇ
'sequential/rnn/peephole_lstm_cell/splitSplit:sequential/rnn/peephole_lstm_cell/split/split_dim:output:02sequential/rnn/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2)
'sequential/rnn/peephole_lstm_cell/splitÚ
0sequential/rnn/peephole_lstm_cell/ReadVariableOpReadVariableOp9sequential_rnn_peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype022
0sequential/rnn/peephole_lstm_cell/ReadVariableOpâ
%sequential/rnn/peephole_lstm_cell/mulMul8sequential/rnn/peephole_lstm_cell/ReadVariableOp:value:0sequential/rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/rnn/peephole_lstm_cell/mulê
'sequential/rnn/peephole_lstm_cell/add_1AddV20sequential/rnn/peephole_lstm_cell/split:output:0)sequential/rnn/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/add_1À
)sequential/rnn/peephole_lstm_cell/SigmoidSigmoid+sequential/rnn/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)sequential/rnn/peephole_lstm_cell/Sigmoidà
2sequential/rnn/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp;sequential_rnn_peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype024
2sequential/rnn/peephole_lstm_cell/ReadVariableOp_1è
'sequential/rnn/peephole_lstm_cell/mul_1Mul:sequential/rnn/peephole_lstm_cell/ReadVariableOp_1:value:0sequential/rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/mul_1ì
'sequential/rnn/peephole_lstm_cell/add_2AddV20sequential/rnn/peephole_lstm_cell/split:output:1+sequential/rnn/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/add_2Ä
+sequential/rnn/peephole_lstm_cell/Sigmoid_1Sigmoid+sequential/rnn/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn/peephole_lstm_cell/Sigmoid_1Ý
'sequential/rnn/peephole_lstm_cell/mul_2Mul/sequential/rnn/peephole_lstm_cell/Sigmoid_1:y:0sequential/rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/mul_2¼
&sequential/rnn/peephole_lstm_cell/TanhTanh0sequential/rnn/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rnn/peephole_lstm_cell/Tanhæ
'sequential/rnn/peephole_lstm_cell/mul_3Mul-sequential/rnn/peephole_lstm_cell/Sigmoid:y:0*sequential/rnn/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/mul_3ç
'sequential/rnn/peephole_lstm_cell/add_3AddV2+sequential/rnn/peephole_lstm_cell/mul_2:z:0+sequential/rnn/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/add_3à
2sequential/rnn/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp;sequential_rnn_peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype024
2sequential/rnn/peephole_lstm_cell/ReadVariableOp_2ô
'sequential/rnn/peephole_lstm_cell/mul_4Mul:sequential/rnn/peephole_lstm_cell/ReadVariableOp_2:value:0+sequential/rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/mul_4ì
'sequential/rnn/peephole_lstm_cell/add_4AddV20sequential/rnn/peephole_lstm_cell/split:output:3+sequential/rnn/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/add_4Ä
+sequential/rnn/peephole_lstm_cell/Sigmoid_2Sigmoid+sequential/rnn/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn/peephole_lstm_cell/Sigmoid_2»
(sequential/rnn/peephole_lstm_cell/Tanh_1Tanh+sequential/rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/rnn/peephole_lstm_cell/Tanh_1ê
'sequential/rnn/peephole_lstm_cell/mul_5Mul/sequential/rnn/peephole_lstm_cell/Sigmoid_2:y:0,sequential/rnn/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rnn/peephole_lstm_cell/mul_5­
,sequential/rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2.
,sequential/rnn/TensorArrayV2_1/element_shapeô
sequential/rnn/TensorArrayV2_1TensorListReserve5sequential/rnn/TensorArrayV2_1/element_shape:output:0'sequential/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential/rnn/TensorArrayV2_1l
sequential/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rnn/time
'sequential/rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'sequential/rnn/while/maximum_iterations
!sequential/rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/rnn/while/loop_counterª	
sequential/rnn/whileWhile*sequential/rnn/while/loop_counter:output:00sequential/rnn/while/maximum_iterations:output:0sequential/rnn/time:output:0'sequential/rnn/TensorArrayV2_1:handle:0sequential/rnn/zeros:output:0sequential/rnn/zeros_1:output:0'sequential/rnn/strided_slice_1:output:0Fsequential/rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_rnn_peephole_lstm_cell_matmul_readvariableop_resourceBsequential_rnn_peephole_lstm_cell_matmul_1_readvariableop_resourceAsequential_rnn_peephole_lstm_cell_biasadd_readvariableop_resource9sequential_rnn_peephole_lstm_cell_readvariableop_resource;sequential_rnn_peephole_lstm_cell_readvariableop_1_resource;sequential_rnn_peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*-
body%R#
!sequential_rnn_while_body_1664865*-
cond%R#
!sequential_rnn_while_cond_1664864*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/rnn/whileÓ
?sequential/rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2A
?sequential/rnn/TensorArrayV2Stack/TensorListStack/element_shape¤
1sequential/rnn/TensorArrayV2Stack/TensorListStackTensorListStacksequential/rnn/while:output:3Hsequential/rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype023
1sequential/rnn/TensorArrayV2Stack/TensorListStack
$sequential/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$sequential/rnn/strided_slice_3/stack
&sequential/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/rnn/strided_slice_3/stack_1
&sequential/rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn/strided_slice_3/stack_2ô
sequential/rnn/strided_slice_3StridedSlice:sequential/rnn/TensorArrayV2Stack/TensorListStack:tensor:0-sequential/rnn/strided_slice_3/stack:output:0/sequential/rnn/strided_slice_3/stack_1:output:0/sequential/rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2 
sequential/rnn/strided_slice_3
sequential/rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/rnn/transpose_1/permá
sequential/rnn/transpose_1	Transpose:sequential/rnn/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rnn/transpose_1~
sequential/rnn_1/ShapeShapesequential/rnn/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/rnn_1/Shape
$sequential/rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/rnn_1/strided_slice/stack
&sequential/rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn_1/strided_slice/stack_1
&sequential/rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/rnn_1/strided_slice/stack_2È
sequential/rnn_1/strided_sliceStridedSlicesequential/rnn_1/Shape:output:0-sequential/rnn_1/strided_slice/stack:output:0/sequential/rnn_1/strided_slice/stack_1:output:0/sequential/rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential/rnn_1/strided_slice~
sequential/rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rnn_1/zeros/mul/y°
sequential/rnn_1/zeros/mulMul'sequential/rnn_1/strided_slice:output:0%sequential/rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros/mul
sequential/rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
sequential/rnn_1/zeros/Less/y«
sequential/rnn_1/zeros/LessLesssequential/rnn_1/zeros/mul:z:0&sequential/rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros/Less
sequential/rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2!
sequential/rnn_1/zeros/packed/1Ç
sequential/rnn_1/zeros/packedPack'sequential/rnn_1/strided_slice:output:0(sequential/rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/rnn_1/zeros/packed
sequential/rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/rnn_1/zeros/Const¹
sequential/rnn_1/zerosFill&sequential/rnn_1/zeros/packed:output:0%sequential/rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rnn_1/zeros
sequential/rnn_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential/rnn_1/zeros_1/mul/y¶
sequential/rnn_1/zeros_1/mulMul'sequential/rnn_1/strided_slice:output:0'sequential/rnn_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros_1/mul
sequential/rnn_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2!
sequential/rnn_1/zeros_1/Less/y³
sequential/rnn_1/zeros_1/LessLess sequential/rnn_1/zeros_1/mul:z:0(sequential/rnn_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/rnn_1/zeros_1/Less
!sequential/rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/rnn_1/zeros_1/packed/1Í
sequential/rnn_1/zeros_1/packedPack'sequential/rnn_1/strided_slice:output:0*sequential/rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
sequential/rnn_1/zeros_1/packed
sequential/rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential/rnn_1/zeros_1/ConstÁ
sequential/rnn_1/zeros_1Fill(sequential/rnn_1/zeros_1/packed:output:0'sequential/rnn_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rnn_1/zeros_1
sequential/rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/rnn_1/transpose/permÅ
sequential/rnn_1/transpose	Transposesequential/rnn/transpose_1:y:0(sequential/rnn_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rnn_1/transpose
sequential/rnn_1/Shape_1Shapesequential/rnn_1/transpose:y:0*
T0*
_output_shapes
:2
sequential/rnn_1/Shape_1
&sequential/rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/rnn_1/strided_slice_1/stack
(sequential/rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_1/stack_1
(sequential/rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_1/stack_2Ô
 sequential/rnn_1/strided_slice_1StridedSlice!sequential/rnn_1/Shape_1:output:0/sequential/rnn_1/strided_slice_1/stack:output:01sequential/rnn_1/strided_slice_1/stack_1:output:01sequential/rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/rnn_1/strided_slice_1§
,sequential/rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,sequential/rnn_1/TensorArrayV2/element_shapeö
sequential/rnn_1/TensorArrayV2TensorListReserve5sequential/rnn_1/TensorArrayV2/element_shape:output:0)sequential/rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential/rnn_1/TensorArrayV2á
Fsequential/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape¼
8sequential/rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/rnn_1/transpose:y:0Osequential/rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8sequential/rnn_1/TensorArrayUnstack/TensorListFromTensor
&sequential/rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/rnn_1/strided_slice_2/stack
(sequential/rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_2/stack_1
(sequential/rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_2/stack_2â
 sequential/rnn_1/strided_slice_2StridedSlicesequential/rnn_1/transpose:y:0/sequential/rnn_1/strided_slice_2/stack:output:01sequential/rnn_1/strided_slice_2/stack_1:output:01sequential/rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2"
 sequential/rnn_1/strided_slice_2
;sequential/rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOpDsequential_rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02=
;sequential/rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp
,sequential/rnn_1/peephole_lstm_cell_1/MatMulMatMul)sequential/rnn_1/strided_slice_2:output:0Csequential/rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential/rnn_1/peephole_lstm_cell_1/MatMul
=sequential/rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpFsequential_rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02?
=sequential/rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp
.sequential/rnn_1/peephole_lstm_cell_1/MatMul_1MatMulsequential/rnn_1/zeros:output:0Esequential/rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/rnn_1/peephole_lstm_cell_1/MatMul_1
)sequential/rnn_1/peephole_lstm_cell_1/addAddV26sequential/rnn_1/peephole_lstm_cell_1/MatMul:product:08sequential/rnn_1/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/rnn_1/peephole_lstm_cell_1/addÿ
<sequential/rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpEsequential_rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<sequential/rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp
-sequential/rnn_1/peephole_lstm_cell_1/BiasAddBiasAdd-sequential/rnn_1/peephole_lstm_cell_1/add:z:0Dsequential/rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/rnn_1/peephole_lstm_cell_1/BiasAdd°
5sequential/rnn_1/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential/rnn_1/peephole_lstm_cell_1/split/split_dim×
+sequential/rnn_1/peephole_lstm_cell_1/splitSplit>sequential/rnn_1/peephole_lstm_cell_1/split/split_dim:output:06sequential/rnn_1/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2-
+sequential/rnn_1/peephole_lstm_cell_1/splitæ
4sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOpReadVariableOp=sequential_rnn_1_peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOpð
)sequential/rnn_1/peephole_lstm_cell_1/mulMul<sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp:value:0!sequential/rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)sequential/rnn_1/peephole_lstm_cell_1/mulú
+sequential/rnn_1/peephole_lstm_cell_1/add_1AddV24sequential/rnn_1/peephole_lstm_cell_1/split:output:0-sequential/rnn_1/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/add_1Ì
-sequential/rnn_1/peephole_lstm_cell_1/SigmoidSigmoid/sequential/rnn_1/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rnn_1/peephole_lstm_cell_1/Sigmoidì
6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp?sequential_rnn_1_peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype028
6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_1ö
+sequential/rnn_1/peephole_lstm_cell_1/mul_1Mul>sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_1:value:0!sequential/rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/mul_1ü
+sequential/rnn_1/peephole_lstm_cell_1/add_2AddV24sequential/rnn_1/peephole_lstm_cell_1/split:output:1/sequential/rnn_1/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/add_2Ð
/sequential/rnn_1/peephole_lstm_cell_1/Sigmoid_1Sigmoid/sequential/rnn_1/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/sequential/rnn_1/peephole_lstm_cell_1/Sigmoid_1ë
+sequential/rnn_1/peephole_lstm_cell_1/mul_2Mul3sequential/rnn_1/peephole_lstm_cell_1/Sigmoid_1:y:0!sequential/rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/mul_2È
*sequential/rnn_1/peephole_lstm_cell_1/TanhTanh4sequential/rnn_1/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/rnn_1/peephole_lstm_cell_1/Tanhö
+sequential/rnn_1/peephole_lstm_cell_1/mul_3Mul1sequential/rnn_1/peephole_lstm_cell_1/Sigmoid:y:0.sequential/rnn_1/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/mul_3÷
+sequential/rnn_1/peephole_lstm_cell_1/add_3AddV2/sequential/rnn_1/peephole_lstm_cell_1/mul_2:z:0/sequential/rnn_1/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/add_3ì
6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp?sequential_rnn_1_peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype028
6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_2
+sequential/rnn_1/peephole_lstm_cell_1/mul_4Mul>sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_2:value:0/sequential/rnn_1/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/mul_4ü
+sequential/rnn_1/peephole_lstm_cell_1/add_4AddV24sequential/rnn_1/peephole_lstm_cell_1/split:output:3/sequential/rnn_1/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/add_4Ð
/sequential/rnn_1/peephole_lstm_cell_1/Sigmoid_2Sigmoid/sequential/rnn_1/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/sequential/rnn_1/peephole_lstm_cell_1/Sigmoid_2Ç
,sequential/rnn_1/peephole_lstm_cell_1/Tanh_1Tanh/sequential/rnn_1/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rnn_1/peephole_lstm_cell_1/Tanh_1ú
+sequential/rnn_1/peephole_lstm_cell_1/mul_5Mul3sequential/rnn_1/peephole_lstm_cell_1/Sigmoid_2:y:00sequential/rnn_1/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rnn_1/peephole_lstm_cell_1/mul_5±
.sequential/rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    20
.sequential/rnn_1/TensorArrayV2_1/element_shapeü
 sequential/rnn_1/TensorArrayV2_1TensorListReserve7sequential/rnn_1/TensorArrayV2_1/element_shape:output:0)sequential/rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sequential/rnn_1/TensorArrayV2_1p
sequential/rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rnn_1/time¡
)sequential/rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)sequential/rnn_1/while/maximum_iterations
#sequential/rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/rnn_1/while/loop_counterÚ	
sequential/rnn_1/whileWhile,sequential/rnn_1/while/loop_counter:output:02sequential/rnn_1/while/maximum_iterations:output:0sequential/rnn_1/time:output:0)sequential/rnn_1/TensorArrayV2_1:handle:0sequential/rnn_1/zeros:output:0!sequential/rnn_1/zeros_1:output:0)sequential/rnn_1/strided_slice_1:output:0Hsequential/rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dsequential_rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resourceFsequential_rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resourceEsequential_rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource=sequential_rnn_1_peephole_lstm_cell_1_readvariableop_resource?sequential_rnn_1_peephole_lstm_cell_1_readvariableop_1_resource?sequential_rnn_1_peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*/
body'R%
#sequential_rnn_1_while_body_1665041*/
cond'R%
#sequential_rnn_1_while_cond_1665040*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/rnn_1/while×
Asequential/rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2C
Asequential/rnn_1/TensorArrayV2Stack/TensorListStack/element_shape¬
3sequential/rnn_1/TensorArrayV2Stack/TensorListStackTensorListStacksequential/rnn_1/while:output:3Jsequential/rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype025
3sequential/rnn_1/TensorArrayV2Stack/TensorListStack£
&sequential/rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2(
&sequential/rnn_1/strided_slice_3/stack
(sequential/rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential/rnn_1/strided_slice_3/stack_1
(sequential/rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/rnn_1/strided_slice_3/stack_2
 sequential/rnn_1/strided_slice_3StridedSlice<sequential/rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/rnn_1/strided_slice_3/stack:output:01sequential/rnn_1/strided_slice_3/stack_1:output:01sequential/rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2"
 sequential/rnn_1/strided_slice_3
!sequential/rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!sequential/rnn_1/transpose_1/permé
sequential/rnn_1/transpose_1	Transpose<sequential/rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0*sequential/rnn_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rnn_1/transpose_1À
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&sequential/dense/MatMul/ReadVariableOpÉ
sequential/dense/MatMulMatMul)sequential/rnn_1/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/MatMul¿
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAdd¡
IdentityIdentity!sequential/dense/BiasAdd:output:05^sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp<^sequential/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp9^sequential/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp8^sequential/rnn/peephole_lstm_cell/MatMul/ReadVariableOp:^sequential/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp1^sequential/rnn/peephole_lstm_cell/ReadVariableOp3^sequential/rnn/peephole_lstm_cell/ReadVariableOp_13^sequential/rnn/peephole_lstm_cell/ReadVariableOp_2^sequential/rnn/while=^sequential/rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp<^sequential/rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp>^sequential/rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp5^sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp7^sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_17^sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_2^sequential/rnn_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2l
4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp2z
;sequential/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp;sequential/conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2t
8sequential/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp8sequential/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp2r
7sequential/rnn/peephole_lstm_cell/MatMul/ReadVariableOp7sequential/rnn/peephole_lstm_cell/MatMul/ReadVariableOp2v
9sequential/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp9sequential/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp2d
0sequential/rnn/peephole_lstm_cell/ReadVariableOp0sequential/rnn/peephole_lstm_cell/ReadVariableOp2h
2sequential/rnn/peephole_lstm_cell/ReadVariableOp_12sequential/rnn/peephole_lstm_cell/ReadVariableOp_12h
2sequential/rnn/peephole_lstm_cell/ReadVariableOp_22sequential/rnn/peephole_lstm_cell/ReadVariableOp_22,
sequential/rnn/whilesequential/rnn/while2|
<sequential/rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp<sequential/rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2z
;sequential/rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp;sequential/rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp2~
=sequential/rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp=sequential/rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2l
4sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp4sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp2p
6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_16sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_12p
6sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_26sequential/rnn_1/peephole_lstm_cell_1/ReadVariableOp_220
sequential/rnn_1/whilesequential/rnn_1/while:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv1d_input
p
Þ
@__inference_rnn_layer_call_and_return_conditional_losses_1669566

inputsD
1peephole_lstm_cell_matmul_readvariableop_resource:	F
3peephole_lstm_cell_matmul_1_readvariableop_resource:	 A
2peephole_lstm_cell_biasadd_readvariableop_resource:	8
*peephole_lstm_cell_readvariableop_resource: :
,peephole_lstm_cell_readvariableop_1_resource: :
,peephole_lstm_cell_readvariableop_2_resource: 
identity¢)peephole_lstm_cell/BiasAdd/ReadVariableOp¢(peephole_lstm_cell/MatMul/ReadVariableOp¢*peephole_lstm_cell/MatMul_1/ReadVariableOp¢!peephole_lstm_cell/ReadVariableOp¢#peephole_lstm_cell/ReadVariableOp_1¢#peephole_lstm_cell/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ç
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp¿
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMulÍ
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp»
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMul_1¸
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/addÆ
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOpÅ
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/BiasAdd
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell/split­
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp¦
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul®
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_1
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid³
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1¬
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_1°
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_2
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_1¡
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_2
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanhª
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_3«
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_3³
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2¸
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_4°
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_4
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_2
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanh_1®
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1669465*
condR
while_cond_1669464*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ã
IdentityIdentitytranspose_1:y:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

'__inference_rnn_1_layer_call_fn_1670371
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16660932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


í
while_cond_1667488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1667488___redundant_placeholder05
1while_while_cond_1667488___redundant_placeholder15
1while_while_cond_1667488___redundant_placeholder25
1while_while_cond_1667488___redundant_placeholder35
1while_while_cond_1667488___redundant_placeholder45
1while_while_cond_1667488___redundant_placeholder55
1while_while_cond_1667488___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¹'
»
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1665422

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	 .
biasadd_readvariableop_resource:	%
readvariableop_resource: '
readvariableop_1_resource: '
readvariableop_2_resource: 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_5ß
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityã

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1ã

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
±s
±
rnn_1_while_body_1668197(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3'
#rnn_1_while_rnn_1_strided_slice_1_0c
_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0T
Arnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 V
Crnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 Q
Brnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	H
:rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0: J
<rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0: J
<rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
rnn_1_while_identity
rnn_1_while_identity_1
rnn_1_while_identity_2
rnn_1_while_identity_3
rnn_1_while_identity_4
rnn_1_while_identity_5%
!rnn_1_while_rnn_1_strided_slice_1a
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensorR
?rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 T
Arnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 O
@rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	F
8rnn_1_while_peephole_lstm_cell_1_readvariableop_resource: H
:rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource: H
:rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource: ¢7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp¢1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1¢1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2Ï
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2?
=rnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape÷
/rnn_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0rnn_1_while_placeholderFrnn_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype021
/rnn_1/while/TensorArrayV2Read/TensorListGetItemó
6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOpArnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype028
6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp
'rnn_1/while/peephole_lstm_cell_1/MatMulMatMul6rnn_1/while/TensorArrayV2Read/TensorListGetItem:item:0>rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn_1/while/peephole_lstm_cell_1/MatMulù
8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpCrnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02:
8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpð
)rnn_1/while/peephole_lstm_cell_1/MatMul_1MatMulrnn_1_while_placeholder_2@rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn_1/while/peephole_lstm_cell_1/MatMul_1ð
$rnn_1/while/peephole_lstm_cell_1/addAddV21rnn_1/while/peephole_lstm_cell_1/MatMul:product:03rnn_1/while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rnn_1/while/peephole_lstm_cell_1/addò
7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpBrnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype029
7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpý
(rnn_1/while/peephole_lstm_cell_1/BiasAddBiasAdd(rnn_1/while/peephole_lstm_cell_1/add:z:0?rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn_1/while/peephole_lstm_cell_1/BiasAdd¦
0rnn_1/while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0rnn_1/while/peephole_lstm_cell_1/split/split_dimÃ
&rnn_1/while/peephole_lstm_cell_1/splitSplit9rnn_1/while/peephole_lstm_cell_1/split/split_dim:output:01rnn_1/while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&rnn_1/while/peephole_lstm_cell_1/splitÙ
/rnn_1/while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp:rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype021
/rnn_1/while/peephole_lstm_cell_1/ReadVariableOpÙ
$rnn_1/while/peephole_lstm_cell_1/mulMul7rnn_1/while/peephole_lstm_cell_1/ReadVariableOp:value:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn_1/while/peephole_lstm_cell_1/mulæ
&rnn_1/while/peephole_lstm_cell_1/add_1AddV2/rnn_1/while/peephole_lstm_cell_1/split:output:0(rnn_1/while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_1½
(rnn_1/while/peephole_lstm_cell_1/SigmoidSigmoid*rnn_1/while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(rnn_1/while/peephole_lstm_cell_1/Sigmoidß
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp<rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype023
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1ß
&rnn_1/while/peephole_lstm_cell_1/mul_1Mul9rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_1:value:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_1è
&rnn_1/while/peephole_lstm_cell_1/add_2AddV2/rnn_1/while/peephole_lstm_cell_1/split:output:1*rnn_1/while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_2Á
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_1Sigmoid*rnn_1/while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_1Ô
&rnn_1/while/peephole_lstm_cell_1/mul_2Mul.rnn_1/while/peephole_lstm_cell_1/Sigmoid_1:y:0rnn_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_2¹
%rnn_1/while/peephole_lstm_cell_1/TanhTanh/rnn_1/while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%rnn_1/while/peephole_lstm_cell_1/Tanhâ
&rnn_1/while/peephole_lstm_cell_1/mul_3Mul,rnn_1/while/peephole_lstm_cell_1/Sigmoid:y:0)rnn_1/while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_3ã
&rnn_1/while/peephole_lstm_cell_1/add_3AddV2*rnn_1/while/peephole_lstm_cell_1/mul_2:z:0*rnn_1/while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_3ß
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp<rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype023
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2ð
&rnn_1/while/peephole_lstm_cell_1/mul_4Mul9rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2:value:0*rnn_1/while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_4è
&rnn_1/while/peephole_lstm_cell_1/add_4AddV2/rnn_1/while/peephole_lstm_cell_1/split:output:3*rnn_1/while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/add_4Á
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_2Sigmoid*rnn_1/while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*rnn_1/while/peephole_lstm_cell_1/Sigmoid_2¸
'rnn_1/while/peephole_lstm_cell_1/Tanh_1Tanh*rnn_1/while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'rnn_1/while/peephole_lstm_cell_1/Tanh_1æ
&rnn_1/while/peephole_lstm_cell_1/mul_5Mul.rnn_1/while/peephole_lstm_cell_1/Sigmoid_2:y:0+rnn_1/while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn_1/while/peephole_lstm_cell_1/mul_5
0rnn_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_1_while_placeholder_1rnn_1_while_placeholder*rnn_1/while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype022
0rnn_1/while/TensorArrayV2Write/TensorListSetItemh
rnn_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add/y
rnn_1/while/addAddV2rnn_1_while_placeholderrnn_1/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/addl
rnn_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_1/while/add_1/y
rnn_1/while/add_1AddV2$rnn_1_while_rnn_1_while_loop_counterrnn_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn_1/while/add_1¸
rnn_1/while/IdentityIdentityrnn_1/while/add_1:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/IdentityÑ
rnn_1/while/Identity_1Identity*rnn_1_while_rnn_1_while_maximum_iterations8^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/Identity_1º
rnn_1/while/Identity_2Identityrnn_1/while/add:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/Identity_2ç
rnn_1/while/Identity_3Identity@rnn_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn_1/while/Identity_3â
rnn_1/while/Identity_4Identity*rnn_1/while/peephole_lstm_cell_1/mul_5:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/while/Identity_4â
rnn_1/while/Identity_5Identity*rnn_1/while/peephole_lstm_cell_1/add_3:z:08^rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7^rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp9^rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp0^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12^rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/while/Identity_5"5
rnn_1_while_identityrnn_1/while/Identity:output:0"9
rnn_1_while_identity_1rnn_1/while/Identity_1:output:0"9
rnn_1_while_identity_2rnn_1/while/Identity_2:output:0"9
rnn_1_while_identity_3rnn_1/while/Identity_3:output:0"9
rnn_1_while_identity_4rnn_1/while/Identity_4:output:0"9
rnn_1_while_identity_5rnn_1/while/Identity_5:output:0"
@rnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resourceBrnn_1_while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"
Arnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resourceCrnn_1_while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"
?rnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resourceArnn_1_while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"z
:rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource<rnn_1_while_peephole_lstm_cell_1_readvariableop_1_resource_0"z
:rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource<rnn_1_while_peephole_lstm_cell_1_readvariableop_2_resource_0"v
8rnn_1_while_peephole_lstm_cell_1_readvariableop_resource:rnn_1_while_peephole_lstm_cell_1_readvariableop_resource_0"H
!rnn_1_while_rnn_1_strided_slice_1#rnn_1_while_rnn_1_strided_slice_1_0"À
]rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_rnn_1_while_tensorarrayv2read_tensorlistgetitem_rnn_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2r
7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp7rnn_1/while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2p
6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp6rnn_1/while/peephole_lstm_cell_1/MatMul/ReadVariableOp2t
8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp8rnn_1/while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2b
/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp/rnn_1/while/peephole_lstm_cell_1/ReadVariableOp2f
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_11rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_12f
1rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_21rnn_1/while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ïp
à
@__inference_rnn_layer_call_and_return_conditional_losses_1669206
inputs_0D
1peephole_lstm_cell_matmul_readvariableop_resource:	F
3peephole_lstm_cell_matmul_1_readvariableop_resource:	 A
2peephole_lstm_cell_biasadd_readvariableop_resource:	8
*peephole_lstm_cell_readvariableop_resource: :
,peephole_lstm_cell_readvariableop_1_resource: :
,peephole_lstm_cell_readvariableop_2_resource: 
identity¢)peephole_lstm_cell/BiasAdd/ReadVariableOp¢(peephole_lstm_cell/MatMul/ReadVariableOp¢*peephole_lstm_cell/MatMul_1/ReadVariableOp¢!peephole_lstm_cell/ReadVariableOp¢#peephole_lstm_cell/ReadVariableOp_1¢#peephole_lstm_cell/ReadVariableOp_2¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ç
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp¿
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMulÍ
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp»
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/MatMul_1¸
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/addÆ
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOpÅ
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell/BiasAdd
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell/split­
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp¦
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul®
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_1
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid³
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1¬
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_1°
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_2
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_1¡
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_2
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanhª
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_3«
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_3³
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2¸
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_4°
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/add_4
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Sigmoid_2
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/Tanh_1®
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1669105*
condR
while_cond_1669104*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1ì
IdentityIdentitytranspose_1:y:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ä

rnn_1_while_cond_1668600(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3*
&rnn_1_while_less_rnn_1_strided_slice_1A
=rnn_1_while_rnn_1_while_cond_1668600___redundant_placeholder0A
=rnn_1_while_rnn_1_while_cond_1668600___redundant_placeholder1A
=rnn_1_while_rnn_1_while_cond_1668600___redundant_placeholder2A
=rnn_1_while_rnn_1_while_cond_1668600___redundant_placeholder3A
=rnn_1_while_rnn_1_while_cond_1668600___redundant_placeholder4A
=rnn_1_while_rnn_1_while_cond_1668600___redundant_placeholder5A
=rnn_1_while_rnn_1_while_cond_1668600___redundant_placeholder6
rnn_1_while_identity

rnn_1/while/LessLessrnn_1_while_placeholder&rnn_1_while_less_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
rnn_1/while/Lesso
rnn_1/while/IdentityIdentityrnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn_1/while/Identity"5
rnn_1_while_identityrnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
r
ø
B__inference_rnn_1_layer_call_and_return_conditional_losses_1670174

inputsF
3peephole_lstm_cell_1_matmul_readvariableop_resource:	 H
5peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 C
4peephole_lstm_cell_1_biasadd_readvariableop_resource:	:
,peephole_lstm_cell_1_readvariableop_resource: <
.peephole_lstm_cell_1_readvariableop_1_resource: <
.peephole_lstm_cell_1_readvariableop_2_resource: 
identity¢+peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢*peephole_lstm_cell_1/MatMul/ReadVariableOp¢,peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢#peephole_lstm_cell_1/ReadVariableOp¢%peephole_lstm_cell_1/ReadVariableOp_1¢%peephole_lstm_cell_1/ReadVariableOp_2¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2Í
*peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp3peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02,
*peephole_lstm_cell_1/MatMul/ReadVariableOpÅ
peephole_lstm_cell_1/MatMulMatMulstrided_slice_2:output:02peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMulÓ
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp5peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02.
,peephole_lstm_cell_1/MatMul_1/ReadVariableOpÁ
peephole_lstm_cell_1/MatMul_1MatMulzeros:output:04peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/MatMul_1À
peephole_lstm_cell_1/addAddV2%peephole_lstm_cell_1/MatMul:product:0'peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/addÌ
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp4peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+peephole_lstm_cell_1/BiasAdd/ReadVariableOpÍ
peephole_lstm_cell_1/BiasAddBiasAddpeephole_lstm_cell_1/add:z:03peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
peephole_lstm_cell_1/BiasAdd
$peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$peephole_lstm_cell_1/split/split_dim
peephole_lstm_cell_1/splitSplit-peephole_lstm_cell_1/split/split_dim:output:0%peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
peephole_lstm_cell_1/split³
#peephole_lstm_cell_1/ReadVariableOpReadVariableOp,peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell_1/ReadVariableOp¬
peephole_lstm_cell_1/mulMul+peephole_lstm_cell_1/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul¶
peephole_lstm_cell_1/add_1AddV2#peephole_lstm_cell_1/split:output:0peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_1
peephole_lstm_cell_1/SigmoidSigmoidpeephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Sigmoid¹
%peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp.peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_1²
peephole_lstm_cell_1/mul_1Mul-peephole_lstm_cell_1/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_1¸
peephole_lstm_cell_1/add_2AddV2#peephole_lstm_cell_1/split:output:1peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_2
peephole_lstm_cell_1/Sigmoid_1Sigmoidpeephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_1§
peephole_lstm_cell_1/mul_2Mul"peephole_lstm_cell_1/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_2
peephole_lstm_cell_1/TanhTanh#peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh²
peephole_lstm_cell_1/mul_3Mul peephole_lstm_cell_1/Sigmoid:y:0peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_3³
peephole_lstm_cell_1/add_3AddV2peephole_lstm_cell_1/mul_2:z:0peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_3¹
%peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp.peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02'
%peephole_lstm_cell_1/ReadVariableOp_2À
peephole_lstm_cell_1/mul_4Mul-peephole_lstm_cell_1/ReadVariableOp_2:value:0peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_4¸
peephole_lstm_cell_1/add_4AddV2#peephole_lstm_cell_1/split:output:3peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/add_4
peephole_lstm_cell_1/Sigmoid_2Sigmoidpeephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
peephole_lstm_cell_1/Sigmoid_2
peephole_lstm_cell_1/Tanh_1Tanhpeephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/Tanh_1¶
peephole_lstm_cell_1/mul_5Mul"peephole_lstm_cell_1/Sigmoid_2:y:0peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
peephole_lstm_cell_1/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¨
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:03peephole_lstm_cell_1_matmul_readvariableop_resource5peephole_lstm_cell_1_matmul_1_readvariableop_resource4peephole_lstm_cell_1_biasadd_readvariableop_resource,peephole_lstm_cell_1_readvariableop_resource.peephole_lstm_cell_1_readvariableop_1_resource.peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1670073*
condR
while_cond_1670072*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1ô
IdentityIdentitystrided_slice_3:output:0,^peephole_lstm_cell_1/BiasAdd/ReadVariableOp+^peephole_lstm_cell_1/MatMul/ReadVariableOp-^peephole_lstm_cell_1/MatMul_1/ReadVariableOp$^peephole_lstm_cell_1/ReadVariableOp&^peephole_lstm_cell_1/ReadVariableOp_1&^peephole_lstm_cell_1/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2Z
+peephole_lstm_cell_1/BiasAdd/ReadVariableOp+peephole_lstm_cell_1/BiasAdd/ReadVariableOp2X
*peephole_lstm_cell_1/MatMul/ReadVariableOp*peephole_lstm_cell_1/MatMul/ReadVariableOp2\
,peephole_lstm_cell_1/MatMul_1/ReadVariableOp,peephole_lstm_cell_1/MatMul_1/ReadVariableOp2J
#peephole_lstm_cell_1/ReadVariableOp#peephole_lstm_cell_1/ReadVariableOp2N
%peephole_lstm_cell_1/ReadVariableOp_1%peephole_lstm_cell_1/ReadVariableOp_12N
%peephole_lstm_cell_1/ReadVariableOp_2%peephole_lstm_cell_1/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_1670252
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1670252___redundant_placeholder05
1while_while_cond_1670252___redundant_placeholder15
1while_while_cond_1670252___redundant_placeholder25
1while_while_cond_1670252___redundant_placeholder35
1while_while_cond_1670252___redundant_placeholder45
1while_while_cond_1670252___redundant_placeholder55
1while_while_cond_1670252___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
©-
Ô	
while_body_1666276
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
$while_peephole_lstm_cell_1_1666300_0:	 7
$while_peephole_lstm_cell_1_1666302_0:	 3
$while_peephole_lstm_cell_1_1666304_0:	2
$while_peephole_lstm_cell_1_1666306_0: 2
$while_peephole_lstm_cell_1_1666308_0: 2
$while_peephole_lstm_cell_1_1666310_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
"while_peephole_lstm_cell_1_1666300:	 5
"while_peephole_lstm_cell_1_1666302:	 1
"while_peephole_lstm_cell_1_1666304:	0
"while_peephole_lstm_cell_1_1666306: 0
"while_peephole_lstm_cell_1_1666308: 0
"while_peephole_lstm_cell_1_1666310: ¢2while/peephole_lstm_cell_1/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
2while/peephole_lstm_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3$while_peephole_lstm_cell_1_1666300_0$while_peephole_lstm_cell_1_1666302_0$while_peephole_lstm_cell_1_1666304_0$while_peephole_lstm_cell_1_1666306_0$while_peephole_lstm_cell_1_1666308_0$while_peephole_lstm_cell_1_1666310_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_166618024
2while/peephole_lstm_cell_1/StatefulPartitionedCallÿ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder;while/peephole_lstm_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:03^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity¦
while/Identity_1Identitywhile_while_maximum_iterations3^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:03^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2Â
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:03^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ô
while/Identity_4Identity;while/peephole_lstm_cell_1/StatefulPartitionedCall:output:13^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4Ô
while/Identity_5Identity;while/peephole_lstm_cell_1/StatefulPartitionedCall:output:23^while/peephole_lstm_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"J
"while_peephole_lstm_cell_1_1666300$while_peephole_lstm_cell_1_1666300_0"J
"while_peephole_lstm_cell_1_1666302$while_peephole_lstm_cell_1_1666302_0"J
"while_peephole_lstm_cell_1_1666304$while_peephole_lstm_cell_1_1666304_0"J
"while_peephole_lstm_cell_1_1666306$while_peephole_lstm_cell_1_1666306_0"J
"while_peephole_lstm_cell_1_1666308$while_peephole_lstm_cell_1_1666308_0"J
"while_peephole_lstm_cell_1_1666310$while_peephole_lstm_cell_1_1666310_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2while/peephole_lstm_cell_1/StatefulPartitionedCall2while/peephole_lstm_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ð
Í
G__inference_sequential_layer_call_and_return_conditional_losses_1667132

inputs$
conv1d_1666709:
conv1d_1666711:
rnn_1666909:	
rnn_1666911:	 
rnn_1666913:	
rnn_1666915: 
rnn_1666917: 
rnn_1666919:  
rnn_1_1667102:	  
rnn_1_1667104:	 
rnn_1_1667106:	
rnn_1_1667108: 
rnn_1_1667110: 
rnn_1_1667112: 
dense_1667126: 
dense_1667128:
identity¢conv1d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢rnn/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1666709conv1d_1666711*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_16667082 
conv1d/StatefulPartitionedCallú
reshape/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_16667272
reshape/PartitionedCallÛ
rnn/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0rnn_1666909rnn_1666911rnn_1666913rnn_1666915rnn_1666917rnn_1666919*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16669082
rnn/StatefulPartitionedCallí
rnn_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0rnn_1_1667102rnn_1_1667104rnn_1_1667106rnn_1_1667108rnn_1_1667110rnn_1_1667112*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16671012
rnn_1/StatefulPartitionedCall«
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_1667126dense_1667128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16671252
dense/StatefulPartitionedCallù
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã¯
«
#__inference__traced_restore_1670976
file_prefix4
assignvariableop_conv1d_kernel:,
assignvariableop_1_conv1d_bias:1
assignvariableop_2_dense_kernel: +
assignvariableop_3_dense_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: C
0assignvariableop_9_rnn_peephole_lstm_cell_kernel:	N
;assignvariableop_10_rnn_peephole_lstm_cell_recurrent_kernel:	 >
/assignvariableop_11_rnn_peephole_lstm_cell_bias:	T
Fassignvariableop_12_rnn_peephole_lstm_cell_input_gate_peephole_weights: U
Gassignvariableop_13_rnn_peephole_lstm_cell_forget_gate_peephole_weights: U
Gassignvariableop_14_rnn_peephole_lstm_cell_output_gate_peephole_weights: H
5assignvariableop_15_rnn_1_peephole_lstm_cell_1_kernel:	 R
?assignvariableop_16_rnn_1_peephole_lstm_cell_1_recurrent_kernel:	 B
3assignvariableop_17_rnn_1_peephole_lstm_cell_1_bias:	X
Jassignvariableop_18_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weights: Y
Kassignvariableop_19_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weights: Y
Kassignvariableop_20_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: C
-assignvariableop_23_rmsprop_conv1d_kernel_rms:9
+assignvariableop_24_rmsprop_conv1d_bias_rms:>
,assignvariableop_25_rmsprop_dense_kernel_rms: 8
*assignvariableop_26_rmsprop_dense_bias_rms:P
=assignvariableop_27_rmsprop_rnn_peephole_lstm_cell_kernel_rms:	Z
Gassignvariableop_28_rmsprop_rnn_peephole_lstm_cell_recurrent_kernel_rms:	 J
;assignvariableop_29_rmsprop_rnn_peephole_lstm_cell_bias_rms:	`
Rassignvariableop_30_rmsprop_rnn_peephole_lstm_cell_input_gate_peephole_weights_rms: a
Sassignvariableop_31_rmsprop_rnn_peephole_lstm_cell_forget_gate_peephole_weights_rms: a
Sassignvariableop_32_rmsprop_rnn_peephole_lstm_cell_output_gate_peephole_weights_rms: T
Aassignvariableop_33_rmsprop_rnn_1_peephole_lstm_cell_1_kernel_rms:	 ^
Kassignvariableop_34_rmsprop_rnn_1_peephole_lstm_cell_1_recurrent_kernel_rms:	 N
?assignvariableop_35_rmsprop_rnn_1_peephole_lstm_cell_1_bias_rms:	d
Vassignvariableop_36_rmsprop_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weights_rms: e
Wassignvariableop_37_rmsprop_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weights_rms: e
Wassignvariableop_38_rmsprop_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weights_rms: 
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9×
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*ã
valueÙBÖ(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÞ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¤
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_rmsprop_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6­
AssignVariableOp_6AssignVariableOp(assignvariableop_6_rmsprop_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¨
AssignVariableOp_7AssignVariableOp#assignvariableop_7_rmsprop_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_rhoIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9µ
AssignVariableOp_9AssignVariableOp0assignvariableop_9_rnn_peephole_lstm_cell_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ã
AssignVariableOp_10AssignVariableOp;assignvariableop_10_rnn_peephole_lstm_cell_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_rnn_peephole_lstm_cell_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Î
AssignVariableOp_12AssignVariableOpFassignvariableop_12_rnn_peephole_lstm_cell_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ï
AssignVariableOp_13AssignVariableOpGassignvariableop_13_rnn_peephole_lstm_cell_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ï
AssignVariableOp_14AssignVariableOpGassignvariableop_14_rnn_peephole_lstm_cell_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15½
AssignVariableOp_15AssignVariableOp5assignvariableop_15_rnn_1_peephole_lstm_cell_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ç
AssignVariableOp_16AssignVariableOp?assignvariableop_16_rnn_1_peephole_lstm_cell_1_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17»
AssignVariableOp_17AssignVariableOp3assignvariableop_17_rnn_1_peephole_lstm_cell_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ò
AssignVariableOp_18AssignVariableOpJassignvariableop_18_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ó
AssignVariableOp_19AssignVariableOpKassignvariableop_19_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ó
AssignVariableOp_20AssignVariableOpKassignvariableop_20_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¡
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¡
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23µ
AssignVariableOp_23AssignVariableOp-assignvariableop_23_rmsprop_conv1d_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24³
AssignVariableOp_24AssignVariableOp+assignvariableop_24_rmsprop_conv1d_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25´
AssignVariableOp_25AssignVariableOp,assignvariableop_25_rmsprop_dense_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26²
AssignVariableOp_26AssignVariableOp*assignvariableop_26_rmsprop_dense_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Å
AssignVariableOp_27AssignVariableOp=assignvariableop_27_rmsprop_rnn_peephole_lstm_cell_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ï
AssignVariableOp_28AssignVariableOpGassignvariableop_28_rmsprop_rnn_peephole_lstm_cell_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ã
AssignVariableOp_29AssignVariableOp;assignvariableop_29_rmsprop_rnn_peephole_lstm_cell_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ú
AssignVariableOp_30AssignVariableOpRassignvariableop_30_rmsprop_rnn_peephole_lstm_cell_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Û
AssignVariableOp_31AssignVariableOpSassignvariableop_31_rmsprop_rnn_peephole_lstm_cell_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Û
AssignVariableOp_32AssignVariableOpSassignvariableop_32_rmsprop_rnn_peephole_lstm_cell_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33É
AssignVariableOp_33AssignVariableOpAassignvariableop_33_rmsprop_rnn_1_peephole_lstm_cell_1_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ó
AssignVariableOp_34AssignVariableOpKassignvariableop_34_rmsprop_rnn_1_peephole_lstm_cell_1_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ç
AssignVariableOp_35AssignVariableOp?assignvariableop_35_rmsprop_rnn_1_peephole_lstm_cell_1_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Þ
AssignVariableOp_36AssignVariableOpVassignvariableop_36_rmsprop_rnn_1_peephole_lstm_cell_1_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ß
AssignVariableOp_37AssignVariableOpWassignvariableop_37_rmsprop_rnn_1_peephole_lstm_cell_1_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ß
AssignVariableOp_38AssignVariableOpWassignvariableop_38_rmsprop_rnn_1_peephole_lstm_cell_1_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¸
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39«
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
g
»
while_body_1667000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0N
;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 P
=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 K
<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	B
4while_peephole_lstm_cell_1_readvariableop_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_1_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorL
9while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2while_peephole_lstm_cell_1_readvariableop_resource: B
4while_peephole_lstm_cell_1_readvariableop_1_resource: B
4while_peephole_lstm_cell_1_readvariableop_2_resource: ¢1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)while/peephole_lstm_cell_1/ReadVariableOp¢+while/peephole_lstm_cell_1/ReadVariableOp_1¢+while/peephole_lstm_cell_1/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpï
!while/peephole_lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:08while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell_1/MatMulç
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype024
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpØ
#while/peephole_lstm_cell_1/MatMul_1MatMulwhile_placeholder_2:while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#while/peephole_lstm_cell_1/MatMul_1Ø
while/peephole_lstm_cell_1/addAddV2+while/peephole_lstm_cell_1/MatMul:product:0-while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/peephole_lstm_cell_1/addà
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"while/peephole_lstm_cell_1/BiasAddBiasAdd"while/peephole_lstm_cell_1/add:z:09while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"while/peephole_lstm_cell_1/BiasAdd
*while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*while/peephole_lstm_cell_1/split/split_dim«
 while/peephole_lstm_cell_1/splitSplit3while/peephole_lstm_cell_1/split/split_dim:output:0+while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 while/peephole_lstm_cell_1/splitÇ
)while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp4while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell_1/ReadVariableOpÁ
while/peephole_lstm_cell_1/mulMul1while/peephole_lstm_cell_1/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell_1/mulÎ
 while/peephole_lstm_cell_1/add_1AddV2)while/peephole_lstm_cell_1/split:output:0"while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_1«
"while/peephole_lstm_cell_1/SigmoidSigmoid$while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell_1/SigmoidÍ
+while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_1Ç
 while/peephole_lstm_cell_1/mul_1Mul3while/peephole_lstm_cell_1/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_1Ð
 while/peephole_lstm_cell_1/add_2AddV2)while/peephole_lstm_cell_1/split:output:1$while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_2¯
$while/peephole_lstm_cell_1/Sigmoid_1Sigmoid$while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_1¼
 while/peephole_lstm_cell_1/mul_2Mul(while/peephole_lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_2§
while/peephole_lstm_cell_1/TanhTanh)while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell_1/TanhÊ
 while/peephole_lstm_cell_1/mul_3Mul&while/peephole_lstm_cell_1/Sigmoid:y:0#while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_3Ë
 while/peephole_lstm_cell_1/add_3AddV2$while/peephole_lstm_cell_1/mul_2:z:0$while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_3Í
+while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_2Ø
 while/peephole_lstm_cell_1/mul_4Mul3while/peephole_lstm_cell_1/ReadVariableOp_2:value:0$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_4Ð
 while/peephole_lstm_cell_1/add_4AddV2)while/peephole_lstm_cell_1/split:output:3$while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_4¯
$while/peephole_lstm_cell_1/Sigmoid_2Sigmoid$while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_2¦
!while/peephole_lstm_cell_1/Tanh_1Tanh$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!while/peephole_lstm_cell_1/Tanh_1Î
 while/peephole_lstm_cell_1/mul_5Mul(while/peephole_lstm_cell_1/Sigmoid_2:y:0%while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_5è
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations2^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2±
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3¬
while/Identity_4Identity$while/peephole_lstm_cell_1/mul_5:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4¬
while/Identity_5Identity$while/peephole_lstm_cell_1/add_3:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"|
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"x
9while_peephole_lstm_cell_1_matmul_readvariableop_resource;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_1_resource6while_peephole_lstm_cell_1_readvariableop_1_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_2_resource6while_peephole_lstm_cell_1_readvariableop_2_resource_0"j
2while_peephole_lstm_cell_1_readvariableop_resource4while_peephole_lstm_cell_1_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2f
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0while/peephole_lstm_cell_1/MatMul/ReadVariableOp0while/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)while/peephole_lstm_cell_1/ReadVariableOp)while/peephole_lstm_cell_1/ReadVariableOp2Z
+while/peephole_lstm_cell_1/ReadVariableOp_1+while/peephole_lstm_cell_1/ReadVariableOp_12Z
+while/peephole_lstm_cell_1/ReadVariableOp_2+while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


'__inference_dense_layer_call_fn_1670441

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16671252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
©
ó
!sequential_rnn_while_cond_1664864:
6sequential_rnn_while_sequential_rnn_while_loop_counter@
<sequential_rnn_while_sequential_rnn_while_maximum_iterations$
 sequential_rnn_while_placeholder&
"sequential_rnn_while_placeholder_1&
"sequential_rnn_while_placeholder_2&
"sequential_rnn_while_placeholder_3<
8sequential_rnn_while_less_sequential_rnn_strided_slice_1S
Osequential_rnn_while_sequential_rnn_while_cond_1664864___redundant_placeholder0S
Osequential_rnn_while_sequential_rnn_while_cond_1664864___redundant_placeholder1S
Osequential_rnn_while_sequential_rnn_while_cond_1664864___redundant_placeholder2S
Osequential_rnn_while_sequential_rnn_while_cond_1664864___redundant_placeholder3S
Osequential_rnn_while_sequential_rnn_while_cond_1664864___redundant_placeholder4S
Osequential_rnn_while_sequential_rnn_while_cond_1664864___redundant_placeholder5S
Osequential_rnn_while_sequential_rnn_while_cond_1664864___redundant_placeholder6!
sequential_rnn_while_identity
»
sequential/rnn/while/LessLess sequential_rnn_while_placeholder8sequential_rnn_while_less_sequential_rnn_strided_slice_1*
T0*
_output_shapes
: 2
sequential/rnn/while/Less
sequential/rnn/while/IdentityIdentitysequential/rnn/while/Less:z:0*
T0
*
_output_shapes
: 2
sequential/rnn/while/Identity"G
sequential_rnn_while_identity&sequential/rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:


í
while_cond_1666806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1666806___redundant_placeholder05
1while_while_cond_1666806___redundant_placeholder15
1while_while_cond_1666806___redundant_placeholder25
1while_while_cond_1666806___redundant_placeholder35
1while_while_cond_1666806___redundant_placeholder45
1while_while_cond_1666806___redundant_placeholder55
1while_while_cond_1666806___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ä

rnn_1_while_cond_1668196(
$rnn_1_while_rnn_1_while_loop_counter.
*rnn_1_while_rnn_1_while_maximum_iterations
rnn_1_while_placeholder
rnn_1_while_placeholder_1
rnn_1_while_placeholder_2
rnn_1_while_placeholder_3*
&rnn_1_while_less_rnn_1_strided_slice_1A
=rnn_1_while_rnn_1_while_cond_1668196___redundant_placeholder0A
=rnn_1_while_rnn_1_while_cond_1668196___redundant_placeholder1A
=rnn_1_while_rnn_1_while_cond_1668196___redundant_placeholder2A
=rnn_1_while_rnn_1_while_cond_1668196___redundant_placeholder3A
=rnn_1_while_rnn_1_while_cond_1668196___redundant_placeholder4A
=rnn_1_while_rnn_1_while_cond_1668196___redundant_placeholder5A
=rnn_1_while_rnn_1_while_cond_1668196___redundant_placeholder6
rnn_1_while_identity

rnn_1/while/LessLessrnn_1_while_placeholder&rnn_1_while_less_rnn_1_strided_slice_1*
T0*
_output_shapes
: 2
rnn_1/while/Lesso
rnn_1/while/IdentityIdentityrnn_1/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn_1/while/Identity"5
rnn_1_while_identityrnn_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¸d

while_body_1666807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
9while_peephole_lstm_cell_matmul_readvariableop_resource_0:	N
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 I
:while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	@
2while_peephole_lstm_cell_readvariableop_resource_0: B
4while_peephole_lstm_cell_readvariableop_1_resource_0: B
4while_peephole_lstm_cell_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
7while_peephole_lstm_cell_matmul_readvariableop_resource:	L
9while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 G
8while_peephole_lstm_cell_biasadd_readvariableop_resource:	>
0while_peephole_lstm_cell_readvariableop_resource: @
2while_peephole_lstm_cell_readvariableop_1_resource: @
2while_peephole_lstm_cell_readvariableop_2_resource: ¢/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢.while/peephole_lstm_cell/MatMul/ReadVariableOp¢0while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢'while/peephole_lstm_cell/ReadVariableOp¢)while/peephole_lstm_cell/ReadVariableOp_1¢)while/peephole_lstm_cell/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÛ
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOpé
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
while/peephole_lstm_cell/MatMulá
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpÒ
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell/MatMul_1Ð
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/peephole_lstm_cell/addÚ
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpÝ
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 while/peephole_lstm_cell/BiasAdd
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim£
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2 
while/peephole_lstm_cell/splitÁ
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp»
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/mulÆ
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_1¥
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell/SigmoidÇ
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1Á
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_1È
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_2©
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_1¶
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_2¡
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/peephole_lstm_cell/TanhÂ
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_3Ã
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_3Ç
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2Ð
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_4È
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/add_4©
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell/Sigmoid_2 
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell/Tanh_1Æ
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell/mul_5æ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ö
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1ø
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2¥
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


,__inference_sequential_layer_call_fn_1668745

inputs
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:	 
	unknown_3:	
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7:	 
	unknown_8:	 
	unknown_9:	

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14:
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16671322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼

G__inference_sequential_layer_call_and_return_conditional_losses_1668708

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource:G
9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource:H
5rnn_peephole_lstm_cell_matmul_readvariableop_resource:	J
7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource:	 E
6rnn_peephole_lstm_cell_biasadd_readvariableop_resource:	<
.rnn_peephole_lstm_cell_readvariableop_resource: >
0rnn_peephole_lstm_cell_readvariableop_1_resource: >
0rnn_peephole_lstm_cell_readvariableop_2_resource: L
9rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2rnn_1_peephole_lstm_cell_1_readvariableop_resource: B
4rnn_1_peephole_lstm_cell_1_readvariableop_1_resource: B
4rnn_1_peephole_lstm_cell_1_readvariableop_2_resource: 6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource:
identity¢)conv1d/conv1d/ExpandDims_1/ReadVariableOp¢0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp¢,rnn/peephole_lstm_cell/MatMul/ReadVariableOp¢.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp¢%rnn/peephole_lstm_cell/ReadVariableOp¢'rnn/peephole_lstm_cell/ReadVariableOp_1¢'rnn/peephole_lstm_cell/ReadVariableOp_2¢	rnn/while¢1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)rnn_1/peephole_lstm_cell_1/ReadVariableOp¢+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1¢+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2¢rnn_1/while
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDims/dim¯
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDimsÍ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimÓ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1{
conv1d/conv1d/ShapeShape!conv1d/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
conv1d/conv1d/Shape
!conv1d/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv1d/conv1d/strided_slice/stack
#conv1d/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2%
#conv1d/conv1d/strided_slice/stack_1
#conv1d/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv1d/conv1d/strided_slice/stack_2´
conv1d/conv1d/strided_sliceStridedSliceconv1d/conv1d/Shape:output:0*conv1d/conv1d/strided_slice/stack:output:0,conv1d/conv1d/strided_slice/stack_1:output:0,conv1d/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv1d/conv1d/strided_slice
conv1d/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2
conv1d/conv1d/Reshape/shape¼
conv1d/conv1d/ReshapeReshape!conv1d/conv1d/ExpandDims:output:0$conv1d/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ReshapeÞ
conv1d/conv1d/Conv2DConv2Dconv1d/conv1d/Reshape:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv1d/conv1d/Conv2D
conv1d/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2
conv1d/conv1d/concat/values_1
conv1d/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/concat/axisØ
conv1d/conv1d/concatConcatV2$conv1d/conv1d/strided_slice:output:0&conv1d/conv1d/concat/values_1:output:0"conv1d/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d/conv1d/concat¹
conv1d/conv1d/Reshape_1Reshapeconv1d/conv1d/Conv2D:output:0conv1d/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/conv1d/Reshape_1µ
conv1d/conv1d/SqueezeSqueeze conv1d/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/Squeeze
conv1d/squeeze_batch_dims/ShapeShapeconv1d/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2!
conv1d/squeeze_batch_dims/Shape¨
-conv1d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv1d/squeeze_batch_dims/strided_slice/stackµ
/conv1d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ21
/conv1d/squeeze_batch_dims/strided_slice/stack_1¬
/conv1d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv1d/squeeze_batch_dims/strided_slice/stack_2ü
'conv1d/squeeze_batch_dims/strided_sliceStridedSlice(conv1d/squeeze_batch_dims/Shape:output:06conv1d/squeeze_batch_dims/strided_slice/stack:output:08conv1d/squeeze_batch_dims/strided_slice/stack_1:output:08conv1d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv1d/squeeze_batch_dims/strided_slice§
'conv1d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2)
'conv1d/squeeze_batch_dims/Reshape/shapeÙ
!conv1d/squeeze_batch_dims/ReshapeReshapeconv1d/conv1d/Squeeze:output:00conv1d/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!conv1d/squeeze_batch_dims/ReshapeÚ
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv1d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOpí
!conv1d/squeeze_batch_dims/BiasAddBiasAdd*conv1d/squeeze_batch_dims/Reshape:output:08conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!conv1d/squeeze_batch_dims/BiasAdd§
)conv1d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)conv1d/squeeze_batch_dims/concat/values_1
%conv1d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%conv1d/squeeze_batch_dims/concat/axis
 conv1d/squeeze_batch_dims/concatConcatV20conv1d/squeeze_batch_dims/strided_slice:output:02conv1d/squeeze_batch_dims/concat/values_1:output:0.conv1d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv1d/squeeze_batch_dims/concatæ
#conv1d/squeeze_batch_dims/Reshape_1Reshape*conv1d/squeeze_batch_dims/BiasAdd:output:0)conv1d/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#conv1d/squeeze_batch_dims/Reshape_1z
reshape/ShapeShape,conv1d/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2È
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape±
reshape/ReshapeReshape,conv1d/squeeze_batch_dims/Reshape_1:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshape^
	rnn/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2ú
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	rnn/zerosh
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/mul/y
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lessn
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/packed/1
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/zeros_1}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm
rnn/transpose	Transposereshape/Reshape:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
rnn/TensorArrayV2/element_shapeÂ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ç
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
rnn/strided_slice_2Ó
,rnn/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp5rnn_peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,rnn/peephole_lstm_cell/MatMul/ReadVariableOpÏ
rnn/peephole_lstm_cell/MatMulMatMulrnn/strided_slice_2:output:04rnn/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/peephole_lstm_cell/MatMulÙ
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype020
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpË
rnn/peephole_lstm_cell/MatMul_1MatMulrnn/zeros:output:06rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
rnn/peephole_lstm_cell/MatMul_1È
rnn/peephole_lstm_cell/addAddV2'rnn/peephole_lstm_cell/MatMul:product:0)rnn/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/peephole_lstm_cell/addÒ
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6rnn_peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpÕ
rnn/peephole_lstm_cell/BiasAddBiasAddrnn/peephole_lstm_cell/add:z:05rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rnn/peephole_lstm_cell/BiasAdd
&rnn/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&rnn/peephole_lstm_cell/split/split_dim
rnn/peephole_lstm_cell/splitSplit/rnn/peephole_lstm_cell/split/split_dim:output:0'rnn/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rnn/peephole_lstm_cell/split¹
%rnn/peephole_lstm_cell/ReadVariableOpReadVariableOp.rnn_peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02'
%rnn/peephole_lstm_cell/ReadVariableOp¶
rnn/peephole_lstm_cell/mulMul-rnn/peephole_lstm_cell/ReadVariableOp:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul¾
rnn/peephole_lstm_cell/add_1AddV2%rnn/peephole_lstm_cell/split:output:0rnn/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_1
rnn/peephole_lstm_cell/SigmoidSigmoid rnn/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
rnn/peephole_lstm_cell/Sigmoid¿
'rnn/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_1¼
rnn/peephole_lstm_cell/mul_1Mul/rnn/peephole_lstm_cell/ReadVariableOp_1:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_1À
rnn/peephole_lstm_cell/add_2AddV2%rnn/peephole_lstm_cell/split:output:1 rnn/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_2£
 rnn/peephole_lstm_cell/Sigmoid_1Sigmoid rnn/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn/peephole_lstm_cell/Sigmoid_1±
rnn/peephole_lstm_cell/mul_2Mul$rnn/peephole_lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_2
rnn/peephole_lstm_cell/TanhTanh%rnn/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/Tanhº
rnn/peephole_lstm_cell/mul_3Mul"rnn/peephole_lstm_cell/Sigmoid:y:0rnn/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_3»
rnn/peephole_lstm_cell/add_3AddV2 rnn/peephole_lstm_cell/mul_2:z:0 rnn/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_3¿
'rnn/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_2È
rnn/peephole_lstm_cell/mul_4Mul/rnn/peephole_lstm_cell/ReadVariableOp_2:value:0 rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_4À
rnn/peephole_lstm_cell/add_4AddV2%rnn/peephole_lstm_cell/split:output:3 rnn/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/add_4£
 rnn/peephole_lstm_cell/Sigmoid_2Sigmoid rnn/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn/peephole_lstm_cell/Sigmoid_2
rnn/peephole_lstm_cell/Tanh_1Tanh rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/Tanh_1¾
rnn/peephole_lstm_cell/mul_5Mul$rnn/peephole_lstm_cell/Sigmoid_2:y:0!rnn/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/peephole_lstm_cell/mul_5
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2#
!rnn/TensorArrayV2_1/element_shapeÈ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counterä
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_peephole_lstm_cell_matmul_readvariableop_resource7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource6rnn_peephole_lstm_cell_biasadd_readvariableop_resource.rnn_peephole_lstm_cell_readvariableop_resource0rnn_peephole_lstm_cell_readvariableop_1_resource0rnn_peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*"
bodyR
rnn_while_body_1668425*"
condR
rnn_while_cond_1668424*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
	rnn/while½
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeø
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2²
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/permµ
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/transpose_1]
rnn_1/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:2
rnn_1/Shape
rnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice/stack
rnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_1
rnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice/stack_2
rnn_1/strided_sliceStridedSlicernn_1/Shape:output:0"rnn_1/strided_slice/stack:output:0$rnn_1/strided_slice/stack_1:output:0$rnn_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_sliceh
rnn_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros/mul/y
rnn_1/zeros/mulMulrnn_1/strided_slice:output:0rnn_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/mulk
rnn_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros/Less/y
rnn_1/zeros/LessLessrnn_1/zeros/mul:z:0rnn_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros/Lessn
rnn_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros/packed/1
rnn_1/zeros/packedPackrnn_1/strided_slice:output:0rnn_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros/packedk
rnn_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros/Const
rnn_1/zerosFillrnn_1/zeros/packed:output:0rnn_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/zerosl
rnn_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros_1/mul/y
rnn_1/zeros_1/mulMulrnn_1/strided_slice:output:0rnn_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/mulo
rnn_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn_1/zeros_1/Less/y
rnn_1/zeros_1/LessLessrnn_1/zeros_1/mul:z:0rnn_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_1/zeros_1/Lessr
rnn_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/zeros_1/packed/1¡
rnn_1/zeros_1/packedPackrnn_1/strided_slice:output:0rnn_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_1/zeros_1/packedo
rnn_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_1/zeros_1/Const
rnn_1/zeros_1Fillrnn_1/zeros_1/packed:output:0rnn_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/zeros_1
rnn_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose/perm
rnn_1/transpose	Transposernn/transpose_1:y:0rnn_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/transposea
rnn_1/Shape_1Shapernn_1/transpose:y:0*
T0*
_output_shapes
:2
rnn_1/Shape_1
rnn_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_1/stack
rnn_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_1
rnn_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_1/stack_2
rnn_1/strided_slice_1StridedSlicernn_1/Shape_1:output:0$rnn_1/strided_slice_1/stack:output:0&rnn_1/strided_slice_1/stack_1:output:0&rnn_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_1/strided_slice_1
!rnn_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!rnn_1/TensorArrayV2/element_shapeÊ
rnn_1/TensorArrayV2TensorListReserve*rnn_1/TensorArrayV2/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2Ë
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;rnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-rnn_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_1/transpose:y:0Drnn_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-rnn_1/TensorArrayUnstack/TensorListFromTensor
rnn_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_2/stack
rnn_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_1
rnn_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_2/stack_2 
rnn_1/strided_slice_2StridedSlicernn_1/transpose:y:0$rnn_1/strided_slice_2/stack:output:0&rnn_1/strided_slice_2/stack_1:output:0&rnn_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rnn_1/strided_slice_2ß
0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp9rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype022
0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOpÝ
!rnn_1/peephole_lstm_cell_1/MatMulMatMulrnn_1/strided_slice_2:output:08rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!rnn_1/peephole_lstm_cell_1/MatMulå
2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp;rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype024
2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOpÙ
#rnn_1/peephole_lstm_cell_1/MatMul_1MatMulrnn_1/zeros:output:0:rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn_1/peephole_lstm_cell_1/MatMul_1Ø
rnn_1/peephole_lstm_cell_1/addAddV2+rnn_1/peephole_lstm_cell_1/MatMul:product:0-rnn_1/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rnn_1/peephole_lstm_cell_1/addÞ
1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp:rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"rnn_1/peephole_lstm_cell_1/BiasAddBiasAdd"rnn_1/peephole_lstm_cell_1/add:z:09rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"rnn_1/peephole_lstm_cell_1/BiasAdd
*rnn_1/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*rnn_1/peephole_lstm_cell_1/split/split_dim«
 rnn_1/peephole_lstm_cell_1/splitSplit3rnn_1/peephole_lstm_cell_1/split/split_dim:output:0+rnn_1/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 rnn_1/peephole_lstm_cell_1/splitÅ
)rnn_1/peephole_lstm_cell_1/ReadVariableOpReadVariableOp2rnn_1_peephole_lstm_cell_1_readvariableop_resource*
_output_shapes
: *
dtype02+
)rnn_1/peephole_lstm_cell_1/ReadVariableOpÄ
rnn_1/peephole_lstm_cell_1/mulMul1rnn_1/peephole_lstm_cell_1/ReadVariableOp:value:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
rnn_1/peephole_lstm_cell_1/mulÎ
 rnn_1/peephole_lstm_cell_1/add_1AddV2)rnn_1/peephole_lstm_cell_1/split:output:0"rnn_1/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_1«
"rnn_1/peephole_lstm_cell_1/SigmoidSigmoid$rnn_1/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn_1/peephole_lstm_cell_1/SigmoidË
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp4rnn_1_peephole_lstm_cell_1_readvariableop_1_resource*
_output_shapes
: *
dtype02-
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1Ê
 rnn_1/peephole_lstm_cell_1/mul_1Mul3rnn_1/peephole_lstm_cell_1/ReadVariableOp_1:value:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_1Ð
 rnn_1/peephole_lstm_cell_1/add_2AddV2)rnn_1/peephole_lstm_cell_1/split:output:1$rnn_1/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_2¯
$rnn_1/peephole_lstm_cell_1/Sigmoid_1Sigmoid$rnn_1/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn_1/peephole_lstm_cell_1/Sigmoid_1¿
 rnn_1/peephole_lstm_cell_1/mul_2Mul(rnn_1/peephole_lstm_cell_1/Sigmoid_1:y:0rnn_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_2§
rnn_1/peephole_lstm_cell_1/TanhTanh)rnn_1/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rnn_1/peephole_lstm_cell_1/TanhÊ
 rnn_1/peephole_lstm_cell_1/mul_3Mul&rnn_1/peephole_lstm_cell_1/Sigmoid:y:0#rnn_1/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_3Ë
 rnn_1/peephole_lstm_cell_1/add_3AddV2$rnn_1/peephole_lstm_cell_1/mul_2:z:0$rnn_1/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_3Ë
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp4rnn_1_peephole_lstm_cell_1_readvariableop_2_resource*
_output_shapes
: *
dtype02-
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2Ø
 rnn_1/peephole_lstm_cell_1/mul_4Mul3rnn_1/peephole_lstm_cell_1/ReadVariableOp_2:value:0$rnn_1/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_4Ð
 rnn_1/peephole_lstm_cell_1/add_4AddV2)rnn_1/peephole_lstm_cell_1/split:output:3$rnn_1/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/add_4¯
$rnn_1/peephole_lstm_cell_1/Sigmoid_2Sigmoid$rnn_1/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn_1/peephole_lstm_cell_1/Sigmoid_2¦
!rnn_1/peephole_lstm_cell_1/Tanh_1Tanh$rnn_1/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rnn_1/peephole_lstm_cell_1/Tanh_1Î
 rnn_1/peephole_lstm_cell_1/mul_5Mul(rnn_1/peephole_lstm_cell_1/Sigmoid_2:y:0%rnn_1/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn_1/peephole_lstm_cell_1/mul_5
#rnn_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2%
#rnn_1/TensorArrayV2_1/element_shapeÐ
rnn_1/TensorArrayV2_1TensorListReserve,rnn_1/TensorArrayV2_1/element_shape:output:0rnn_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_1/TensorArrayV2_1Z

rnn_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn_1/time
rnn_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
rnn_1/while/maximum_iterationsv
rnn_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_1/while/loop_counter
rnn_1/whileWhile!rnn_1/while/loop_counter:output:0'rnn_1/while/maximum_iterations:output:0rnn_1/time:output:0rnn_1/TensorArrayV2_1:handle:0rnn_1/zeros:output:0rnn_1/zeros_1:output:0rnn_1/strided_slice_1:output:0=rnn_1/TensorArrayUnstack/TensorListFromTensor:output_handle:09rnn_1_peephole_lstm_cell_1_matmul_readvariableop_resource;rnn_1_peephole_lstm_cell_1_matmul_1_readvariableop_resource:rnn_1_peephole_lstm_cell_1_biasadd_readvariableop_resource2rnn_1_peephole_lstm_cell_1_readvariableop_resource4rnn_1_peephole_lstm_cell_1_readvariableop_1_resource4rnn_1_peephole_lstm_cell_1_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*$
bodyR
rnn_1_while_body_1668601*$
condR
rnn_1_while_cond_1668600*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
rnn_1/whileÁ
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    28
6rnn_1/TensorArrayV2Stack/TensorListStack/element_shape
(rnn_1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_1/while:output:3?rnn_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02*
(rnn_1/TensorArrayV2Stack/TensorListStack
rnn_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn_1/strided_slice_3/stack
rnn_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn_1/strided_slice_3/stack_1
rnn_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn_1/strided_slice_3/stack_2¾
rnn_1/strided_slice_3StridedSlice1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0$rnn_1/strided_slice_3/stack:output:0&rnn_1/strided_slice_3/stack_1:output:0&rnn_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rnn_1/strided_slice_3
rnn_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_1/transpose_1/perm½
rnn_1/transpose_1	Transpose1rnn_1/TensorArrayV2Stack/TensorListStack:tensor:0rnn_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn_1/transpose_1
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulrnn_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddÐ
IdentityIdentitydense/BiasAdd:output:0*^conv1d/conv1d/ExpandDims_1/ReadVariableOp1^conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp.^rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-^rnn/peephole_lstm_cell/MatMul/ReadVariableOp/^rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp&^rnn/peephole_lstm_cell/ReadVariableOp(^rnn/peephole_lstm_cell/ReadVariableOp_1(^rnn/peephole_lstm_cell/ReadVariableOp_2
^rnn/while2^rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp3^rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^rnn_1/peephole_lstm_cell_1/ReadVariableOp,^rnn_1/peephole_lstm_cell_1/ReadVariableOp_1,^rnn_1/peephole_lstm_cell_1/ReadVariableOp_2^rnn_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2d
0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv1d/squeeze_batch_dims/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2^
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp2\
,rnn/peephole_lstm_cell/MatMul/ReadVariableOp,rnn/peephole_lstm_cell/MatMul/ReadVariableOp2`
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp2N
%rnn/peephole_lstm_cell/ReadVariableOp%rnn/peephole_lstm_cell/ReadVariableOp2R
'rnn/peephole_lstm_cell/ReadVariableOp_1'rnn/peephole_lstm_cell/ReadVariableOp_12R
'rnn/peephole_lstm_cell/ReadVariableOp_2'rnn/peephole_lstm_cell/ReadVariableOp_22
	rnn/while	rnn/while2f
1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1rnn_1/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp0rnn_1/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2rnn_1/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)rnn_1/peephole_lstm_cell_1/ReadVariableOp)rnn_1/peephole_lstm_cell_1/ReadVariableOp2Z
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_1+rnn_1/peephole_lstm_cell_1/ReadVariableOp_12Z
+rnn_1/peephole_lstm_cell_1/ReadVariableOp_2+rnn_1/peephole_lstm_cell_1/ReadVariableOp_22
rnn_1/whilernn_1/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
g
»
while_body_1669893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0N
;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 P
=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 K
<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	B
4while_peephole_lstm_cell_1_readvariableop_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_1_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorL
9while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2while_peephole_lstm_cell_1_readvariableop_resource: B
4while_peephole_lstm_cell_1_readvariableop_1_resource: B
4while_peephole_lstm_cell_1_readvariableop_2_resource: ¢1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)while/peephole_lstm_cell_1/ReadVariableOp¢+while/peephole_lstm_cell_1/ReadVariableOp_1¢+while/peephole_lstm_cell_1/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpï
!while/peephole_lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:08while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell_1/MatMulç
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype024
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpØ
#while/peephole_lstm_cell_1/MatMul_1MatMulwhile_placeholder_2:while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#while/peephole_lstm_cell_1/MatMul_1Ø
while/peephole_lstm_cell_1/addAddV2+while/peephole_lstm_cell_1/MatMul:product:0-while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/peephole_lstm_cell_1/addà
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"while/peephole_lstm_cell_1/BiasAddBiasAdd"while/peephole_lstm_cell_1/add:z:09while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"while/peephole_lstm_cell_1/BiasAdd
*while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*while/peephole_lstm_cell_1/split/split_dim«
 while/peephole_lstm_cell_1/splitSplit3while/peephole_lstm_cell_1/split/split_dim:output:0+while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 while/peephole_lstm_cell_1/splitÇ
)while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp4while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell_1/ReadVariableOpÁ
while/peephole_lstm_cell_1/mulMul1while/peephole_lstm_cell_1/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell_1/mulÎ
 while/peephole_lstm_cell_1/add_1AddV2)while/peephole_lstm_cell_1/split:output:0"while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_1«
"while/peephole_lstm_cell_1/SigmoidSigmoid$while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell_1/SigmoidÍ
+while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_1Ç
 while/peephole_lstm_cell_1/mul_1Mul3while/peephole_lstm_cell_1/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_1Ð
 while/peephole_lstm_cell_1/add_2AddV2)while/peephole_lstm_cell_1/split:output:1$while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_2¯
$while/peephole_lstm_cell_1/Sigmoid_1Sigmoid$while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_1¼
 while/peephole_lstm_cell_1/mul_2Mul(while/peephole_lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_2§
while/peephole_lstm_cell_1/TanhTanh)while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell_1/TanhÊ
 while/peephole_lstm_cell_1/mul_3Mul&while/peephole_lstm_cell_1/Sigmoid:y:0#while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_3Ë
 while/peephole_lstm_cell_1/add_3AddV2$while/peephole_lstm_cell_1/mul_2:z:0$while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_3Í
+while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_2Ø
 while/peephole_lstm_cell_1/mul_4Mul3while/peephole_lstm_cell_1/ReadVariableOp_2:value:0$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_4Ð
 while/peephole_lstm_cell_1/add_4AddV2)while/peephole_lstm_cell_1/split:output:3$while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_4¯
$while/peephole_lstm_cell_1/Sigmoid_2Sigmoid$while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_2¦
!while/peephole_lstm_cell_1/Tanh_1Tanh$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!while/peephole_lstm_cell_1/Tanh_1Î
 while/peephole_lstm_cell_1/mul_5Mul(while/peephole_lstm_cell_1/Sigmoid_2:y:0%while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_5è
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations2^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2±
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3¬
while/Identity_4Identity$while/peephole_lstm_cell_1/mul_5:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4¬
while/Identity_5Identity$while/peephole_lstm_cell_1/add_3:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"|
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"x
9while_peephole_lstm_cell_1_matmul_readvariableop_resource;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_1_resource6while_peephole_lstm_cell_1_readvariableop_1_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_2_resource6while_peephole_lstm_cell_1_readvariableop_2_resource_0"j
2while_peephole_lstm_cell_1_readvariableop_resource4while_peephole_lstm_cell_1_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2f
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0while/peephole_lstm_cell_1/MatMul/ReadVariableOp0while/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)while/peephole_lstm_cell_1/ReadVariableOp)while/peephole_lstm_cell_1/ReadVariableOp2Z
+while/peephole_lstm_cell_1/ReadVariableOp_1+while/peephole_lstm_cell_1/ReadVariableOp_12Z
+while/peephole_lstm_cell_1/ReadVariableOp_2+while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


í
while_cond_1667274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1667274___redundant_placeholder05
1while_while_cond_1667274___redundant_placeholder15
1while_while_cond_1667274___redundant_placeholder25
1while_while_cond_1667274___redundant_placeholder35
1while_while_cond_1667274___redundant_placeholder45
1while_while_cond_1667274___redundant_placeholder55
1while_while_cond_1667274___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ôG

@__inference_rnn_layer_call_and_return_conditional_losses_1665335

inputs-
peephole_lstm_cell_1665236:	-
peephole_lstm_cell_1665238:	 )
peephole_lstm_cell_1665240:	(
peephole_lstm_cell_1665242: (
peephole_lstm_cell_1665244: (
peephole_lstm_cell_1665246: 
identity¢*peephole_lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
*peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0peephole_lstm_cell_1665236peephole_lstm_cell_1665238peephole_lstm_cell_1665240peephole_lstm_cell_1665242peephole_lstm_cell_1665244peephole_lstm_cell_1665246*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_16652352,
*peephole_lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0peephole_lstm_cell_1665236peephole_lstm_cell_1665238peephole_lstm_cell_1665240peephole_lstm_cell_1665242peephole_lstm_cell_1665244peephole_lstm_cell_1665246*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_1665255*
condR
while_cond_1665254*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1¥
IdentityIdentitytranspose_1:y:0+^peephole_lstm_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2X
*peephole_lstm_cell/StatefulPartitionedCall*peephole_lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
g
»
while_body_1669713
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0N
;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0:	 P
=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0:	 K
<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0:	B
4while_peephole_lstm_cell_1_readvariableop_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_1_resource_0: D
6while_peephole_lstm_cell_1_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorL
9while_peephole_lstm_cell_1_matmul_readvariableop_resource:	 N
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource:	 I
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource:	@
2while_peephole_lstm_cell_1_readvariableop_resource: B
4while_peephole_lstm_cell_1_readvariableop_1_resource: B
4while_peephole_lstm_cell_1_readvariableop_2_resource: ¢1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp¢0while/peephole_lstm_cell_1/MatMul/ReadVariableOp¢2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp¢)while/peephole_lstm_cell_1/ReadVariableOp¢+while/peephole_lstm_cell_1/ReadVariableOp_1¢+while/peephole_lstm_cell_1/ReadVariableOp_2Ã
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemá
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype022
0while/peephole_lstm_cell_1/MatMul/ReadVariableOpï
!while/peephole_lstm_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:08while/peephole_lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!while/peephole_lstm_cell_1/MatMulç
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype024
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOpØ
#while/peephole_lstm_cell_1/MatMul_1MatMulwhile_placeholder_2:while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#while/peephole_lstm_cell_1/MatMul_1Ø
while/peephole_lstm_cell_1/addAddV2+while/peephole_lstm_cell_1/MatMul:product:0-while/peephole_lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
while/peephole_lstm_cell_1/addà
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOpå
"while/peephole_lstm_cell_1/BiasAddBiasAdd"while/peephole_lstm_cell_1/add:z:09while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"while/peephole_lstm_cell_1/BiasAdd
*while/peephole_lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*while/peephole_lstm_cell_1/split/split_dim«
 while/peephole_lstm_cell_1/splitSplit3while/peephole_lstm_cell_1/split/split_dim:output:0+while/peephole_lstm_cell_1/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2"
 while/peephole_lstm_cell_1/splitÇ
)while/peephole_lstm_cell_1/ReadVariableOpReadVariableOp4while_peephole_lstm_cell_1_readvariableop_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell_1/ReadVariableOpÁ
while/peephole_lstm_cell_1/mulMul1while/peephole_lstm_cell_1/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
while/peephole_lstm_cell_1/mulÎ
 while/peephole_lstm_cell_1/add_1AddV2)while/peephole_lstm_cell_1/split:output:0"while/peephole_lstm_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_1«
"while/peephole_lstm_cell_1/SigmoidSigmoid$while/peephole_lstm_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"while/peephole_lstm_cell_1/SigmoidÍ
+while/peephole_lstm_cell_1/ReadVariableOp_1ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_1_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_1Ç
 while/peephole_lstm_cell_1/mul_1Mul3while/peephole_lstm_cell_1/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_1Ð
 while/peephole_lstm_cell_1/add_2AddV2)while/peephole_lstm_cell_1/split:output:1$while/peephole_lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_2¯
$while/peephole_lstm_cell_1/Sigmoid_1Sigmoid$while/peephole_lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_1¼
 while/peephole_lstm_cell_1/mul_2Mul(while/peephole_lstm_cell_1/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_2§
while/peephole_lstm_cell_1/TanhTanh)while/peephole_lstm_cell_1/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
while/peephole_lstm_cell_1/TanhÊ
 while/peephole_lstm_cell_1/mul_3Mul&while/peephole_lstm_cell_1/Sigmoid:y:0#while/peephole_lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_3Ë
 while/peephole_lstm_cell_1/add_3AddV2$while/peephole_lstm_cell_1/mul_2:z:0$while/peephole_lstm_cell_1/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_3Í
+while/peephole_lstm_cell_1/ReadVariableOp_2ReadVariableOp6while_peephole_lstm_cell_1_readvariableop_2_resource_0*
_output_shapes
: *
dtype02-
+while/peephole_lstm_cell_1/ReadVariableOp_2Ø
 while/peephole_lstm_cell_1/mul_4Mul3while/peephole_lstm_cell_1/ReadVariableOp_2:value:0$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_4Ð
 while/peephole_lstm_cell_1/add_4AddV2)while/peephole_lstm_cell_1/split:output:3$while/peephole_lstm_cell_1/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/add_4¯
$while/peephole_lstm_cell_1/Sigmoid_2Sigmoid$while/peephole_lstm_cell_1/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$while/peephole_lstm_cell_1/Sigmoid_2¦
!while/peephole_lstm_cell_1/Tanh_1Tanh$while/peephole_lstm_cell_1/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!while/peephole_lstm_cell_1/Tanh_1Î
 while/peephole_lstm_cell_1/mul_5Mul(while/peephole_lstm_cell_1/Sigmoid_2:y:0%while/peephole_lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 while/peephole_lstm_cell_1/mul_5è
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder$while/peephole_lstm_cell_1/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations2^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2±
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3¬
while/Identity_4Identity$while/peephole_lstm_cell_1/mul_5:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4¬
while/Identity_5Identity$while/peephole_lstm_cell_1/add_3:z:02^while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1^while/peephole_lstm_cell_1/MatMul/ReadVariableOp3^while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp*^while/peephole_lstm_cell_1/ReadVariableOp,^while/peephole_lstm_cell_1/ReadVariableOp_1,^while/peephole_lstm_cell_1/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"z
:while_peephole_lstm_cell_1_biasadd_readvariableop_resource<while_peephole_lstm_cell_1_biasadd_readvariableop_resource_0"|
;while_peephole_lstm_cell_1_matmul_1_readvariableop_resource=while_peephole_lstm_cell_1_matmul_1_readvariableop_resource_0"x
9while_peephole_lstm_cell_1_matmul_readvariableop_resource;while_peephole_lstm_cell_1_matmul_readvariableop_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_1_resource6while_peephole_lstm_cell_1_readvariableop_1_resource_0"n
4while_peephole_lstm_cell_1_readvariableop_2_resource6while_peephole_lstm_cell_1_readvariableop_2_resource_0"j
2while_peephole_lstm_cell_1_readvariableop_resource4while_peephole_lstm_cell_1_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2f
1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp1while/peephole_lstm_cell_1/BiasAdd/ReadVariableOp2d
0while/peephole_lstm_cell_1/MatMul/ReadVariableOp0while/peephole_lstm_cell_1/MatMul/ReadVariableOp2h
2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2while/peephole_lstm_cell_1/MatMul_1/ReadVariableOp2V
)while/peephole_lstm_cell_1/ReadVariableOp)while/peephole_lstm_cell_1/ReadVariableOp2Z
+while/peephole_lstm_cell_1/ReadVariableOp_1+while/peephole_lstm_cell_1/ReadVariableOp_12Z
+while/peephole_lstm_cell_1/ReadVariableOp_2+while/peephole_lstm_cell_1/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


í
while_cond_1669284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1669284___redundant_placeholder05
1while_while_cond_1669284___redundant_placeholder15
1while_while_cond_1669284___redundant_placeholder25
1while_while_cond_1669284___redundant_placeholder35
1while_while_cond_1669284___redundant_placeholder45
1while_while_cond_1669284___redundant_placeholder55
1while_while_cond_1669284___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:

`
D__inference_reshape_layer_call_and_return_conditional_losses_1666727

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
ÿ
'__inference_rnn_1_layer_call_fn_1670422

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16673762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Îl
»
rnn_while_body_1668021$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0P
=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0:	R
?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0:	 M
>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0:	D
6rnn_while_peephole_lstm_cell_readvariableop_resource_0: F
8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0: F
8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0: 
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorN
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resource:	P
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource:	 K
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource:	B
4rnn_while_peephole_lstm_cell_readvariableop_resource: D
6rnn_while_peephole_lstm_cell_readvariableop_1_resource: D
6rnn_while_peephole_lstm_cell_readvariableop_2_resource: ¢3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp¢2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp¢4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp¢+rnn/while/peephole_lstm_cell/ReadVariableOp¢-rnn/while/peephole_lstm_cell/ReadVariableOp_1¢-rnn/while/peephole_lstm_cell/ReadVariableOp_2Ë
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeë
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItemç
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype024
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpù
#rnn/while/peephole_lstm_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0:rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn/while/peephole_lstm_cell/MatMulí
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype026
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpâ
%rnn/while/peephole_lstm_cell/MatMul_1MatMulrnn_while_placeholder_2<rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/while/peephole_lstm_cell/MatMul_1à
 rnn/while/peephole_lstm_cell/addAddV2-rnn/while/peephole_lstm_cell/MatMul:product:0/rnn/while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 rnn/while/peephole_lstm_cell/addæ
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype025
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpí
$rnn/while/peephole_lstm_cell/BiasAddBiasAdd$rnn/while/peephole_lstm_cell/add:z:0;rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rnn/while/peephole_lstm_cell/BiasAdd
,rnn/while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,rnn/while/peephole_lstm_cell/split/split_dim³
"rnn/while/peephole_lstm_cell/splitSplit5rnn/while/peephole_lstm_cell/split/split_dim:output:0-rnn/while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2$
"rnn/while/peephole_lstm_cell/splitÍ
+rnn/while/peephole_lstm_cell/ReadVariableOpReadVariableOp6rnn_while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02-
+rnn/while/peephole_lstm_cell/ReadVariableOpË
 rnn/while/peephole_lstm_cell/mulMul3rnn/while/peephole_lstm_cell/ReadVariableOp:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rnn/while/peephole_lstm_cell/mulÖ
"rnn/while/peephole_lstm_cell/add_1AddV2+rnn/while/peephole_lstm_cell/split:output:0$rnn/while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_1±
$rnn/while/peephole_lstm_cell/SigmoidSigmoid&rnn/while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$rnn/while/peephole_lstm_cell/SigmoidÓ
-rnn/while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_1Ñ
"rnn/while/peephole_lstm_cell/mul_1Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_1:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_1Ø
"rnn/while/peephole_lstm_cell/add_2AddV2+rnn/while/peephole_lstm_cell/split:output:1&rnn/while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_2µ
&rnn/while/peephole_lstm_cell/Sigmoid_1Sigmoid&rnn/while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn/while/peephole_lstm_cell/Sigmoid_1Æ
"rnn/while/peephole_lstm_cell/mul_2Mul*rnn/while/peephole_lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_2­
!rnn/while/peephole_lstm_cell/TanhTanh+rnn/while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rnn/while/peephole_lstm_cell/TanhÒ
"rnn/while/peephole_lstm_cell/mul_3Mul(rnn/while/peephole_lstm_cell/Sigmoid:y:0%rnn/while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_3Ó
"rnn/while/peephole_lstm_cell/add_3AddV2&rnn/while/peephole_lstm_cell/mul_2:z:0&rnn/while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_3Ó
-rnn/while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_2à
"rnn/while/peephole_lstm_cell/mul_4Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_2:value:0&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_4Ø
"rnn/while/peephole_lstm_cell/add_4AddV2+rnn/while/peephole_lstm_cell/split:output:3&rnn/while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/add_4µ
&rnn/while/peephole_lstm_cell/Sigmoid_2Sigmoid&rnn/while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&rnn/while/peephole_lstm_cell/Sigmoid_2¬
#rnn/while/peephole_lstm_cell/Tanh_1Tanh&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#rnn/while/peephole_lstm_cell/Tanh_1Ö
"rnn/while/peephole_lstm_cell/mul_5Mul*rnn/while/peephole_lstm_cell/Sigmoid_2:y:0'rnn/while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rnn/while/peephole_lstm_cell/mul_5ú
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder&rnn/while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1
rnn/while/IdentityIdentityrnn/while/add_1:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity±
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations4^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_1
rnn/while/Identity_2Identityrnn/while/add:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_2É
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_3Â
rnn/while/Identity_4Identity&rnn/while/peephole_lstm_cell/mul_5:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/while/Identity_4Â
rnn/while/Identity_5Identity&rnn/while/peephole_lstm_cell/add_3:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rnn/while/Identity_5"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"~
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0"
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"|
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resource=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_1_resource8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_2_resource8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0"n
4rnn_while_peephole_lstm_cell_readvariableop_resource6rnn_while_peephole_lstm_cell_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"¸
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2j
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2h
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2l
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp2Z
+rnn/while/peephole_lstm_cell/ReadVariableOp+rnn/while/peephole_lstm_cell/ReadVariableOp2^
-rnn/while/peephole_lstm_cell/ReadVariableOp_1-rnn/while/peephole_lstm_cell/ReadVariableOp_12^
-rnn/while/peephole_lstm_cell/ReadVariableOp_2-rnn/while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


í
while_cond_1669104
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1669104___redundant_placeholder05
1while_while_cond_1669104___redundant_placeholder15
1while_while_cond_1669104___redundant_placeholder25
1while_while_cond_1669104___redundant_placeholder35
1while_while_cond_1669104___redundant_placeholder45
1while_while_cond_1669104___redundant_placeholder55
1while_while_cond_1669104___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
â
Ó
G__inference_sequential_layer_call_and_return_conditional_losses_1667814
conv1d_input$
conv1d_1667776:
conv1d_1667778:
rnn_1667782:	
rnn_1667784:	 
rnn_1667786:	
rnn_1667788: 
rnn_1667790: 
rnn_1667792:  
rnn_1_1667795:	  
rnn_1_1667797:	 
rnn_1_1667799:	
rnn_1_1667801: 
rnn_1_1667803: 
rnn_1_1667805: 
dense_1667808: 
dense_1667810:
identity¢conv1d/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢rnn/StatefulPartitionedCall¢rnn_1/StatefulPartitionedCall
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_1667776conv1d_1667778*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_16667082 
conv1d/StatefulPartitionedCallú
reshape/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_16667272
reshape/PartitionedCallÛ
rnn/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0rnn_1667782rnn_1667784rnn_1667786rnn_1667788rnn_1667790rnn_1667792*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_rnn_layer_call_and_return_conditional_losses_16669082
rnn/StatefulPartitionedCallí
rnn_1/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0rnn_1_1667795rnn_1_1667797rnn_1_1667799rnn_1_1667801rnn_1_1667803rnn_1_1667805*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_rnn_1_layer_call_and_return_conditional_losses_16671012
rnn_1/StatefulPartitionedCall«
dense/StatefulPartitionedCallStatefulPartitionedCall&rnn_1/StatefulPartitionedCall:output:0dense_1667808dense_1667810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16671252
dense/StatefulPartitionedCallù
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall^rnn_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2>
rnn_1/StatefulPartitionedCallrnn_1/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv1d_input"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*º
serving_default¦
M
conv1d_input=
serving_default_conv1d_input:0ÿÿÿÿÿÿÿÿÿ9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:º
³D
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
_default_save_signature
+&call_and_return_all_conditional_losses
__call__"¼A
_tf_keras_sequentialA{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "rnn_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "conv1d_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv1d_input"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "rnn_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
Ä

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"

_tf_keras_layer
{"name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ù
_tf_keras_layerß{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
§
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ü
_tf_keras_rnn_layerÞ{"name": "rnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
³
cell

state_spec
	variables
regularization_losses
 trainable_variables
!	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layerê{"name": "rnn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "rnn_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ï

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"¨
_tf_keras_layer{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}

(iter
	)decay
*learning_rate
+momentum
,rho	rmsr	rmss	"rmst	#rmsu	-rmsv	.rmsw	/rmsx	0rmsy	1rmsz	2rms{	3rms|	4rms}	5rms~	6rms
7rms
8rms"
	optimizer

0
1
-2
.3
/4
05
16
27
38
49
510
611
712
813
"14
#15"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
-2
.3
/4
05
16
27
38
49
510
611
712
813
"14
#15"
trackable_list_wrapper
Î
9non_trainable_variables
	variables

:layers
regularization_losses
;layer_metrics
<metrics
=layer_regularization_losses
	trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
#:!2conv1d/kernel
:2conv1d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
>non_trainable_variables
?layer_metrics

@layers
	variables
regularization_losses
Ametrics
Blayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Cnon_trainable_variables
Dlayer_metrics

Elayers
	variables
regularization_losses
Fmetrics
Glayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


H
state_size

-kernel
.recurrent_kernel
/bias
0input_gate_peephole_weights
 1forget_gate_peephole_weights
 2output_gate_peephole_weights
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+&call_and_return_all_conditional_losses
__call__"æ
_tf_keras_layerÌ{"name": "peephole_lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
 "
trackable_list_wrapper
J
-0
.1
/2
03
14
25"
trackable_list_wrapper
 "
trackable_list_wrapper
J
-0
.1
/2
03
14
25"
trackable_list_wrapper
¼
Mnon_trainable_variables
	variables

Nlayers
Olayer_metrics
regularization_losses

Pstates
Qmetrics
Rlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 

S
state_size

3kernel
4recurrent_kernel
5bias
6input_gate_peephole_weights
 7forget_gate_peephole_weights
 8output_gate_peephole_weights
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+&call_and_return_all_conditional_losses
__call__"î
_tf_keras_layerÔ{"name": "peephole_lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
 "
trackable_list_wrapper
J
30
41
52
63
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
J
30
41
52
63
74
85"
trackable_list_wrapper
¼
Xnon_trainable_variables
	variables

Ylayers
Zlayer_metrics
regularization_losses

[states
\metrics
]layer_regularization_losses
 trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
: 2dense/kernel
:2
dense/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
°
^non_trainable_variables
_layer_metrics

`layers
$	variables
%regularization_losses
ametrics
blayer_regularization_losses
&trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
0:.	2rnn/peephole_lstm_cell/kernel
::8	 2'rnn/peephole_lstm_cell/recurrent_kernel
*:(2rnn/peephole_lstm_cell/bias
@:> 22rnn/peephole_lstm_cell/input_gate_peephole_weights
A:? 23rnn/peephole_lstm_cell/forget_gate_peephole_weights
A:? 23rnn/peephole_lstm_cell/output_gate_peephole_weights
4:2	 2!rnn_1/peephole_lstm_cell_1/kernel
>:<	 2+rnn_1/peephole_lstm_cell_1/recurrent_kernel
.:,2rnn_1/peephole_lstm_cell_1/bias
D:B 26rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights
E:C 27rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights
E:C 27rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
-0
.1
/2
03
14
25"
trackable_list_wrapper
 "
trackable_list_wrapper
J
-0
.1
/2
03
14
25"
trackable_list_wrapper
°
dnon_trainable_variables
elayer_metrics

flayers
I	variables
Jregularization_losses
gmetrics
hlayer_regularization_losses
Ktrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
30
41
52
63
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
J
30
41
52
63
74
85"
trackable_list_wrapper
°
inon_trainable_variables
jlayer_metrics

klayers
T	variables
Uregularization_losses
lmetrics
mlayer_regularization_losses
Vtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ô
	ntotal
	ocount
p	variables
q	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
-:+2RMSprop/conv1d/kernel/rms
#:!2RMSprop/conv1d/bias/rms
(:& 2RMSprop/dense/kernel/rms
": 2RMSprop/dense/bias/rms
::8	2)RMSprop/rnn/peephole_lstm_cell/kernel/rms
D:B	 23RMSprop/rnn/peephole_lstm_cell/recurrent_kernel/rms
4:22'RMSprop/rnn/peephole_lstm_cell/bias/rms
J:H 2>RMSprop/rnn/peephole_lstm_cell/input_gate_peephole_weights/rms
K:I 2?RMSprop/rnn/peephole_lstm_cell/forget_gate_peephole_weights/rms
K:I 2?RMSprop/rnn/peephole_lstm_cell/output_gate_peephole_weights/rms
>:<	 2-RMSprop/rnn_1/peephole_lstm_cell_1/kernel/rms
H:F	 27RMSprop/rnn_1/peephole_lstm_cell_1/recurrent_kernel/rms
8:62+RMSprop/rnn_1/peephole_lstm_cell_1/bias/rms
N:L 2BRMSprop/rnn_1/peephole_lstm_cell_1/input_gate_peephole_weights/rms
O:M 2CRMSprop/rnn_1/peephole_lstm_cell_1/forget_gate_peephole_weights/rms
O:M 2CRMSprop/rnn_1/peephole_lstm_cell_1/output_gate_peephole_weights/rms
í2ê
"__inference__wrapped_model_1665148Ã
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+
conv1d_inputÿÿÿÿÿÿÿÿÿ
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_1668304
G__inference_sequential_layer_call_and_return_conditional_losses_1668708
G__inference_sequential_layer_call_and_return_conditional_losses_1667814
G__inference_sequential_layer_call_and_return_conditional_losses_1667855À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
,__inference_sequential_layer_call_fn_1667167
,__inference_sequential_layer_call_fn_1668745
,__inference_sequential_layer_call_fn_1668782
,__inference_sequential_layer_call_fn_1667773À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_conv1d_layer_call_and_return_conditional_losses_1668819¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv1d_layer_call_fn_1668828¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_layer_call_and_return_conditional_losses_1668841¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_reshape_layer_call_fn_1668846¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
@__inference_rnn_layer_call_and_return_conditional_losses_1669026
@__inference_rnn_layer_call_and_return_conditional_losses_1669206
@__inference_rnn_layer_call_and_return_conditional_losses_1669386
@__inference_rnn_layer_call_and_return_conditional_losses_1669566æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
%__inference_rnn_layer_call_fn_1669583
%__inference_rnn_layer_call_fn_1669600
%__inference_rnn_layer_call_fn_1669617
%__inference_rnn_layer_call_fn_1669634æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ü2ù
B__inference_rnn_1_layer_call_and_return_conditional_losses_1669814
B__inference_rnn_1_layer_call_and_return_conditional_losses_1669994
B__inference_rnn_1_layer_call_and_return_conditional_losses_1670174
B__inference_rnn_1_layer_call_and_return_conditional_losses_1670354æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
'__inference_rnn_1_layer_call_fn_1670371
'__inference_rnn_1_layer_call_fn_1670388
'__inference_rnn_1_layer_call_fn_1670405
'__inference_rnn_1_layer_call_fn_1670422æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_dense_layer_call_and_return_conditional_losses_1670432¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_layer_call_fn_1670441¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÑBÎ
%__inference_signature_wrapper_1667900conv1d_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
æ2ã
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1670485
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1670529¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
°2­
4__inference_peephole_lstm_cell_layer_call_fn_1670552
4__inference_peephole_lstm_cell_layer_call_fn_1670575¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1670619
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1670663¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
´2±
6__inference_peephole_lstm_cell_1_layer_call_fn_1670686
6__inference_peephole_lstm_cell_1_layer_call_fn_1670709¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 §
"__inference__wrapped_model_1665148-./012345678"#=¢:
3¢0
.+
conv1d_inputÿÿÿÿÿÿÿÿÿ
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ³
C__inference_conv1d_layer_call_and_return_conditional_losses_1668819l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_conv1d_layer_call_fn_1668828_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¢
B__inference_dense_layer_call_and_return_conditional_losses_1670432\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_layer_call_fn_1670441O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÖ
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1670619345678¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ö
Q__inference_peephole_lstm_cell_1_layer_call_and_return_conditional_losses_1670663345678¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 «
6__inference_peephole_lstm_cell_1_layer_call_fn_1670686ð345678¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ «
6__inference_peephole_lstm_cell_1_layer_call_fn_1670709ð345678¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ 
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ Ô
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1670485-./012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 Ô
O__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_1670529-./012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ 
EB

0/1/0ÿÿÿÿÿÿÿÿÿ 

0/1/1ÿÿÿÿÿÿÿÿÿ 
 ©
4__inference_peephole_lstm_cell_layer_call_fn_1670552ð-./012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ ©
4__inference_peephole_lstm_cell_layer_call_fn_1670575ð-./012¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ 
"
states/1ÿÿÿÿÿÿÿÿÿ 
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ 
A>

1/0ÿÿÿÿÿÿÿÿÿ 

1/1ÿÿÿÿÿÿÿÿÿ ¬
D__inference_reshape_layer_call_and_return_conditional_losses_1668841d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_reshape_layer_call_fn_1668846W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿË
B__inference_rnn_1_layer_call_and_return_conditional_losses_1669814345678S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 Ë
B__inference_rnn_1_layer_call_and_return_conditional_losses_1669994345678S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 º
B__inference_rnn_1_layer_call_and_return_conditional_losses_1670174t345678C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p 

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 º
B__inference_rnn_1_layer_call_and_return_conditional_losses_1670354t345678C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p

 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¢
'__inference_rnn_1_layer_call_fn_1670371w345678S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ ¢
'__inference_rnn_1_layer_call_fn_1670388w345678S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ 
'__inference_rnn_1_layer_call_fn_1670405g345678C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ 
'__inference_rnn_1_layer_call_fn_1670422g345678C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ Ö
@__inference_rnn_layer_call_and_return_conditional_losses_1669026-./012S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ö
@__inference_rnn_layer_call_and_return_conditional_losses_1669206-./012S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¼
@__inference_rnn_layer_call_and_return_conditional_losses_1669386x-./012C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ¼
@__inference_rnn_layer_call_and_return_conditional_losses_1669566x-./012C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ®
%__inference_rnn_layer_call_fn_1669583-./012S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ®
%__inference_rnn_layer_call_fn_1669600-./012S¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
%__inference_rnn_layer_call_fn_1669617k-./012C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "ÿÿÿÿÿÿÿÿÿ 
%__inference_rnn_layer_call_fn_1669634k-./012C¢@
9¢6
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "ÿÿÿÿÿÿÿÿÿ Ì
G__inference_sequential_layer_call_and_return_conditional_losses_1667814-./012345678"#E¢B
;¢8
.+
conv1d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
G__inference_sequential_layer_call_and_return_conditional_losses_1667855-./012345678"#E¢B
;¢8
.+
conv1d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_1668304z-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_1668708z-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
,__inference_sequential_layer_call_fn_1667167s-./012345678"#E¢B
;¢8
.+
conv1d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
,__inference_sequential_layer_call_fn_1667773s-./012345678"#E¢B
;¢8
.+
conv1d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1668745m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1668782m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿº
%__inference_signature_wrapper_1667900-./012345678"#M¢J
¢ 
Cª@
>
conv1d_input.+
conv1d_inputÿÿÿÿÿÿÿÿÿ"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ