ý2
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
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718®/

zdelabjare/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namezdelabjare/kernel
{
%zdelabjare/kernel/Read/ReadVariableOpReadVariableOpzdelabjare/kernel*"
_output_shapes
:*
dtype0
v
zdelabjare/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namezdelabjare/bias
o
#zdelabjare/bias/Read/ReadVariableOpReadVariableOpzdelabjare/bias*
_output_shapes
:*
dtype0
~
vfmtgawzzo/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namevfmtgawzzo/kernel
w
%vfmtgawzzo/kernel/Read/ReadVariableOpReadVariableOpvfmtgawzzo/kernel*
_output_shapes

: *
dtype0
v
vfmtgawzzo/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namevfmtgawzzo/bias
o
#vfmtgawzzo/bias/Read/ReadVariableOpReadVariableOpvfmtgawzzo/bias*
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

xedyzswikc/jgtgtymybc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namexedyzswikc/jgtgtymybc/kernel

0xedyzswikc/jgtgtymybc/kernel/Read/ReadVariableOpReadVariableOpxedyzswikc/jgtgtymybc/kernel*
_output_shapes
:	*
dtype0
©
&xedyzswikc/jgtgtymybc/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&xedyzswikc/jgtgtymybc/recurrent_kernel
¢
:xedyzswikc/jgtgtymybc/recurrent_kernel/Read/ReadVariableOpReadVariableOp&xedyzswikc/jgtgtymybc/recurrent_kernel*
_output_shapes
:	 *
dtype0

xedyzswikc/jgtgtymybc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namexedyzswikc/jgtgtymybc/bias

.xedyzswikc/jgtgtymybc/bias/Read/ReadVariableOpReadVariableOpxedyzswikc/jgtgtymybc/bias*
_output_shapes	
:*
dtype0
º
1xedyzswikc/jgtgtymybc/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31xedyzswikc/jgtgtymybc/input_gate_peephole_weights
³
Exedyzswikc/jgtgtymybc/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1xedyzswikc/jgtgtymybc/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2xedyzswikc/jgtgtymybc/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42xedyzswikc/jgtgtymybc/forget_gate_peephole_weights
µ
Fxedyzswikc/jgtgtymybc/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2xedyzswikc/jgtgtymybc/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2xedyzswikc/jgtgtymybc/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42xedyzswikc/jgtgtymybc/output_gate_peephole_weights
µ
Fxedyzswikc/jgtgtymybc/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2xedyzswikc/jgtgtymybc/output_gate_peephole_weights*
_output_shapes
: *
dtype0

ksthobzafc/qjjkgjcvpf/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_nameksthobzafc/qjjkgjcvpf/kernel

0ksthobzafc/qjjkgjcvpf/kernel/Read/ReadVariableOpReadVariableOpksthobzafc/qjjkgjcvpf/kernel*
_output_shapes
:	 *
dtype0
©
&ksthobzafc/qjjkgjcvpf/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&ksthobzafc/qjjkgjcvpf/recurrent_kernel
¢
:ksthobzafc/qjjkgjcvpf/recurrent_kernel/Read/ReadVariableOpReadVariableOp&ksthobzafc/qjjkgjcvpf/recurrent_kernel*
_output_shapes
:	 *
dtype0

ksthobzafc/qjjkgjcvpf/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameksthobzafc/qjjkgjcvpf/bias

.ksthobzafc/qjjkgjcvpf/bias/Read/ReadVariableOpReadVariableOpksthobzafc/qjjkgjcvpf/bias*
_output_shapes	
:*
dtype0
º
1ksthobzafc/qjjkgjcvpf/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights
³
Eksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights
µ
Fksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2ksthobzafc/qjjkgjcvpf/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights
µ
Fksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights*
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

RMSprop/zdelabjare/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/zdelabjare/kernel/rms

1RMSprop/zdelabjare/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/zdelabjare/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/zdelabjare/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/zdelabjare/bias/rms

/RMSprop/zdelabjare/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/zdelabjare/bias/rms*
_output_shapes
:*
dtype0

RMSprop/vfmtgawzzo/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/vfmtgawzzo/kernel/rms

1RMSprop/vfmtgawzzo/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/vfmtgawzzo/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/vfmtgawzzo/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/vfmtgawzzo/bias/rms

/RMSprop/vfmtgawzzo/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/vfmtgawzzo/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/xedyzswikc/jgtgtymybc/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/xedyzswikc/jgtgtymybc/kernel/rms
¦
<RMSprop/xedyzswikc/jgtgtymybc/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/xedyzswikc/jgtgtymybc/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rms
º
FRMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/xedyzswikc/jgtgtymybc/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/xedyzswikc/jgtgtymybc/bias/rms

:RMSprop/xedyzswikc/jgtgtymybc/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/xedyzswikc/jgtgtymybc/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rms
Ë
QRMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rms
Í
RRMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rms
Í
RRMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rms
¦
<RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rms
º
FRMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/ksthobzafc/qjjkgjcvpf/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/ksthobzafc/qjjkgjcvpf/bias/rms

:RMSprop/ksthobzafc/qjjkgjcvpf/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/ksthobzafc/qjjkgjcvpf/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rms
Ë
QRMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rms
Í
RRMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rms
Í
RRMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0

NoOpNoOp
B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*½A
value³AB°A B©A
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
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
ñ
(iter
	)decay
*learning_rate
+momentum
,rho	rmsr	rmss	"rmst	#rmsu	-rmsv	.rmsw	/rmsx	0rmsy	1rmsz	2rms{	3rms|	4rms}	5rms~	6rms
7rms
8rms
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
regularization_losses
	variables
9layer_regularization_losses
:non_trainable_variables
;layer_metrics
	trainable_variables
<metrics

=layers
 
][
VARIABLE_VALUEzdelabjare/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEzdelabjare/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
>layer_regularization_losses
?non_trainable_variables
trainable_variables
@layer_metrics
	variables
Ametrics

Blayers
 
 
 
­
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
trainable_variables
Elayer_metrics
	variables
Fmetrics

Glayers
ó
H
state_size

-kernel
.recurrent_kernel
/bias
0input_gate_peephole_weights
 1forget_gate_peephole_weights
 2output_gate_peephole_weights
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
 
 
*
-0
.1
/2
03
14
25
*
-0
.1
/2
03
14
25
¹
regularization_losses
	variables
Mlayer_regularization_losses

Nstates
Onon_trainable_variables
Player_metrics
trainable_variables
Qmetrics

Rlayers
ó
S
state_size

3kernel
4recurrent_kernel
5bias
6input_gate_peephole_weights
 7forget_gate_peephole_weights
 8output_gate_peephole_weights
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
 
 
*
30
41
52
63
74
85
*
30
41
52
63
74
85
¹
regularization_losses
	variables
Xlayer_regularization_losses

Ystates
Znon_trainable_variables
[layer_metrics
 trainable_variables
\metrics

]layers
][
VARIABLE_VALUEvfmtgawzzo/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvfmtgawzzo/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
­
$regularization_losses
^layer_regularization_losses
_non_trainable_variables
%trainable_variables
`layer_metrics
&	variables
ametrics

blayers
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
XV
VARIABLE_VALUExedyzswikc/jgtgtymybc/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&xedyzswikc/jgtgtymybc/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUExedyzswikc/jgtgtymybc/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1xedyzswikc/jgtgtymybc/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2xedyzswikc/jgtgtymybc/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2xedyzswikc/jgtgtymybc/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEksthobzafc/qjjkgjcvpf/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&ksthobzafc/qjjkgjcvpf/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEksthobzafc/qjjkgjcvpf/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

c0
#
0
1
2
3
4
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
*
-0
.1
/2
03
14
25
­
Iregularization_losses
dlayer_regularization_losses
enon_trainable_variables
Jtrainable_variables
flayer_metrics
K	variables
gmetrics

hlayers
 
 
 
 
 

0
 
 
*
30
41
52
63
74
85
*
30
41
52
63
74
85
­
Tregularization_losses
ilayer_regularization_losses
jnon_trainable_variables
Utrainable_variables
klayer_metrics
V	variables
lmetrics

mlayers
 
 
 
 
 

0
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

VARIABLE_VALUERMSprop/zdelabjare/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/zdelabjare/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/vfmtgawzzo/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/vfmtgawzzo/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/xedyzswikc/jgtgtymybc/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/xedyzswikc/jgtgtymybc/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/ksthobzafc/qjjkgjcvpf/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_qtjxeibmnqPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_qtjxeibmnqzdelabjare/kernelzdelabjare/biasxedyzswikc/jgtgtymybc/kernel&xedyzswikc/jgtgtymybc/recurrent_kernelxedyzswikc/jgtgtymybc/bias1xedyzswikc/jgtgtymybc/input_gate_peephole_weights2xedyzswikc/jgtgtymybc/forget_gate_peephole_weights2xedyzswikc/jgtgtymybc/output_gate_peephole_weightsksthobzafc/qjjkgjcvpf/kernel&ksthobzafc/qjjkgjcvpf/recurrent_kernelksthobzafc/qjjkgjcvpf/bias1ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights2ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights2ksthobzafc/qjjkgjcvpf/output_gate_peephole_weightsvfmtgawzzo/kernelvfmtgawzzo/bias*
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
%__inference_signature_wrapper_1734175
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%zdelabjare/kernel/Read/ReadVariableOp#zdelabjare/bias/Read/ReadVariableOp%vfmtgawzzo/kernel/Read/ReadVariableOp#vfmtgawzzo/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0xedyzswikc/jgtgtymybc/kernel/Read/ReadVariableOp:xedyzswikc/jgtgtymybc/recurrent_kernel/Read/ReadVariableOp.xedyzswikc/jgtgtymybc/bias/Read/ReadVariableOpExedyzswikc/jgtgtymybc/input_gate_peephole_weights/Read/ReadVariableOpFxedyzswikc/jgtgtymybc/forget_gate_peephole_weights/Read/ReadVariableOpFxedyzswikc/jgtgtymybc/output_gate_peephole_weights/Read/ReadVariableOp0ksthobzafc/qjjkgjcvpf/kernel/Read/ReadVariableOp:ksthobzafc/qjjkgjcvpf/recurrent_kernel/Read/ReadVariableOp.ksthobzafc/qjjkgjcvpf/bias/Read/ReadVariableOpEksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/Read/ReadVariableOpFksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/Read/ReadVariableOpFksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/zdelabjare/kernel/rms/Read/ReadVariableOp/RMSprop/zdelabjare/bias/rms/Read/ReadVariableOp1RMSprop/vfmtgawzzo/kernel/rms/Read/ReadVariableOp/RMSprop/vfmtgawzzo/bias/rms/Read/ReadVariableOp<RMSprop/xedyzswikc/jgtgtymybc/kernel/rms/Read/ReadVariableOpFRMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/xedyzswikc/jgtgtymybc/bias/rms/Read/ReadVariableOpQRMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rms/Read/ReadVariableOpFRMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/ksthobzafc/qjjkgjcvpf/bias/rms/Read/ReadVariableOpQRMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
 __inference__traced_save_1737124
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamezdelabjare/kernelzdelabjare/biasvfmtgawzzo/kernelvfmtgawzzo/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoxedyzswikc/jgtgtymybc/kernel&xedyzswikc/jgtgtymybc/recurrent_kernelxedyzswikc/jgtgtymybc/bias1xedyzswikc/jgtgtymybc/input_gate_peephole_weights2xedyzswikc/jgtgtymybc/forget_gate_peephole_weights2xedyzswikc/jgtgtymybc/output_gate_peephole_weightsksthobzafc/qjjkgjcvpf/kernel&ksthobzafc/qjjkgjcvpf/recurrent_kernelksthobzafc/qjjkgjcvpf/bias1ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights2ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights2ksthobzafc/qjjkgjcvpf/output_gate_peephole_weightstotalcountRMSprop/zdelabjare/kernel/rmsRMSprop/zdelabjare/bias/rmsRMSprop/vfmtgawzzo/kernel/rmsRMSprop/vfmtgawzzo/bias/rms(RMSprop/xedyzswikc/jgtgtymybc/kernel/rms2RMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rms&RMSprop/xedyzswikc/jgtgtymybc/bias/rms=RMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rms>RMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rms>RMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rms(RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rms2RMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rms&RMSprop/ksthobzafc/qjjkgjcvpf/bias/rms=RMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rms>RMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rms>RMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rms*3
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
#__inference__traced_restore_1737251Âà-
p
Ê
xedyzswikc_while_body_17347742
.xedyzswikc_while_xedyzswikc_while_loop_counter8
4xedyzswikc_while_xedyzswikc_while_maximum_iterations 
xedyzswikc_while_placeholder"
xedyzswikc_while_placeholder_1"
xedyzswikc_while_placeholder_2"
xedyzswikc_while_placeholder_31
-xedyzswikc_while_xedyzswikc_strided_slice_1_0m
ixedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0O
<xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0:	Q
>xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 L
=xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0:	C
5xedyzswikc_while_jgtgtymybc_readvariableop_resource_0: E
7xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0: E
7xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0: 
xedyzswikc_while_identity
xedyzswikc_while_identity_1
xedyzswikc_while_identity_2
xedyzswikc_while_identity_3
xedyzswikc_while_identity_4
xedyzswikc_while_identity_5/
+xedyzswikc_while_xedyzswikc_strided_slice_1k
gxedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensorM
:xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource:	O
<xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource:	 J
;xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource:	A
3xedyzswikc_while_jgtgtymybc_readvariableop_resource: C
5xedyzswikc_while_jgtgtymybc_readvariableop_1_resource: C
5xedyzswikc_while_jgtgtymybc_readvariableop_2_resource: ¢2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp¢1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp¢3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp¢*xedyzswikc/while/jgtgtymybc/ReadVariableOp¢,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1¢,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2Ù
Bxedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bxedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4xedyzswikc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemixedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0xedyzswikc_while_placeholderKxedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4xedyzswikc/while/TensorArrayV2Read/TensorListGetItemä
1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp<xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOpý
"xedyzswikc/while/jgtgtymybc/MatMulMatMul;xedyzswikc/while/TensorArrayV2Read/TensorListGetItem:item:09xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"xedyzswikc/while/jgtgtymybc/MatMulê
3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp>xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOpæ
$xedyzswikc/while/jgtgtymybc/MatMul_1MatMulxedyzswikc_while_placeholder_2;xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$xedyzswikc/while/jgtgtymybc/MatMul_1Ü
xedyzswikc/while/jgtgtymybc/addAddV2,xedyzswikc/while/jgtgtymybc/MatMul:product:0.xedyzswikc/while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
xedyzswikc/while/jgtgtymybc/addã
2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp=xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOpé
#xedyzswikc/while/jgtgtymybc/BiasAddBiasAdd#xedyzswikc/while/jgtgtymybc/add:z:0:xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#xedyzswikc/while/jgtgtymybc/BiasAdd
+xedyzswikc/while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+xedyzswikc/while/jgtgtymybc/split/split_dim¯
!xedyzswikc/while/jgtgtymybc/splitSplit4xedyzswikc/while/jgtgtymybc/split/split_dim:output:0,xedyzswikc/while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!xedyzswikc/while/jgtgtymybc/splitÊ
*xedyzswikc/while/jgtgtymybc/ReadVariableOpReadVariableOp5xedyzswikc_while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*xedyzswikc/while/jgtgtymybc/ReadVariableOpÏ
xedyzswikc/while/jgtgtymybc/mulMul2xedyzswikc/while/jgtgtymybc/ReadVariableOp:value:0xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
xedyzswikc/while/jgtgtymybc/mulÒ
!xedyzswikc/while/jgtgtymybc/add_1AddV2*xedyzswikc/while/jgtgtymybc/split:output:0#xedyzswikc/while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_1®
#xedyzswikc/while/jgtgtymybc/SigmoidSigmoid%xedyzswikc/while/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#xedyzswikc/while/jgtgtymybc/SigmoidÐ
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1ReadVariableOp7xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1Õ
!xedyzswikc/while/jgtgtymybc/mul_1Mul4xedyzswikc/while/jgtgtymybc/ReadVariableOp_1:value:0xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_1Ô
!xedyzswikc/while/jgtgtymybc/add_2AddV2*xedyzswikc/while/jgtgtymybc/split:output:1%xedyzswikc/while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_2²
%xedyzswikc/while/jgtgtymybc/Sigmoid_1Sigmoid%xedyzswikc/while/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%xedyzswikc/while/jgtgtymybc/Sigmoid_1Ê
!xedyzswikc/while/jgtgtymybc/mul_2Mul)xedyzswikc/while/jgtgtymybc/Sigmoid_1:y:0xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_2ª
 xedyzswikc/while/jgtgtymybc/TanhTanh*xedyzswikc/while/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 xedyzswikc/while/jgtgtymybc/TanhÎ
!xedyzswikc/while/jgtgtymybc/mul_3Mul'xedyzswikc/while/jgtgtymybc/Sigmoid:y:0$xedyzswikc/while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_3Ï
!xedyzswikc/while/jgtgtymybc/add_3AddV2%xedyzswikc/while/jgtgtymybc/mul_2:z:0%xedyzswikc/while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_3Ð
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2ReadVariableOp7xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2Ü
!xedyzswikc/while/jgtgtymybc/mul_4Mul4xedyzswikc/while/jgtgtymybc/ReadVariableOp_2:value:0%xedyzswikc/while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_4Ô
!xedyzswikc/while/jgtgtymybc/add_4AddV2*xedyzswikc/while/jgtgtymybc/split:output:3%xedyzswikc/while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_4²
%xedyzswikc/while/jgtgtymybc/Sigmoid_2Sigmoid%xedyzswikc/while/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%xedyzswikc/while/jgtgtymybc/Sigmoid_2©
"xedyzswikc/while/jgtgtymybc/Tanh_1Tanh%xedyzswikc/while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"xedyzswikc/while/jgtgtymybc/Tanh_1Ò
!xedyzswikc/while/jgtgtymybc/mul_5Mul)xedyzswikc/while/jgtgtymybc/Sigmoid_2:y:0&xedyzswikc/while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_5
5xedyzswikc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemxedyzswikc_while_placeholder_1xedyzswikc_while_placeholder%xedyzswikc/while/jgtgtymybc/mul_5:z:0*
_output_shapes
: *
element_dtype027
5xedyzswikc/while/TensorArrayV2Write/TensorListSetItemr
xedyzswikc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
xedyzswikc/while/add/y
xedyzswikc/while/addAddV2xedyzswikc_while_placeholderxedyzswikc/while/add/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/while/addv
xedyzswikc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
xedyzswikc/while/add_1/y­
xedyzswikc/while/add_1AddV2.xedyzswikc_while_xedyzswikc_while_loop_counter!xedyzswikc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/while/add_1©
xedyzswikc/while/IdentityIdentityxedyzswikc/while/add_1:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/IdentityÇ
xedyzswikc/while/Identity_1Identity4xedyzswikc_while_xedyzswikc_while_maximum_iterations3^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/Identity_1«
xedyzswikc/while/Identity_2Identityxedyzswikc/while/add:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/Identity_2Ø
xedyzswikc/while/Identity_3IdentityExedyzswikc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/Identity_3É
xedyzswikc/while/Identity_4Identity%xedyzswikc/while/jgtgtymybc/mul_5:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/while/Identity_4É
xedyzswikc/while/Identity_5Identity%xedyzswikc/while/jgtgtymybc/add_3:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/while/Identity_5"?
xedyzswikc_while_identity"xedyzswikc/while/Identity:output:0"C
xedyzswikc_while_identity_1$xedyzswikc/while/Identity_1:output:0"C
xedyzswikc_while_identity_2$xedyzswikc/while/Identity_2:output:0"C
xedyzswikc_while_identity_3$xedyzswikc/while/Identity_3:output:0"C
xedyzswikc_while_identity_4$xedyzswikc/while/Identity_4:output:0"C
xedyzswikc_while_identity_5$xedyzswikc/while/Identity_5:output:0"|
;xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource=xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0"~
<xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource>xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0"z
:xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource<xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0"p
5xedyzswikc_while_jgtgtymybc_readvariableop_1_resource7xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0"p
5xedyzswikc_while_jgtgtymybc_readvariableop_2_resource7xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0"l
3xedyzswikc_while_jgtgtymybc_readvariableop_resource5xedyzswikc_while_jgtgtymybc_readvariableop_resource_0"Ô
gxedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensorixedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0"\
+xedyzswikc_while_xedyzswikc_strided_slice_1-xedyzswikc_while_xedyzswikc_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2f
1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp2j
3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp2X
*xedyzswikc/while/jgtgtymybc/ReadVariableOp*xedyzswikc/while/jgtgtymybc/ReadVariableOp2\
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1,xedyzswikc/while/jgtgtymybc/ReadVariableOp_12\
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2: 
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
while_cond_1735267
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1735267___redundant_placeholder05
1while_while_cond_1735267___redundant_placeholder15
1while_while_cond_1735267___redundant_placeholder25
1while_while_cond_1735267___redundant_placeholder35
1while_while_cond_1735267___redundant_placeholder45
1while_while_cond_1735267___redundant_placeholder55
1while_while_cond_1735267___redundant_placeholder6
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
¡h

G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1733865

inputs<
)jgtgtymybc_matmul_readvariableop_resource:	>
+jgtgtymybc_matmul_1_readvariableop_resource:	 9
*jgtgtymybc_biasadd_readvariableop_resource:	0
"jgtgtymybc_readvariableop_resource: 2
$jgtgtymybc_readvariableop_1_resource: 2
$jgtgtymybc_readvariableop_2_resource: 
identity¢!jgtgtymybc/BiasAdd/ReadVariableOp¢ jgtgtymybc/MatMul/ReadVariableOp¢"jgtgtymybc/MatMul_1/ReadVariableOp¢jgtgtymybc/ReadVariableOp¢jgtgtymybc/ReadVariableOp_1¢jgtgtymybc/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 jgtgtymybc/MatMul/ReadVariableOpReadVariableOp)jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jgtgtymybc/MatMul/ReadVariableOp§
jgtgtymybc/MatMulMatMulstrided_slice_2:output:0(jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMulµ
"jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp+jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jgtgtymybc/MatMul_1/ReadVariableOp£
jgtgtymybc/MatMul_1MatMulzeros:output:0*jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMul_1
jgtgtymybc/addAddV2jgtgtymybc/MatMul:product:0jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/add®
!jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp*jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jgtgtymybc/BiasAdd/ReadVariableOp¥
jgtgtymybc/BiasAddBiasAddjgtgtymybc/add:z:0)jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/BiasAddz
jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jgtgtymybc/split/split_dimë
jgtgtymybc/splitSplit#jgtgtymybc/split/split_dim:output:0jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jgtgtymybc/split
jgtgtymybc/ReadVariableOpReadVariableOp"jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp
jgtgtymybc/mulMul!jgtgtymybc/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul
jgtgtymybc/add_1AddV2jgtgtymybc/split:output:0jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_1{
jgtgtymybc/SigmoidSigmoidjgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid
jgtgtymybc/ReadVariableOp_1ReadVariableOp$jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_1
jgtgtymybc/mul_1Mul#jgtgtymybc/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_1
jgtgtymybc/add_2AddV2jgtgtymybc/split:output:1jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_2
jgtgtymybc/Sigmoid_1Sigmoidjgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_1
jgtgtymybc/mul_2Muljgtgtymybc/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_2w
jgtgtymybc/TanhTanhjgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh
jgtgtymybc/mul_3Muljgtgtymybc/Sigmoid:y:0jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_3
jgtgtymybc/add_3AddV2jgtgtymybc/mul_2:z:0jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_3
jgtgtymybc/ReadVariableOp_2ReadVariableOp$jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_2
jgtgtymybc/mul_4Mul#jgtgtymybc/ReadVariableOp_2:value:0jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_4
jgtgtymybc/add_4AddV2jgtgtymybc/split:output:3jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_4
jgtgtymybc/Sigmoid_2Sigmoidjgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_2v
jgtgtymybc/Tanh_1Tanhjgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh_1
jgtgtymybc/mul_5Muljgtgtymybc/Sigmoid_2:y:0jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jgtgtymybc_matmul_readvariableop_resource+jgtgtymybc_matmul_1_readvariableop_resource*jgtgtymybc_biasadd_readvariableop_resource"jgtgtymybc_readvariableop_resource$jgtgtymybc_readvariableop_1_resource$jgtgtymybc_readvariableop_2_resource*
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
while_body_1733764*
condR
while_cond_1733763*Q
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
transpose_1³
IdentityIdentitytranspose_1:y:0"^jgtgtymybc/BiasAdd/ReadVariableOp!^jgtgtymybc/MatMul/ReadVariableOp#^jgtgtymybc/MatMul_1/ReadVariableOp^jgtgtymybc/ReadVariableOp^jgtgtymybc/ReadVariableOp_1^jgtgtymybc/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jgtgtymybc/BiasAdd/ReadVariableOp!jgtgtymybc/BiasAdd/ReadVariableOp2D
 jgtgtymybc/MatMul/ReadVariableOp jgtgtymybc/MatMul/ReadVariableOp2H
"jgtgtymybc/MatMul_1/ReadVariableOp"jgtgtymybc/MatMul_1/ReadVariableOp26
jgtgtymybc/ReadVariableOpjgtgtymybc/ReadVariableOp2:
jgtgtymybc/ReadVariableOp_1jgtgtymybc/ReadVariableOp_12:
jgtgtymybc/ReadVariableOp_2jgtgtymybc/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Üh

G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736157
inputs_0<
)qjjkgjcvpf_matmul_readvariableop_resource:	 >
+qjjkgjcvpf_matmul_1_readvariableop_resource:	 9
*qjjkgjcvpf_biasadd_readvariableop_resource:	0
"qjjkgjcvpf_readvariableop_resource: 2
$qjjkgjcvpf_readvariableop_1_resource: 2
$qjjkgjcvpf_readvariableop_2_resource: 
identity¢!qjjkgjcvpf/BiasAdd/ReadVariableOp¢ qjjkgjcvpf/MatMul/ReadVariableOp¢"qjjkgjcvpf/MatMul_1/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp_1¢qjjkgjcvpf/ReadVariableOp_2¢whileF
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
strided_slice_2¯
 qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp)qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 qjjkgjcvpf/MatMul/ReadVariableOp§
qjjkgjcvpf/MatMulMatMulstrided_slice_2:output:0(qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMulµ
"qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp+qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"qjjkgjcvpf/MatMul_1/ReadVariableOp£
qjjkgjcvpf/MatMul_1MatMulzeros:output:0*qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMul_1
qjjkgjcvpf/addAddV2qjjkgjcvpf/MatMul:product:0qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/add®
!qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp*qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!qjjkgjcvpf/BiasAdd/ReadVariableOp¥
qjjkgjcvpf/BiasAddBiasAddqjjkgjcvpf/add:z:0)qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/BiasAddz
qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
qjjkgjcvpf/split/split_dimë
qjjkgjcvpf/splitSplit#qjjkgjcvpf/split/split_dim:output:0qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qjjkgjcvpf/split
qjjkgjcvpf/ReadVariableOpReadVariableOp"qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp
qjjkgjcvpf/mulMul!qjjkgjcvpf/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul
qjjkgjcvpf/add_1AddV2qjjkgjcvpf/split:output:0qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_1{
qjjkgjcvpf/SigmoidSigmoidqjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid
qjjkgjcvpf/ReadVariableOp_1ReadVariableOp$qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_1
qjjkgjcvpf/mul_1Mul#qjjkgjcvpf/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_1
qjjkgjcvpf/add_2AddV2qjjkgjcvpf/split:output:1qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_2
qjjkgjcvpf/Sigmoid_1Sigmoidqjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_1
qjjkgjcvpf/mul_2Mulqjjkgjcvpf/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_2w
qjjkgjcvpf/TanhTanhqjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh
qjjkgjcvpf/mul_3Mulqjjkgjcvpf/Sigmoid:y:0qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_3
qjjkgjcvpf/add_3AddV2qjjkgjcvpf/mul_2:z:0qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_3
qjjkgjcvpf/ReadVariableOp_2ReadVariableOp$qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_2
qjjkgjcvpf/mul_4Mul#qjjkgjcvpf/ReadVariableOp_2:value:0qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_4
qjjkgjcvpf/add_4AddV2qjjkgjcvpf/split:output:3qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_4
qjjkgjcvpf/Sigmoid_2Sigmoidqjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_2v
qjjkgjcvpf/Tanh_1Tanhqjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh_1
qjjkgjcvpf/mul_5Mulqjjkgjcvpf/Sigmoid_2:y:0qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)qjjkgjcvpf_matmul_readvariableop_resource+qjjkgjcvpf_matmul_1_readvariableop_resource*qjjkgjcvpf_biasadd_readvariableop_resource"qjjkgjcvpf_readvariableop_resource$qjjkgjcvpf_readvariableop_1_resource$qjjkgjcvpf_readvariableop_2_resource*
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
while_body_1736056*
condR
while_cond_1736055*Q
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
transpose_1¸
IdentityIdentitystrided_slice_3:output:0"^qjjkgjcvpf/BiasAdd/ReadVariableOp!^qjjkgjcvpf/MatMul/ReadVariableOp#^qjjkgjcvpf/MatMul_1/ReadVariableOp^qjjkgjcvpf/ReadVariableOp^qjjkgjcvpf/ReadVariableOp_1^qjjkgjcvpf/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!qjjkgjcvpf/BiasAdd/ReadVariableOp!qjjkgjcvpf/BiasAdd/ReadVariableOp2D
 qjjkgjcvpf/MatMul/ReadVariableOp qjjkgjcvpf/MatMul/ReadVariableOp2H
"qjjkgjcvpf/MatMul_1/ReadVariableOp"qjjkgjcvpf/MatMul_1/ReadVariableOp26
qjjkgjcvpf/ReadVariableOpqjjkgjcvpf/ReadVariableOp2:
qjjkgjcvpf/ReadVariableOp_1qjjkgjcvpf/ReadVariableOp_12:
qjjkgjcvpf/ReadVariableOp_2qjjkgjcvpf/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
¯F
ê
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1732631

inputs%
qjjkgjcvpf_1732532:	 %
qjjkgjcvpf_1732534:	 !
qjjkgjcvpf_1732536:	 
qjjkgjcvpf_1732538:  
qjjkgjcvpf_1732540:  
qjjkgjcvpf_1732542: 
identity¢"qjjkgjcvpf/StatefulPartitionedCall¢whileD
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
strided_slice_2Ú
"qjjkgjcvpf/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0qjjkgjcvpf_1732532qjjkgjcvpf_1732534qjjkgjcvpf_1732536qjjkgjcvpf_1732538qjjkgjcvpf_1732540qjjkgjcvpf_1732542*
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
GPU2*0J 8 *P
fKRI
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_17324552$
"qjjkgjcvpf/StatefulPartitionedCall
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
while/loop_counterð
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0qjjkgjcvpf_1732532qjjkgjcvpf_1732534qjjkgjcvpf_1732536qjjkgjcvpf_1732538qjjkgjcvpf_1732540qjjkgjcvpf_1732542*
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
while_body_1732551*
condR
while_cond_1732550*Q
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
transpose_1
IdentityIdentitystrided_slice_3:output:0#^qjjkgjcvpf/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"qjjkgjcvpf/StatefulPartitionedCall"qjjkgjcvpf/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó	
ø
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_1736716

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


í
while_cond_1735447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1735447___redundant_placeholder05
1while_while_cond_1735447___redundant_placeholder15
1while_while_cond_1735447___redundant_placeholder25
1while_while_cond_1735447___redundant_placeholder35
1while_while_cond_1735447___redundant_placeholder45
1while_while_cond_1735447___redundant_placeholder55
1while_while_cond_1735447___redundant_placeholder6
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
àY

while_body_1736416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 F
3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 A
2while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	8
*while_qjjkgjcvpf_readvariableop_resource_0: :
,while_qjjkgjcvpf_readvariableop_1_resource_0: :
,while_qjjkgjcvpf_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_qjjkgjcvpf_matmul_readvariableop_resource:	 D
1while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 ?
0while_qjjkgjcvpf_biasadd_readvariableop_resource:	6
(while_qjjkgjcvpf_readvariableop_resource: 8
*while_qjjkgjcvpf_readvariableop_1_resource: 8
*while_qjjkgjcvpf_readvariableop_2_resource: ¢'while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢&while/qjjkgjcvpf/MatMul/ReadVariableOp¢(while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢while/qjjkgjcvpf/ReadVariableOp¢!while/qjjkgjcvpf/ReadVariableOp_1¢!while/qjjkgjcvpf/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp1while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/qjjkgjcvpf/MatMul/ReadVariableOpÑ
while/qjjkgjcvpf/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMulÉ
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpº
while/qjjkgjcvpf/MatMul_1MatMulwhile_placeholder_20while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMul_1°
while/qjjkgjcvpf/addAddV2!while/qjjkgjcvpf/MatMul:product:0#while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/addÂ
'while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp2while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp½
while/qjjkgjcvpf/BiasAddBiasAddwhile/qjjkgjcvpf/add:z:0/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/BiasAdd
 while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/qjjkgjcvpf/split/split_dim
while/qjjkgjcvpf/splitSplit)while/qjjkgjcvpf/split/split_dim:output:0!while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/qjjkgjcvpf/split©
while/qjjkgjcvpf/ReadVariableOpReadVariableOp*while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/qjjkgjcvpf/ReadVariableOp£
while/qjjkgjcvpf/mulMul'while/qjjkgjcvpf/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul¦
while/qjjkgjcvpf/add_1AddV2while/qjjkgjcvpf/split:output:0while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_1
while/qjjkgjcvpf/SigmoidSigmoidwhile/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid¯
!while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp,while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_1©
while/qjjkgjcvpf/mul_1Mul)while/qjjkgjcvpf/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_1¨
while/qjjkgjcvpf/add_2AddV2while/qjjkgjcvpf/split:output:1while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_2
while/qjjkgjcvpf/Sigmoid_1Sigmoidwhile/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_1
while/qjjkgjcvpf/mul_2Mulwhile/qjjkgjcvpf/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_2
while/qjjkgjcvpf/TanhTanhwhile/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh¢
while/qjjkgjcvpf/mul_3Mulwhile/qjjkgjcvpf/Sigmoid:y:0while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_3£
while/qjjkgjcvpf/add_3AddV2while/qjjkgjcvpf/mul_2:z:0while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_3¯
!while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp,while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_2°
while/qjjkgjcvpf/mul_4Mul)while/qjjkgjcvpf/ReadVariableOp_2:value:0while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_4¨
while/qjjkgjcvpf/add_4AddV2while/qjjkgjcvpf/split:output:3while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_4
while/qjjkgjcvpf/Sigmoid_2Sigmoidwhile/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_2
while/qjjkgjcvpf/Tanh_1Tanhwhile/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh_1¦
while/qjjkgjcvpf/mul_5Mulwhile/qjjkgjcvpf/Sigmoid_2:y:0while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/qjjkgjcvpf/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/qjjkgjcvpf/mul_5:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/qjjkgjcvpf/add_3:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_qjjkgjcvpf_biasadd_readvariableop_resource2while_qjjkgjcvpf_biasadd_readvariableop_resource_0"h
1while_qjjkgjcvpf_matmul_1_readvariableop_resource3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"d
/while_qjjkgjcvpf_matmul_readvariableop_resource1while_qjjkgjcvpf_matmul_readvariableop_resource_0"Z
*while_qjjkgjcvpf_readvariableop_1_resource,while_qjjkgjcvpf_readvariableop_1_resource_0"Z
*while_qjjkgjcvpf_readvariableop_2_resource,while_qjjkgjcvpf_readvariableop_2_resource_0"V
(while_qjjkgjcvpf_readvariableop_resource*while_qjjkgjcvpf_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp'while/qjjkgjcvpf/BiasAdd/ReadVariableOp2P
&while/qjjkgjcvpf/MatMul/ReadVariableOp&while/qjjkgjcvpf/MatMul/ReadVariableOp2T
(while/qjjkgjcvpf/MatMul_1/ReadVariableOp(while/qjjkgjcvpf/MatMul_1/ReadVariableOp2B
while/qjjkgjcvpf/ReadVariableOpwhile/qjjkgjcvpf/ReadVariableOp2F
!while/qjjkgjcvpf/ReadVariableOp_1!while/qjjkgjcvpf/ReadVariableOp_12F
!while/qjjkgjcvpf/ReadVariableOp_2!while/qjjkgjcvpf/ReadVariableOp_2: 
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
¦h

G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1733376

inputs<
)qjjkgjcvpf_matmul_readvariableop_resource:	 >
+qjjkgjcvpf_matmul_1_readvariableop_resource:	 9
*qjjkgjcvpf_biasadd_readvariableop_resource:	0
"qjjkgjcvpf_readvariableop_resource: 2
$qjjkgjcvpf_readvariableop_1_resource: 2
$qjjkgjcvpf_readvariableop_2_resource: 
identity¢!qjjkgjcvpf/BiasAdd/ReadVariableOp¢ qjjkgjcvpf/MatMul/ReadVariableOp¢"qjjkgjcvpf/MatMul_1/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp_1¢qjjkgjcvpf/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp)qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 qjjkgjcvpf/MatMul/ReadVariableOp§
qjjkgjcvpf/MatMulMatMulstrided_slice_2:output:0(qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMulµ
"qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp+qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"qjjkgjcvpf/MatMul_1/ReadVariableOp£
qjjkgjcvpf/MatMul_1MatMulzeros:output:0*qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMul_1
qjjkgjcvpf/addAddV2qjjkgjcvpf/MatMul:product:0qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/add®
!qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp*qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!qjjkgjcvpf/BiasAdd/ReadVariableOp¥
qjjkgjcvpf/BiasAddBiasAddqjjkgjcvpf/add:z:0)qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/BiasAddz
qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
qjjkgjcvpf/split/split_dimë
qjjkgjcvpf/splitSplit#qjjkgjcvpf/split/split_dim:output:0qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qjjkgjcvpf/split
qjjkgjcvpf/ReadVariableOpReadVariableOp"qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp
qjjkgjcvpf/mulMul!qjjkgjcvpf/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul
qjjkgjcvpf/add_1AddV2qjjkgjcvpf/split:output:0qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_1{
qjjkgjcvpf/SigmoidSigmoidqjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid
qjjkgjcvpf/ReadVariableOp_1ReadVariableOp$qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_1
qjjkgjcvpf/mul_1Mul#qjjkgjcvpf/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_1
qjjkgjcvpf/add_2AddV2qjjkgjcvpf/split:output:1qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_2
qjjkgjcvpf/Sigmoid_1Sigmoidqjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_1
qjjkgjcvpf/mul_2Mulqjjkgjcvpf/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_2w
qjjkgjcvpf/TanhTanhqjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh
qjjkgjcvpf/mul_3Mulqjjkgjcvpf/Sigmoid:y:0qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_3
qjjkgjcvpf/add_3AddV2qjjkgjcvpf/mul_2:z:0qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_3
qjjkgjcvpf/ReadVariableOp_2ReadVariableOp$qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_2
qjjkgjcvpf/mul_4Mul#qjjkgjcvpf/ReadVariableOp_2:value:0qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_4
qjjkgjcvpf/add_4AddV2qjjkgjcvpf/split:output:3qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_4
qjjkgjcvpf/Sigmoid_2Sigmoidqjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_2v
qjjkgjcvpf/Tanh_1Tanhqjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh_1
qjjkgjcvpf/mul_5Mulqjjkgjcvpf/Sigmoid_2:y:0qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)qjjkgjcvpf_matmul_readvariableop_resource+qjjkgjcvpf_matmul_1_readvariableop_resource*qjjkgjcvpf_biasadd_readvariableop_resource"qjjkgjcvpf_readvariableop_resource$qjjkgjcvpf_readvariableop_1_resource$qjjkgjcvpf_readvariableop_2_resource*
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
while_body_1733275*
condR
while_cond_1733274*Q
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
transpose_1¸
IdentityIdentitystrided_slice_3:output:0"^qjjkgjcvpf/BiasAdd/ReadVariableOp!^qjjkgjcvpf/MatMul/ReadVariableOp#^qjjkgjcvpf/MatMul_1/ReadVariableOp^qjjkgjcvpf/ReadVariableOp^qjjkgjcvpf/ReadVariableOp_1^qjjkgjcvpf/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!qjjkgjcvpf/BiasAdd/ReadVariableOp!qjjkgjcvpf/BiasAdd/ReadVariableOp2D
 qjjkgjcvpf/MatMul/ReadVariableOp qjjkgjcvpf/MatMul/ReadVariableOp2H
"qjjkgjcvpf/MatMul_1/ReadVariableOp"qjjkgjcvpf/MatMul_1/ReadVariableOp26
qjjkgjcvpf/ReadVariableOpqjjkgjcvpf/ReadVariableOp2:
qjjkgjcvpf/ReadVariableOp_1qjjkgjcvpf/ReadVariableOp_12:
qjjkgjcvpf/ReadVariableOp_2qjjkgjcvpf/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ksthobzafc_while_cond_17349492
.ksthobzafc_while_ksthobzafc_while_loop_counter8
4ksthobzafc_while_ksthobzafc_while_maximum_iterations 
ksthobzafc_while_placeholder"
ksthobzafc_while_placeholder_1"
ksthobzafc_while_placeholder_2"
ksthobzafc_while_placeholder_34
0ksthobzafc_while_less_ksthobzafc_strided_slice_1K
Gksthobzafc_while_ksthobzafc_while_cond_1734949___redundant_placeholder0K
Gksthobzafc_while_ksthobzafc_while_cond_1734949___redundant_placeholder1K
Gksthobzafc_while_ksthobzafc_while_cond_1734949___redundant_placeholder2K
Gksthobzafc_while_ksthobzafc_while_cond_1734949___redundant_placeholder3K
Gksthobzafc_while_ksthobzafc_while_cond_1734949___redundant_placeholder4K
Gksthobzafc_while_ksthobzafc_while_cond_1734949___redundant_placeholder5K
Gksthobzafc_while_ksthobzafc_while_cond_1734949___redundant_placeholder6
ksthobzafc_while_identity
§
ksthobzafc/while/LessLessksthobzafc_while_placeholder0ksthobzafc_while_less_ksthobzafc_strided_slice_1*
T0*
_output_shapes
: 2
ksthobzafc/while/Less~
ksthobzafc/while/IdentityIdentityksthobzafc/while/Less:z:0*
T0
*
_output_shapes
: 2
ksthobzafc/while/Identity"?
ksthobzafc_while_identity"ksthobzafc/while/Identity:output:0*(
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
µ
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1736806

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
¦h

G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736517

inputs<
)qjjkgjcvpf_matmul_readvariableop_resource:	 >
+qjjkgjcvpf_matmul_1_readvariableop_resource:	 9
*qjjkgjcvpf_biasadd_readvariableop_resource:	0
"qjjkgjcvpf_readvariableop_resource: 2
$qjjkgjcvpf_readvariableop_1_resource: 2
$qjjkgjcvpf_readvariableop_2_resource: 
identity¢!qjjkgjcvpf/BiasAdd/ReadVariableOp¢ qjjkgjcvpf/MatMul/ReadVariableOp¢"qjjkgjcvpf/MatMul_1/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp_1¢qjjkgjcvpf/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp)qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 qjjkgjcvpf/MatMul/ReadVariableOp§
qjjkgjcvpf/MatMulMatMulstrided_slice_2:output:0(qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMulµ
"qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp+qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"qjjkgjcvpf/MatMul_1/ReadVariableOp£
qjjkgjcvpf/MatMul_1MatMulzeros:output:0*qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMul_1
qjjkgjcvpf/addAddV2qjjkgjcvpf/MatMul:product:0qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/add®
!qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp*qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!qjjkgjcvpf/BiasAdd/ReadVariableOp¥
qjjkgjcvpf/BiasAddBiasAddqjjkgjcvpf/add:z:0)qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/BiasAddz
qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
qjjkgjcvpf/split/split_dimë
qjjkgjcvpf/splitSplit#qjjkgjcvpf/split/split_dim:output:0qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qjjkgjcvpf/split
qjjkgjcvpf/ReadVariableOpReadVariableOp"qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp
qjjkgjcvpf/mulMul!qjjkgjcvpf/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul
qjjkgjcvpf/add_1AddV2qjjkgjcvpf/split:output:0qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_1{
qjjkgjcvpf/SigmoidSigmoidqjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid
qjjkgjcvpf/ReadVariableOp_1ReadVariableOp$qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_1
qjjkgjcvpf/mul_1Mul#qjjkgjcvpf/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_1
qjjkgjcvpf/add_2AddV2qjjkgjcvpf/split:output:1qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_2
qjjkgjcvpf/Sigmoid_1Sigmoidqjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_1
qjjkgjcvpf/mul_2Mulqjjkgjcvpf/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_2w
qjjkgjcvpf/TanhTanhqjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh
qjjkgjcvpf/mul_3Mulqjjkgjcvpf/Sigmoid:y:0qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_3
qjjkgjcvpf/add_3AddV2qjjkgjcvpf/mul_2:z:0qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_3
qjjkgjcvpf/ReadVariableOp_2ReadVariableOp$qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_2
qjjkgjcvpf/mul_4Mul#qjjkgjcvpf/ReadVariableOp_2:value:0qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_4
qjjkgjcvpf/add_4AddV2qjjkgjcvpf/split:output:3qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_4
qjjkgjcvpf/Sigmoid_2Sigmoidqjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_2v
qjjkgjcvpf/Tanh_1Tanhqjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh_1
qjjkgjcvpf/mul_5Mulqjjkgjcvpf/Sigmoid_2:y:0qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)qjjkgjcvpf_matmul_readvariableop_resource+qjjkgjcvpf_matmul_1_readvariableop_resource*qjjkgjcvpf_biasadd_readvariableop_resource"qjjkgjcvpf_readvariableop_resource$qjjkgjcvpf_readvariableop_1_resource$qjjkgjcvpf_readvariableop_2_resource*
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
while_body_1736416*
condR
while_cond_1736415*Q
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
transpose_1¸
IdentityIdentitystrided_slice_3:output:0"^qjjkgjcvpf/BiasAdd/ReadVariableOp!^qjjkgjcvpf/MatMul/ReadVariableOp#^qjjkgjcvpf/MatMul_1/ReadVariableOp^qjjkgjcvpf/ReadVariableOp^qjjkgjcvpf/ReadVariableOp_1^qjjkgjcvpf/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!qjjkgjcvpf/BiasAdd/ReadVariableOp!qjjkgjcvpf/BiasAdd/ReadVariableOp2D
 qjjkgjcvpf/MatMul/ReadVariableOp qjjkgjcvpf/MatMul/ReadVariableOp2H
"qjjkgjcvpf/MatMul_1/ReadVariableOp"qjjkgjcvpf/MatMul_1/ReadVariableOp26
qjjkgjcvpf/ReadVariableOpqjjkgjcvpf/ReadVariableOp2:
qjjkgjcvpf/ReadVariableOp_1qjjkgjcvpf/ReadVariableOp_12:
qjjkgjcvpf/ReadVariableOp_2qjjkgjcvpf/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç)
Ò
while_body_1731530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_jgtgtymybc_1731554_0:	-
while_jgtgtymybc_1731556_0:	 )
while_jgtgtymybc_1731558_0:	(
while_jgtgtymybc_1731560_0: (
while_jgtgtymybc_1731562_0: (
while_jgtgtymybc_1731564_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_jgtgtymybc_1731554:	+
while_jgtgtymybc_1731556:	 '
while_jgtgtymybc_1731558:	&
while_jgtgtymybc_1731560: &
while_jgtgtymybc_1731562: &
while_jgtgtymybc_1731564: ¢(while/jgtgtymybc/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¶
(while/jgtgtymybc/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_jgtgtymybc_1731554_0while_jgtgtymybc_1731556_0while_jgtgtymybc_1731558_0while_jgtgtymybc_1731560_0while_jgtgtymybc_1731562_0while_jgtgtymybc_1731564_0*
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
GPU2*0J 8 *P
fKRI
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_17315102*
(while/jgtgtymybc/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/jgtgtymybc/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/jgtgtymybc/StatefulPartitionedCall:output:1)^while/jgtgtymybc/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/jgtgtymybc/StatefulPartitionedCall:output:2)^while/jgtgtymybc/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_jgtgtymybc_1731554while_jgtgtymybc_1731554_0"6
while_jgtgtymybc_1731556while_jgtgtymybc_1731556_0"6
while_jgtgtymybc_1731558while_jgtgtymybc_1731558_0"6
while_jgtgtymybc_1731560while_jgtgtymybc_1731560_0"6
while_jgtgtymybc_1731562while_jgtgtymybc_1731562_0"6
while_jgtgtymybc_1731564while_jgtgtymybc_1731564_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/jgtgtymybc/StatefulPartitionedCall(while/jgtgtymybc/StatefulPartitionedCall: 
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
p
Ê
xedyzswikc_while_body_17343702
.xedyzswikc_while_xedyzswikc_while_loop_counter8
4xedyzswikc_while_xedyzswikc_while_maximum_iterations 
xedyzswikc_while_placeholder"
xedyzswikc_while_placeholder_1"
xedyzswikc_while_placeholder_2"
xedyzswikc_while_placeholder_31
-xedyzswikc_while_xedyzswikc_strided_slice_1_0m
ixedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0O
<xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0:	Q
>xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 L
=xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0:	C
5xedyzswikc_while_jgtgtymybc_readvariableop_resource_0: E
7xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0: E
7xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0: 
xedyzswikc_while_identity
xedyzswikc_while_identity_1
xedyzswikc_while_identity_2
xedyzswikc_while_identity_3
xedyzswikc_while_identity_4
xedyzswikc_while_identity_5/
+xedyzswikc_while_xedyzswikc_strided_slice_1k
gxedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensorM
:xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource:	O
<xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource:	 J
;xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource:	A
3xedyzswikc_while_jgtgtymybc_readvariableop_resource: C
5xedyzswikc_while_jgtgtymybc_readvariableop_1_resource: C
5xedyzswikc_while_jgtgtymybc_readvariableop_2_resource: ¢2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp¢1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp¢3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp¢*xedyzswikc/while/jgtgtymybc/ReadVariableOp¢,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1¢,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2Ù
Bxedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bxedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4xedyzswikc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemixedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0xedyzswikc_while_placeholderKxedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4xedyzswikc/while/TensorArrayV2Read/TensorListGetItemä
1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp<xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOpý
"xedyzswikc/while/jgtgtymybc/MatMulMatMul;xedyzswikc/while/TensorArrayV2Read/TensorListGetItem:item:09xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"xedyzswikc/while/jgtgtymybc/MatMulê
3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp>xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOpæ
$xedyzswikc/while/jgtgtymybc/MatMul_1MatMulxedyzswikc_while_placeholder_2;xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$xedyzswikc/while/jgtgtymybc/MatMul_1Ü
xedyzswikc/while/jgtgtymybc/addAddV2,xedyzswikc/while/jgtgtymybc/MatMul:product:0.xedyzswikc/while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
xedyzswikc/while/jgtgtymybc/addã
2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp=xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOpé
#xedyzswikc/while/jgtgtymybc/BiasAddBiasAdd#xedyzswikc/while/jgtgtymybc/add:z:0:xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#xedyzswikc/while/jgtgtymybc/BiasAdd
+xedyzswikc/while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+xedyzswikc/while/jgtgtymybc/split/split_dim¯
!xedyzswikc/while/jgtgtymybc/splitSplit4xedyzswikc/while/jgtgtymybc/split/split_dim:output:0,xedyzswikc/while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!xedyzswikc/while/jgtgtymybc/splitÊ
*xedyzswikc/while/jgtgtymybc/ReadVariableOpReadVariableOp5xedyzswikc_while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*xedyzswikc/while/jgtgtymybc/ReadVariableOpÏ
xedyzswikc/while/jgtgtymybc/mulMul2xedyzswikc/while/jgtgtymybc/ReadVariableOp:value:0xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
xedyzswikc/while/jgtgtymybc/mulÒ
!xedyzswikc/while/jgtgtymybc/add_1AddV2*xedyzswikc/while/jgtgtymybc/split:output:0#xedyzswikc/while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_1®
#xedyzswikc/while/jgtgtymybc/SigmoidSigmoid%xedyzswikc/while/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#xedyzswikc/while/jgtgtymybc/SigmoidÐ
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1ReadVariableOp7xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1Õ
!xedyzswikc/while/jgtgtymybc/mul_1Mul4xedyzswikc/while/jgtgtymybc/ReadVariableOp_1:value:0xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_1Ô
!xedyzswikc/while/jgtgtymybc/add_2AddV2*xedyzswikc/while/jgtgtymybc/split:output:1%xedyzswikc/while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_2²
%xedyzswikc/while/jgtgtymybc/Sigmoid_1Sigmoid%xedyzswikc/while/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%xedyzswikc/while/jgtgtymybc/Sigmoid_1Ê
!xedyzswikc/while/jgtgtymybc/mul_2Mul)xedyzswikc/while/jgtgtymybc/Sigmoid_1:y:0xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_2ª
 xedyzswikc/while/jgtgtymybc/TanhTanh*xedyzswikc/while/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 xedyzswikc/while/jgtgtymybc/TanhÎ
!xedyzswikc/while/jgtgtymybc/mul_3Mul'xedyzswikc/while/jgtgtymybc/Sigmoid:y:0$xedyzswikc/while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_3Ï
!xedyzswikc/while/jgtgtymybc/add_3AddV2%xedyzswikc/while/jgtgtymybc/mul_2:z:0%xedyzswikc/while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_3Ð
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2ReadVariableOp7xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2Ü
!xedyzswikc/while/jgtgtymybc/mul_4Mul4xedyzswikc/while/jgtgtymybc/ReadVariableOp_2:value:0%xedyzswikc/while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_4Ô
!xedyzswikc/while/jgtgtymybc/add_4AddV2*xedyzswikc/while/jgtgtymybc/split:output:3%xedyzswikc/while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/add_4²
%xedyzswikc/while/jgtgtymybc/Sigmoid_2Sigmoid%xedyzswikc/while/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%xedyzswikc/while/jgtgtymybc/Sigmoid_2©
"xedyzswikc/while/jgtgtymybc/Tanh_1Tanh%xedyzswikc/while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"xedyzswikc/while/jgtgtymybc/Tanh_1Ò
!xedyzswikc/while/jgtgtymybc/mul_5Mul)xedyzswikc/while/jgtgtymybc/Sigmoid_2:y:0&xedyzswikc/while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!xedyzswikc/while/jgtgtymybc/mul_5
5xedyzswikc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemxedyzswikc_while_placeholder_1xedyzswikc_while_placeholder%xedyzswikc/while/jgtgtymybc/mul_5:z:0*
_output_shapes
: *
element_dtype027
5xedyzswikc/while/TensorArrayV2Write/TensorListSetItemr
xedyzswikc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
xedyzswikc/while/add/y
xedyzswikc/while/addAddV2xedyzswikc_while_placeholderxedyzswikc/while/add/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/while/addv
xedyzswikc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
xedyzswikc/while/add_1/y­
xedyzswikc/while/add_1AddV2.xedyzswikc_while_xedyzswikc_while_loop_counter!xedyzswikc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/while/add_1©
xedyzswikc/while/IdentityIdentityxedyzswikc/while/add_1:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/IdentityÇ
xedyzswikc/while/Identity_1Identity4xedyzswikc_while_xedyzswikc_while_maximum_iterations3^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/Identity_1«
xedyzswikc/while/Identity_2Identityxedyzswikc/while/add:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/Identity_2Ø
xedyzswikc/while/Identity_3IdentityExedyzswikc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
xedyzswikc/while/Identity_3É
xedyzswikc/while/Identity_4Identity%xedyzswikc/while/jgtgtymybc/mul_5:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/while/Identity_4É
xedyzswikc/while/Identity_5Identity%xedyzswikc/while/jgtgtymybc/add_3:z:03^xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2^xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp4^xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp+^xedyzswikc/while/jgtgtymybc/ReadVariableOp-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_1-^xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/while/Identity_5"?
xedyzswikc_while_identity"xedyzswikc/while/Identity:output:0"C
xedyzswikc_while_identity_1$xedyzswikc/while/Identity_1:output:0"C
xedyzswikc_while_identity_2$xedyzswikc/while/Identity_2:output:0"C
xedyzswikc_while_identity_3$xedyzswikc/while/Identity_3:output:0"C
xedyzswikc_while_identity_4$xedyzswikc/while/Identity_4:output:0"C
xedyzswikc_while_identity_5$xedyzswikc/while/Identity_5:output:0"|
;xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource=xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0"~
<xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource>xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0"z
:xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource<xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0"p
5xedyzswikc_while_jgtgtymybc_readvariableop_1_resource7xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0"p
5xedyzswikc_while_jgtgtymybc_readvariableop_2_resource7xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0"l
3xedyzswikc_while_jgtgtymybc_readvariableop_resource5xedyzswikc_while_jgtgtymybc_readvariableop_resource_0"Ô
gxedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensorixedyzswikc_while_tensorarrayv2read_tensorlistgetitem_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0"\
+xedyzswikc_while_xedyzswikc_strided_slice_1-xedyzswikc_while_xedyzswikc_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2f
1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp1xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp2j
3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp3xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp2X
*xedyzswikc/while/jgtgtymybc/ReadVariableOp*xedyzswikc/while/jgtgtymybc/ReadVariableOp2\
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_1,xedyzswikc/while/jgtgtymybc/ReadVariableOp_12\
,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2,xedyzswikc/while/jgtgtymybc/ReadVariableOp_2: 
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
³
×
"__inference__wrapped_model_1731423

qtjxeibmnqW
Asequential_zdelabjare_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_zdelabjare_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_xedyzswikc_jgtgtymybc_matmul_readvariableop_resource:	T
Asequential_xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource:	 O
@sequential_xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource:	F
8sequential_xedyzswikc_jgtgtymybc_readvariableop_resource: H
:sequential_xedyzswikc_jgtgtymybc_readvariableop_1_resource: H
:sequential_xedyzswikc_jgtgtymybc_readvariableop_2_resource: R
?sequential_ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource:	 T
Asequential_ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource:	 O
@sequential_ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource:	F
8sequential_ksthobzafc_qjjkgjcvpf_readvariableop_resource: H
:sequential_ksthobzafc_qjjkgjcvpf_readvariableop_1_resource: H
:sequential_ksthobzafc_qjjkgjcvpf_readvariableop_2_resource: F
4sequential_vfmtgawzzo_matmul_readvariableop_resource: C
5sequential_vfmtgawzzo_biasadd_readvariableop_resource:
identity¢7sequential/ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp¢6sequential/ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp¢8sequential/ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp¢/sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp¢1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_1¢1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_2¢sequential/ksthobzafc/while¢,sequential/vfmtgawzzo/BiasAdd/ReadVariableOp¢+sequential/vfmtgawzzo/MatMul/ReadVariableOp¢7sequential/xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp¢6sequential/xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp¢8sequential/xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp¢/sequential/xedyzswikc/jgtgtymybc/ReadVariableOp¢1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_1¢1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_2¢sequential/xedyzswikc/while¢8sequential/zdelabjare/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp¥
+sequential/zdelabjare/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/zdelabjare/conv1d/ExpandDims/dimà
'sequential/zdelabjare/conv1d/ExpandDims
ExpandDims
qtjxeibmnq4sequential/zdelabjare/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/zdelabjare/conv1d/ExpandDimsú
8sequential/zdelabjare/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_zdelabjare_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/zdelabjare/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/zdelabjare/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/zdelabjare/conv1d/ExpandDims_1/dim
)sequential/zdelabjare/conv1d/ExpandDims_1
ExpandDims@sequential/zdelabjare/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/zdelabjare/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/zdelabjare/conv1d/ExpandDims_1¨
"sequential/zdelabjare/conv1d/ShapeShape0sequential/zdelabjare/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/zdelabjare/conv1d/Shape®
0sequential/zdelabjare/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/zdelabjare/conv1d/strided_slice/stack»
2sequential/zdelabjare/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/zdelabjare/conv1d/strided_slice/stack_1²
2sequential/zdelabjare/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/zdelabjare/conv1d/strided_slice/stack_2
*sequential/zdelabjare/conv1d/strided_sliceStridedSlice+sequential/zdelabjare/conv1d/Shape:output:09sequential/zdelabjare/conv1d/strided_slice/stack:output:0;sequential/zdelabjare/conv1d/strided_slice/stack_1:output:0;sequential/zdelabjare/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/zdelabjare/conv1d/strided_slice±
*sequential/zdelabjare/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/zdelabjare/conv1d/Reshape/shapeø
$sequential/zdelabjare/conv1d/ReshapeReshape0sequential/zdelabjare/conv1d/ExpandDims:output:03sequential/zdelabjare/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/zdelabjare/conv1d/Reshape
#sequential/zdelabjare/conv1d/Conv2DConv2D-sequential/zdelabjare/conv1d/Reshape:output:02sequential/zdelabjare/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/zdelabjare/conv1d/Conv2D±
,sequential/zdelabjare/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/zdelabjare/conv1d/concat/values_1
(sequential/zdelabjare/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/zdelabjare/conv1d/concat/axis£
#sequential/zdelabjare/conv1d/concatConcatV23sequential/zdelabjare/conv1d/strided_slice:output:05sequential/zdelabjare/conv1d/concat/values_1:output:01sequential/zdelabjare/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/zdelabjare/conv1d/concatõ
&sequential/zdelabjare/conv1d/Reshape_1Reshape,sequential/zdelabjare/conv1d/Conv2D:output:0,sequential/zdelabjare/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/zdelabjare/conv1d/Reshape_1â
$sequential/zdelabjare/conv1d/SqueezeSqueeze/sequential/zdelabjare/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/zdelabjare/conv1d/Squeeze½
.sequential/zdelabjare/squeeze_batch_dims/ShapeShape-sequential/zdelabjare/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/zdelabjare/squeeze_batch_dims/ShapeÆ
<sequential/zdelabjare/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/zdelabjare/squeeze_batch_dims/strided_slice/stackÓ
>sequential/zdelabjare/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/zdelabjare/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/zdelabjare/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/zdelabjare/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/zdelabjare/squeeze_batch_dims/strided_sliceStridedSlice7sequential/zdelabjare/squeeze_batch_dims/Shape:output:0Esequential/zdelabjare/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/zdelabjare/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/zdelabjare/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/zdelabjare/squeeze_batch_dims/strided_sliceÅ
6sequential/zdelabjare/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/zdelabjare/squeeze_batch_dims/Reshape/shape
0sequential/zdelabjare/squeeze_batch_dims/ReshapeReshape-sequential/zdelabjare/conv1d/Squeeze:output:0?sequential/zdelabjare/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/zdelabjare/squeeze_batch_dims/Reshape
?sequential/zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_zdelabjare_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/zdelabjare/squeeze_batch_dims/BiasAddBiasAdd9sequential/zdelabjare/squeeze_batch_dims/Reshape:output:0Gsequential/zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/zdelabjare/squeeze_batch_dims/BiasAddÅ
8sequential/zdelabjare/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/zdelabjare/squeeze_batch_dims/concat/values_1·
4sequential/zdelabjare/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/zdelabjare/squeeze_batch_dims/concat/axisß
/sequential/zdelabjare/squeeze_batch_dims/concatConcatV2?sequential/zdelabjare/squeeze_batch_dims/strided_slice:output:0Asequential/zdelabjare/squeeze_batch_dims/concat/values_1:output:0=sequential/zdelabjare/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/zdelabjare/squeeze_batch_dims/concat¢
2sequential/zdelabjare/squeeze_batch_dims/Reshape_1Reshape9sequential/zdelabjare/squeeze_batch_dims/BiasAdd:output:08sequential/zdelabjare/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/zdelabjare/squeeze_batch_dims/Reshape_1¥
sequential/ojrtxmspqi/ShapeShape;sequential/zdelabjare/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/ojrtxmspqi/Shape 
)sequential/ojrtxmspqi/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/ojrtxmspqi/strided_slice/stack¤
+sequential/ojrtxmspqi/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ojrtxmspqi/strided_slice/stack_1¤
+sequential/ojrtxmspqi/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ojrtxmspqi/strided_slice/stack_2æ
#sequential/ojrtxmspqi/strided_sliceStridedSlice$sequential/ojrtxmspqi/Shape:output:02sequential/ojrtxmspqi/strided_slice/stack:output:04sequential/ojrtxmspqi/strided_slice/stack_1:output:04sequential/ojrtxmspqi/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/ojrtxmspqi/strided_slice
%sequential/ojrtxmspqi/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ojrtxmspqi/Reshape/shape/1
%sequential/ojrtxmspqi/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ojrtxmspqi/Reshape/shape/2
#sequential/ojrtxmspqi/Reshape/shapePack,sequential/ojrtxmspqi/strided_slice:output:0.sequential/ojrtxmspqi/Reshape/shape/1:output:0.sequential/ojrtxmspqi/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/ojrtxmspqi/Reshape/shapeê
sequential/ojrtxmspqi/ReshapeReshape;sequential/zdelabjare/squeeze_batch_dims/Reshape_1:output:0,sequential/ojrtxmspqi/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/ojrtxmspqi/Reshape
sequential/xedyzswikc/ShapeShape&sequential/ojrtxmspqi/Reshape:output:0*
T0*
_output_shapes
:2
sequential/xedyzswikc/Shape 
)sequential/xedyzswikc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/xedyzswikc/strided_slice/stack¤
+sequential/xedyzswikc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xedyzswikc/strided_slice/stack_1¤
+sequential/xedyzswikc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xedyzswikc/strided_slice/stack_2æ
#sequential/xedyzswikc/strided_sliceStridedSlice$sequential/xedyzswikc/Shape:output:02sequential/xedyzswikc/strided_slice/stack:output:04sequential/xedyzswikc/strided_slice/stack_1:output:04sequential/xedyzswikc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/xedyzswikc/strided_slice
!sequential/xedyzswikc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/xedyzswikc/zeros/mul/yÄ
sequential/xedyzswikc/zeros/mulMul,sequential/xedyzswikc/strided_slice:output:0*sequential/xedyzswikc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/xedyzswikc/zeros/mul
"sequential/xedyzswikc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/xedyzswikc/zeros/Less/y¿
 sequential/xedyzswikc/zeros/LessLess#sequential/xedyzswikc/zeros/mul:z:0+sequential/xedyzswikc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/xedyzswikc/zeros/Less
$sequential/xedyzswikc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/xedyzswikc/zeros/packed/1Û
"sequential/xedyzswikc/zeros/packedPack,sequential/xedyzswikc/strided_slice:output:0-sequential/xedyzswikc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/xedyzswikc/zeros/packed
!sequential/xedyzswikc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/xedyzswikc/zeros/ConstÍ
sequential/xedyzswikc/zerosFill+sequential/xedyzswikc/zeros/packed:output:0*sequential/xedyzswikc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/xedyzswikc/zeros
#sequential/xedyzswikc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/xedyzswikc/zeros_1/mul/yÊ
!sequential/xedyzswikc/zeros_1/mulMul,sequential/xedyzswikc/strided_slice:output:0,sequential/xedyzswikc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/xedyzswikc/zeros_1/mul
$sequential/xedyzswikc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/xedyzswikc/zeros_1/Less/yÇ
"sequential/xedyzswikc/zeros_1/LessLess%sequential/xedyzswikc/zeros_1/mul:z:0-sequential/xedyzswikc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/xedyzswikc/zeros_1/Less
&sequential/xedyzswikc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/xedyzswikc/zeros_1/packed/1á
$sequential/xedyzswikc/zeros_1/packedPack,sequential/xedyzswikc/strided_slice:output:0/sequential/xedyzswikc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/xedyzswikc/zeros_1/packed
#sequential/xedyzswikc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/xedyzswikc/zeros_1/ConstÕ
sequential/xedyzswikc/zeros_1Fill-sequential/xedyzswikc/zeros_1/packed:output:0,sequential/xedyzswikc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/xedyzswikc/zeros_1¡
$sequential/xedyzswikc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/xedyzswikc/transpose/permÜ
sequential/xedyzswikc/transpose	Transpose&sequential/ojrtxmspqi/Reshape:output:0-sequential/xedyzswikc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/xedyzswikc/transpose
sequential/xedyzswikc/Shape_1Shape#sequential/xedyzswikc/transpose:y:0*
T0*
_output_shapes
:2
sequential/xedyzswikc/Shape_1¤
+sequential/xedyzswikc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/xedyzswikc/strided_slice_1/stack¨
-sequential/xedyzswikc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/xedyzswikc/strided_slice_1/stack_1¨
-sequential/xedyzswikc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/xedyzswikc/strided_slice_1/stack_2ò
%sequential/xedyzswikc/strided_slice_1StridedSlice&sequential/xedyzswikc/Shape_1:output:04sequential/xedyzswikc/strided_slice_1/stack:output:06sequential/xedyzswikc/strided_slice_1/stack_1:output:06sequential/xedyzswikc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/xedyzswikc/strided_slice_1±
1sequential/xedyzswikc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/xedyzswikc/TensorArrayV2/element_shape
#sequential/xedyzswikc/TensorArrayV2TensorListReserve:sequential/xedyzswikc/TensorArrayV2/element_shape:output:0.sequential/xedyzswikc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/xedyzswikc/TensorArrayV2ë
Ksequential/xedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/xedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/xedyzswikc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/xedyzswikc/transpose:y:0Tsequential/xedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/xedyzswikc/TensorArrayUnstack/TensorListFromTensor¤
+sequential/xedyzswikc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/xedyzswikc/strided_slice_2/stack¨
-sequential/xedyzswikc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/xedyzswikc/strided_slice_2/stack_1¨
-sequential/xedyzswikc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/xedyzswikc/strided_slice_2/stack_2
%sequential/xedyzswikc/strided_slice_2StridedSlice#sequential/xedyzswikc/transpose:y:04sequential/xedyzswikc/strided_slice_2/stack:output:06sequential/xedyzswikc/strided_slice_2/stack_1:output:06sequential/xedyzswikc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/xedyzswikc/strided_slice_2ñ
6sequential/xedyzswikc/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp?sequential_xedyzswikc_jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/xedyzswikc/jgtgtymybc/MatMul/ReadVariableOpÿ
'sequential/xedyzswikc/jgtgtymybc/MatMulMatMul.sequential/xedyzswikc/strided_slice_2:output:0>sequential/xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/xedyzswikc/jgtgtymybc/MatMul÷
8sequential/xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOpAsequential_xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOpû
)sequential/xedyzswikc/jgtgtymybc/MatMul_1MatMul$sequential/xedyzswikc/zeros:output:0@sequential/xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/xedyzswikc/jgtgtymybc/MatMul_1ð
$sequential/xedyzswikc/jgtgtymybc/addAddV21sequential/xedyzswikc/jgtgtymybc/MatMul:product:03sequential/xedyzswikc/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/xedyzswikc/jgtgtymybc/addð
7sequential/xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp@sequential_xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOpý
(sequential/xedyzswikc/jgtgtymybc/BiasAddBiasAdd(sequential/xedyzswikc/jgtgtymybc/add:z:0?sequential/xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/xedyzswikc/jgtgtymybc/BiasAdd¦
0sequential/xedyzswikc/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/xedyzswikc/jgtgtymybc/split/split_dimÃ
&sequential/xedyzswikc/jgtgtymybc/splitSplit9sequential/xedyzswikc/jgtgtymybc/split/split_dim:output:01sequential/xedyzswikc/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/xedyzswikc/jgtgtymybc/split×
/sequential/xedyzswikc/jgtgtymybc/ReadVariableOpReadVariableOp8sequential_xedyzswikc_jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/xedyzswikc/jgtgtymybc/ReadVariableOpæ
$sequential/xedyzswikc/jgtgtymybc/mulMul7sequential/xedyzswikc/jgtgtymybc/ReadVariableOp:value:0&sequential/xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/xedyzswikc/jgtgtymybc/mulæ
&sequential/xedyzswikc/jgtgtymybc/add_1AddV2/sequential/xedyzswikc/jgtgtymybc/split:output:0(sequential/xedyzswikc/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/add_1½
(sequential/xedyzswikc/jgtgtymybc/SigmoidSigmoid*sequential/xedyzswikc/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/xedyzswikc/jgtgtymybc/SigmoidÝ
1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_1ReadVariableOp:sequential_xedyzswikc_jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_1ì
&sequential/xedyzswikc/jgtgtymybc/mul_1Mul9sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_1:value:0&sequential/xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/mul_1è
&sequential/xedyzswikc/jgtgtymybc/add_2AddV2/sequential/xedyzswikc/jgtgtymybc/split:output:1*sequential/xedyzswikc/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/add_2Á
*sequential/xedyzswikc/jgtgtymybc/Sigmoid_1Sigmoid*sequential/xedyzswikc/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/xedyzswikc/jgtgtymybc/Sigmoid_1á
&sequential/xedyzswikc/jgtgtymybc/mul_2Mul.sequential/xedyzswikc/jgtgtymybc/Sigmoid_1:y:0&sequential/xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/mul_2¹
%sequential/xedyzswikc/jgtgtymybc/TanhTanh/sequential/xedyzswikc/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/xedyzswikc/jgtgtymybc/Tanhâ
&sequential/xedyzswikc/jgtgtymybc/mul_3Mul,sequential/xedyzswikc/jgtgtymybc/Sigmoid:y:0)sequential/xedyzswikc/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/mul_3ã
&sequential/xedyzswikc/jgtgtymybc/add_3AddV2*sequential/xedyzswikc/jgtgtymybc/mul_2:z:0*sequential/xedyzswikc/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/add_3Ý
1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_2ReadVariableOp:sequential_xedyzswikc_jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_2ð
&sequential/xedyzswikc/jgtgtymybc/mul_4Mul9sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_2:value:0*sequential/xedyzswikc/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/mul_4è
&sequential/xedyzswikc/jgtgtymybc/add_4AddV2/sequential/xedyzswikc/jgtgtymybc/split:output:3*sequential/xedyzswikc/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/add_4Á
*sequential/xedyzswikc/jgtgtymybc/Sigmoid_2Sigmoid*sequential/xedyzswikc/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/xedyzswikc/jgtgtymybc/Sigmoid_2¸
'sequential/xedyzswikc/jgtgtymybc/Tanh_1Tanh*sequential/xedyzswikc/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/xedyzswikc/jgtgtymybc/Tanh_1æ
&sequential/xedyzswikc/jgtgtymybc/mul_5Mul.sequential/xedyzswikc/jgtgtymybc/Sigmoid_2:y:0+sequential/xedyzswikc/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/jgtgtymybc/mul_5»
3sequential/xedyzswikc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/xedyzswikc/TensorArrayV2_1/element_shape
%sequential/xedyzswikc/TensorArrayV2_1TensorListReserve<sequential/xedyzswikc/TensorArrayV2_1/element_shape:output:0.sequential/xedyzswikc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/xedyzswikc/TensorArrayV2_1z
sequential/xedyzswikc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/xedyzswikc/time«
.sequential/xedyzswikc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/xedyzswikc/while/maximum_iterations
(sequential/xedyzswikc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/xedyzswikc/while/loop_counterø	
sequential/xedyzswikc/whileWhile1sequential/xedyzswikc/while/loop_counter:output:07sequential/xedyzswikc/while/maximum_iterations:output:0#sequential/xedyzswikc/time:output:0.sequential/xedyzswikc/TensorArrayV2_1:handle:0$sequential/xedyzswikc/zeros:output:0&sequential/xedyzswikc/zeros_1:output:0.sequential/xedyzswikc/strided_slice_1:output:0Msequential/xedyzswikc/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_xedyzswikc_jgtgtymybc_matmul_readvariableop_resourceAsequential_xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource@sequential_xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource8sequential_xedyzswikc_jgtgtymybc_readvariableop_resource:sequential_xedyzswikc_jgtgtymybc_readvariableop_1_resource:sequential_xedyzswikc_jgtgtymybc_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*4
body,R*
(sequential_xedyzswikc_while_body_1731140*4
cond,R*
(sequential_xedyzswikc_while_cond_1731139*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/xedyzswikc/whileá
Fsequential/xedyzswikc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/xedyzswikc/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/xedyzswikc/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/xedyzswikc/while:output:3Osequential/xedyzswikc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/xedyzswikc/TensorArrayV2Stack/TensorListStack­
+sequential/xedyzswikc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/xedyzswikc/strided_slice_3/stack¨
-sequential/xedyzswikc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/xedyzswikc/strided_slice_3/stack_1¨
-sequential/xedyzswikc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/xedyzswikc/strided_slice_3/stack_2
%sequential/xedyzswikc/strided_slice_3StridedSliceAsequential/xedyzswikc/TensorArrayV2Stack/TensorListStack:tensor:04sequential/xedyzswikc/strided_slice_3/stack:output:06sequential/xedyzswikc/strided_slice_3/stack_1:output:06sequential/xedyzswikc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/xedyzswikc/strided_slice_3¥
&sequential/xedyzswikc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/xedyzswikc/transpose_1/permý
!sequential/xedyzswikc/transpose_1	TransposeAsequential/xedyzswikc/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/xedyzswikc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/xedyzswikc/transpose_1
sequential/ksthobzafc/ShapeShape%sequential/xedyzswikc/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/ksthobzafc/Shape 
)sequential/ksthobzafc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/ksthobzafc/strided_slice/stack¤
+sequential/ksthobzafc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ksthobzafc/strided_slice/stack_1¤
+sequential/ksthobzafc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ksthobzafc/strided_slice/stack_2æ
#sequential/ksthobzafc/strided_sliceStridedSlice$sequential/ksthobzafc/Shape:output:02sequential/ksthobzafc/strided_slice/stack:output:04sequential/ksthobzafc/strided_slice/stack_1:output:04sequential/ksthobzafc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/ksthobzafc/strided_slice
!sequential/ksthobzafc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/ksthobzafc/zeros/mul/yÄ
sequential/ksthobzafc/zeros/mulMul,sequential/ksthobzafc/strided_slice:output:0*sequential/ksthobzafc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/ksthobzafc/zeros/mul
"sequential/ksthobzafc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/ksthobzafc/zeros/Less/y¿
 sequential/ksthobzafc/zeros/LessLess#sequential/ksthobzafc/zeros/mul:z:0+sequential/ksthobzafc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/ksthobzafc/zeros/Less
$sequential/ksthobzafc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/ksthobzafc/zeros/packed/1Û
"sequential/ksthobzafc/zeros/packedPack,sequential/ksthobzafc/strided_slice:output:0-sequential/ksthobzafc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/ksthobzafc/zeros/packed
!sequential/ksthobzafc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/ksthobzafc/zeros/ConstÍ
sequential/ksthobzafc/zerosFill+sequential/ksthobzafc/zeros/packed:output:0*sequential/ksthobzafc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/ksthobzafc/zeros
#sequential/ksthobzafc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/ksthobzafc/zeros_1/mul/yÊ
!sequential/ksthobzafc/zeros_1/mulMul,sequential/ksthobzafc/strided_slice:output:0,sequential/ksthobzafc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/ksthobzafc/zeros_1/mul
$sequential/ksthobzafc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/ksthobzafc/zeros_1/Less/yÇ
"sequential/ksthobzafc/zeros_1/LessLess%sequential/ksthobzafc/zeros_1/mul:z:0-sequential/ksthobzafc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/ksthobzafc/zeros_1/Less
&sequential/ksthobzafc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/ksthobzafc/zeros_1/packed/1á
$sequential/ksthobzafc/zeros_1/packedPack,sequential/ksthobzafc/strided_slice:output:0/sequential/ksthobzafc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/ksthobzafc/zeros_1/packed
#sequential/ksthobzafc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/ksthobzafc/zeros_1/ConstÕ
sequential/ksthobzafc/zeros_1Fill-sequential/ksthobzafc/zeros_1/packed:output:0,sequential/ksthobzafc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/ksthobzafc/zeros_1¡
$sequential/ksthobzafc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/ksthobzafc/transpose/permÛ
sequential/ksthobzafc/transpose	Transpose%sequential/xedyzswikc/transpose_1:y:0-sequential/ksthobzafc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/ksthobzafc/transpose
sequential/ksthobzafc/Shape_1Shape#sequential/ksthobzafc/transpose:y:0*
T0*
_output_shapes
:2
sequential/ksthobzafc/Shape_1¤
+sequential/ksthobzafc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/ksthobzafc/strided_slice_1/stack¨
-sequential/ksthobzafc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ksthobzafc/strided_slice_1/stack_1¨
-sequential/ksthobzafc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ksthobzafc/strided_slice_1/stack_2ò
%sequential/ksthobzafc/strided_slice_1StridedSlice&sequential/ksthobzafc/Shape_1:output:04sequential/ksthobzafc/strided_slice_1/stack:output:06sequential/ksthobzafc/strided_slice_1/stack_1:output:06sequential/ksthobzafc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/ksthobzafc/strided_slice_1±
1sequential/ksthobzafc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/ksthobzafc/TensorArrayV2/element_shape
#sequential/ksthobzafc/TensorArrayV2TensorListReserve:sequential/ksthobzafc/TensorArrayV2/element_shape:output:0.sequential/ksthobzafc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/ksthobzafc/TensorArrayV2ë
Ksequential/ksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/ksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/ksthobzafc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/ksthobzafc/transpose:y:0Tsequential/ksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/ksthobzafc/TensorArrayUnstack/TensorListFromTensor¤
+sequential/ksthobzafc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/ksthobzafc/strided_slice_2/stack¨
-sequential/ksthobzafc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ksthobzafc/strided_slice_2/stack_1¨
-sequential/ksthobzafc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ksthobzafc/strided_slice_2/stack_2
%sequential/ksthobzafc/strided_slice_2StridedSlice#sequential/ksthobzafc/transpose:y:04sequential/ksthobzafc/strided_slice_2/stack:output:06sequential/ksthobzafc/strided_slice_2/stack_1:output:06sequential/ksthobzafc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/ksthobzafc/strided_slice_2ñ
6sequential/ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp?sequential_ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOpÿ
'sequential/ksthobzafc/qjjkgjcvpf/MatMulMatMul.sequential/ksthobzafc/strided_slice_2:output:0>sequential/ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/ksthobzafc/qjjkgjcvpf/MatMul÷
8sequential/ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOpAsequential_ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOpû
)sequential/ksthobzafc/qjjkgjcvpf/MatMul_1MatMul$sequential/ksthobzafc/zeros:output:0@sequential/ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/ksthobzafc/qjjkgjcvpf/MatMul_1ð
$sequential/ksthobzafc/qjjkgjcvpf/addAddV21sequential/ksthobzafc/qjjkgjcvpf/MatMul:product:03sequential/ksthobzafc/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/ksthobzafc/qjjkgjcvpf/addð
7sequential/ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp@sequential_ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOpý
(sequential/ksthobzafc/qjjkgjcvpf/BiasAddBiasAdd(sequential/ksthobzafc/qjjkgjcvpf/add:z:0?sequential/ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/ksthobzafc/qjjkgjcvpf/BiasAdd¦
0sequential/ksthobzafc/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/ksthobzafc/qjjkgjcvpf/split/split_dimÃ
&sequential/ksthobzafc/qjjkgjcvpf/splitSplit9sequential/ksthobzafc/qjjkgjcvpf/split/split_dim:output:01sequential/ksthobzafc/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/ksthobzafc/qjjkgjcvpf/split×
/sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOpReadVariableOp8sequential_ksthobzafc_qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOpæ
$sequential/ksthobzafc/qjjkgjcvpf/mulMul7sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp:value:0&sequential/ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/ksthobzafc/qjjkgjcvpf/mulæ
&sequential/ksthobzafc/qjjkgjcvpf/add_1AddV2/sequential/ksthobzafc/qjjkgjcvpf/split:output:0(sequential/ksthobzafc/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/add_1½
(sequential/ksthobzafc/qjjkgjcvpf/SigmoidSigmoid*sequential/ksthobzafc/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/ksthobzafc/qjjkgjcvpf/SigmoidÝ
1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp:sequential_ksthobzafc_qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_1ì
&sequential/ksthobzafc/qjjkgjcvpf/mul_1Mul9sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_1:value:0&sequential/ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/mul_1è
&sequential/ksthobzafc/qjjkgjcvpf/add_2AddV2/sequential/ksthobzafc/qjjkgjcvpf/split:output:1*sequential/ksthobzafc/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/add_2Á
*sequential/ksthobzafc/qjjkgjcvpf/Sigmoid_1Sigmoid*sequential/ksthobzafc/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/ksthobzafc/qjjkgjcvpf/Sigmoid_1á
&sequential/ksthobzafc/qjjkgjcvpf/mul_2Mul.sequential/ksthobzafc/qjjkgjcvpf/Sigmoid_1:y:0&sequential/ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/mul_2¹
%sequential/ksthobzafc/qjjkgjcvpf/TanhTanh/sequential/ksthobzafc/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/ksthobzafc/qjjkgjcvpf/Tanhâ
&sequential/ksthobzafc/qjjkgjcvpf/mul_3Mul,sequential/ksthobzafc/qjjkgjcvpf/Sigmoid:y:0)sequential/ksthobzafc/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/mul_3ã
&sequential/ksthobzafc/qjjkgjcvpf/add_3AddV2*sequential/ksthobzafc/qjjkgjcvpf/mul_2:z:0*sequential/ksthobzafc/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/add_3Ý
1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp:sequential_ksthobzafc_qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_2ð
&sequential/ksthobzafc/qjjkgjcvpf/mul_4Mul9sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_2:value:0*sequential/ksthobzafc/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/mul_4è
&sequential/ksthobzafc/qjjkgjcvpf/add_4AddV2/sequential/ksthobzafc/qjjkgjcvpf/split:output:3*sequential/ksthobzafc/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/add_4Á
*sequential/ksthobzafc/qjjkgjcvpf/Sigmoid_2Sigmoid*sequential/ksthobzafc/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/ksthobzafc/qjjkgjcvpf/Sigmoid_2¸
'sequential/ksthobzafc/qjjkgjcvpf/Tanh_1Tanh*sequential/ksthobzafc/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/ksthobzafc/qjjkgjcvpf/Tanh_1æ
&sequential/ksthobzafc/qjjkgjcvpf/mul_5Mul.sequential/ksthobzafc/qjjkgjcvpf/Sigmoid_2:y:0+sequential/ksthobzafc/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/qjjkgjcvpf/mul_5»
3sequential/ksthobzafc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/ksthobzafc/TensorArrayV2_1/element_shape
%sequential/ksthobzafc/TensorArrayV2_1TensorListReserve<sequential/ksthobzafc/TensorArrayV2_1/element_shape:output:0.sequential/ksthobzafc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/ksthobzafc/TensorArrayV2_1z
sequential/ksthobzafc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/ksthobzafc/time«
.sequential/ksthobzafc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/ksthobzafc/while/maximum_iterations
(sequential/ksthobzafc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/ksthobzafc/while/loop_counterø	
sequential/ksthobzafc/whileWhile1sequential/ksthobzafc/while/loop_counter:output:07sequential/ksthobzafc/while/maximum_iterations:output:0#sequential/ksthobzafc/time:output:0.sequential/ksthobzafc/TensorArrayV2_1:handle:0$sequential/ksthobzafc/zeros:output:0&sequential/ksthobzafc/zeros_1:output:0.sequential/ksthobzafc/strided_slice_1:output:0Msequential/ksthobzafc/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resourceAsequential_ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource@sequential_ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource8sequential_ksthobzafc_qjjkgjcvpf_readvariableop_resource:sequential_ksthobzafc_qjjkgjcvpf_readvariableop_1_resource:sequential_ksthobzafc_qjjkgjcvpf_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*4
body,R*
(sequential_ksthobzafc_while_body_1731316*4
cond,R*
(sequential_ksthobzafc_while_cond_1731315*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/ksthobzafc/whileá
Fsequential/ksthobzafc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/ksthobzafc/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/ksthobzafc/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/ksthobzafc/while:output:3Osequential/ksthobzafc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/ksthobzafc/TensorArrayV2Stack/TensorListStack­
+sequential/ksthobzafc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/ksthobzafc/strided_slice_3/stack¨
-sequential/ksthobzafc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/ksthobzafc/strided_slice_3/stack_1¨
-sequential/ksthobzafc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ksthobzafc/strided_slice_3/stack_2
%sequential/ksthobzafc/strided_slice_3StridedSliceAsequential/ksthobzafc/TensorArrayV2Stack/TensorListStack:tensor:04sequential/ksthobzafc/strided_slice_3/stack:output:06sequential/ksthobzafc/strided_slice_3/stack_1:output:06sequential/ksthobzafc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/ksthobzafc/strided_slice_3¥
&sequential/ksthobzafc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/ksthobzafc/transpose_1/permý
!sequential/ksthobzafc/transpose_1	TransposeAsequential/ksthobzafc/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/ksthobzafc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/ksthobzafc/transpose_1Ï
+sequential/vfmtgawzzo/MatMul/ReadVariableOpReadVariableOp4sequential_vfmtgawzzo_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/vfmtgawzzo/MatMul/ReadVariableOpÝ
sequential/vfmtgawzzo/MatMulMatMul.sequential/ksthobzafc/strided_slice_3:output:03sequential/vfmtgawzzo/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/vfmtgawzzo/MatMulÎ
,sequential/vfmtgawzzo/BiasAdd/ReadVariableOpReadVariableOp5sequential_vfmtgawzzo_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/vfmtgawzzo/BiasAdd/ReadVariableOpÙ
sequential/vfmtgawzzo/BiasAddBiasAdd&sequential/vfmtgawzzo/MatMul:product:04sequential/vfmtgawzzo/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/vfmtgawzzo/BiasAdd 
IdentityIdentity&sequential/vfmtgawzzo/BiasAdd:output:08^sequential/ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp7^sequential/ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp9^sequential/ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp0^sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp2^sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_12^sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_2^sequential/ksthobzafc/while-^sequential/vfmtgawzzo/BiasAdd/ReadVariableOp,^sequential/vfmtgawzzo/MatMul/ReadVariableOp8^sequential/xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp7^sequential/xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp9^sequential/xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp0^sequential/xedyzswikc/jgtgtymybc/ReadVariableOp2^sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_12^sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_2^sequential/xedyzswikc/while9^sequential/zdelabjare/conv1d/ExpandDims_1/ReadVariableOp@^sequential/zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2r
7sequential/ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp7sequential/ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp2p
6sequential/ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp6sequential/ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp2t
8sequential/ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp8sequential/ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp2b
/sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp/sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp2f
1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_11sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_12f
1sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_21sequential/ksthobzafc/qjjkgjcvpf/ReadVariableOp_22:
sequential/ksthobzafc/whilesequential/ksthobzafc/while2\
,sequential/vfmtgawzzo/BiasAdd/ReadVariableOp,sequential/vfmtgawzzo/BiasAdd/ReadVariableOp2Z
+sequential/vfmtgawzzo/MatMul/ReadVariableOp+sequential/vfmtgawzzo/MatMul/ReadVariableOp2r
7sequential/xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp7sequential/xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp2p
6sequential/xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp6sequential/xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp2t
8sequential/xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp8sequential/xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp2b
/sequential/xedyzswikc/jgtgtymybc/ReadVariableOp/sequential/xedyzswikc/jgtgtymybc/ReadVariableOp2f
1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_11sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_12f
1sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_21sequential/xedyzswikc/jgtgtymybc/ReadVariableOp_22:
sequential/xedyzswikc/whilesequential/xedyzswikc/while2t
8sequential/zdelabjare/conv1d/ExpandDims_1/ReadVariableOp8sequential/zdelabjare/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
qtjxeibmnq
àY

while_body_1736236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 F
3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 A
2while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	8
*while_qjjkgjcvpf_readvariableop_resource_0: :
,while_qjjkgjcvpf_readvariableop_1_resource_0: :
,while_qjjkgjcvpf_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_qjjkgjcvpf_matmul_readvariableop_resource:	 D
1while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 ?
0while_qjjkgjcvpf_biasadd_readvariableop_resource:	6
(while_qjjkgjcvpf_readvariableop_resource: 8
*while_qjjkgjcvpf_readvariableop_1_resource: 8
*while_qjjkgjcvpf_readvariableop_2_resource: ¢'while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢&while/qjjkgjcvpf/MatMul/ReadVariableOp¢(while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢while/qjjkgjcvpf/ReadVariableOp¢!while/qjjkgjcvpf/ReadVariableOp_1¢!while/qjjkgjcvpf/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp1while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/qjjkgjcvpf/MatMul/ReadVariableOpÑ
while/qjjkgjcvpf/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMulÉ
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpº
while/qjjkgjcvpf/MatMul_1MatMulwhile_placeholder_20while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMul_1°
while/qjjkgjcvpf/addAddV2!while/qjjkgjcvpf/MatMul:product:0#while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/addÂ
'while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp2while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp½
while/qjjkgjcvpf/BiasAddBiasAddwhile/qjjkgjcvpf/add:z:0/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/BiasAdd
 while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/qjjkgjcvpf/split/split_dim
while/qjjkgjcvpf/splitSplit)while/qjjkgjcvpf/split/split_dim:output:0!while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/qjjkgjcvpf/split©
while/qjjkgjcvpf/ReadVariableOpReadVariableOp*while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/qjjkgjcvpf/ReadVariableOp£
while/qjjkgjcvpf/mulMul'while/qjjkgjcvpf/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul¦
while/qjjkgjcvpf/add_1AddV2while/qjjkgjcvpf/split:output:0while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_1
while/qjjkgjcvpf/SigmoidSigmoidwhile/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid¯
!while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp,while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_1©
while/qjjkgjcvpf/mul_1Mul)while/qjjkgjcvpf/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_1¨
while/qjjkgjcvpf/add_2AddV2while/qjjkgjcvpf/split:output:1while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_2
while/qjjkgjcvpf/Sigmoid_1Sigmoidwhile/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_1
while/qjjkgjcvpf/mul_2Mulwhile/qjjkgjcvpf/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_2
while/qjjkgjcvpf/TanhTanhwhile/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh¢
while/qjjkgjcvpf/mul_3Mulwhile/qjjkgjcvpf/Sigmoid:y:0while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_3£
while/qjjkgjcvpf/add_3AddV2while/qjjkgjcvpf/mul_2:z:0while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_3¯
!while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp,while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_2°
while/qjjkgjcvpf/mul_4Mul)while/qjjkgjcvpf/ReadVariableOp_2:value:0while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_4¨
while/qjjkgjcvpf/add_4AddV2while/qjjkgjcvpf/split:output:3while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_4
while/qjjkgjcvpf/Sigmoid_2Sigmoidwhile/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_2
while/qjjkgjcvpf/Tanh_1Tanhwhile/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh_1¦
while/qjjkgjcvpf/mul_5Mulwhile/qjjkgjcvpf/Sigmoid_2:y:0while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/qjjkgjcvpf/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/qjjkgjcvpf/mul_5:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/qjjkgjcvpf/add_3:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_qjjkgjcvpf_biasadd_readvariableop_resource2while_qjjkgjcvpf_biasadd_readvariableop_resource_0"h
1while_qjjkgjcvpf_matmul_1_readvariableop_resource3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"d
/while_qjjkgjcvpf_matmul_readvariableop_resource1while_qjjkgjcvpf_matmul_readvariableop_resource_0"Z
*while_qjjkgjcvpf_readvariableop_1_resource,while_qjjkgjcvpf_readvariableop_1_resource_0"Z
*while_qjjkgjcvpf_readvariableop_2_resource,while_qjjkgjcvpf_readvariableop_2_resource_0"V
(while_qjjkgjcvpf_readvariableop_resource*while_qjjkgjcvpf_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp'while/qjjkgjcvpf/BiasAdd/ReadVariableOp2P
&while/qjjkgjcvpf/MatMul/ReadVariableOp&while/qjjkgjcvpf/MatMul/ReadVariableOp2T
(while/qjjkgjcvpf/MatMul_1/ReadVariableOp(while/qjjkgjcvpf/MatMul_1/ReadVariableOp2B
while/qjjkgjcvpf/ReadVariableOpwhile/qjjkgjcvpf/ReadVariableOp2F
!while/qjjkgjcvpf/ReadVariableOp_1!while/qjjkgjcvpf/ReadVariableOp_12F
!while/qjjkgjcvpf/ReadVariableOp_2!while/qjjkgjcvpf/ReadVariableOp_2: 
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
¹'
µ
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1736850

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


,__inference_sequential_layer_call_fn_1734249

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
G__inference_sequential_layer_call_and_return_conditional_losses_17339762
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

À
,__inference_jgtgtymybc_layer_call_fn_1736739

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

identity_2¢StatefulPartitionedCallì
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
GPU2*0J 8 *P
fKRI
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_17315102
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
ë

,__inference_ksthobzafc_layer_call_fn_1735926
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall°
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17323682
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
àY

while_body_1735268
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jgtgtymybc_matmul_readvariableop_resource_0:	F
3while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 A
2while_jgtgtymybc_biasadd_readvariableop_resource_0:	8
*while_jgtgtymybc_readvariableop_resource_0: :
,while_jgtgtymybc_readvariableop_1_resource_0: :
,while_jgtgtymybc_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jgtgtymybc_matmul_readvariableop_resource:	D
1while_jgtgtymybc_matmul_1_readvariableop_resource:	 ?
0while_jgtgtymybc_biasadd_readvariableop_resource:	6
(while_jgtgtymybc_readvariableop_resource: 8
*while_jgtgtymybc_readvariableop_1_resource: 8
*while_jgtgtymybc_readvariableop_2_resource: ¢'while/jgtgtymybc/BiasAdd/ReadVariableOp¢&while/jgtgtymybc/MatMul/ReadVariableOp¢(while/jgtgtymybc/MatMul_1/ReadVariableOp¢while/jgtgtymybc/ReadVariableOp¢!while/jgtgtymybc/ReadVariableOp_1¢!while/jgtgtymybc/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp1while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jgtgtymybc/MatMul/ReadVariableOpÑ
while/jgtgtymybc/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMulÉ
(while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp3while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jgtgtymybc/MatMul_1/ReadVariableOpº
while/jgtgtymybc/MatMul_1MatMulwhile_placeholder_20while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMul_1°
while/jgtgtymybc/addAddV2!while/jgtgtymybc/MatMul:product:0#while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/addÂ
'while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp2while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jgtgtymybc/BiasAdd/ReadVariableOp½
while/jgtgtymybc/BiasAddBiasAddwhile/jgtgtymybc/add:z:0/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/BiasAdd
 while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jgtgtymybc/split/split_dim
while/jgtgtymybc/splitSplit)while/jgtgtymybc/split/split_dim:output:0!while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jgtgtymybc/split©
while/jgtgtymybc/ReadVariableOpReadVariableOp*while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jgtgtymybc/ReadVariableOp£
while/jgtgtymybc/mulMul'while/jgtgtymybc/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul¦
while/jgtgtymybc/add_1AddV2while/jgtgtymybc/split:output:0while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_1
while/jgtgtymybc/SigmoidSigmoidwhile/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid¯
!while/jgtgtymybc/ReadVariableOp_1ReadVariableOp,while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_1©
while/jgtgtymybc/mul_1Mul)while/jgtgtymybc/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_1¨
while/jgtgtymybc/add_2AddV2while/jgtgtymybc/split:output:1while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_2
while/jgtgtymybc/Sigmoid_1Sigmoidwhile/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_1
while/jgtgtymybc/mul_2Mulwhile/jgtgtymybc/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_2
while/jgtgtymybc/TanhTanhwhile/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh¢
while/jgtgtymybc/mul_3Mulwhile/jgtgtymybc/Sigmoid:y:0while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_3£
while/jgtgtymybc/add_3AddV2while/jgtgtymybc/mul_2:z:0while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_3¯
!while/jgtgtymybc/ReadVariableOp_2ReadVariableOp,while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_2°
while/jgtgtymybc/mul_4Mul)while/jgtgtymybc/ReadVariableOp_2:value:0while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_4¨
while/jgtgtymybc/add_4AddV2while/jgtgtymybc/split:output:3while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_4
while/jgtgtymybc/Sigmoid_2Sigmoidwhile/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_2
while/jgtgtymybc/Tanh_1Tanhwhile/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh_1¦
while/jgtgtymybc/mul_5Mulwhile/jgtgtymybc/Sigmoid_2:y:0while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jgtgtymybc/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jgtgtymybc/mul_5:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jgtgtymybc/add_3:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_jgtgtymybc_biasadd_readvariableop_resource2while_jgtgtymybc_biasadd_readvariableop_resource_0"h
1while_jgtgtymybc_matmul_1_readvariableop_resource3while_jgtgtymybc_matmul_1_readvariableop_resource_0"d
/while_jgtgtymybc_matmul_readvariableop_resource1while_jgtgtymybc_matmul_readvariableop_resource_0"Z
*while_jgtgtymybc_readvariableop_1_resource,while_jgtgtymybc_readvariableop_1_resource_0"Z
*while_jgtgtymybc_readvariableop_2_resource,while_jgtgtymybc_readvariableop_2_resource_0"V
(while_jgtgtymybc_readvariableop_resource*while_jgtgtymybc_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jgtgtymybc/BiasAdd/ReadVariableOp'while/jgtgtymybc/BiasAdd/ReadVariableOp2P
&while/jgtgtymybc/MatMul/ReadVariableOp&while/jgtgtymybc/MatMul/ReadVariableOp2T
(while/jgtgtymybc/MatMul_1/ReadVariableOp(while/jgtgtymybc/MatMul_1/ReadVariableOp2B
while/jgtgtymybc/ReadVariableOpwhile/jgtgtymybc/ReadVariableOp2F
!while/jgtgtymybc/ReadVariableOp_1!while/jgtgtymybc/ReadVariableOp_12F
!while/jgtgtymybc/ReadVariableOp_2!while/jgtgtymybc/ReadVariableOp_2: 
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
while_cond_1732550
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1732550___redundant_placeholder05
1while_while_cond_1732550___redundant_placeholder15
1while_while_cond_1732550___redundant_placeholder25
1while_while_cond_1732550___redundant_placeholder35
1while_while_cond_1732550___redundant_placeholder45
1while_while_cond_1732550___redundant_placeholder55
1while_while_cond_1732550___redundant_placeholder6
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
àY

while_body_1735448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jgtgtymybc_matmul_readvariableop_resource_0:	F
3while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 A
2while_jgtgtymybc_biasadd_readvariableop_resource_0:	8
*while_jgtgtymybc_readvariableop_resource_0: :
,while_jgtgtymybc_readvariableop_1_resource_0: :
,while_jgtgtymybc_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jgtgtymybc_matmul_readvariableop_resource:	D
1while_jgtgtymybc_matmul_1_readvariableop_resource:	 ?
0while_jgtgtymybc_biasadd_readvariableop_resource:	6
(while_jgtgtymybc_readvariableop_resource: 8
*while_jgtgtymybc_readvariableop_1_resource: 8
*while_jgtgtymybc_readvariableop_2_resource: ¢'while/jgtgtymybc/BiasAdd/ReadVariableOp¢&while/jgtgtymybc/MatMul/ReadVariableOp¢(while/jgtgtymybc/MatMul_1/ReadVariableOp¢while/jgtgtymybc/ReadVariableOp¢!while/jgtgtymybc/ReadVariableOp_1¢!while/jgtgtymybc/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp1while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jgtgtymybc/MatMul/ReadVariableOpÑ
while/jgtgtymybc/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMulÉ
(while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp3while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jgtgtymybc/MatMul_1/ReadVariableOpº
while/jgtgtymybc/MatMul_1MatMulwhile_placeholder_20while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMul_1°
while/jgtgtymybc/addAddV2!while/jgtgtymybc/MatMul:product:0#while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/addÂ
'while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp2while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jgtgtymybc/BiasAdd/ReadVariableOp½
while/jgtgtymybc/BiasAddBiasAddwhile/jgtgtymybc/add:z:0/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/BiasAdd
 while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jgtgtymybc/split/split_dim
while/jgtgtymybc/splitSplit)while/jgtgtymybc/split/split_dim:output:0!while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jgtgtymybc/split©
while/jgtgtymybc/ReadVariableOpReadVariableOp*while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jgtgtymybc/ReadVariableOp£
while/jgtgtymybc/mulMul'while/jgtgtymybc/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul¦
while/jgtgtymybc/add_1AddV2while/jgtgtymybc/split:output:0while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_1
while/jgtgtymybc/SigmoidSigmoidwhile/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid¯
!while/jgtgtymybc/ReadVariableOp_1ReadVariableOp,while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_1©
while/jgtgtymybc/mul_1Mul)while/jgtgtymybc/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_1¨
while/jgtgtymybc/add_2AddV2while/jgtgtymybc/split:output:1while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_2
while/jgtgtymybc/Sigmoid_1Sigmoidwhile/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_1
while/jgtgtymybc/mul_2Mulwhile/jgtgtymybc/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_2
while/jgtgtymybc/TanhTanhwhile/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh¢
while/jgtgtymybc/mul_3Mulwhile/jgtgtymybc/Sigmoid:y:0while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_3£
while/jgtgtymybc/add_3AddV2while/jgtgtymybc/mul_2:z:0while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_3¯
!while/jgtgtymybc/ReadVariableOp_2ReadVariableOp,while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_2°
while/jgtgtymybc/mul_4Mul)while/jgtgtymybc/ReadVariableOp_2:value:0while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_4¨
while/jgtgtymybc/add_4AddV2while/jgtgtymybc/split:output:3while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_4
while/jgtgtymybc/Sigmoid_2Sigmoidwhile/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_2
while/jgtgtymybc/Tanh_1Tanhwhile/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh_1¦
while/jgtgtymybc/mul_5Mulwhile/jgtgtymybc/Sigmoid_2:y:0while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jgtgtymybc/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jgtgtymybc/mul_5:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jgtgtymybc/add_3:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_jgtgtymybc_biasadd_readvariableop_resource2while_jgtgtymybc_biasadd_readvariableop_resource_0"h
1while_jgtgtymybc_matmul_1_readvariableop_resource3while_jgtgtymybc_matmul_1_readvariableop_resource_0"d
/while_jgtgtymybc_matmul_readvariableop_resource1while_jgtgtymybc_matmul_readvariableop_resource_0"Z
*while_jgtgtymybc_readvariableop_1_resource,while_jgtgtymybc_readvariableop_1_resource_0"Z
*while_jgtgtymybc_readvariableop_2_resource,while_jgtgtymybc_readvariableop_2_resource_0"V
(while_jgtgtymybc_readvariableop_resource*while_jgtgtymybc_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jgtgtymybc/BiasAdd/ReadVariableOp'while/jgtgtymybc/BiasAdd/ReadVariableOp2P
&while/jgtgtymybc/MatMul/ReadVariableOp&while/jgtgtymybc/MatMul/ReadVariableOp2T
(while/jgtgtymybc/MatMul_1/ReadVariableOp(while/jgtgtymybc/MatMul_1/ReadVariableOp2B
while/jgtgtymybc/ReadVariableOpwhile/jgtgtymybc/ReadVariableOp2F
!while/jgtgtymybc/ReadVariableOp_1!while/jgtgtymybc/ReadVariableOp_12F
!while/jgtgtymybc/ReadVariableOp_2!while/jgtgtymybc/ReadVariableOp_2: 
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

À
,__inference_qjjkgjcvpf_layer_call_fn_1736896

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

identity_2¢StatefulPartitionedCallì
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
GPU2*0J 8 *P
fKRI
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_17324552
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
ç)
Ò
while_body_1731793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_jgtgtymybc_1731817_0:	-
while_jgtgtymybc_1731819_0:	 )
while_jgtgtymybc_1731821_0:	(
while_jgtgtymybc_1731823_0: (
while_jgtgtymybc_1731825_0: (
while_jgtgtymybc_1731827_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_jgtgtymybc_1731817:	+
while_jgtgtymybc_1731819:	 '
while_jgtgtymybc_1731821:	&
while_jgtgtymybc_1731823: &
while_jgtgtymybc_1731825: &
while_jgtgtymybc_1731827: ¢(while/jgtgtymybc/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¶
(while/jgtgtymybc/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_jgtgtymybc_1731817_0while_jgtgtymybc_1731819_0while_jgtgtymybc_1731821_0while_jgtgtymybc_1731823_0while_jgtgtymybc_1731825_0while_jgtgtymybc_1731827_0*
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
GPU2*0J 8 *P
fKRI
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_17316972*
(while/jgtgtymybc/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/jgtgtymybc/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/jgtgtymybc/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/jgtgtymybc/StatefulPartitionedCall:output:1)^while/jgtgtymybc/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/jgtgtymybc/StatefulPartitionedCall:output:2)^while/jgtgtymybc/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_jgtgtymybc_1731817while_jgtgtymybc_1731817_0"6
while_jgtgtymybc_1731819while_jgtgtymybc_1731819_0"6
while_jgtgtymybc_1731821while_jgtgtymybc_1731821_0"6
while_jgtgtymybc_1731823while_jgtgtymybc_1731823_0"6
while_jgtgtymybc_1731825while_jgtgtymybc_1731825_0"6
while_jgtgtymybc_1731827while_jgtgtymybc_1731827_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/jgtgtymybc/StatefulPartitionedCall(while/jgtgtymybc/StatefulPartitionedCall: 
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

c
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_1735121

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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1734653

inputsL
6zdelabjare_conv1d_expanddims_1_readvariableop_resource:K
=zdelabjare_squeeze_batch_dims_biasadd_readvariableop_resource:G
4xedyzswikc_jgtgtymybc_matmul_readvariableop_resource:	I
6xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource:	 D
5xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource:	;
-xedyzswikc_jgtgtymybc_readvariableop_resource: =
/xedyzswikc_jgtgtymybc_readvariableop_1_resource: =
/xedyzswikc_jgtgtymybc_readvariableop_2_resource: G
4ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource:	 I
6ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource:	 D
5ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource:	;
-ksthobzafc_qjjkgjcvpf_readvariableop_resource: =
/ksthobzafc_qjjkgjcvpf_readvariableop_1_resource: =
/ksthobzafc_qjjkgjcvpf_readvariableop_2_resource: ;
)vfmtgawzzo_matmul_readvariableop_resource: 8
*vfmtgawzzo_biasadd_readvariableop_resource:
identity¢,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp¢+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp¢-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp¢$ksthobzafc/qjjkgjcvpf/ReadVariableOp¢&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1¢&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2¢ksthobzafc/while¢!vfmtgawzzo/BiasAdd/ReadVariableOp¢ vfmtgawzzo/MatMul/ReadVariableOp¢,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp¢+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp¢-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp¢$xedyzswikc/jgtgtymybc/ReadVariableOp¢&xedyzswikc/jgtgtymybc/ReadVariableOp_1¢&xedyzswikc/jgtgtymybc/ReadVariableOp_2¢xedyzswikc/while¢-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp¢4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp
 zdelabjare/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 zdelabjare/conv1d/ExpandDims/dim»
zdelabjare/conv1d/ExpandDims
ExpandDimsinputs)zdelabjare/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/ExpandDimsÙ
-zdelabjare/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6zdelabjare_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp
"zdelabjare/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"zdelabjare/conv1d/ExpandDims_1/dimã
zdelabjare/conv1d/ExpandDims_1
ExpandDims5zdelabjare/conv1d/ExpandDims_1/ReadVariableOp:value:0+zdelabjare/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
zdelabjare/conv1d/ExpandDims_1
zdelabjare/conv1d/ShapeShape%zdelabjare/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
zdelabjare/conv1d/Shape
%zdelabjare/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%zdelabjare/conv1d/strided_slice/stack¥
'zdelabjare/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'zdelabjare/conv1d/strided_slice/stack_1
'zdelabjare/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'zdelabjare/conv1d/strided_slice/stack_2Ì
zdelabjare/conv1d/strided_sliceStridedSlice zdelabjare/conv1d/Shape:output:0.zdelabjare/conv1d/strided_slice/stack:output:00zdelabjare/conv1d/strided_slice/stack_1:output:00zdelabjare/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
zdelabjare/conv1d/strided_slice
zdelabjare/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
zdelabjare/conv1d/Reshape/shapeÌ
zdelabjare/conv1d/ReshapeReshape%zdelabjare/conv1d/ExpandDims:output:0(zdelabjare/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/Reshapeî
zdelabjare/conv1d/Conv2DConv2D"zdelabjare/conv1d/Reshape:output:0'zdelabjare/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
zdelabjare/conv1d/Conv2D
!zdelabjare/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!zdelabjare/conv1d/concat/values_1
zdelabjare/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/concat/axisì
zdelabjare/conv1d/concatConcatV2(zdelabjare/conv1d/strided_slice:output:0*zdelabjare/conv1d/concat/values_1:output:0&zdelabjare/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
zdelabjare/conv1d/concatÉ
zdelabjare/conv1d/Reshape_1Reshape!zdelabjare/conv1d/Conv2D:output:0!zdelabjare/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/Reshape_1Á
zdelabjare/conv1d/SqueezeSqueeze$zdelabjare/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/Squeeze
#zdelabjare/squeeze_batch_dims/ShapeShape"zdelabjare/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#zdelabjare/squeeze_batch_dims/Shape°
1zdelabjare/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1zdelabjare/squeeze_batch_dims/strided_slice/stack½
3zdelabjare/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3zdelabjare/squeeze_batch_dims/strided_slice/stack_1´
3zdelabjare/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3zdelabjare/squeeze_batch_dims/strided_slice/stack_2
+zdelabjare/squeeze_batch_dims/strided_sliceStridedSlice,zdelabjare/squeeze_batch_dims/Shape:output:0:zdelabjare/squeeze_batch_dims/strided_slice/stack:output:0<zdelabjare/squeeze_batch_dims/strided_slice/stack_1:output:0<zdelabjare/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+zdelabjare/squeeze_batch_dims/strided_slice¯
+zdelabjare/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+zdelabjare/squeeze_batch_dims/Reshape/shapeé
%zdelabjare/squeeze_batch_dims/ReshapeReshape"zdelabjare/conv1d/Squeeze:output:04zdelabjare/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%zdelabjare/squeeze_batch_dims/Reshapeæ
4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=zdelabjare_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%zdelabjare/squeeze_batch_dims/BiasAddBiasAdd.zdelabjare/squeeze_batch_dims/Reshape:output:0<zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%zdelabjare/squeeze_batch_dims/BiasAdd¯
-zdelabjare/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-zdelabjare/squeeze_batch_dims/concat/values_1¡
)zdelabjare/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)zdelabjare/squeeze_batch_dims/concat/axis¨
$zdelabjare/squeeze_batch_dims/concatConcatV24zdelabjare/squeeze_batch_dims/strided_slice:output:06zdelabjare/squeeze_batch_dims/concat/values_1:output:02zdelabjare/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$zdelabjare/squeeze_batch_dims/concatö
'zdelabjare/squeeze_batch_dims/Reshape_1Reshape.zdelabjare/squeeze_batch_dims/BiasAdd:output:0-zdelabjare/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'zdelabjare/squeeze_batch_dims/Reshape_1
ojrtxmspqi/ShapeShape0zdelabjare/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ojrtxmspqi/Shape
ojrtxmspqi/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ojrtxmspqi/strided_slice/stack
 ojrtxmspqi/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojrtxmspqi/strided_slice/stack_1
 ojrtxmspqi/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojrtxmspqi/strided_slice/stack_2¤
ojrtxmspqi/strided_sliceStridedSliceojrtxmspqi/Shape:output:0'ojrtxmspqi/strided_slice/stack:output:0)ojrtxmspqi/strided_slice/stack_1:output:0)ojrtxmspqi/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ojrtxmspqi/strided_slicez
ojrtxmspqi/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ojrtxmspqi/Reshape/shape/1z
ojrtxmspqi/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ojrtxmspqi/Reshape/shape/2×
ojrtxmspqi/Reshape/shapePack!ojrtxmspqi/strided_slice:output:0#ojrtxmspqi/Reshape/shape/1:output:0#ojrtxmspqi/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ojrtxmspqi/Reshape/shape¾
ojrtxmspqi/ReshapeReshape0zdelabjare/squeeze_batch_dims/Reshape_1:output:0!ojrtxmspqi/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ojrtxmspqi/Reshapeo
xedyzswikc/ShapeShapeojrtxmspqi/Reshape:output:0*
T0*
_output_shapes
:2
xedyzswikc/Shape
xedyzswikc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xedyzswikc/strided_slice/stack
 xedyzswikc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xedyzswikc/strided_slice/stack_1
 xedyzswikc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xedyzswikc/strided_slice/stack_2¤
xedyzswikc/strided_sliceStridedSlicexedyzswikc/Shape:output:0'xedyzswikc/strided_slice/stack:output:0)xedyzswikc/strided_slice/stack_1:output:0)xedyzswikc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xedyzswikc/strided_slicer
xedyzswikc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros/mul/y
xedyzswikc/zeros/mulMul!xedyzswikc/strided_slice:output:0xedyzswikc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros/mulu
xedyzswikc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
xedyzswikc/zeros/Less/y
xedyzswikc/zeros/LessLessxedyzswikc/zeros/mul:z:0 xedyzswikc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros/Lessx
xedyzswikc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros/packed/1¯
xedyzswikc/zeros/packedPack!xedyzswikc/strided_slice:output:0"xedyzswikc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
xedyzswikc/zeros/packedu
xedyzswikc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xedyzswikc/zeros/Const¡
xedyzswikc/zerosFill xedyzswikc/zeros/packed:output:0xedyzswikc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/zerosv
xedyzswikc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros_1/mul/y
xedyzswikc/zeros_1/mulMul!xedyzswikc/strided_slice:output:0!xedyzswikc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros_1/muly
xedyzswikc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
xedyzswikc/zeros_1/Less/y
xedyzswikc/zeros_1/LessLessxedyzswikc/zeros_1/mul:z:0"xedyzswikc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros_1/Less|
xedyzswikc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros_1/packed/1µ
xedyzswikc/zeros_1/packedPack!xedyzswikc/strided_slice:output:0$xedyzswikc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
xedyzswikc/zeros_1/packedy
xedyzswikc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xedyzswikc/zeros_1/Const©
xedyzswikc/zeros_1Fill"xedyzswikc/zeros_1/packed:output:0!xedyzswikc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/zeros_1
xedyzswikc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
xedyzswikc/transpose/perm°
xedyzswikc/transpose	Transposeojrtxmspqi/Reshape:output:0"xedyzswikc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/transposep
xedyzswikc/Shape_1Shapexedyzswikc/transpose:y:0*
T0*
_output_shapes
:2
xedyzswikc/Shape_1
 xedyzswikc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 xedyzswikc/strided_slice_1/stack
"xedyzswikc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_1/stack_1
"xedyzswikc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_1/stack_2°
xedyzswikc/strided_slice_1StridedSlicexedyzswikc/Shape_1:output:0)xedyzswikc/strided_slice_1/stack:output:0+xedyzswikc/strided_slice_1/stack_1:output:0+xedyzswikc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xedyzswikc/strided_slice_1
&xedyzswikc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&xedyzswikc/TensorArrayV2/element_shapeÞ
xedyzswikc/TensorArrayV2TensorListReserve/xedyzswikc/TensorArrayV2/element_shape:output:0#xedyzswikc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
xedyzswikc/TensorArrayV2Õ
@xedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@xedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2xedyzswikc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorxedyzswikc/transpose:y:0Ixedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2xedyzswikc/TensorArrayUnstack/TensorListFromTensor
 xedyzswikc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 xedyzswikc/strided_slice_2/stack
"xedyzswikc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_2/stack_1
"xedyzswikc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_2/stack_2¾
xedyzswikc/strided_slice_2StridedSlicexedyzswikc/transpose:y:0)xedyzswikc/strided_slice_2/stack:output:0+xedyzswikc/strided_slice_2/stack_1:output:0+xedyzswikc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
xedyzswikc/strided_slice_2Ð
+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp4xedyzswikc_jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOpÓ
xedyzswikc/jgtgtymybc/MatMulMatMul#xedyzswikc/strided_slice_2:output:03xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/jgtgtymybc/MatMulÖ
-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp6xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOpÏ
xedyzswikc/jgtgtymybc/MatMul_1MatMulxedyzswikc/zeros:output:05xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
xedyzswikc/jgtgtymybc/MatMul_1Ä
xedyzswikc/jgtgtymybc/addAddV2&xedyzswikc/jgtgtymybc/MatMul:product:0(xedyzswikc/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/jgtgtymybc/addÏ
,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp5xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOpÑ
xedyzswikc/jgtgtymybc/BiasAddBiasAddxedyzswikc/jgtgtymybc/add:z:04xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/jgtgtymybc/BiasAdd
%xedyzswikc/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%xedyzswikc/jgtgtymybc/split/split_dim
xedyzswikc/jgtgtymybc/splitSplit.xedyzswikc/jgtgtymybc/split/split_dim:output:0&xedyzswikc/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
xedyzswikc/jgtgtymybc/split¶
$xedyzswikc/jgtgtymybc/ReadVariableOpReadVariableOp-xedyzswikc_jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02&
$xedyzswikc/jgtgtymybc/ReadVariableOpº
xedyzswikc/jgtgtymybc/mulMul,xedyzswikc/jgtgtymybc/ReadVariableOp:value:0xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mulº
xedyzswikc/jgtgtymybc/add_1AddV2$xedyzswikc/jgtgtymybc/split:output:0xedyzswikc/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_1
xedyzswikc/jgtgtymybc/SigmoidSigmoidxedyzswikc/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/Sigmoid¼
&xedyzswikc/jgtgtymybc/ReadVariableOp_1ReadVariableOp/xedyzswikc_jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&xedyzswikc/jgtgtymybc/ReadVariableOp_1À
xedyzswikc/jgtgtymybc/mul_1Mul.xedyzswikc/jgtgtymybc/ReadVariableOp_1:value:0xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_1¼
xedyzswikc/jgtgtymybc/add_2AddV2$xedyzswikc/jgtgtymybc/split:output:1xedyzswikc/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_2 
xedyzswikc/jgtgtymybc/Sigmoid_1Sigmoidxedyzswikc/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
xedyzswikc/jgtgtymybc/Sigmoid_1µ
xedyzswikc/jgtgtymybc/mul_2Mul#xedyzswikc/jgtgtymybc/Sigmoid_1:y:0xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_2
xedyzswikc/jgtgtymybc/TanhTanh$xedyzswikc/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/Tanh¶
xedyzswikc/jgtgtymybc/mul_3Mul!xedyzswikc/jgtgtymybc/Sigmoid:y:0xedyzswikc/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_3·
xedyzswikc/jgtgtymybc/add_3AddV2xedyzswikc/jgtgtymybc/mul_2:z:0xedyzswikc/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_3¼
&xedyzswikc/jgtgtymybc/ReadVariableOp_2ReadVariableOp/xedyzswikc_jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&xedyzswikc/jgtgtymybc/ReadVariableOp_2Ä
xedyzswikc/jgtgtymybc/mul_4Mul.xedyzswikc/jgtgtymybc/ReadVariableOp_2:value:0xedyzswikc/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_4¼
xedyzswikc/jgtgtymybc/add_4AddV2$xedyzswikc/jgtgtymybc/split:output:3xedyzswikc/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_4 
xedyzswikc/jgtgtymybc/Sigmoid_2Sigmoidxedyzswikc/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
xedyzswikc/jgtgtymybc/Sigmoid_2
xedyzswikc/jgtgtymybc/Tanh_1Tanhxedyzswikc/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/Tanh_1º
xedyzswikc/jgtgtymybc/mul_5Mul#xedyzswikc/jgtgtymybc/Sigmoid_2:y:0 xedyzswikc/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_5¥
(xedyzswikc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(xedyzswikc/TensorArrayV2_1/element_shapeä
xedyzswikc/TensorArrayV2_1TensorListReserve1xedyzswikc/TensorArrayV2_1/element_shape:output:0#xedyzswikc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
xedyzswikc/TensorArrayV2_1d
xedyzswikc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/time
#xedyzswikc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#xedyzswikc/while/maximum_iterations
xedyzswikc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/while/loop_counter²
xedyzswikc/whileWhile&xedyzswikc/while/loop_counter:output:0,xedyzswikc/while/maximum_iterations:output:0xedyzswikc/time:output:0#xedyzswikc/TensorArrayV2_1:handle:0xedyzswikc/zeros:output:0xedyzswikc/zeros_1:output:0#xedyzswikc/strided_slice_1:output:0Bxedyzswikc/TensorArrayUnstack/TensorListFromTensor:output_handle:04xedyzswikc_jgtgtymybc_matmul_readvariableop_resource6xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource5xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource-xedyzswikc_jgtgtymybc_readvariableop_resource/xedyzswikc_jgtgtymybc_readvariableop_1_resource/xedyzswikc_jgtgtymybc_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*)
body!R
xedyzswikc_while_body_1734370*)
cond!R
xedyzswikc_while_cond_1734369*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
xedyzswikc/whileË
;xedyzswikc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;xedyzswikc/TensorArrayV2Stack/TensorListStack/element_shape
-xedyzswikc/TensorArrayV2Stack/TensorListStackTensorListStackxedyzswikc/while:output:3Dxedyzswikc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-xedyzswikc/TensorArrayV2Stack/TensorListStack
 xedyzswikc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 xedyzswikc/strided_slice_3/stack
"xedyzswikc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"xedyzswikc/strided_slice_3/stack_1
"xedyzswikc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_3/stack_2Ü
xedyzswikc/strided_slice_3StridedSlice6xedyzswikc/TensorArrayV2Stack/TensorListStack:tensor:0)xedyzswikc/strided_slice_3/stack:output:0+xedyzswikc/strided_slice_3/stack_1:output:0+xedyzswikc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
xedyzswikc/strided_slice_3
xedyzswikc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
xedyzswikc/transpose_1/permÑ
xedyzswikc/transpose_1	Transpose6xedyzswikc/TensorArrayV2Stack/TensorListStack:tensor:0$xedyzswikc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/transpose_1n
ksthobzafc/ShapeShapexedyzswikc/transpose_1:y:0*
T0*
_output_shapes
:2
ksthobzafc/Shape
ksthobzafc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ksthobzafc/strided_slice/stack
 ksthobzafc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ksthobzafc/strided_slice/stack_1
 ksthobzafc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ksthobzafc/strided_slice/stack_2¤
ksthobzafc/strided_sliceStridedSliceksthobzafc/Shape:output:0'ksthobzafc/strided_slice/stack:output:0)ksthobzafc/strided_slice/stack_1:output:0)ksthobzafc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ksthobzafc/strided_slicer
ksthobzafc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros/mul/y
ksthobzafc/zeros/mulMul!ksthobzafc/strided_slice:output:0ksthobzafc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros/mulu
ksthobzafc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ksthobzafc/zeros/Less/y
ksthobzafc/zeros/LessLessksthobzafc/zeros/mul:z:0 ksthobzafc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros/Lessx
ksthobzafc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros/packed/1¯
ksthobzafc/zeros/packedPack!ksthobzafc/strided_slice:output:0"ksthobzafc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
ksthobzafc/zeros/packedu
ksthobzafc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ksthobzafc/zeros/Const¡
ksthobzafc/zerosFill ksthobzafc/zeros/packed:output:0ksthobzafc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/zerosv
ksthobzafc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros_1/mul/y
ksthobzafc/zeros_1/mulMul!ksthobzafc/strided_slice:output:0!ksthobzafc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros_1/muly
ksthobzafc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ksthobzafc/zeros_1/Less/y
ksthobzafc/zeros_1/LessLessksthobzafc/zeros_1/mul:z:0"ksthobzafc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros_1/Less|
ksthobzafc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros_1/packed/1µ
ksthobzafc/zeros_1/packedPack!ksthobzafc/strided_slice:output:0$ksthobzafc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
ksthobzafc/zeros_1/packedy
ksthobzafc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ksthobzafc/zeros_1/Const©
ksthobzafc/zeros_1Fill"ksthobzafc/zeros_1/packed:output:0!ksthobzafc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/zeros_1
ksthobzafc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ksthobzafc/transpose/perm¯
ksthobzafc/transpose	Transposexedyzswikc/transpose_1:y:0"ksthobzafc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/transposep
ksthobzafc/Shape_1Shapeksthobzafc/transpose:y:0*
T0*
_output_shapes
:2
ksthobzafc/Shape_1
 ksthobzafc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ksthobzafc/strided_slice_1/stack
"ksthobzafc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_1/stack_1
"ksthobzafc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_1/stack_2°
ksthobzafc/strided_slice_1StridedSliceksthobzafc/Shape_1:output:0)ksthobzafc/strided_slice_1/stack:output:0+ksthobzafc/strided_slice_1/stack_1:output:0+ksthobzafc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ksthobzafc/strided_slice_1
&ksthobzafc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&ksthobzafc/TensorArrayV2/element_shapeÞ
ksthobzafc/TensorArrayV2TensorListReserve/ksthobzafc/TensorArrayV2/element_shape:output:0#ksthobzafc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ksthobzafc/TensorArrayV2Õ
@ksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@ksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2ksthobzafc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorksthobzafc/transpose:y:0Iksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2ksthobzafc/TensorArrayUnstack/TensorListFromTensor
 ksthobzafc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ksthobzafc/strided_slice_2/stack
"ksthobzafc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_2/stack_1
"ksthobzafc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_2/stack_2¾
ksthobzafc/strided_slice_2StridedSliceksthobzafc/transpose:y:0)ksthobzafc/strided_slice_2/stack:output:0+ksthobzafc/strided_slice_2/stack_1:output:0+ksthobzafc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ksthobzafc/strided_slice_2Ð
+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp4ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOpÓ
ksthobzafc/qjjkgjcvpf/MatMulMatMul#ksthobzafc/strided_slice_2:output:03ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ksthobzafc/qjjkgjcvpf/MatMulÖ
-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp6ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOpÏ
ksthobzafc/qjjkgjcvpf/MatMul_1MatMulksthobzafc/zeros:output:05ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
ksthobzafc/qjjkgjcvpf/MatMul_1Ä
ksthobzafc/qjjkgjcvpf/addAddV2&ksthobzafc/qjjkgjcvpf/MatMul:product:0(ksthobzafc/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ksthobzafc/qjjkgjcvpf/addÏ
,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp5ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOpÑ
ksthobzafc/qjjkgjcvpf/BiasAddBiasAddksthobzafc/qjjkgjcvpf/add:z:04ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ksthobzafc/qjjkgjcvpf/BiasAdd
%ksthobzafc/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%ksthobzafc/qjjkgjcvpf/split/split_dim
ksthobzafc/qjjkgjcvpf/splitSplit.ksthobzafc/qjjkgjcvpf/split/split_dim:output:0&ksthobzafc/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ksthobzafc/qjjkgjcvpf/split¶
$ksthobzafc/qjjkgjcvpf/ReadVariableOpReadVariableOp-ksthobzafc_qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02&
$ksthobzafc/qjjkgjcvpf/ReadVariableOpº
ksthobzafc/qjjkgjcvpf/mulMul,ksthobzafc/qjjkgjcvpf/ReadVariableOp:value:0ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mulº
ksthobzafc/qjjkgjcvpf/add_1AddV2$ksthobzafc/qjjkgjcvpf/split:output:0ksthobzafc/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_1
ksthobzafc/qjjkgjcvpf/SigmoidSigmoidksthobzafc/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/Sigmoid¼
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp/ksthobzafc_qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1À
ksthobzafc/qjjkgjcvpf/mul_1Mul.ksthobzafc/qjjkgjcvpf/ReadVariableOp_1:value:0ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_1¼
ksthobzafc/qjjkgjcvpf/add_2AddV2$ksthobzafc/qjjkgjcvpf/split:output:1ksthobzafc/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_2 
ksthobzafc/qjjkgjcvpf/Sigmoid_1Sigmoidksthobzafc/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ksthobzafc/qjjkgjcvpf/Sigmoid_1µ
ksthobzafc/qjjkgjcvpf/mul_2Mul#ksthobzafc/qjjkgjcvpf/Sigmoid_1:y:0ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_2
ksthobzafc/qjjkgjcvpf/TanhTanh$ksthobzafc/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/Tanh¶
ksthobzafc/qjjkgjcvpf/mul_3Mul!ksthobzafc/qjjkgjcvpf/Sigmoid:y:0ksthobzafc/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_3·
ksthobzafc/qjjkgjcvpf/add_3AddV2ksthobzafc/qjjkgjcvpf/mul_2:z:0ksthobzafc/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_3¼
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp/ksthobzafc_qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2Ä
ksthobzafc/qjjkgjcvpf/mul_4Mul.ksthobzafc/qjjkgjcvpf/ReadVariableOp_2:value:0ksthobzafc/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_4¼
ksthobzafc/qjjkgjcvpf/add_4AddV2$ksthobzafc/qjjkgjcvpf/split:output:3ksthobzafc/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_4 
ksthobzafc/qjjkgjcvpf/Sigmoid_2Sigmoidksthobzafc/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ksthobzafc/qjjkgjcvpf/Sigmoid_2
ksthobzafc/qjjkgjcvpf/Tanh_1Tanhksthobzafc/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/Tanh_1º
ksthobzafc/qjjkgjcvpf/mul_5Mul#ksthobzafc/qjjkgjcvpf/Sigmoid_2:y:0 ksthobzafc/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_5¥
(ksthobzafc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(ksthobzafc/TensorArrayV2_1/element_shapeä
ksthobzafc/TensorArrayV2_1TensorListReserve1ksthobzafc/TensorArrayV2_1/element_shape:output:0#ksthobzafc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ksthobzafc/TensorArrayV2_1d
ksthobzafc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/time
#ksthobzafc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#ksthobzafc/while/maximum_iterations
ksthobzafc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/while/loop_counter²
ksthobzafc/whileWhile&ksthobzafc/while/loop_counter:output:0,ksthobzafc/while/maximum_iterations:output:0ksthobzafc/time:output:0#ksthobzafc/TensorArrayV2_1:handle:0ksthobzafc/zeros:output:0ksthobzafc/zeros_1:output:0#ksthobzafc/strided_slice_1:output:0Bksthobzafc/TensorArrayUnstack/TensorListFromTensor:output_handle:04ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource6ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource5ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource-ksthobzafc_qjjkgjcvpf_readvariableop_resource/ksthobzafc_qjjkgjcvpf_readvariableop_1_resource/ksthobzafc_qjjkgjcvpf_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*)
body!R
ksthobzafc_while_body_1734546*)
cond!R
ksthobzafc_while_cond_1734545*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
ksthobzafc/whileË
;ksthobzafc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;ksthobzafc/TensorArrayV2Stack/TensorListStack/element_shape
-ksthobzafc/TensorArrayV2Stack/TensorListStackTensorListStackksthobzafc/while:output:3Dksthobzafc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-ksthobzafc/TensorArrayV2Stack/TensorListStack
 ksthobzafc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ksthobzafc/strided_slice_3/stack
"ksthobzafc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ksthobzafc/strided_slice_3/stack_1
"ksthobzafc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_3/stack_2Ü
ksthobzafc/strided_slice_3StridedSlice6ksthobzafc/TensorArrayV2Stack/TensorListStack:tensor:0)ksthobzafc/strided_slice_3/stack:output:0+ksthobzafc/strided_slice_3/stack_1:output:0+ksthobzafc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ksthobzafc/strided_slice_3
ksthobzafc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ksthobzafc/transpose_1/permÑ
ksthobzafc/transpose_1	Transpose6ksthobzafc/TensorArrayV2Stack/TensorListStack:tensor:0$ksthobzafc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/transpose_1®
 vfmtgawzzo/MatMul/ReadVariableOpReadVariableOp)vfmtgawzzo_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 vfmtgawzzo/MatMul/ReadVariableOp±
vfmtgawzzo/MatMulMatMul#ksthobzafc/strided_slice_3:output:0(vfmtgawzzo/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vfmtgawzzo/MatMul­
!vfmtgawzzo/BiasAdd/ReadVariableOpReadVariableOp*vfmtgawzzo_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!vfmtgawzzo/BiasAdd/ReadVariableOp­
vfmtgawzzo/BiasAddBiasAddvfmtgawzzo/MatMul:product:0)vfmtgawzzo/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vfmtgawzzo/BiasAddÏ
IdentityIdentityvfmtgawzzo/BiasAdd:output:0-^ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp,^ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp.^ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp%^ksthobzafc/qjjkgjcvpf/ReadVariableOp'^ksthobzafc/qjjkgjcvpf/ReadVariableOp_1'^ksthobzafc/qjjkgjcvpf/ReadVariableOp_2^ksthobzafc/while"^vfmtgawzzo/BiasAdd/ReadVariableOp!^vfmtgawzzo/MatMul/ReadVariableOp-^xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp,^xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp.^xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp%^xedyzswikc/jgtgtymybc/ReadVariableOp'^xedyzswikc/jgtgtymybc/ReadVariableOp_1'^xedyzswikc/jgtgtymybc/ReadVariableOp_2^xedyzswikc/while.^zdelabjare/conv1d/ExpandDims_1/ReadVariableOp5^zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp2Z
+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp2^
-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp2L
$ksthobzafc/qjjkgjcvpf/ReadVariableOp$ksthobzafc/qjjkgjcvpf/ReadVariableOp2P
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1&ksthobzafc/qjjkgjcvpf/ReadVariableOp_12P
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2&ksthobzafc/qjjkgjcvpf/ReadVariableOp_22$
ksthobzafc/whileksthobzafc/while2F
!vfmtgawzzo/BiasAdd/ReadVariableOp!vfmtgawzzo/BiasAdd/ReadVariableOp2D
 vfmtgawzzo/MatMul/ReadVariableOp vfmtgawzzo/MatMul/ReadVariableOp2\
,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp2Z
+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp2^
-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp2L
$xedyzswikc/jgtgtymybc/ReadVariableOp$xedyzswikc/jgtgtymybc/ReadVariableOp2P
&xedyzswikc/jgtgtymybc/ReadVariableOp_1&xedyzswikc/jgtgtymybc/ReadVariableOp_12P
&xedyzswikc/jgtgtymybc/ReadVariableOp_2&xedyzswikc/jgtgtymybc/ReadVariableOp_22$
xedyzswikc/whilexedyzswikc/while2^
-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp2l
4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹'
µ
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1736940

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
p
Ê
ksthobzafc_while_body_17345462
.ksthobzafc_while_ksthobzafc_while_loop_counter8
4ksthobzafc_while_ksthobzafc_while_maximum_iterations 
ksthobzafc_while_placeholder"
ksthobzafc_while_placeholder_1"
ksthobzafc_while_placeholder_2"
ksthobzafc_while_placeholder_31
-ksthobzafc_while_ksthobzafc_strided_slice_1_0m
iksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0O
<ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 Q
>ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 L
=ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	C
5ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0: E
7ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0: E
7ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0: 
ksthobzafc_while_identity
ksthobzafc_while_identity_1
ksthobzafc_while_identity_2
ksthobzafc_while_identity_3
ksthobzafc_while_identity_4
ksthobzafc_while_identity_5/
+ksthobzafc_while_ksthobzafc_strided_slice_1k
gksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensorM
:ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource:	 O
<ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 J
;ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource:	A
3ksthobzafc_while_qjjkgjcvpf_readvariableop_resource: C
5ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource: C
5ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource: ¢2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp¢3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢*ksthobzafc/while/qjjkgjcvpf/ReadVariableOp¢,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1¢,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2Ù
Bksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4ksthobzafc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0ksthobzafc_while_placeholderKksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4ksthobzafc/while/TensorArrayV2Read/TensorListGetItemä
1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp<ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOpý
"ksthobzafc/while/qjjkgjcvpf/MatMulMatMul;ksthobzafc/while/TensorArrayV2Read/TensorListGetItem:item:09ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"ksthobzafc/while/qjjkgjcvpf/MatMulê
3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp>ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOpæ
$ksthobzafc/while/qjjkgjcvpf/MatMul_1MatMulksthobzafc_while_placeholder_2;ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$ksthobzafc/while/qjjkgjcvpf/MatMul_1Ü
ksthobzafc/while/qjjkgjcvpf/addAddV2,ksthobzafc/while/qjjkgjcvpf/MatMul:product:0.ksthobzafc/while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
ksthobzafc/while/qjjkgjcvpf/addã
2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp=ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOpé
#ksthobzafc/while/qjjkgjcvpf/BiasAddBiasAdd#ksthobzafc/while/qjjkgjcvpf/add:z:0:ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ksthobzafc/while/qjjkgjcvpf/BiasAdd
+ksthobzafc/while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+ksthobzafc/while/qjjkgjcvpf/split/split_dim¯
!ksthobzafc/while/qjjkgjcvpf/splitSplit4ksthobzafc/while/qjjkgjcvpf/split/split_dim:output:0,ksthobzafc/while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!ksthobzafc/while/qjjkgjcvpf/splitÊ
*ksthobzafc/while/qjjkgjcvpf/ReadVariableOpReadVariableOp5ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*ksthobzafc/while/qjjkgjcvpf/ReadVariableOpÏ
ksthobzafc/while/qjjkgjcvpf/mulMul2ksthobzafc/while/qjjkgjcvpf/ReadVariableOp:value:0ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ksthobzafc/while/qjjkgjcvpf/mulÒ
!ksthobzafc/while/qjjkgjcvpf/add_1AddV2*ksthobzafc/while/qjjkgjcvpf/split:output:0#ksthobzafc/while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_1®
#ksthobzafc/while/qjjkgjcvpf/SigmoidSigmoid%ksthobzafc/while/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#ksthobzafc/while/qjjkgjcvpf/SigmoidÐ
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp7ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1Õ
!ksthobzafc/while/qjjkgjcvpf/mul_1Mul4ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1:value:0ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_1Ô
!ksthobzafc/while/qjjkgjcvpf/add_2AddV2*ksthobzafc/while/qjjkgjcvpf/split:output:1%ksthobzafc/while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_2²
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_1Sigmoid%ksthobzafc/while/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_1Ê
!ksthobzafc/while/qjjkgjcvpf/mul_2Mul)ksthobzafc/while/qjjkgjcvpf/Sigmoid_1:y:0ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_2ª
 ksthobzafc/while/qjjkgjcvpf/TanhTanh*ksthobzafc/while/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 ksthobzafc/while/qjjkgjcvpf/TanhÎ
!ksthobzafc/while/qjjkgjcvpf/mul_3Mul'ksthobzafc/while/qjjkgjcvpf/Sigmoid:y:0$ksthobzafc/while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_3Ï
!ksthobzafc/while/qjjkgjcvpf/add_3AddV2%ksthobzafc/while/qjjkgjcvpf/mul_2:z:0%ksthobzafc/while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_3Ð
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp7ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2Ü
!ksthobzafc/while/qjjkgjcvpf/mul_4Mul4ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2:value:0%ksthobzafc/while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_4Ô
!ksthobzafc/while/qjjkgjcvpf/add_4AddV2*ksthobzafc/while/qjjkgjcvpf/split:output:3%ksthobzafc/while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_4²
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_2Sigmoid%ksthobzafc/while/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_2©
"ksthobzafc/while/qjjkgjcvpf/Tanh_1Tanh%ksthobzafc/while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"ksthobzafc/while/qjjkgjcvpf/Tanh_1Ò
!ksthobzafc/while/qjjkgjcvpf/mul_5Mul)ksthobzafc/while/qjjkgjcvpf/Sigmoid_2:y:0&ksthobzafc/while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_5
5ksthobzafc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemksthobzafc_while_placeholder_1ksthobzafc_while_placeholder%ksthobzafc/while/qjjkgjcvpf/mul_5:z:0*
_output_shapes
: *
element_dtype027
5ksthobzafc/while/TensorArrayV2Write/TensorListSetItemr
ksthobzafc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
ksthobzafc/while/add/y
ksthobzafc/while/addAddV2ksthobzafc_while_placeholderksthobzafc/while/add/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/while/addv
ksthobzafc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
ksthobzafc/while/add_1/y­
ksthobzafc/while/add_1AddV2.ksthobzafc_while_ksthobzafc_while_loop_counter!ksthobzafc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/while/add_1©
ksthobzafc/while/IdentityIdentityksthobzafc/while/add_1:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/IdentityÇ
ksthobzafc/while/Identity_1Identity4ksthobzafc_while_ksthobzafc_while_maximum_iterations3^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/Identity_1«
ksthobzafc/while/Identity_2Identityksthobzafc/while/add:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/Identity_2Ø
ksthobzafc/while/Identity_3IdentityEksthobzafc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/Identity_3É
ksthobzafc/while/Identity_4Identity%ksthobzafc/while/qjjkgjcvpf/mul_5:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/while/Identity_4É
ksthobzafc/while/Identity_5Identity%ksthobzafc/while/qjjkgjcvpf/add_3:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/while/Identity_5"?
ksthobzafc_while_identity"ksthobzafc/while/Identity:output:0"C
ksthobzafc_while_identity_1$ksthobzafc/while/Identity_1:output:0"C
ksthobzafc_while_identity_2$ksthobzafc/while/Identity_2:output:0"C
ksthobzafc_while_identity_3$ksthobzafc/while/Identity_3:output:0"C
ksthobzafc_while_identity_4$ksthobzafc/while/Identity_4:output:0"C
ksthobzafc_while_identity_5$ksthobzafc/while/Identity_5:output:0"\
+ksthobzafc_while_ksthobzafc_strided_slice_1-ksthobzafc_while_ksthobzafc_strided_slice_1_0"|
;ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource=ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0"~
<ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource>ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"z
:ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource<ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0"p
5ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource7ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0"p
5ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource7ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0"l
3ksthobzafc_while_qjjkgjcvpf_readvariableop_resource5ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0"Ô
gksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensoriksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2f
1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp2j
3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp2X
*ksthobzafc/while/qjjkgjcvpf/ReadVariableOp*ksthobzafc/while/qjjkgjcvpf/ReadVariableOp2\
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_12\
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2: 
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
while_cond_1733763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1733763___redundant_placeholder05
1while_while_cond_1733763___redundant_placeholder15
1while_while_cond_1733763___redundant_placeholder25
1while_while_cond_1733763___redundant_placeholder35
1while_while_cond_1733763___redundant_placeholder45
1while_while_cond_1733763___redundant_placeholder55
1while_while_cond_1733763___redundant_placeholder6
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
¥
©	
(sequential_xedyzswikc_while_cond_1731139H
Dsequential_xedyzswikc_while_sequential_xedyzswikc_while_loop_counterN
Jsequential_xedyzswikc_while_sequential_xedyzswikc_while_maximum_iterations+
'sequential_xedyzswikc_while_placeholder-
)sequential_xedyzswikc_while_placeholder_1-
)sequential_xedyzswikc_while_placeholder_2-
)sequential_xedyzswikc_while_placeholder_3J
Fsequential_xedyzswikc_while_less_sequential_xedyzswikc_strided_slice_1a
]sequential_xedyzswikc_while_sequential_xedyzswikc_while_cond_1731139___redundant_placeholder0a
]sequential_xedyzswikc_while_sequential_xedyzswikc_while_cond_1731139___redundant_placeholder1a
]sequential_xedyzswikc_while_sequential_xedyzswikc_while_cond_1731139___redundant_placeholder2a
]sequential_xedyzswikc_while_sequential_xedyzswikc_while_cond_1731139___redundant_placeholder3a
]sequential_xedyzswikc_while_sequential_xedyzswikc_while_cond_1731139___redundant_placeholder4a
]sequential_xedyzswikc_while_sequential_xedyzswikc_while_cond_1731139___redundant_placeholder5a
]sequential_xedyzswikc_while_sequential_xedyzswikc_while_cond_1731139___redundant_placeholder6(
$sequential_xedyzswikc_while_identity
Þ
 sequential/xedyzswikc/while/LessLess'sequential_xedyzswikc_while_placeholderFsequential_xedyzswikc_while_less_sequential_xedyzswikc_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/xedyzswikc/while/Less
$sequential/xedyzswikc/while/IdentityIdentity$sequential/xedyzswikc/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/xedyzswikc/while/Identity"U
$sequential_xedyzswikc_while_identity-sequential/xedyzswikc/while/Identity:output:0*(
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
ç)
Ò
while_body_1732551
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_qjjkgjcvpf_1732575_0:	 -
while_qjjkgjcvpf_1732577_0:	 )
while_qjjkgjcvpf_1732579_0:	(
while_qjjkgjcvpf_1732581_0: (
while_qjjkgjcvpf_1732583_0: (
while_qjjkgjcvpf_1732585_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_qjjkgjcvpf_1732575:	 +
while_qjjkgjcvpf_1732577:	 '
while_qjjkgjcvpf_1732579:	&
while_qjjkgjcvpf_1732581: &
while_qjjkgjcvpf_1732583: &
while_qjjkgjcvpf_1732585: ¢(while/qjjkgjcvpf/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¶
(while/qjjkgjcvpf/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_qjjkgjcvpf_1732575_0while_qjjkgjcvpf_1732577_0while_qjjkgjcvpf_1732579_0while_qjjkgjcvpf_1732581_0while_qjjkgjcvpf_1732583_0while_qjjkgjcvpf_1732585_0*
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
GPU2*0J 8 *P
fKRI
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_17324552*
(while/qjjkgjcvpf/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/qjjkgjcvpf/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/qjjkgjcvpf/StatefulPartitionedCall:output:1)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/qjjkgjcvpf/StatefulPartitionedCall:output:2)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_qjjkgjcvpf_1732575while_qjjkgjcvpf_1732575_0"6
while_qjjkgjcvpf_1732577while_qjjkgjcvpf_1732577_0"6
while_qjjkgjcvpf_1732579while_qjjkgjcvpf_1732579_0"6
while_qjjkgjcvpf_1732581while_qjjkgjcvpf_1732581_0"6
while_qjjkgjcvpf_1732583while_qjjkgjcvpf_1732583_0"6
while_qjjkgjcvpf_1732585while_qjjkgjcvpf_1732585_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/qjjkgjcvpf/StatefulPartitionedCall(while/qjjkgjcvpf/StatefulPartitionedCall: 
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
È

,__inference_zdelabjare_layer_call_fn_1735066

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *P
fKRI
G__inference_zdelabjare_layer_call_and_return_conditional_losses_17329832
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


,__inference_sequential_layer_call_fn_1734212

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
G__inference_sequential_layer_call_and_return_conditional_losses_17334072
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
±'
³
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1732455

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


,__inference_sequential_layer_call_fn_1733442

qtjxeibmnq
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
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCall
qtjxeibmnqunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_17334072
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
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
qtjxeibmnq
Ó

,__inference_ksthobzafc_layer_call_fn_1735977

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall®
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17336512
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
©0
¼
G__inference_zdelabjare_layer_call_and_return_conditional_losses_1732983

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
2
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
©0
¼
G__inference_zdelabjare_layer_call_and_return_conditional_losses_1735103

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
2
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


xedyzswikc_while_cond_17347732
.xedyzswikc_while_xedyzswikc_while_loop_counter8
4xedyzswikc_while_xedyzswikc_while_maximum_iterations 
xedyzswikc_while_placeholder"
xedyzswikc_while_placeholder_1"
xedyzswikc_while_placeholder_2"
xedyzswikc_while_placeholder_34
0xedyzswikc_while_less_xedyzswikc_strided_slice_1K
Gxedyzswikc_while_xedyzswikc_while_cond_1734773___redundant_placeholder0K
Gxedyzswikc_while_xedyzswikc_while_cond_1734773___redundant_placeholder1K
Gxedyzswikc_while_xedyzswikc_while_cond_1734773___redundant_placeholder2K
Gxedyzswikc_while_xedyzswikc_while_cond_1734773___redundant_placeholder3K
Gxedyzswikc_while_xedyzswikc_while_cond_1734773___redundant_placeholder4K
Gxedyzswikc_while_xedyzswikc_while_cond_1734773___redundant_placeholder5K
Gxedyzswikc_while_xedyzswikc_while_cond_1734773___redundant_placeholder6
xedyzswikc_while_identity
§
xedyzswikc/while/LessLessxedyzswikc_while_placeholder0xedyzswikc_while_less_xedyzswikc_strided_slice_1*
T0*
_output_shapes
: 2
xedyzswikc/while/Less~
xedyzswikc/while/IdentityIdentityxedyzswikc/while/Less:z:0*
T0
*
_output_shapes
: 2
xedyzswikc/while/Identity"?
xedyzswikc_while_identity"xedyzswikc/while/Identity:output:0*(
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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1735057

inputsL
6zdelabjare_conv1d_expanddims_1_readvariableop_resource:K
=zdelabjare_squeeze_batch_dims_biasadd_readvariableop_resource:G
4xedyzswikc_jgtgtymybc_matmul_readvariableop_resource:	I
6xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource:	 D
5xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource:	;
-xedyzswikc_jgtgtymybc_readvariableop_resource: =
/xedyzswikc_jgtgtymybc_readvariableop_1_resource: =
/xedyzswikc_jgtgtymybc_readvariableop_2_resource: G
4ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource:	 I
6ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource:	 D
5ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource:	;
-ksthobzafc_qjjkgjcvpf_readvariableop_resource: =
/ksthobzafc_qjjkgjcvpf_readvariableop_1_resource: =
/ksthobzafc_qjjkgjcvpf_readvariableop_2_resource: ;
)vfmtgawzzo_matmul_readvariableop_resource: 8
*vfmtgawzzo_biasadd_readvariableop_resource:
identity¢,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp¢+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp¢-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp¢$ksthobzafc/qjjkgjcvpf/ReadVariableOp¢&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1¢&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2¢ksthobzafc/while¢!vfmtgawzzo/BiasAdd/ReadVariableOp¢ vfmtgawzzo/MatMul/ReadVariableOp¢,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp¢+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp¢-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp¢$xedyzswikc/jgtgtymybc/ReadVariableOp¢&xedyzswikc/jgtgtymybc/ReadVariableOp_1¢&xedyzswikc/jgtgtymybc/ReadVariableOp_2¢xedyzswikc/while¢-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp¢4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp
 zdelabjare/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 zdelabjare/conv1d/ExpandDims/dim»
zdelabjare/conv1d/ExpandDims
ExpandDimsinputs)zdelabjare/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/ExpandDimsÙ
-zdelabjare/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6zdelabjare_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp
"zdelabjare/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"zdelabjare/conv1d/ExpandDims_1/dimã
zdelabjare/conv1d/ExpandDims_1
ExpandDims5zdelabjare/conv1d/ExpandDims_1/ReadVariableOp:value:0+zdelabjare/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
zdelabjare/conv1d/ExpandDims_1
zdelabjare/conv1d/ShapeShape%zdelabjare/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
zdelabjare/conv1d/Shape
%zdelabjare/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%zdelabjare/conv1d/strided_slice/stack¥
'zdelabjare/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'zdelabjare/conv1d/strided_slice/stack_1
'zdelabjare/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'zdelabjare/conv1d/strided_slice/stack_2Ì
zdelabjare/conv1d/strided_sliceStridedSlice zdelabjare/conv1d/Shape:output:0.zdelabjare/conv1d/strided_slice/stack:output:00zdelabjare/conv1d/strided_slice/stack_1:output:00zdelabjare/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
zdelabjare/conv1d/strided_slice
zdelabjare/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
zdelabjare/conv1d/Reshape/shapeÌ
zdelabjare/conv1d/ReshapeReshape%zdelabjare/conv1d/ExpandDims:output:0(zdelabjare/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/Reshapeî
zdelabjare/conv1d/Conv2DConv2D"zdelabjare/conv1d/Reshape:output:0'zdelabjare/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
zdelabjare/conv1d/Conv2D
!zdelabjare/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!zdelabjare/conv1d/concat/values_1
zdelabjare/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/concat/axisì
zdelabjare/conv1d/concatConcatV2(zdelabjare/conv1d/strided_slice:output:0*zdelabjare/conv1d/concat/values_1:output:0&zdelabjare/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
zdelabjare/conv1d/concatÉ
zdelabjare/conv1d/Reshape_1Reshape!zdelabjare/conv1d/Conv2D:output:0!zdelabjare/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/Reshape_1Á
zdelabjare/conv1d/SqueezeSqueeze$zdelabjare/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
zdelabjare/conv1d/Squeeze
#zdelabjare/squeeze_batch_dims/ShapeShape"zdelabjare/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#zdelabjare/squeeze_batch_dims/Shape°
1zdelabjare/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1zdelabjare/squeeze_batch_dims/strided_slice/stack½
3zdelabjare/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3zdelabjare/squeeze_batch_dims/strided_slice/stack_1´
3zdelabjare/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3zdelabjare/squeeze_batch_dims/strided_slice/stack_2
+zdelabjare/squeeze_batch_dims/strided_sliceStridedSlice,zdelabjare/squeeze_batch_dims/Shape:output:0:zdelabjare/squeeze_batch_dims/strided_slice/stack:output:0<zdelabjare/squeeze_batch_dims/strided_slice/stack_1:output:0<zdelabjare/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+zdelabjare/squeeze_batch_dims/strided_slice¯
+zdelabjare/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+zdelabjare/squeeze_batch_dims/Reshape/shapeé
%zdelabjare/squeeze_batch_dims/ReshapeReshape"zdelabjare/conv1d/Squeeze:output:04zdelabjare/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%zdelabjare/squeeze_batch_dims/Reshapeæ
4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=zdelabjare_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%zdelabjare/squeeze_batch_dims/BiasAddBiasAdd.zdelabjare/squeeze_batch_dims/Reshape:output:0<zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%zdelabjare/squeeze_batch_dims/BiasAdd¯
-zdelabjare/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-zdelabjare/squeeze_batch_dims/concat/values_1¡
)zdelabjare/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)zdelabjare/squeeze_batch_dims/concat/axis¨
$zdelabjare/squeeze_batch_dims/concatConcatV24zdelabjare/squeeze_batch_dims/strided_slice:output:06zdelabjare/squeeze_batch_dims/concat/values_1:output:02zdelabjare/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$zdelabjare/squeeze_batch_dims/concatö
'zdelabjare/squeeze_batch_dims/Reshape_1Reshape.zdelabjare/squeeze_batch_dims/BiasAdd:output:0-zdelabjare/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'zdelabjare/squeeze_batch_dims/Reshape_1
ojrtxmspqi/ShapeShape0zdelabjare/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ojrtxmspqi/Shape
ojrtxmspqi/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ojrtxmspqi/strided_slice/stack
 ojrtxmspqi/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojrtxmspqi/strided_slice/stack_1
 ojrtxmspqi/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojrtxmspqi/strided_slice/stack_2¤
ojrtxmspqi/strided_sliceStridedSliceojrtxmspqi/Shape:output:0'ojrtxmspqi/strided_slice/stack:output:0)ojrtxmspqi/strided_slice/stack_1:output:0)ojrtxmspqi/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ojrtxmspqi/strided_slicez
ojrtxmspqi/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ojrtxmspqi/Reshape/shape/1z
ojrtxmspqi/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ojrtxmspqi/Reshape/shape/2×
ojrtxmspqi/Reshape/shapePack!ojrtxmspqi/strided_slice:output:0#ojrtxmspqi/Reshape/shape/1:output:0#ojrtxmspqi/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ojrtxmspqi/Reshape/shape¾
ojrtxmspqi/ReshapeReshape0zdelabjare/squeeze_batch_dims/Reshape_1:output:0!ojrtxmspqi/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ojrtxmspqi/Reshapeo
xedyzswikc/ShapeShapeojrtxmspqi/Reshape:output:0*
T0*
_output_shapes
:2
xedyzswikc/Shape
xedyzswikc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xedyzswikc/strided_slice/stack
 xedyzswikc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xedyzswikc/strided_slice/stack_1
 xedyzswikc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xedyzswikc/strided_slice/stack_2¤
xedyzswikc/strided_sliceStridedSlicexedyzswikc/Shape:output:0'xedyzswikc/strided_slice/stack:output:0)xedyzswikc/strided_slice/stack_1:output:0)xedyzswikc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xedyzswikc/strided_slicer
xedyzswikc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros/mul/y
xedyzswikc/zeros/mulMul!xedyzswikc/strided_slice:output:0xedyzswikc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros/mulu
xedyzswikc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
xedyzswikc/zeros/Less/y
xedyzswikc/zeros/LessLessxedyzswikc/zeros/mul:z:0 xedyzswikc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros/Lessx
xedyzswikc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros/packed/1¯
xedyzswikc/zeros/packedPack!xedyzswikc/strided_slice:output:0"xedyzswikc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
xedyzswikc/zeros/packedu
xedyzswikc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xedyzswikc/zeros/Const¡
xedyzswikc/zerosFill xedyzswikc/zeros/packed:output:0xedyzswikc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/zerosv
xedyzswikc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros_1/mul/y
xedyzswikc/zeros_1/mulMul!xedyzswikc/strided_slice:output:0!xedyzswikc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros_1/muly
xedyzswikc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
xedyzswikc/zeros_1/Less/y
xedyzswikc/zeros_1/LessLessxedyzswikc/zeros_1/mul:z:0"xedyzswikc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
xedyzswikc/zeros_1/Less|
xedyzswikc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/zeros_1/packed/1µ
xedyzswikc/zeros_1/packedPack!xedyzswikc/strided_slice:output:0$xedyzswikc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
xedyzswikc/zeros_1/packedy
xedyzswikc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xedyzswikc/zeros_1/Const©
xedyzswikc/zeros_1Fill"xedyzswikc/zeros_1/packed:output:0!xedyzswikc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/zeros_1
xedyzswikc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
xedyzswikc/transpose/perm°
xedyzswikc/transpose	Transposeojrtxmspqi/Reshape:output:0"xedyzswikc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/transposep
xedyzswikc/Shape_1Shapexedyzswikc/transpose:y:0*
T0*
_output_shapes
:2
xedyzswikc/Shape_1
 xedyzswikc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 xedyzswikc/strided_slice_1/stack
"xedyzswikc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_1/stack_1
"xedyzswikc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_1/stack_2°
xedyzswikc/strided_slice_1StridedSlicexedyzswikc/Shape_1:output:0)xedyzswikc/strided_slice_1/stack:output:0+xedyzswikc/strided_slice_1/stack_1:output:0+xedyzswikc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xedyzswikc/strided_slice_1
&xedyzswikc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&xedyzswikc/TensorArrayV2/element_shapeÞ
xedyzswikc/TensorArrayV2TensorListReserve/xedyzswikc/TensorArrayV2/element_shape:output:0#xedyzswikc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
xedyzswikc/TensorArrayV2Õ
@xedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@xedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2xedyzswikc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorxedyzswikc/transpose:y:0Ixedyzswikc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2xedyzswikc/TensorArrayUnstack/TensorListFromTensor
 xedyzswikc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 xedyzswikc/strided_slice_2/stack
"xedyzswikc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_2/stack_1
"xedyzswikc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_2/stack_2¾
xedyzswikc/strided_slice_2StridedSlicexedyzswikc/transpose:y:0)xedyzswikc/strided_slice_2/stack:output:0+xedyzswikc/strided_slice_2/stack_1:output:0+xedyzswikc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
xedyzswikc/strided_slice_2Ð
+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp4xedyzswikc_jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOpÓ
xedyzswikc/jgtgtymybc/MatMulMatMul#xedyzswikc/strided_slice_2:output:03xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/jgtgtymybc/MatMulÖ
-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp6xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOpÏ
xedyzswikc/jgtgtymybc/MatMul_1MatMulxedyzswikc/zeros:output:05xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
xedyzswikc/jgtgtymybc/MatMul_1Ä
xedyzswikc/jgtgtymybc/addAddV2&xedyzswikc/jgtgtymybc/MatMul:product:0(xedyzswikc/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/jgtgtymybc/addÏ
,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp5xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOpÑ
xedyzswikc/jgtgtymybc/BiasAddBiasAddxedyzswikc/jgtgtymybc/add:z:04xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xedyzswikc/jgtgtymybc/BiasAdd
%xedyzswikc/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%xedyzswikc/jgtgtymybc/split/split_dim
xedyzswikc/jgtgtymybc/splitSplit.xedyzswikc/jgtgtymybc/split/split_dim:output:0&xedyzswikc/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
xedyzswikc/jgtgtymybc/split¶
$xedyzswikc/jgtgtymybc/ReadVariableOpReadVariableOp-xedyzswikc_jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02&
$xedyzswikc/jgtgtymybc/ReadVariableOpº
xedyzswikc/jgtgtymybc/mulMul,xedyzswikc/jgtgtymybc/ReadVariableOp:value:0xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mulº
xedyzswikc/jgtgtymybc/add_1AddV2$xedyzswikc/jgtgtymybc/split:output:0xedyzswikc/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_1
xedyzswikc/jgtgtymybc/SigmoidSigmoidxedyzswikc/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/Sigmoid¼
&xedyzswikc/jgtgtymybc/ReadVariableOp_1ReadVariableOp/xedyzswikc_jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&xedyzswikc/jgtgtymybc/ReadVariableOp_1À
xedyzswikc/jgtgtymybc/mul_1Mul.xedyzswikc/jgtgtymybc/ReadVariableOp_1:value:0xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_1¼
xedyzswikc/jgtgtymybc/add_2AddV2$xedyzswikc/jgtgtymybc/split:output:1xedyzswikc/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_2 
xedyzswikc/jgtgtymybc/Sigmoid_1Sigmoidxedyzswikc/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
xedyzswikc/jgtgtymybc/Sigmoid_1µ
xedyzswikc/jgtgtymybc/mul_2Mul#xedyzswikc/jgtgtymybc/Sigmoid_1:y:0xedyzswikc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_2
xedyzswikc/jgtgtymybc/TanhTanh$xedyzswikc/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/Tanh¶
xedyzswikc/jgtgtymybc/mul_3Mul!xedyzswikc/jgtgtymybc/Sigmoid:y:0xedyzswikc/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_3·
xedyzswikc/jgtgtymybc/add_3AddV2xedyzswikc/jgtgtymybc/mul_2:z:0xedyzswikc/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_3¼
&xedyzswikc/jgtgtymybc/ReadVariableOp_2ReadVariableOp/xedyzswikc_jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&xedyzswikc/jgtgtymybc/ReadVariableOp_2Ä
xedyzswikc/jgtgtymybc/mul_4Mul.xedyzswikc/jgtgtymybc/ReadVariableOp_2:value:0xedyzswikc/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_4¼
xedyzswikc/jgtgtymybc/add_4AddV2$xedyzswikc/jgtgtymybc/split:output:3xedyzswikc/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/add_4 
xedyzswikc/jgtgtymybc/Sigmoid_2Sigmoidxedyzswikc/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
xedyzswikc/jgtgtymybc/Sigmoid_2
xedyzswikc/jgtgtymybc/Tanh_1Tanhxedyzswikc/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/Tanh_1º
xedyzswikc/jgtgtymybc/mul_5Mul#xedyzswikc/jgtgtymybc/Sigmoid_2:y:0 xedyzswikc/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/jgtgtymybc/mul_5¥
(xedyzswikc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(xedyzswikc/TensorArrayV2_1/element_shapeä
xedyzswikc/TensorArrayV2_1TensorListReserve1xedyzswikc/TensorArrayV2_1/element_shape:output:0#xedyzswikc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
xedyzswikc/TensorArrayV2_1d
xedyzswikc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/time
#xedyzswikc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#xedyzswikc/while/maximum_iterations
xedyzswikc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
xedyzswikc/while/loop_counter²
xedyzswikc/whileWhile&xedyzswikc/while/loop_counter:output:0,xedyzswikc/while/maximum_iterations:output:0xedyzswikc/time:output:0#xedyzswikc/TensorArrayV2_1:handle:0xedyzswikc/zeros:output:0xedyzswikc/zeros_1:output:0#xedyzswikc/strided_slice_1:output:0Bxedyzswikc/TensorArrayUnstack/TensorListFromTensor:output_handle:04xedyzswikc_jgtgtymybc_matmul_readvariableop_resource6xedyzswikc_jgtgtymybc_matmul_1_readvariableop_resource5xedyzswikc_jgtgtymybc_biasadd_readvariableop_resource-xedyzswikc_jgtgtymybc_readvariableop_resource/xedyzswikc_jgtgtymybc_readvariableop_1_resource/xedyzswikc_jgtgtymybc_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*)
body!R
xedyzswikc_while_body_1734774*)
cond!R
xedyzswikc_while_cond_1734773*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
xedyzswikc/whileË
;xedyzswikc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;xedyzswikc/TensorArrayV2Stack/TensorListStack/element_shape
-xedyzswikc/TensorArrayV2Stack/TensorListStackTensorListStackxedyzswikc/while:output:3Dxedyzswikc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-xedyzswikc/TensorArrayV2Stack/TensorListStack
 xedyzswikc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 xedyzswikc/strided_slice_3/stack
"xedyzswikc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"xedyzswikc/strided_slice_3/stack_1
"xedyzswikc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"xedyzswikc/strided_slice_3/stack_2Ü
xedyzswikc/strided_slice_3StridedSlice6xedyzswikc/TensorArrayV2Stack/TensorListStack:tensor:0)xedyzswikc/strided_slice_3/stack:output:0+xedyzswikc/strided_slice_3/stack_1:output:0+xedyzswikc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
xedyzswikc/strided_slice_3
xedyzswikc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
xedyzswikc/transpose_1/permÑ
xedyzswikc/transpose_1	Transpose6xedyzswikc/TensorArrayV2Stack/TensorListStack:tensor:0$xedyzswikc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
xedyzswikc/transpose_1n
ksthobzafc/ShapeShapexedyzswikc/transpose_1:y:0*
T0*
_output_shapes
:2
ksthobzafc/Shape
ksthobzafc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ksthobzafc/strided_slice/stack
 ksthobzafc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ksthobzafc/strided_slice/stack_1
 ksthobzafc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ksthobzafc/strided_slice/stack_2¤
ksthobzafc/strided_sliceStridedSliceksthobzafc/Shape:output:0'ksthobzafc/strided_slice/stack:output:0)ksthobzafc/strided_slice/stack_1:output:0)ksthobzafc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ksthobzafc/strided_slicer
ksthobzafc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros/mul/y
ksthobzafc/zeros/mulMul!ksthobzafc/strided_slice:output:0ksthobzafc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros/mulu
ksthobzafc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ksthobzafc/zeros/Less/y
ksthobzafc/zeros/LessLessksthobzafc/zeros/mul:z:0 ksthobzafc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros/Lessx
ksthobzafc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros/packed/1¯
ksthobzafc/zeros/packedPack!ksthobzafc/strided_slice:output:0"ksthobzafc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
ksthobzafc/zeros/packedu
ksthobzafc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ksthobzafc/zeros/Const¡
ksthobzafc/zerosFill ksthobzafc/zeros/packed:output:0ksthobzafc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/zerosv
ksthobzafc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros_1/mul/y
ksthobzafc/zeros_1/mulMul!ksthobzafc/strided_slice:output:0!ksthobzafc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros_1/muly
ksthobzafc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ksthobzafc/zeros_1/Less/y
ksthobzafc/zeros_1/LessLessksthobzafc/zeros_1/mul:z:0"ksthobzafc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/zeros_1/Less|
ksthobzafc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/zeros_1/packed/1µ
ksthobzafc/zeros_1/packedPack!ksthobzafc/strided_slice:output:0$ksthobzafc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
ksthobzafc/zeros_1/packedy
ksthobzafc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ksthobzafc/zeros_1/Const©
ksthobzafc/zeros_1Fill"ksthobzafc/zeros_1/packed:output:0!ksthobzafc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/zeros_1
ksthobzafc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ksthobzafc/transpose/perm¯
ksthobzafc/transpose	Transposexedyzswikc/transpose_1:y:0"ksthobzafc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/transposep
ksthobzafc/Shape_1Shapeksthobzafc/transpose:y:0*
T0*
_output_shapes
:2
ksthobzafc/Shape_1
 ksthobzafc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ksthobzafc/strided_slice_1/stack
"ksthobzafc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_1/stack_1
"ksthobzafc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_1/stack_2°
ksthobzafc/strided_slice_1StridedSliceksthobzafc/Shape_1:output:0)ksthobzafc/strided_slice_1/stack:output:0+ksthobzafc/strided_slice_1/stack_1:output:0+ksthobzafc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ksthobzafc/strided_slice_1
&ksthobzafc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&ksthobzafc/TensorArrayV2/element_shapeÞ
ksthobzafc/TensorArrayV2TensorListReserve/ksthobzafc/TensorArrayV2/element_shape:output:0#ksthobzafc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ksthobzafc/TensorArrayV2Õ
@ksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@ksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2ksthobzafc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorksthobzafc/transpose:y:0Iksthobzafc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2ksthobzafc/TensorArrayUnstack/TensorListFromTensor
 ksthobzafc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ksthobzafc/strided_slice_2/stack
"ksthobzafc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_2/stack_1
"ksthobzafc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_2/stack_2¾
ksthobzafc/strided_slice_2StridedSliceksthobzafc/transpose:y:0)ksthobzafc/strided_slice_2/stack:output:0+ksthobzafc/strided_slice_2/stack_1:output:0+ksthobzafc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ksthobzafc/strided_slice_2Ð
+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp4ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOpÓ
ksthobzafc/qjjkgjcvpf/MatMulMatMul#ksthobzafc/strided_slice_2:output:03ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ksthobzafc/qjjkgjcvpf/MatMulÖ
-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp6ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOpÏ
ksthobzafc/qjjkgjcvpf/MatMul_1MatMulksthobzafc/zeros:output:05ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
ksthobzafc/qjjkgjcvpf/MatMul_1Ä
ksthobzafc/qjjkgjcvpf/addAddV2&ksthobzafc/qjjkgjcvpf/MatMul:product:0(ksthobzafc/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ksthobzafc/qjjkgjcvpf/addÏ
,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp5ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOpÑ
ksthobzafc/qjjkgjcvpf/BiasAddBiasAddksthobzafc/qjjkgjcvpf/add:z:04ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ksthobzafc/qjjkgjcvpf/BiasAdd
%ksthobzafc/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%ksthobzafc/qjjkgjcvpf/split/split_dim
ksthobzafc/qjjkgjcvpf/splitSplit.ksthobzafc/qjjkgjcvpf/split/split_dim:output:0&ksthobzafc/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ksthobzafc/qjjkgjcvpf/split¶
$ksthobzafc/qjjkgjcvpf/ReadVariableOpReadVariableOp-ksthobzafc_qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02&
$ksthobzafc/qjjkgjcvpf/ReadVariableOpº
ksthobzafc/qjjkgjcvpf/mulMul,ksthobzafc/qjjkgjcvpf/ReadVariableOp:value:0ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mulº
ksthobzafc/qjjkgjcvpf/add_1AddV2$ksthobzafc/qjjkgjcvpf/split:output:0ksthobzafc/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_1
ksthobzafc/qjjkgjcvpf/SigmoidSigmoidksthobzafc/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/Sigmoid¼
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp/ksthobzafc_qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1À
ksthobzafc/qjjkgjcvpf/mul_1Mul.ksthobzafc/qjjkgjcvpf/ReadVariableOp_1:value:0ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_1¼
ksthobzafc/qjjkgjcvpf/add_2AddV2$ksthobzafc/qjjkgjcvpf/split:output:1ksthobzafc/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_2 
ksthobzafc/qjjkgjcvpf/Sigmoid_1Sigmoidksthobzafc/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ksthobzafc/qjjkgjcvpf/Sigmoid_1µ
ksthobzafc/qjjkgjcvpf/mul_2Mul#ksthobzafc/qjjkgjcvpf/Sigmoid_1:y:0ksthobzafc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_2
ksthobzafc/qjjkgjcvpf/TanhTanh$ksthobzafc/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/Tanh¶
ksthobzafc/qjjkgjcvpf/mul_3Mul!ksthobzafc/qjjkgjcvpf/Sigmoid:y:0ksthobzafc/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_3·
ksthobzafc/qjjkgjcvpf/add_3AddV2ksthobzafc/qjjkgjcvpf/mul_2:z:0ksthobzafc/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_3¼
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp/ksthobzafc_qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2Ä
ksthobzafc/qjjkgjcvpf/mul_4Mul.ksthobzafc/qjjkgjcvpf/ReadVariableOp_2:value:0ksthobzafc/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_4¼
ksthobzafc/qjjkgjcvpf/add_4AddV2$ksthobzafc/qjjkgjcvpf/split:output:3ksthobzafc/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/add_4 
ksthobzafc/qjjkgjcvpf/Sigmoid_2Sigmoidksthobzafc/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ksthobzafc/qjjkgjcvpf/Sigmoid_2
ksthobzafc/qjjkgjcvpf/Tanh_1Tanhksthobzafc/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/Tanh_1º
ksthobzafc/qjjkgjcvpf/mul_5Mul#ksthobzafc/qjjkgjcvpf/Sigmoid_2:y:0 ksthobzafc/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/qjjkgjcvpf/mul_5¥
(ksthobzafc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(ksthobzafc/TensorArrayV2_1/element_shapeä
ksthobzafc/TensorArrayV2_1TensorListReserve1ksthobzafc/TensorArrayV2_1/element_shape:output:0#ksthobzafc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ksthobzafc/TensorArrayV2_1d
ksthobzafc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/time
#ksthobzafc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#ksthobzafc/while/maximum_iterations
ksthobzafc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
ksthobzafc/while/loop_counter²
ksthobzafc/whileWhile&ksthobzafc/while/loop_counter:output:0,ksthobzafc/while/maximum_iterations:output:0ksthobzafc/time:output:0#ksthobzafc/TensorArrayV2_1:handle:0ksthobzafc/zeros:output:0ksthobzafc/zeros_1:output:0#ksthobzafc/strided_slice_1:output:0Bksthobzafc/TensorArrayUnstack/TensorListFromTensor:output_handle:04ksthobzafc_qjjkgjcvpf_matmul_readvariableop_resource6ksthobzafc_qjjkgjcvpf_matmul_1_readvariableop_resource5ksthobzafc_qjjkgjcvpf_biasadd_readvariableop_resource-ksthobzafc_qjjkgjcvpf_readvariableop_resource/ksthobzafc_qjjkgjcvpf_readvariableop_1_resource/ksthobzafc_qjjkgjcvpf_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*)
body!R
ksthobzafc_while_body_1734950*)
cond!R
ksthobzafc_while_cond_1734949*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
ksthobzafc/whileË
;ksthobzafc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;ksthobzafc/TensorArrayV2Stack/TensorListStack/element_shape
-ksthobzafc/TensorArrayV2Stack/TensorListStackTensorListStackksthobzafc/while:output:3Dksthobzafc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-ksthobzafc/TensorArrayV2Stack/TensorListStack
 ksthobzafc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ksthobzafc/strided_slice_3/stack
"ksthobzafc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ksthobzafc/strided_slice_3/stack_1
"ksthobzafc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ksthobzafc/strided_slice_3/stack_2Ü
ksthobzafc/strided_slice_3StridedSlice6ksthobzafc/TensorArrayV2Stack/TensorListStack:tensor:0)ksthobzafc/strided_slice_3/stack:output:0+ksthobzafc/strided_slice_3/stack_1:output:0+ksthobzafc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ksthobzafc/strided_slice_3
ksthobzafc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ksthobzafc/transpose_1/permÑ
ksthobzafc/transpose_1	Transpose6ksthobzafc/TensorArrayV2Stack/TensorListStack:tensor:0$ksthobzafc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/transpose_1®
 vfmtgawzzo/MatMul/ReadVariableOpReadVariableOp)vfmtgawzzo_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 vfmtgawzzo/MatMul/ReadVariableOp±
vfmtgawzzo/MatMulMatMul#ksthobzafc/strided_slice_3:output:0(vfmtgawzzo/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vfmtgawzzo/MatMul­
!vfmtgawzzo/BiasAdd/ReadVariableOpReadVariableOp*vfmtgawzzo_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!vfmtgawzzo/BiasAdd/ReadVariableOp­
vfmtgawzzo/BiasAddBiasAddvfmtgawzzo/MatMul:product:0)vfmtgawzzo/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vfmtgawzzo/BiasAddÏ
IdentityIdentityvfmtgawzzo/BiasAdd:output:0-^ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp,^ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp.^ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp%^ksthobzafc/qjjkgjcvpf/ReadVariableOp'^ksthobzafc/qjjkgjcvpf/ReadVariableOp_1'^ksthobzafc/qjjkgjcvpf/ReadVariableOp_2^ksthobzafc/while"^vfmtgawzzo/BiasAdd/ReadVariableOp!^vfmtgawzzo/MatMul/ReadVariableOp-^xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp,^xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp.^xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp%^xedyzswikc/jgtgtymybc/ReadVariableOp'^xedyzswikc/jgtgtymybc/ReadVariableOp_1'^xedyzswikc/jgtgtymybc/ReadVariableOp_2^xedyzswikc/while.^zdelabjare/conv1d/ExpandDims_1/ReadVariableOp5^zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp,ksthobzafc/qjjkgjcvpf/BiasAdd/ReadVariableOp2Z
+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp+ksthobzafc/qjjkgjcvpf/MatMul/ReadVariableOp2^
-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp-ksthobzafc/qjjkgjcvpf/MatMul_1/ReadVariableOp2L
$ksthobzafc/qjjkgjcvpf/ReadVariableOp$ksthobzafc/qjjkgjcvpf/ReadVariableOp2P
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_1&ksthobzafc/qjjkgjcvpf/ReadVariableOp_12P
&ksthobzafc/qjjkgjcvpf/ReadVariableOp_2&ksthobzafc/qjjkgjcvpf/ReadVariableOp_22$
ksthobzafc/whileksthobzafc/while2F
!vfmtgawzzo/BiasAdd/ReadVariableOp!vfmtgawzzo/BiasAdd/ReadVariableOp2D
 vfmtgawzzo/MatMul/ReadVariableOp vfmtgawzzo/MatMul/ReadVariableOp2\
,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp,xedyzswikc/jgtgtymybc/BiasAdd/ReadVariableOp2Z
+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp+xedyzswikc/jgtgtymybc/MatMul/ReadVariableOp2^
-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp-xedyzswikc/jgtgtymybc/MatMul_1/ReadVariableOp2L
$xedyzswikc/jgtgtymybc/ReadVariableOp$xedyzswikc/jgtgtymybc/ReadVariableOp2P
&xedyzswikc/jgtgtymybc/ReadVariableOp_1&xedyzswikc/jgtgtymybc/ReadVariableOp_12P
&xedyzswikc/jgtgtymybc/ReadVariableOp_2&xedyzswikc/jgtgtymybc/ReadVariableOp_22$
xedyzswikc/whilexedyzswikc/while2^
-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp-zdelabjare/conv1d/ExpandDims_1/ReadVariableOp2l
4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp4zdelabjare/squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

(sequential_ksthobzafc_while_body_1731316H
Dsequential_ksthobzafc_while_sequential_ksthobzafc_while_loop_counterN
Jsequential_ksthobzafc_while_sequential_ksthobzafc_while_maximum_iterations+
'sequential_ksthobzafc_while_placeholder-
)sequential_ksthobzafc_while_placeholder_1-
)sequential_ksthobzafc_while_placeholder_2-
)sequential_ksthobzafc_while_placeholder_3G
Csequential_ksthobzafc_while_sequential_ksthobzafc_strided_slice_1_0
sequential_ksthobzafc_while_tensorarrayv2read_tensorlistgetitem_sequential_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 \
Isequential_ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 W
Hsequential_ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	N
@sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0: P
Bsequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0: P
Bsequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0: (
$sequential_ksthobzafc_while_identity*
&sequential_ksthobzafc_while_identity_1*
&sequential_ksthobzafc_while_identity_2*
&sequential_ksthobzafc_while_identity_3*
&sequential_ksthobzafc_while_identity_4*
&sequential_ksthobzafc_while_identity_5E
Asequential_ksthobzafc_while_sequential_ksthobzafc_strided_slice_1
}sequential_ksthobzafc_while_tensorarrayv2read_tensorlistgetitem_sequential_ksthobzafc_tensorarrayunstack_tensorlistfromtensorX
Esequential_ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource:	 Z
Gsequential_ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 U
Fsequential_ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource:	L
>sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_resource: N
@sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource: N
@sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource: ¢=sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢<sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp¢>sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢5sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp¢7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1¢7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2ï
Msequential/ksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/ksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/ksthobzafc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_ksthobzafc_while_tensorarrayv2read_tensorlistgetitem_sequential_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0'sequential_ksthobzafc_while_placeholderVsequential/ksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/ksthobzafc/while/TensorArrayV2Read/TensorListGetItem
<sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOpGsequential_ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp©
-sequential/ksthobzafc/while/qjjkgjcvpf/MatMulMatMulFsequential/ksthobzafc/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/ksthobzafc/while/qjjkgjcvpf/MatMul
>sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOpIsequential_ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp
/sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1MatMul)sequential_ksthobzafc_while_placeholder_2Fsequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1
*sequential/ksthobzafc/while/qjjkgjcvpf/addAddV27sequential/ksthobzafc/while/qjjkgjcvpf/MatMul:product:09sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/ksthobzafc/while/qjjkgjcvpf/add
=sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOpHsequential_ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp
.sequential/ksthobzafc/while/qjjkgjcvpf/BiasAddBiasAdd.sequential/ksthobzafc/while/qjjkgjcvpf/add:z:0Esequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd²
6sequential/ksthobzafc/while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/ksthobzafc/while/qjjkgjcvpf/split/split_dimÛ
,sequential/ksthobzafc/while/qjjkgjcvpf/splitSplit?sequential/ksthobzafc/while/qjjkgjcvpf/split/split_dim:output:07sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/ksthobzafc/while/qjjkgjcvpf/splitë
5sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOpReadVariableOp@sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOpû
*sequential/ksthobzafc/while/qjjkgjcvpf/mulMul=sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp:value:0)sequential_ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/ksthobzafc/while/qjjkgjcvpf/mulþ
,sequential/ksthobzafc/while/qjjkgjcvpf/add_1AddV25sequential/ksthobzafc/while/qjjkgjcvpf/split:output:0.sequential/ksthobzafc/while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/add_1Ï
.sequential/ksthobzafc/while/qjjkgjcvpf/SigmoidSigmoid0sequential/ksthobzafc/while/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoidñ
7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOpBsequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_1Mul?sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1:value:0)sequential_ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_1
,sequential/ksthobzafc/while/qjjkgjcvpf/add_2AddV25sequential/ksthobzafc/while/qjjkgjcvpf/split:output:10sequential/ksthobzafc/while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/add_2Ó
0sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoid_1Sigmoid0sequential/ksthobzafc/while/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoid_1ö
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_2Mul4sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoid_1:y:0)sequential_ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_2Ë
+sequential/ksthobzafc/while/qjjkgjcvpf/TanhTanh5sequential/ksthobzafc/while/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/ksthobzafc/while/qjjkgjcvpf/Tanhú
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_3Mul2sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoid:y:0/sequential/ksthobzafc/while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_3û
,sequential/ksthobzafc/while/qjjkgjcvpf/add_3AddV20sequential/ksthobzafc/while/qjjkgjcvpf/mul_2:z:00sequential/ksthobzafc/while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/add_3ñ
7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOpBsequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_4Mul?sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2:value:00sequential/ksthobzafc/while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_4
,sequential/ksthobzafc/while/qjjkgjcvpf/add_4AddV25sequential/ksthobzafc/while/qjjkgjcvpf/split:output:30sequential/ksthobzafc/while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/add_4Ó
0sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoid_2Sigmoid0sequential/ksthobzafc/while/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoid_2Ê
-sequential/ksthobzafc/while/qjjkgjcvpf/Tanh_1Tanh0sequential/ksthobzafc/while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/ksthobzafc/while/qjjkgjcvpf/Tanh_1þ
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_5Mul4sequential/ksthobzafc/while/qjjkgjcvpf/Sigmoid_2:y:01sequential/ksthobzafc/while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ksthobzafc/while/qjjkgjcvpf/mul_5Ì
@sequential/ksthobzafc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_ksthobzafc_while_placeholder_1'sequential_ksthobzafc_while_placeholder0sequential/ksthobzafc/while/qjjkgjcvpf/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/ksthobzafc/while/TensorArrayV2Write/TensorListSetItem
!sequential/ksthobzafc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/ksthobzafc/while/add/yÁ
sequential/ksthobzafc/while/addAddV2'sequential_ksthobzafc_while_placeholder*sequential/ksthobzafc/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/ksthobzafc/while/add
#sequential/ksthobzafc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/ksthobzafc/while/add_1/yä
!sequential/ksthobzafc/while/add_1AddV2Dsequential_ksthobzafc_while_sequential_ksthobzafc_while_loop_counter,sequential/ksthobzafc/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/ksthobzafc/while/add_1
$sequential/ksthobzafc/while/IdentityIdentity%sequential/ksthobzafc/while/add_1:z:0>^sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp=^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp?^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp6^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp8^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_18^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/ksthobzafc/while/Identityµ
&sequential/ksthobzafc/while/Identity_1IdentityJsequential_ksthobzafc_while_sequential_ksthobzafc_while_maximum_iterations>^sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp=^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp?^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp6^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp8^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_18^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/ksthobzafc/while/Identity_1
&sequential/ksthobzafc/while/Identity_2Identity#sequential/ksthobzafc/while/add:z:0>^sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp=^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp?^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp6^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp8^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_18^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/ksthobzafc/while/Identity_2»
&sequential/ksthobzafc/while/Identity_3IdentityPsequential/ksthobzafc/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp=^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp?^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp6^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp8^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_18^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/ksthobzafc/while/Identity_3¬
&sequential/ksthobzafc/while/Identity_4Identity0sequential/ksthobzafc/while/qjjkgjcvpf/mul_5:z:0>^sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp=^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp?^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp6^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp8^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_18^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/while/Identity_4¬
&sequential/ksthobzafc/while/Identity_5Identity0sequential/ksthobzafc/while/qjjkgjcvpf/add_3:z:0>^sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp=^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp?^sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp6^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp8^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_18^sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ksthobzafc/while/Identity_5"U
$sequential_ksthobzafc_while_identity-sequential/ksthobzafc/while/Identity:output:0"Y
&sequential_ksthobzafc_while_identity_1/sequential/ksthobzafc/while/Identity_1:output:0"Y
&sequential_ksthobzafc_while_identity_2/sequential/ksthobzafc/while/Identity_2:output:0"Y
&sequential_ksthobzafc_while_identity_3/sequential/ksthobzafc/while/Identity_3:output:0"Y
&sequential_ksthobzafc_while_identity_4/sequential/ksthobzafc/while/Identity_4:output:0"Y
&sequential_ksthobzafc_while_identity_5/sequential/ksthobzafc/while/Identity_5:output:0"
Fsequential_ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resourceHsequential_ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0"
Gsequential_ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resourceIsequential_ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"
Esequential_ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resourceGsequential_ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0"
@sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resourceBsequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0"
@sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resourceBsequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0"
>sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_resource@sequential_ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0"
Asequential_ksthobzafc_while_sequential_ksthobzafc_strided_slice_1Csequential_ksthobzafc_while_sequential_ksthobzafc_strided_slice_1_0"
}sequential_ksthobzafc_while_tensorarrayv2read_tensorlistgetitem_sequential_ksthobzafc_tensorarrayunstack_tensorlistfromtensorsequential_ksthobzafc_while_tensorarrayv2read_tensorlistgetitem_sequential_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp=sequential/ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2|
<sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp<sequential/ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp2
>sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp>sequential/ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp2n
5sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp5sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp2r
7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_17sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_12r
7sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_27sequential/ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2: 
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


,__inference_sequential_layer_call_fn_1734048

qtjxeibmnq
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
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCall
qtjxeibmnqunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_17339762
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
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
qtjxeibmnq
Û

,__inference_xedyzswikc_layer_call_fn_1735189

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall²
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17338652
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
àY

while_body_1736056
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 F
3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 A
2while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	8
*while_qjjkgjcvpf_readvariableop_resource_0: :
,while_qjjkgjcvpf_readvariableop_1_resource_0: :
,while_qjjkgjcvpf_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_qjjkgjcvpf_matmul_readvariableop_resource:	 D
1while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 ?
0while_qjjkgjcvpf_biasadd_readvariableop_resource:	6
(while_qjjkgjcvpf_readvariableop_resource: 8
*while_qjjkgjcvpf_readvariableop_1_resource: 8
*while_qjjkgjcvpf_readvariableop_2_resource: ¢'while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢&while/qjjkgjcvpf/MatMul/ReadVariableOp¢(while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢while/qjjkgjcvpf/ReadVariableOp¢!while/qjjkgjcvpf/ReadVariableOp_1¢!while/qjjkgjcvpf/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp1while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/qjjkgjcvpf/MatMul/ReadVariableOpÑ
while/qjjkgjcvpf/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMulÉ
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpº
while/qjjkgjcvpf/MatMul_1MatMulwhile_placeholder_20while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMul_1°
while/qjjkgjcvpf/addAddV2!while/qjjkgjcvpf/MatMul:product:0#while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/addÂ
'while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp2while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp½
while/qjjkgjcvpf/BiasAddBiasAddwhile/qjjkgjcvpf/add:z:0/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/BiasAdd
 while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/qjjkgjcvpf/split/split_dim
while/qjjkgjcvpf/splitSplit)while/qjjkgjcvpf/split/split_dim:output:0!while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/qjjkgjcvpf/split©
while/qjjkgjcvpf/ReadVariableOpReadVariableOp*while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/qjjkgjcvpf/ReadVariableOp£
while/qjjkgjcvpf/mulMul'while/qjjkgjcvpf/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul¦
while/qjjkgjcvpf/add_1AddV2while/qjjkgjcvpf/split:output:0while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_1
while/qjjkgjcvpf/SigmoidSigmoidwhile/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid¯
!while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp,while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_1©
while/qjjkgjcvpf/mul_1Mul)while/qjjkgjcvpf/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_1¨
while/qjjkgjcvpf/add_2AddV2while/qjjkgjcvpf/split:output:1while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_2
while/qjjkgjcvpf/Sigmoid_1Sigmoidwhile/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_1
while/qjjkgjcvpf/mul_2Mulwhile/qjjkgjcvpf/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_2
while/qjjkgjcvpf/TanhTanhwhile/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh¢
while/qjjkgjcvpf/mul_3Mulwhile/qjjkgjcvpf/Sigmoid:y:0while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_3£
while/qjjkgjcvpf/add_3AddV2while/qjjkgjcvpf/mul_2:z:0while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_3¯
!while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp,while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_2°
while/qjjkgjcvpf/mul_4Mul)while/qjjkgjcvpf/ReadVariableOp_2:value:0while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_4¨
while/qjjkgjcvpf/add_4AddV2while/qjjkgjcvpf/split:output:3while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_4
while/qjjkgjcvpf/Sigmoid_2Sigmoidwhile/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_2
while/qjjkgjcvpf/Tanh_1Tanhwhile/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh_1¦
while/qjjkgjcvpf/mul_5Mulwhile/qjjkgjcvpf/Sigmoid_2:y:0while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/qjjkgjcvpf/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/qjjkgjcvpf/mul_5:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/qjjkgjcvpf/add_3:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_qjjkgjcvpf_biasadd_readvariableop_resource2while_qjjkgjcvpf_biasadd_readvariableop_resource_0"h
1while_qjjkgjcvpf_matmul_1_readvariableop_resource3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"d
/while_qjjkgjcvpf_matmul_readvariableop_resource1while_qjjkgjcvpf_matmul_readvariableop_resource_0"Z
*while_qjjkgjcvpf_readvariableop_1_resource,while_qjjkgjcvpf_readvariableop_1_resource_0"Z
*while_qjjkgjcvpf_readvariableop_2_resource,while_qjjkgjcvpf_readvariableop_2_resource_0"V
(while_qjjkgjcvpf_readvariableop_resource*while_qjjkgjcvpf_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp'while/qjjkgjcvpf/BiasAdd/ReadVariableOp2P
&while/qjjkgjcvpf/MatMul/ReadVariableOp&while/qjjkgjcvpf/MatMul/ReadVariableOp2T
(while/qjjkgjcvpf/MatMul_1/ReadVariableOp(while/qjjkgjcvpf/MatMul_1/ReadVariableOp2B
while/qjjkgjcvpf/ReadVariableOpwhile/qjjkgjcvpf/ReadVariableOp2F
!while/qjjkgjcvpf/ReadVariableOp_1!while/qjjkgjcvpf/ReadVariableOp_12F
!while/qjjkgjcvpf/ReadVariableOp_2!while/qjjkgjcvpf/ReadVariableOp_2: 
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
Ý
H
,__inference_ojrtxmspqi_layer_call_fn_1735108

inputs
identityÌ
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
GPU2*0J 8 *P
fKRI
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_17330022
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


í
while_cond_1735807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1735807___redundant_placeholder05
1while_while_cond_1735807___redundant_placeholder15
1while_while_cond_1735807___redundant_placeholder25
1while_while_cond_1735807___redundant_placeholder35
1while_while_cond_1735807___redundant_placeholder45
1while_while_cond_1735807___redundant_placeholder55
1while_while_cond_1735807___redundant_placeholder6
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
àh

G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735549
inputs_0<
)jgtgtymybc_matmul_readvariableop_resource:	>
+jgtgtymybc_matmul_1_readvariableop_resource:	 9
*jgtgtymybc_biasadd_readvariableop_resource:	0
"jgtgtymybc_readvariableop_resource: 2
$jgtgtymybc_readvariableop_1_resource: 2
$jgtgtymybc_readvariableop_2_resource: 
identity¢!jgtgtymybc/BiasAdd/ReadVariableOp¢ jgtgtymybc/MatMul/ReadVariableOp¢"jgtgtymybc/MatMul_1/ReadVariableOp¢jgtgtymybc/ReadVariableOp¢jgtgtymybc/ReadVariableOp_1¢jgtgtymybc/ReadVariableOp_2¢whileF
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
strided_slice_2¯
 jgtgtymybc/MatMul/ReadVariableOpReadVariableOp)jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jgtgtymybc/MatMul/ReadVariableOp§
jgtgtymybc/MatMulMatMulstrided_slice_2:output:0(jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMulµ
"jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp+jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jgtgtymybc/MatMul_1/ReadVariableOp£
jgtgtymybc/MatMul_1MatMulzeros:output:0*jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMul_1
jgtgtymybc/addAddV2jgtgtymybc/MatMul:product:0jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/add®
!jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp*jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jgtgtymybc/BiasAdd/ReadVariableOp¥
jgtgtymybc/BiasAddBiasAddjgtgtymybc/add:z:0)jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/BiasAddz
jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jgtgtymybc/split/split_dimë
jgtgtymybc/splitSplit#jgtgtymybc/split/split_dim:output:0jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jgtgtymybc/split
jgtgtymybc/ReadVariableOpReadVariableOp"jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp
jgtgtymybc/mulMul!jgtgtymybc/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul
jgtgtymybc/add_1AddV2jgtgtymybc/split:output:0jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_1{
jgtgtymybc/SigmoidSigmoidjgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid
jgtgtymybc/ReadVariableOp_1ReadVariableOp$jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_1
jgtgtymybc/mul_1Mul#jgtgtymybc/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_1
jgtgtymybc/add_2AddV2jgtgtymybc/split:output:1jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_2
jgtgtymybc/Sigmoid_1Sigmoidjgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_1
jgtgtymybc/mul_2Muljgtgtymybc/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_2w
jgtgtymybc/TanhTanhjgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh
jgtgtymybc/mul_3Muljgtgtymybc/Sigmoid:y:0jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_3
jgtgtymybc/add_3AddV2jgtgtymybc/mul_2:z:0jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_3
jgtgtymybc/ReadVariableOp_2ReadVariableOp$jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_2
jgtgtymybc/mul_4Mul#jgtgtymybc/ReadVariableOp_2:value:0jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_4
jgtgtymybc/add_4AddV2jgtgtymybc/split:output:3jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_4
jgtgtymybc/Sigmoid_2Sigmoidjgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_2v
jgtgtymybc/Tanh_1Tanhjgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh_1
jgtgtymybc/mul_5Muljgtgtymybc/Sigmoid_2:y:0jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jgtgtymybc_matmul_readvariableop_resource+jgtgtymybc_matmul_1_readvariableop_resource*jgtgtymybc_biasadd_readvariableop_resource"jgtgtymybc_readvariableop_resource$jgtgtymybc_readvariableop_1_resource$jgtgtymybc_readvariableop_2_resource*
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
while_body_1735448*
condR
while_cond_1735447*Q
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
transpose_1¼
IdentityIdentitytranspose_1:y:0"^jgtgtymybc/BiasAdd/ReadVariableOp!^jgtgtymybc/MatMul/ReadVariableOp#^jgtgtymybc/MatMul_1/ReadVariableOp^jgtgtymybc/ReadVariableOp^jgtgtymybc/ReadVariableOp_1^jgtgtymybc/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jgtgtymybc/BiasAdd/ReadVariableOp!jgtgtymybc/BiasAdd/ReadVariableOp2D
 jgtgtymybc/MatMul/ReadVariableOp jgtgtymybc/MatMul/ReadVariableOp2H
"jgtgtymybc/MatMul_1/ReadVariableOp"jgtgtymybc/MatMul_1/ReadVariableOp26
jgtgtymybc/ReadVariableOpjgtgtymybc/ReadVariableOp2:
jgtgtymybc/ReadVariableOp_1jgtgtymybc/ReadVariableOp_12:
jgtgtymybc/ReadVariableOp_2jgtgtymybc/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ó

,__inference_ksthobzafc_layer_call_fn_1735960

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall®
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17333762
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
±'
³
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1731697

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
àY

while_body_1735628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jgtgtymybc_matmul_readvariableop_resource_0:	F
3while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 A
2while_jgtgtymybc_biasadd_readvariableop_resource_0:	8
*while_jgtgtymybc_readvariableop_resource_0: :
,while_jgtgtymybc_readvariableop_1_resource_0: :
,while_jgtgtymybc_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jgtgtymybc_matmul_readvariableop_resource:	D
1while_jgtgtymybc_matmul_1_readvariableop_resource:	 ?
0while_jgtgtymybc_biasadd_readvariableop_resource:	6
(while_jgtgtymybc_readvariableop_resource: 8
*while_jgtgtymybc_readvariableop_1_resource: 8
*while_jgtgtymybc_readvariableop_2_resource: ¢'while/jgtgtymybc/BiasAdd/ReadVariableOp¢&while/jgtgtymybc/MatMul/ReadVariableOp¢(while/jgtgtymybc/MatMul_1/ReadVariableOp¢while/jgtgtymybc/ReadVariableOp¢!while/jgtgtymybc/ReadVariableOp_1¢!while/jgtgtymybc/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp1while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jgtgtymybc/MatMul/ReadVariableOpÑ
while/jgtgtymybc/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMulÉ
(while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp3while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jgtgtymybc/MatMul_1/ReadVariableOpº
while/jgtgtymybc/MatMul_1MatMulwhile_placeholder_20while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMul_1°
while/jgtgtymybc/addAddV2!while/jgtgtymybc/MatMul:product:0#while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/addÂ
'while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp2while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jgtgtymybc/BiasAdd/ReadVariableOp½
while/jgtgtymybc/BiasAddBiasAddwhile/jgtgtymybc/add:z:0/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/BiasAdd
 while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jgtgtymybc/split/split_dim
while/jgtgtymybc/splitSplit)while/jgtgtymybc/split/split_dim:output:0!while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jgtgtymybc/split©
while/jgtgtymybc/ReadVariableOpReadVariableOp*while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jgtgtymybc/ReadVariableOp£
while/jgtgtymybc/mulMul'while/jgtgtymybc/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul¦
while/jgtgtymybc/add_1AddV2while/jgtgtymybc/split:output:0while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_1
while/jgtgtymybc/SigmoidSigmoidwhile/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid¯
!while/jgtgtymybc/ReadVariableOp_1ReadVariableOp,while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_1©
while/jgtgtymybc/mul_1Mul)while/jgtgtymybc/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_1¨
while/jgtgtymybc/add_2AddV2while/jgtgtymybc/split:output:1while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_2
while/jgtgtymybc/Sigmoid_1Sigmoidwhile/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_1
while/jgtgtymybc/mul_2Mulwhile/jgtgtymybc/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_2
while/jgtgtymybc/TanhTanhwhile/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh¢
while/jgtgtymybc/mul_3Mulwhile/jgtgtymybc/Sigmoid:y:0while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_3£
while/jgtgtymybc/add_3AddV2while/jgtgtymybc/mul_2:z:0while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_3¯
!while/jgtgtymybc/ReadVariableOp_2ReadVariableOp,while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_2°
while/jgtgtymybc/mul_4Mul)while/jgtgtymybc/ReadVariableOp_2:value:0while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_4¨
while/jgtgtymybc/add_4AddV2while/jgtgtymybc/split:output:3while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_4
while/jgtgtymybc/Sigmoid_2Sigmoidwhile/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_2
while/jgtgtymybc/Tanh_1Tanhwhile/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh_1¦
while/jgtgtymybc/mul_5Mulwhile/jgtgtymybc/Sigmoid_2:y:0while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jgtgtymybc/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jgtgtymybc/mul_5:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jgtgtymybc/add_3:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_jgtgtymybc_biasadd_readvariableop_resource2while_jgtgtymybc_biasadd_readvariableop_resource_0"h
1while_jgtgtymybc_matmul_1_readvariableop_resource3while_jgtgtymybc_matmul_1_readvariableop_resource_0"d
/while_jgtgtymybc_matmul_readvariableop_resource1while_jgtgtymybc_matmul_readvariableop_resource_0"Z
*while_jgtgtymybc_readvariableop_1_resource,while_jgtgtymybc_readvariableop_1_resource_0"Z
*while_jgtgtymybc_readvariableop_2_resource,while_jgtgtymybc_readvariableop_2_resource_0"V
(while_jgtgtymybc_readvariableop_resource*while_jgtgtymybc_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jgtgtymybc/BiasAdd/ReadVariableOp'while/jgtgtymybc/BiasAdd/ReadVariableOp2P
&while/jgtgtymybc/MatMul/ReadVariableOp&while/jgtgtymybc/MatMul/ReadVariableOp2T
(while/jgtgtymybc/MatMul_1/ReadVariableOp(while/jgtgtymybc/MatMul_1/ReadVariableOp2B
while/jgtgtymybc/ReadVariableOpwhile/jgtgtymybc/ReadVariableOp2F
!while/jgtgtymybc/ReadVariableOp_1!while/jgtgtymybc/ReadVariableOp_12F
!while/jgtgtymybc/ReadVariableOp_2!while/jgtgtymybc/ReadVariableOp_2: 
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
Z
Ë
 __inference__traced_save_1737124
file_prefix0
,savev2_zdelabjare_kernel_read_readvariableop.
*savev2_zdelabjare_bias_read_readvariableop0
,savev2_vfmtgawzzo_kernel_read_readvariableop.
*savev2_vfmtgawzzo_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_xedyzswikc_jgtgtymybc_kernel_read_readvariableopE
Asavev2_xedyzswikc_jgtgtymybc_recurrent_kernel_read_readvariableop9
5savev2_xedyzswikc_jgtgtymybc_bias_read_readvariableopP
Lsavev2_xedyzswikc_jgtgtymybc_input_gate_peephole_weights_read_readvariableopQ
Msavev2_xedyzswikc_jgtgtymybc_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_xedyzswikc_jgtgtymybc_output_gate_peephole_weights_read_readvariableop;
7savev2_ksthobzafc_qjjkgjcvpf_kernel_read_readvariableopE
Asavev2_ksthobzafc_qjjkgjcvpf_recurrent_kernel_read_readvariableop9
5savev2_ksthobzafc_qjjkgjcvpf_bias_read_readvariableopP
Lsavev2_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weights_read_readvariableopQ
Msavev2_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_zdelabjare_kernel_rms_read_readvariableop:
6savev2_rmsprop_zdelabjare_bias_rms_read_readvariableop<
8savev2_rmsprop_vfmtgawzzo_kernel_rms_read_readvariableop:
6savev2_rmsprop_vfmtgawzzo_bias_rms_read_readvariableopG
Csavev2_rmsprop_xedyzswikc_jgtgtymybc_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_xedyzswikc_jgtgtymybc_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_xedyzswikc_jgtgtymybc_bias_rms_read_readvariableop\
Xsavev2_rmsprop_xedyzswikc_jgtgtymybc_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_xedyzswikc_jgtgtymybc_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_xedyzswikc_jgtgtymybc_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_ksthobzafc_qjjkgjcvpf_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_ksthobzafc_qjjkgjcvpf_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_ksthobzafc_qjjkgjcvpf_bias_rms_read_readvariableop\
Xsavev2_rmsprop_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2/shape_and_slices£
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_zdelabjare_kernel_read_readvariableop*savev2_zdelabjare_bias_read_readvariableop,savev2_vfmtgawzzo_kernel_read_readvariableop*savev2_vfmtgawzzo_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_xedyzswikc_jgtgtymybc_kernel_read_readvariableopAsavev2_xedyzswikc_jgtgtymybc_recurrent_kernel_read_readvariableop5savev2_xedyzswikc_jgtgtymybc_bias_read_readvariableopLsavev2_xedyzswikc_jgtgtymybc_input_gate_peephole_weights_read_readvariableopMsavev2_xedyzswikc_jgtgtymybc_forget_gate_peephole_weights_read_readvariableopMsavev2_xedyzswikc_jgtgtymybc_output_gate_peephole_weights_read_readvariableop7savev2_ksthobzafc_qjjkgjcvpf_kernel_read_readvariableopAsavev2_ksthobzafc_qjjkgjcvpf_recurrent_kernel_read_readvariableop5savev2_ksthobzafc_qjjkgjcvpf_bias_read_readvariableopLsavev2_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weights_read_readvariableopMsavev2_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weights_read_readvariableopMsavev2_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_zdelabjare_kernel_rms_read_readvariableop6savev2_rmsprop_zdelabjare_bias_rms_read_readvariableop8savev2_rmsprop_vfmtgawzzo_kernel_rms_read_readvariableop6savev2_rmsprop_vfmtgawzzo_bias_rms_read_readvariableopCsavev2_rmsprop_xedyzswikc_jgtgtymybc_kernel_rms_read_readvariableopMsavev2_rmsprop_xedyzswikc_jgtgtymybc_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_xedyzswikc_jgtgtymybc_bias_rms_read_readvariableopXsavev2_rmsprop_xedyzswikc_jgtgtymybc_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_xedyzswikc_jgtgtymybc_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_xedyzswikc_jgtgtymybc_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_ksthobzafc_qjjkgjcvpf_kernel_rms_read_readvariableopMsavev2_rmsprop_ksthobzafc_qjjkgjcvpf_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_ksthobzafc_qjjkgjcvpf_bias_rms_read_readvariableopXsavev2_rmsprop_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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


í
while_cond_1736415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1736415___redundant_placeholder05
1while_while_cond_1736415___redundant_placeholder15
1while_while_cond_1736415___redundant_placeholder25
1while_while_cond_1736415___redundant_placeholder35
1while_while_cond_1736415___redundant_placeholder45
1while_while_cond_1736415___redundant_placeholder55
1while_while_cond_1736415___redundant_placeholder6
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
¯

#__inference__traced_restore_1737251
file_prefix8
"assignvariableop_zdelabjare_kernel:0
"assignvariableop_1_zdelabjare_bias:6
$assignvariableop_2_vfmtgawzzo_kernel: 0
"assignvariableop_3_vfmtgawzzo_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_xedyzswikc_jgtgtymybc_kernel:	M
:assignvariableop_10_xedyzswikc_jgtgtymybc_recurrent_kernel:	 =
.assignvariableop_11_xedyzswikc_jgtgtymybc_bias:	S
Eassignvariableop_12_xedyzswikc_jgtgtymybc_input_gate_peephole_weights: T
Fassignvariableop_13_xedyzswikc_jgtgtymybc_forget_gate_peephole_weights: T
Fassignvariableop_14_xedyzswikc_jgtgtymybc_output_gate_peephole_weights: C
0assignvariableop_15_ksthobzafc_qjjkgjcvpf_kernel:	 M
:assignvariableop_16_ksthobzafc_qjjkgjcvpf_recurrent_kernel:	 =
.assignvariableop_17_ksthobzafc_qjjkgjcvpf_bias:	S
Eassignvariableop_18_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weights: T
Fassignvariableop_19_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weights: T
Fassignvariableop_20_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_zdelabjare_kernel_rms:=
/assignvariableop_24_rmsprop_zdelabjare_bias_rms:C
1assignvariableop_25_rmsprop_vfmtgawzzo_kernel_rms: =
/assignvariableop_26_rmsprop_vfmtgawzzo_bias_rms:O
<assignvariableop_27_rmsprop_xedyzswikc_jgtgtymybc_kernel_rms:	Y
Fassignvariableop_28_rmsprop_xedyzswikc_jgtgtymybc_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_xedyzswikc_jgtgtymybc_bias_rms:	_
Qassignvariableop_30_rmsprop_xedyzswikc_jgtgtymybc_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_xedyzswikc_jgtgtymybc_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_xedyzswikc_jgtgtymybc_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_ksthobzafc_qjjkgjcvpf_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_ksthobzafc_qjjkgjcvpf_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_ksthobzafc_qjjkgjcvpf_bias_rms:	_
Qassignvariableop_36_rmsprop_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weights_rms: 
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

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_zdelabjare_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_zdelabjare_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_vfmtgawzzo_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_vfmtgawzzo_biasIdentity_3:output:0"/device:CPU:0*
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

Identity_9´
AssignVariableOp_9AssignVariableOp/assignvariableop_9_xedyzswikc_jgtgtymybc_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_xedyzswikc_jgtgtymybc_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_xedyzswikc_jgtgtymybc_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_xedyzswikc_jgtgtymybc_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_xedyzswikc_jgtgtymybc_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_xedyzswikc_jgtgtymybc_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_ksthobzafc_qjjkgjcvpf_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_ksthobzafc_qjjkgjcvpf_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_ksthobzafc_qjjkgjcvpf_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
Identity_23¹
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_zdelabjare_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_zdelabjare_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_vfmtgawzzo_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_vfmtgawzzo_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_xedyzswikc_jgtgtymybc_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_xedyzswikc_jgtgtymybc_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_xedyzswikc_jgtgtymybc_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_xedyzswikc_jgtgtymybc_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_xedyzswikc_jgtgtymybc_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_xedyzswikc_jgtgtymybc_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_ksthobzafc_qjjkgjcvpf_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_ksthobzafc_qjjkgjcvpf_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_ksthobzafc_qjjkgjcvpf_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_ksthobzafc_qjjkgjcvpf_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_ksthobzafc_qjjkgjcvpf_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_ksthobzafc_qjjkgjcvpf_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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

À
,__inference_qjjkgjcvpf_layer_call_fn_1736873

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

identity_2¢StatefulPartitionedCallì
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
GPU2*0J 8 *P
fKRI
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_17322682
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
Û

,__inference_xedyzswikc_layer_call_fn_1735172

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall²
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17331832
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
Ó	
ø
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_1733400

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
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_1734130

qtjxeibmnq(
zdelabjare_1734092: 
zdelabjare_1734094:%
xedyzswikc_1734098:	%
xedyzswikc_1734100:	 !
xedyzswikc_1734102:	 
xedyzswikc_1734104:  
xedyzswikc_1734106:  
xedyzswikc_1734108: %
ksthobzafc_1734111:	 %
ksthobzafc_1734113:	 !
ksthobzafc_1734115:	 
ksthobzafc_1734117:  
ksthobzafc_1734119:  
ksthobzafc_1734121: $
vfmtgawzzo_1734124:  
vfmtgawzzo_1734126:
identity¢"ksthobzafc/StatefulPartitionedCall¢"vfmtgawzzo/StatefulPartitionedCall¢"xedyzswikc/StatefulPartitionedCall¢"zdelabjare/StatefulPartitionedCall°
"zdelabjare/StatefulPartitionedCallStatefulPartitionedCall
qtjxeibmnqzdelabjare_1734092zdelabjare_1734094*
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
GPU2*0J 8 *P
fKRI
G__inference_zdelabjare_layer_call_and_return_conditional_losses_17329832$
"zdelabjare/StatefulPartitionedCall
ojrtxmspqi/PartitionedCallPartitionedCall+zdelabjare/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_17330022
ojrtxmspqi/PartitionedCall
"xedyzswikc/StatefulPartitionedCallStatefulPartitionedCall#ojrtxmspqi/PartitionedCall:output:0xedyzswikc_1734098xedyzswikc_1734100xedyzswikc_1734102xedyzswikc_1734104xedyzswikc_1734106xedyzswikc_1734108*
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17338652$
"xedyzswikc/StatefulPartitionedCall¡
"ksthobzafc/StatefulPartitionedCallStatefulPartitionedCall+xedyzswikc/StatefulPartitionedCall:output:0ksthobzafc_1734111ksthobzafc_1734113ksthobzafc_1734115ksthobzafc_1734117ksthobzafc_1734119ksthobzafc_1734121*
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17336512$
"ksthobzafc/StatefulPartitionedCallÉ
"vfmtgawzzo/StatefulPartitionedCallStatefulPartitionedCall+ksthobzafc/StatefulPartitionedCall:output:0vfmtgawzzo_1734124vfmtgawzzo_1734126*
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
GPU2*0J 8 *P
fKRI
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_17334002$
"vfmtgawzzo/StatefulPartitionedCall
IdentityIdentity+vfmtgawzzo/StatefulPartitionedCall:output:0#^ksthobzafc/StatefulPartitionedCall#^vfmtgawzzo/StatefulPartitionedCall#^xedyzswikc/StatefulPartitionedCall#^zdelabjare/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ksthobzafc/StatefulPartitionedCall"ksthobzafc/StatefulPartitionedCall2H
"vfmtgawzzo/StatefulPartitionedCall"vfmtgawzzo/StatefulPartitionedCall2H
"xedyzswikc/StatefulPartitionedCall"xedyzswikc/StatefulPartitionedCall2H
"zdelabjare/StatefulPartitionedCall"zdelabjare/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
qtjxeibmnq


í
while_cond_1733549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1733549___redundant_placeholder05
1while_while_cond_1733549___redundant_placeholder15
1while_while_cond_1733549___redundant_placeholder25
1while_while_cond_1733549___redundant_placeholder35
1while_while_cond_1733549___redundant_placeholder45
1while_while_cond_1733549___redundant_placeholder55
1while_while_cond_1733549___redundant_placeholder6
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
àY

while_body_1733764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jgtgtymybc_matmul_readvariableop_resource_0:	F
3while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 A
2while_jgtgtymybc_biasadd_readvariableop_resource_0:	8
*while_jgtgtymybc_readvariableop_resource_0: :
,while_jgtgtymybc_readvariableop_1_resource_0: :
,while_jgtgtymybc_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jgtgtymybc_matmul_readvariableop_resource:	D
1while_jgtgtymybc_matmul_1_readvariableop_resource:	 ?
0while_jgtgtymybc_biasadd_readvariableop_resource:	6
(while_jgtgtymybc_readvariableop_resource: 8
*while_jgtgtymybc_readvariableop_1_resource: 8
*while_jgtgtymybc_readvariableop_2_resource: ¢'while/jgtgtymybc/BiasAdd/ReadVariableOp¢&while/jgtgtymybc/MatMul/ReadVariableOp¢(while/jgtgtymybc/MatMul_1/ReadVariableOp¢while/jgtgtymybc/ReadVariableOp¢!while/jgtgtymybc/ReadVariableOp_1¢!while/jgtgtymybc/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp1while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jgtgtymybc/MatMul/ReadVariableOpÑ
while/jgtgtymybc/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMulÉ
(while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp3while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jgtgtymybc/MatMul_1/ReadVariableOpº
while/jgtgtymybc/MatMul_1MatMulwhile_placeholder_20while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMul_1°
while/jgtgtymybc/addAddV2!while/jgtgtymybc/MatMul:product:0#while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/addÂ
'while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp2while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jgtgtymybc/BiasAdd/ReadVariableOp½
while/jgtgtymybc/BiasAddBiasAddwhile/jgtgtymybc/add:z:0/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/BiasAdd
 while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jgtgtymybc/split/split_dim
while/jgtgtymybc/splitSplit)while/jgtgtymybc/split/split_dim:output:0!while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jgtgtymybc/split©
while/jgtgtymybc/ReadVariableOpReadVariableOp*while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jgtgtymybc/ReadVariableOp£
while/jgtgtymybc/mulMul'while/jgtgtymybc/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul¦
while/jgtgtymybc/add_1AddV2while/jgtgtymybc/split:output:0while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_1
while/jgtgtymybc/SigmoidSigmoidwhile/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid¯
!while/jgtgtymybc/ReadVariableOp_1ReadVariableOp,while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_1©
while/jgtgtymybc/mul_1Mul)while/jgtgtymybc/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_1¨
while/jgtgtymybc/add_2AddV2while/jgtgtymybc/split:output:1while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_2
while/jgtgtymybc/Sigmoid_1Sigmoidwhile/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_1
while/jgtgtymybc/mul_2Mulwhile/jgtgtymybc/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_2
while/jgtgtymybc/TanhTanhwhile/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh¢
while/jgtgtymybc/mul_3Mulwhile/jgtgtymybc/Sigmoid:y:0while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_3£
while/jgtgtymybc/add_3AddV2while/jgtgtymybc/mul_2:z:0while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_3¯
!while/jgtgtymybc/ReadVariableOp_2ReadVariableOp,while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_2°
while/jgtgtymybc/mul_4Mul)while/jgtgtymybc/ReadVariableOp_2:value:0while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_4¨
while/jgtgtymybc/add_4AddV2while/jgtgtymybc/split:output:3while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_4
while/jgtgtymybc/Sigmoid_2Sigmoidwhile/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_2
while/jgtgtymybc/Tanh_1Tanhwhile/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh_1¦
while/jgtgtymybc/mul_5Mulwhile/jgtgtymybc/Sigmoid_2:y:0while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jgtgtymybc/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jgtgtymybc/mul_5:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jgtgtymybc/add_3:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_jgtgtymybc_biasadd_readvariableop_resource2while_jgtgtymybc_biasadd_readvariableop_resource_0"h
1while_jgtgtymybc_matmul_1_readvariableop_resource3while_jgtgtymybc_matmul_1_readvariableop_resource_0"d
/while_jgtgtymybc_matmul_readvariableop_resource1while_jgtgtymybc_matmul_readvariableop_resource_0"Z
*while_jgtgtymybc_readvariableop_1_resource,while_jgtgtymybc_readvariableop_1_resource_0"Z
*while_jgtgtymybc_readvariableop_2_resource,while_jgtgtymybc_readvariableop_2_resource_0"V
(while_jgtgtymybc_readvariableop_resource*while_jgtgtymybc_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jgtgtymybc/BiasAdd/ReadVariableOp'while/jgtgtymybc/BiasAdd/ReadVariableOp2P
&while/jgtgtymybc/MatMul/ReadVariableOp&while/jgtgtymybc/MatMul/ReadVariableOp2T
(while/jgtgtymybc/MatMul_1/ReadVariableOp(while/jgtgtymybc/MatMul_1/ReadVariableOp2B
while/jgtgtymybc/ReadVariableOpwhile/jgtgtymybc/ReadVariableOp2F
!while/jgtgtymybc/ReadVariableOp_1!while/jgtgtymybc/ReadVariableOp_12F
!while/jgtgtymybc/ReadVariableOp_2!while/jgtgtymybc/ReadVariableOp_2: 
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
ì

%__inference_signature_wrapper_1734175

qtjxeibmnq
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall
qtjxeibmnqunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_17314232
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
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
qtjxeibmnq
¡h

G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1733183

inputs<
)jgtgtymybc_matmul_readvariableop_resource:	>
+jgtgtymybc_matmul_1_readvariableop_resource:	 9
*jgtgtymybc_biasadd_readvariableop_resource:	0
"jgtgtymybc_readvariableop_resource: 2
$jgtgtymybc_readvariableop_1_resource: 2
$jgtgtymybc_readvariableop_2_resource: 
identity¢!jgtgtymybc/BiasAdd/ReadVariableOp¢ jgtgtymybc/MatMul/ReadVariableOp¢"jgtgtymybc/MatMul_1/ReadVariableOp¢jgtgtymybc/ReadVariableOp¢jgtgtymybc/ReadVariableOp_1¢jgtgtymybc/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 jgtgtymybc/MatMul/ReadVariableOpReadVariableOp)jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jgtgtymybc/MatMul/ReadVariableOp§
jgtgtymybc/MatMulMatMulstrided_slice_2:output:0(jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMulµ
"jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp+jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jgtgtymybc/MatMul_1/ReadVariableOp£
jgtgtymybc/MatMul_1MatMulzeros:output:0*jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMul_1
jgtgtymybc/addAddV2jgtgtymybc/MatMul:product:0jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/add®
!jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp*jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jgtgtymybc/BiasAdd/ReadVariableOp¥
jgtgtymybc/BiasAddBiasAddjgtgtymybc/add:z:0)jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/BiasAddz
jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jgtgtymybc/split/split_dimë
jgtgtymybc/splitSplit#jgtgtymybc/split/split_dim:output:0jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jgtgtymybc/split
jgtgtymybc/ReadVariableOpReadVariableOp"jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp
jgtgtymybc/mulMul!jgtgtymybc/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul
jgtgtymybc/add_1AddV2jgtgtymybc/split:output:0jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_1{
jgtgtymybc/SigmoidSigmoidjgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid
jgtgtymybc/ReadVariableOp_1ReadVariableOp$jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_1
jgtgtymybc/mul_1Mul#jgtgtymybc/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_1
jgtgtymybc/add_2AddV2jgtgtymybc/split:output:1jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_2
jgtgtymybc/Sigmoid_1Sigmoidjgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_1
jgtgtymybc/mul_2Muljgtgtymybc/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_2w
jgtgtymybc/TanhTanhjgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh
jgtgtymybc/mul_3Muljgtgtymybc/Sigmoid:y:0jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_3
jgtgtymybc/add_3AddV2jgtgtymybc/mul_2:z:0jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_3
jgtgtymybc/ReadVariableOp_2ReadVariableOp$jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_2
jgtgtymybc/mul_4Mul#jgtgtymybc/ReadVariableOp_2:value:0jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_4
jgtgtymybc/add_4AddV2jgtgtymybc/split:output:3jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_4
jgtgtymybc/Sigmoid_2Sigmoidjgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_2v
jgtgtymybc/Tanh_1Tanhjgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh_1
jgtgtymybc/mul_5Muljgtgtymybc/Sigmoid_2:y:0jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jgtgtymybc_matmul_readvariableop_resource+jgtgtymybc_matmul_1_readvariableop_resource*jgtgtymybc_biasadd_readvariableop_resource"jgtgtymybc_readvariableop_resource$jgtgtymybc_readvariableop_1_resource$jgtgtymybc_readvariableop_2_resource*
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
while_body_1733082*
condR
while_cond_1733081*Q
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
transpose_1³
IdentityIdentitytranspose_1:y:0"^jgtgtymybc/BiasAdd/ReadVariableOp!^jgtgtymybc/MatMul/ReadVariableOp#^jgtgtymybc/MatMul_1/ReadVariableOp^jgtgtymybc/ReadVariableOp^jgtgtymybc/ReadVariableOp_1^jgtgtymybc/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jgtgtymybc/BiasAdd/ReadVariableOp!jgtgtymybc/BiasAdd/ReadVariableOp2D
 jgtgtymybc/MatMul/ReadVariableOp jgtgtymybc/MatMul/ReadVariableOp2H
"jgtgtymybc/MatMul_1/ReadVariableOp"jgtgtymybc/MatMul_1/ReadVariableOp26
jgtgtymybc/ReadVariableOpjgtgtymybc/ReadVariableOp2:
jgtgtymybc/ReadVariableOp_1jgtgtymybc/ReadVariableOp_12:
jgtgtymybc/ReadVariableOp_2jgtgtymybc/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1735627
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1735627___redundant_placeholder05
1while_while_cond_1735627___redundant_placeholder15
1while_while_cond_1735627___redundant_placeholder25
1while_while_cond_1735627___redundant_placeholder35
1while_while_cond_1735627___redundant_placeholder45
1while_while_cond_1735627___redundant_placeholder55
1while_while_cond_1735627___redundant_placeholder6
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
±'
³
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1732268

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


í
while_cond_1732287
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1732287___redundant_placeholder05
1while_while_cond_1732287___redundant_placeholder15
1while_while_cond_1732287___redundant_placeholder25
1while_while_cond_1732287___redundant_placeholder35
1while_while_cond_1732287___redundant_placeholder45
1while_while_cond_1732287___redundant_placeholder55
1while_while_cond_1732287___redundant_placeholder6
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
while_cond_1731529
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1731529___redundant_placeholder05
1while_while_cond_1731529___redundant_placeholder15
1while_while_cond_1731529___redundant_placeholder25
1while_while_cond_1731529___redundant_placeholder35
1while_while_cond_1731529___redundant_placeholder45
1while_while_cond_1731529___redundant_placeholder55
1while_while_cond_1731529___redundant_placeholder6
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
	

,__inference_xedyzswikc_layer_call_fn_1735155
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall½
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17318732
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
¹'
µ
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1736984

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
±'
³
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1731510

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


í
while_cond_1736595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1736595___redundant_placeholder05
1while_while_cond_1736595___redundant_placeholder15
1while_while_cond_1736595___redundant_placeholder25
1while_while_cond_1736595___redundant_placeholder35
1while_while_cond_1736595___redundant_placeholder45
1while_while_cond_1736595___redundant_placeholder55
1while_while_cond_1736595___redundant_placeholder6
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
while_cond_1733081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1733081___redundant_placeholder05
1while_while_cond_1733081___redundant_placeholder15
1while_while_cond_1733081___redundant_placeholder25
1while_while_cond_1733081___redundant_placeholder35
1while_while_cond_1733081___redundant_placeholder45
1while_while_cond_1733081___redundant_placeholder55
1while_while_cond_1733081___redundant_placeholder6
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


ksthobzafc_while_cond_17345452
.ksthobzafc_while_ksthobzafc_while_loop_counter8
4ksthobzafc_while_ksthobzafc_while_maximum_iterations 
ksthobzafc_while_placeholder"
ksthobzafc_while_placeholder_1"
ksthobzafc_while_placeholder_2"
ksthobzafc_while_placeholder_34
0ksthobzafc_while_less_ksthobzafc_strided_slice_1K
Gksthobzafc_while_ksthobzafc_while_cond_1734545___redundant_placeholder0K
Gksthobzafc_while_ksthobzafc_while_cond_1734545___redundant_placeholder1K
Gksthobzafc_while_ksthobzafc_while_cond_1734545___redundant_placeholder2K
Gksthobzafc_while_ksthobzafc_while_cond_1734545___redundant_placeholder3K
Gksthobzafc_while_ksthobzafc_while_cond_1734545___redundant_placeholder4K
Gksthobzafc_while_ksthobzafc_while_cond_1734545___redundant_placeholder5K
Gksthobzafc_while_ksthobzafc_while_cond_1734545___redundant_placeholder6
ksthobzafc_while_identity
§
ksthobzafc/while/LessLessksthobzafc_while_placeholder0ksthobzafc_while_less_ksthobzafc_strided_slice_1*
T0*
_output_shapes
: 2
ksthobzafc/while/Less~
ksthobzafc/while/IdentityIdentityksthobzafc/while/Less:z:0*
T0
*
_output_shapes
: 2
ksthobzafc/while/Identity"?
ksthobzafc_while_identity"ksthobzafc/while/Identity:output:0*(
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
¦h

G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1733651

inputs<
)qjjkgjcvpf_matmul_readvariableop_resource:	 >
+qjjkgjcvpf_matmul_1_readvariableop_resource:	 9
*qjjkgjcvpf_biasadd_readvariableop_resource:	0
"qjjkgjcvpf_readvariableop_resource: 2
$qjjkgjcvpf_readvariableop_1_resource: 2
$qjjkgjcvpf_readvariableop_2_resource: 
identity¢!qjjkgjcvpf/BiasAdd/ReadVariableOp¢ qjjkgjcvpf/MatMul/ReadVariableOp¢"qjjkgjcvpf/MatMul_1/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp_1¢qjjkgjcvpf/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp)qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 qjjkgjcvpf/MatMul/ReadVariableOp§
qjjkgjcvpf/MatMulMatMulstrided_slice_2:output:0(qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMulµ
"qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp+qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"qjjkgjcvpf/MatMul_1/ReadVariableOp£
qjjkgjcvpf/MatMul_1MatMulzeros:output:0*qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMul_1
qjjkgjcvpf/addAddV2qjjkgjcvpf/MatMul:product:0qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/add®
!qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp*qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!qjjkgjcvpf/BiasAdd/ReadVariableOp¥
qjjkgjcvpf/BiasAddBiasAddqjjkgjcvpf/add:z:0)qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/BiasAddz
qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
qjjkgjcvpf/split/split_dimë
qjjkgjcvpf/splitSplit#qjjkgjcvpf/split/split_dim:output:0qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qjjkgjcvpf/split
qjjkgjcvpf/ReadVariableOpReadVariableOp"qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp
qjjkgjcvpf/mulMul!qjjkgjcvpf/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul
qjjkgjcvpf/add_1AddV2qjjkgjcvpf/split:output:0qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_1{
qjjkgjcvpf/SigmoidSigmoidqjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid
qjjkgjcvpf/ReadVariableOp_1ReadVariableOp$qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_1
qjjkgjcvpf/mul_1Mul#qjjkgjcvpf/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_1
qjjkgjcvpf/add_2AddV2qjjkgjcvpf/split:output:1qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_2
qjjkgjcvpf/Sigmoid_1Sigmoidqjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_1
qjjkgjcvpf/mul_2Mulqjjkgjcvpf/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_2w
qjjkgjcvpf/TanhTanhqjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh
qjjkgjcvpf/mul_3Mulqjjkgjcvpf/Sigmoid:y:0qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_3
qjjkgjcvpf/add_3AddV2qjjkgjcvpf/mul_2:z:0qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_3
qjjkgjcvpf/ReadVariableOp_2ReadVariableOp$qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_2
qjjkgjcvpf/mul_4Mul#qjjkgjcvpf/ReadVariableOp_2:value:0qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_4
qjjkgjcvpf/add_4AddV2qjjkgjcvpf/split:output:3qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_4
qjjkgjcvpf/Sigmoid_2Sigmoidqjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_2v
qjjkgjcvpf/Tanh_1Tanhqjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh_1
qjjkgjcvpf/mul_5Mulqjjkgjcvpf/Sigmoid_2:y:0qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)qjjkgjcvpf_matmul_readvariableop_resource+qjjkgjcvpf_matmul_1_readvariableop_resource*qjjkgjcvpf_biasadd_readvariableop_resource"qjjkgjcvpf_readvariableop_resource$qjjkgjcvpf_readvariableop_1_resource$qjjkgjcvpf_readvariableop_2_resource*
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
while_body_1733550*
condR
while_cond_1733549*Q
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
transpose_1¸
IdentityIdentitystrided_slice_3:output:0"^qjjkgjcvpf/BiasAdd/ReadVariableOp!^qjjkgjcvpf/MatMul/ReadVariableOp#^qjjkgjcvpf/MatMul_1/ReadVariableOp^qjjkgjcvpf/ReadVariableOp^qjjkgjcvpf/ReadVariableOp_1^qjjkgjcvpf/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!qjjkgjcvpf/BiasAdd/ReadVariableOp!qjjkgjcvpf/BiasAdd/ReadVariableOp2D
 qjjkgjcvpf/MatMul/ReadVariableOp qjjkgjcvpf/MatMul/ReadVariableOp2H
"qjjkgjcvpf/MatMul_1/ReadVariableOp"qjjkgjcvpf/MatMul_1/ReadVariableOp26
qjjkgjcvpf/ReadVariableOpqjjkgjcvpf/ReadVariableOp2:
qjjkgjcvpf/ReadVariableOp_1qjjkgjcvpf/ReadVariableOp_12:
qjjkgjcvpf/ReadVariableOp_2qjjkgjcvpf/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
©	
(sequential_ksthobzafc_while_cond_1731315H
Dsequential_ksthobzafc_while_sequential_ksthobzafc_while_loop_counterN
Jsequential_ksthobzafc_while_sequential_ksthobzafc_while_maximum_iterations+
'sequential_ksthobzafc_while_placeholder-
)sequential_ksthobzafc_while_placeholder_1-
)sequential_ksthobzafc_while_placeholder_2-
)sequential_ksthobzafc_while_placeholder_3J
Fsequential_ksthobzafc_while_less_sequential_ksthobzafc_strided_slice_1a
]sequential_ksthobzafc_while_sequential_ksthobzafc_while_cond_1731315___redundant_placeholder0a
]sequential_ksthobzafc_while_sequential_ksthobzafc_while_cond_1731315___redundant_placeholder1a
]sequential_ksthobzafc_while_sequential_ksthobzafc_while_cond_1731315___redundant_placeholder2a
]sequential_ksthobzafc_while_sequential_ksthobzafc_while_cond_1731315___redundant_placeholder3a
]sequential_ksthobzafc_while_sequential_ksthobzafc_while_cond_1731315___redundant_placeholder4a
]sequential_ksthobzafc_while_sequential_ksthobzafc_while_cond_1731315___redundant_placeholder5a
]sequential_ksthobzafc_while_sequential_ksthobzafc_while_cond_1731315___redundant_placeholder6(
$sequential_ksthobzafc_while_identity
Þ
 sequential/ksthobzafc/while/LessLess'sequential_ksthobzafc_while_placeholderFsequential_ksthobzafc_while_less_sequential_ksthobzafc_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/ksthobzafc/while/Less
$sequential/ksthobzafc/while/IdentityIdentity$sequential/ksthobzafc/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/ksthobzafc/while/Identity"U
$sequential_ksthobzafc_while_identity-sequential/ksthobzafc/while/Identity:output:0*(
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
¸
¼
G__inference_sequential_layer_call_and_return_conditional_losses_1733407

inputs(
zdelabjare_1732984: 
zdelabjare_1732986:%
xedyzswikc_1733184:	%
xedyzswikc_1733186:	 !
xedyzswikc_1733188:	 
xedyzswikc_1733190:  
xedyzswikc_1733192:  
xedyzswikc_1733194: %
ksthobzafc_1733377:	 %
ksthobzafc_1733379:	 !
ksthobzafc_1733381:	 
ksthobzafc_1733383:  
ksthobzafc_1733385:  
ksthobzafc_1733387: $
vfmtgawzzo_1733401:  
vfmtgawzzo_1733403:
identity¢"ksthobzafc/StatefulPartitionedCall¢"vfmtgawzzo/StatefulPartitionedCall¢"xedyzswikc/StatefulPartitionedCall¢"zdelabjare/StatefulPartitionedCall¬
"zdelabjare/StatefulPartitionedCallStatefulPartitionedCallinputszdelabjare_1732984zdelabjare_1732986*
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
GPU2*0J 8 *P
fKRI
G__inference_zdelabjare_layer_call_and_return_conditional_losses_17329832$
"zdelabjare/StatefulPartitionedCall
ojrtxmspqi/PartitionedCallPartitionedCall+zdelabjare/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_17330022
ojrtxmspqi/PartitionedCall
"xedyzswikc/StatefulPartitionedCallStatefulPartitionedCall#ojrtxmspqi/PartitionedCall:output:0xedyzswikc_1733184xedyzswikc_1733186xedyzswikc_1733188xedyzswikc_1733190xedyzswikc_1733192xedyzswikc_1733194*
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17331832$
"xedyzswikc/StatefulPartitionedCall¡
"ksthobzafc/StatefulPartitionedCallStatefulPartitionedCall+xedyzswikc/StatefulPartitionedCall:output:0ksthobzafc_1733377ksthobzafc_1733379ksthobzafc_1733381ksthobzafc_1733383ksthobzafc_1733385ksthobzafc_1733387*
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17333762$
"ksthobzafc/StatefulPartitionedCallÉ
"vfmtgawzzo/StatefulPartitionedCallStatefulPartitionedCall+ksthobzafc/StatefulPartitionedCall:output:0vfmtgawzzo_1733401vfmtgawzzo_1733403*
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
GPU2*0J 8 *P
fKRI
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_17334002$
"vfmtgawzzo/StatefulPartitionedCall
IdentityIdentity+vfmtgawzzo/StatefulPartitionedCall:output:0#^ksthobzafc/StatefulPartitionedCall#^vfmtgawzzo/StatefulPartitionedCall#^xedyzswikc/StatefulPartitionedCall#^zdelabjare/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ksthobzafc/StatefulPartitionedCall"ksthobzafc/StatefulPartitionedCall2H
"vfmtgawzzo/StatefulPartitionedCall"vfmtgawzzo/StatefulPartitionedCall2H
"xedyzswikc/StatefulPartitionedCall"xedyzswikc/StatefulPartitionedCall2H
"zdelabjare/StatefulPartitionedCall"zdelabjare/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àY

while_body_1733275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 F
3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 A
2while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	8
*while_qjjkgjcvpf_readvariableop_resource_0: :
,while_qjjkgjcvpf_readvariableop_1_resource_0: :
,while_qjjkgjcvpf_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_qjjkgjcvpf_matmul_readvariableop_resource:	 D
1while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 ?
0while_qjjkgjcvpf_biasadd_readvariableop_resource:	6
(while_qjjkgjcvpf_readvariableop_resource: 8
*while_qjjkgjcvpf_readvariableop_1_resource: 8
*while_qjjkgjcvpf_readvariableop_2_resource: ¢'while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢&while/qjjkgjcvpf/MatMul/ReadVariableOp¢(while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢while/qjjkgjcvpf/ReadVariableOp¢!while/qjjkgjcvpf/ReadVariableOp_1¢!while/qjjkgjcvpf/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp1while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/qjjkgjcvpf/MatMul/ReadVariableOpÑ
while/qjjkgjcvpf/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMulÉ
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpº
while/qjjkgjcvpf/MatMul_1MatMulwhile_placeholder_20while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMul_1°
while/qjjkgjcvpf/addAddV2!while/qjjkgjcvpf/MatMul:product:0#while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/addÂ
'while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp2while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp½
while/qjjkgjcvpf/BiasAddBiasAddwhile/qjjkgjcvpf/add:z:0/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/BiasAdd
 while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/qjjkgjcvpf/split/split_dim
while/qjjkgjcvpf/splitSplit)while/qjjkgjcvpf/split/split_dim:output:0!while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/qjjkgjcvpf/split©
while/qjjkgjcvpf/ReadVariableOpReadVariableOp*while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/qjjkgjcvpf/ReadVariableOp£
while/qjjkgjcvpf/mulMul'while/qjjkgjcvpf/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul¦
while/qjjkgjcvpf/add_1AddV2while/qjjkgjcvpf/split:output:0while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_1
while/qjjkgjcvpf/SigmoidSigmoidwhile/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid¯
!while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp,while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_1©
while/qjjkgjcvpf/mul_1Mul)while/qjjkgjcvpf/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_1¨
while/qjjkgjcvpf/add_2AddV2while/qjjkgjcvpf/split:output:1while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_2
while/qjjkgjcvpf/Sigmoid_1Sigmoidwhile/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_1
while/qjjkgjcvpf/mul_2Mulwhile/qjjkgjcvpf/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_2
while/qjjkgjcvpf/TanhTanhwhile/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh¢
while/qjjkgjcvpf/mul_3Mulwhile/qjjkgjcvpf/Sigmoid:y:0while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_3£
while/qjjkgjcvpf/add_3AddV2while/qjjkgjcvpf/mul_2:z:0while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_3¯
!while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp,while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_2°
while/qjjkgjcvpf/mul_4Mul)while/qjjkgjcvpf/ReadVariableOp_2:value:0while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_4¨
while/qjjkgjcvpf/add_4AddV2while/qjjkgjcvpf/split:output:3while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_4
while/qjjkgjcvpf/Sigmoid_2Sigmoidwhile/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_2
while/qjjkgjcvpf/Tanh_1Tanhwhile/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh_1¦
while/qjjkgjcvpf/mul_5Mulwhile/qjjkgjcvpf/Sigmoid_2:y:0while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/qjjkgjcvpf/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/qjjkgjcvpf/mul_5:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/qjjkgjcvpf/add_3:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_qjjkgjcvpf_biasadd_readvariableop_resource2while_qjjkgjcvpf_biasadd_readvariableop_resource_0"h
1while_qjjkgjcvpf_matmul_1_readvariableop_resource3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"d
/while_qjjkgjcvpf_matmul_readvariableop_resource1while_qjjkgjcvpf_matmul_readvariableop_resource_0"Z
*while_qjjkgjcvpf_readvariableop_1_resource,while_qjjkgjcvpf_readvariableop_1_resource_0"Z
*while_qjjkgjcvpf_readvariableop_2_resource,while_qjjkgjcvpf_readvariableop_2_resource_0"V
(while_qjjkgjcvpf_readvariableop_resource*while_qjjkgjcvpf_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp'while/qjjkgjcvpf/BiasAdd/ReadVariableOp2P
&while/qjjkgjcvpf/MatMul/ReadVariableOp&while/qjjkgjcvpf/MatMul/ReadVariableOp2T
(while/qjjkgjcvpf/MatMul_1/ReadVariableOp(while/qjjkgjcvpf/MatMul_1/ReadVariableOp2B
while/qjjkgjcvpf/ReadVariableOpwhile/qjjkgjcvpf/ReadVariableOp2F
!while/qjjkgjcvpf/ReadVariableOp_1!while/qjjkgjcvpf/ReadVariableOp_12F
!while/qjjkgjcvpf/ReadVariableOp_2!while/qjjkgjcvpf/ReadVariableOp_2: 
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
while_cond_1733274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1733274___redundant_placeholder05
1while_while_cond_1733274___redundant_placeholder15
1while_while_cond_1733274___redundant_placeholder25
1while_while_cond_1733274___redundant_placeholder35
1while_while_cond_1733274___redundant_placeholder45
1while_while_cond_1733274___redundant_placeholder55
1while_while_cond_1733274___redundant_placeholder6
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

À
,__inference_jgtgtymybc_layer_call_fn_1736762

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

identity_2¢StatefulPartitionedCallì
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
GPU2*0J 8 *P
fKRI
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_17316972
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
àY

while_body_1733550
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 F
3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 A
2while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	8
*while_qjjkgjcvpf_readvariableop_resource_0: :
,while_qjjkgjcvpf_readvariableop_1_resource_0: :
,while_qjjkgjcvpf_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_qjjkgjcvpf_matmul_readvariableop_resource:	 D
1while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 ?
0while_qjjkgjcvpf_biasadd_readvariableop_resource:	6
(while_qjjkgjcvpf_readvariableop_resource: 8
*while_qjjkgjcvpf_readvariableop_1_resource: 8
*while_qjjkgjcvpf_readvariableop_2_resource: ¢'while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢&while/qjjkgjcvpf/MatMul/ReadVariableOp¢(while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢while/qjjkgjcvpf/ReadVariableOp¢!while/qjjkgjcvpf/ReadVariableOp_1¢!while/qjjkgjcvpf/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp1while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/qjjkgjcvpf/MatMul/ReadVariableOpÑ
while/qjjkgjcvpf/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMulÉ
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpº
while/qjjkgjcvpf/MatMul_1MatMulwhile_placeholder_20while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMul_1°
while/qjjkgjcvpf/addAddV2!while/qjjkgjcvpf/MatMul:product:0#while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/addÂ
'while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp2while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp½
while/qjjkgjcvpf/BiasAddBiasAddwhile/qjjkgjcvpf/add:z:0/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/BiasAdd
 while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/qjjkgjcvpf/split/split_dim
while/qjjkgjcvpf/splitSplit)while/qjjkgjcvpf/split/split_dim:output:0!while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/qjjkgjcvpf/split©
while/qjjkgjcvpf/ReadVariableOpReadVariableOp*while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/qjjkgjcvpf/ReadVariableOp£
while/qjjkgjcvpf/mulMul'while/qjjkgjcvpf/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul¦
while/qjjkgjcvpf/add_1AddV2while/qjjkgjcvpf/split:output:0while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_1
while/qjjkgjcvpf/SigmoidSigmoidwhile/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid¯
!while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp,while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_1©
while/qjjkgjcvpf/mul_1Mul)while/qjjkgjcvpf/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_1¨
while/qjjkgjcvpf/add_2AddV2while/qjjkgjcvpf/split:output:1while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_2
while/qjjkgjcvpf/Sigmoid_1Sigmoidwhile/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_1
while/qjjkgjcvpf/mul_2Mulwhile/qjjkgjcvpf/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_2
while/qjjkgjcvpf/TanhTanhwhile/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh¢
while/qjjkgjcvpf/mul_3Mulwhile/qjjkgjcvpf/Sigmoid:y:0while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_3£
while/qjjkgjcvpf/add_3AddV2while/qjjkgjcvpf/mul_2:z:0while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_3¯
!while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp,while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_2°
while/qjjkgjcvpf/mul_4Mul)while/qjjkgjcvpf/ReadVariableOp_2:value:0while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_4¨
while/qjjkgjcvpf/add_4AddV2while/qjjkgjcvpf/split:output:3while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_4
while/qjjkgjcvpf/Sigmoid_2Sigmoidwhile/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_2
while/qjjkgjcvpf/Tanh_1Tanhwhile/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh_1¦
while/qjjkgjcvpf/mul_5Mulwhile/qjjkgjcvpf/Sigmoid_2:y:0while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/qjjkgjcvpf/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/qjjkgjcvpf/mul_5:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/qjjkgjcvpf/add_3:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_qjjkgjcvpf_biasadd_readvariableop_resource2while_qjjkgjcvpf_biasadd_readvariableop_resource_0"h
1while_qjjkgjcvpf_matmul_1_readvariableop_resource3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"d
/while_qjjkgjcvpf_matmul_readvariableop_resource1while_qjjkgjcvpf_matmul_readvariableop_resource_0"Z
*while_qjjkgjcvpf_readvariableop_1_resource,while_qjjkgjcvpf_readvariableop_1_resource_0"Z
*while_qjjkgjcvpf_readvariableop_2_resource,while_qjjkgjcvpf_readvariableop_2_resource_0"V
(while_qjjkgjcvpf_readvariableop_resource*while_qjjkgjcvpf_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp'while/qjjkgjcvpf/BiasAdd/ReadVariableOp2P
&while/qjjkgjcvpf/MatMul/ReadVariableOp&while/qjjkgjcvpf/MatMul/ReadVariableOp2T
(while/qjjkgjcvpf/MatMul_1/ReadVariableOp(while/qjjkgjcvpf/MatMul_1/ReadVariableOp2B
while/qjjkgjcvpf/ReadVariableOpwhile/qjjkgjcvpf/ReadVariableOp2F
!while/qjjkgjcvpf/ReadVariableOp_1!while/qjjkgjcvpf/ReadVariableOp_12F
!while/qjjkgjcvpf/ReadVariableOp_2!while/qjjkgjcvpf/ReadVariableOp_2: 
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
³F
ê
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1731873

inputs%
jgtgtymybc_1731774:	%
jgtgtymybc_1731776:	 !
jgtgtymybc_1731778:	 
jgtgtymybc_1731780:  
jgtgtymybc_1731782:  
jgtgtymybc_1731784: 
identity¢"jgtgtymybc/StatefulPartitionedCall¢whileD
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
strided_slice_2Ú
"jgtgtymybc/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0jgtgtymybc_1731774jgtgtymybc_1731776jgtgtymybc_1731778jgtgtymybc_1731780jgtgtymybc_1731782jgtgtymybc_1731784*
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
GPU2*0J 8 *P
fKRI
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_17316972$
"jgtgtymybc/StatefulPartitionedCall
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
while/loop_counterð
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0jgtgtymybc_1731774jgtgtymybc_1731776jgtgtymybc_1731778jgtgtymybc_1731780jgtgtymybc_1731782jgtgtymybc_1731784*
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
while_body_1731793*
condR
while_cond_1731792*Q
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
transpose_1
IdentityIdentitytranspose_1:y:0#^jgtgtymybc/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"jgtgtymybc/StatefulPartitionedCall"jgtgtymybc/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1736055
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1736055___redundant_placeholder05
1while_while_cond_1736055___redundant_placeholder15
1while_while_cond_1736055___redundant_placeholder25
1while_while_cond_1736055___redundant_placeholder35
1while_while_cond_1736055___redundant_placeholder45
1while_while_cond_1736055___redundant_placeholder55
1while_while_cond_1736055___redundant_placeholder6
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
¡h

G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735909

inputs<
)jgtgtymybc_matmul_readvariableop_resource:	>
+jgtgtymybc_matmul_1_readvariableop_resource:	 9
*jgtgtymybc_biasadd_readvariableop_resource:	0
"jgtgtymybc_readvariableop_resource: 2
$jgtgtymybc_readvariableop_1_resource: 2
$jgtgtymybc_readvariableop_2_resource: 
identity¢!jgtgtymybc/BiasAdd/ReadVariableOp¢ jgtgtymybc/MatMul/ReadVariableOp¢"jgtgtymybc/MatMul_1/ReadVariableOp¢jgtgtymybc/ReadVariableOp¢jgtgtymybc/ReadVariableOp_1¢jgtgtymybc/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 jgtgtymybc/MatMul/ReadVariableOpReadVariableOp)jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jgtgtymybc/MatMul/ReadVariableOp§
jgtgtymybc/MatMulMatMulstrided_slice_2:output:0(jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMulµ
"jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp+jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jgtgtymybc/MatMul_1/ReadVariableOp£
jgtgtymybc/MatMul_1MatMulzeros:output:0*jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMul_1
jgtgtymybc/addAddV2jgtgtymybc/MatMul:product:0jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/add®
!jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp*jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jgtgtymybc/BiasAdd/ReadVariableOp¥
jgtgtymybc/BiasAddBiasAddjgtgtymybc/add:z:0)jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/BiasAddz
jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jgtgtymybc/split/split_dimë
jgtgtymybc/splitSplit#jgtgtymybc/split/split_dim:output:0jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jgtgtymybc/split
jgtgtymybc/ReadVariableOpReadVariableOp"jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp
jgtgtymybc/mulMul!jgtgtymybc/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul
jgtgtymybc/add_1AddV2jgtgtymybc/split:output:0jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_1{
jgtgtymybc/SigmoidSigmoidjgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid
jgtgtymybc/ReadVariableOp_1ReadVariableOp$jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_1
jgtgtymybc/mul_1Mul#jgtgtymybc/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_1
jgtgtymybc/add_2AddV2jgtgtymybc/split:output:1jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_2
jgtgtymybc/Sigmoid_1Sigmoidjgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_1
jgtgtymybc/mul_2Muljgtgtymybc/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_2w
jgtgtymybc/TanhTanhjgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh
jgtgtymybc/mul_3Muljgtgtymybc/Sigmoid:y:0jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_3
jgtgtymybc/add_3AddV2jgtgtymybc/mul_2:z:0jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_3
jgtgtymybc/ReadVariableOp_2ReadVariableOp$jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_2
jgtgtymybc/mul_4Mul#jgtgtymybc/ReadVariableOp_2:value:0jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_4
jgtgtymybc/add_4AddV2jgtgtymybc/split:output:3jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_4
jgtgtymybc/Sigmoid_2Sigmoidjgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_2v
jgtgtymybc/Tanh_1Tanhjgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh_1
jgtgtymybc/mul_5Muljgtgtymybc/Sigmoid_2:y:0jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jgtgtymybc_matmul_readvariableop_resource+jgtgtymybc_matmul_1_readvariableop_resource*jgtgtymybc_biasadd_readvariableop_resource"jgtgtymybc_readvariableop_resource$jgtgtymybc_readvariableop_1_resource$jgtgtymybc_readvariableop_2_resource*
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
while_body_1735808*
condR
while_cond_1735807*Q
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
transpose_1³
IdentityIdentitytranspose_1:y:0"^jgtgtymybc/BiasAdd/ReadVariableOp!^jgtgtymybc/MatMul/ReadVariableOp#^jgtgtymybc/MatMul_1/ReadVariableOp^jgtgtymybc/ReadVariableOp^jgtgtymybc/ReadVariableOp_1^jgtgtymybc/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jgtgtymybc/BiasAdd/ReadVariableOp!jgtgtymybc/BiasAdd/ReadVariableOp2D
 jgtgtymybc/MatMul/ReadVariableOp jgtgtymybc/MatMul/ReadVariableOp2H
"jgtgtymybc/MatMul_1/ReadVariableOp"jgtgtymybc/MatMul_1/ReadVariableOp26
jgtgtymybc/ReadVariableOpjgtgtymybc/ReadVariableOp2:
jgtgtymybc/ReadVariableOp_1jgtgtymybc/ReadVariableOp_12:
jgtgtymybc/ReadVariableOp_2jgtgtymybc/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1731792
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1731792___redundant_placeholder05
1while_while_cond_1731792___redundant_placeholder15
1while_while_cond_1731792___redundant_placeholder25
1while_while_cond_1731792___redundant_placeholder35
1while_while_cond_1731792___redundant_placeholder45
1while_while_cond_1731792___redundant_placeholder55
1while_while_cond_1731792___redundant_placeholder6
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


xedyzswikc_while_cond_17343692
.xedyzswikc_while_xedyzswikc_while_loop_counter8
4xedyzswikc_while_xedyzswikc_while_maximum_iterations 
xedyzswikc_while_placeholder"
xedyzswikc_while_placeholder_1"
xedyzswikc_while_placeholder_2"
xedyzswikc_while_placeholder_34
0xedyzswikc_while_less_xedyzswikc_strided_slice_1K
Gxedyzswikc_while_xedyzswikc_while_cond_1734369___redundant_placeholder0K
Gxedyzswikc_while_xedyzswikc_while_cond_1734369___redundant_placeholder1K
Gxedyzswikc_while_xedyzswikc_while_cond_1734369___redundant_placeholder2K
Gxedyzswikc_while_xedyzswikc_while_cond_1734369___redundant_placeholder3K
Gxedyzswikc_while_xedyzswikc_while_cond_1734369___redundant_placeholder4K
Gxedyzswikc_while_xedyzswikc_while_cond_1734369___redundant_placeholder5K
Gxedyzswikc_while_xedyzswikc_while_cond_1734369___redundant_placeholder6
xedyzswikc_while_identity
§
xedyzswikc/while/LessLessxedyzswikc_while_placeholder0xedyzswikc_while_less_xedyzswikc_strided_slice_1*
T0*
_output_shapes
: 2
xedyzswikc/while/Less~
xedyzswikc/while/IdentityIdentityxedyzswikc/while/Less:z:0*
T0
*
_output_shapes
: 2
xedyzswikc/while/Identity"?
xedyzswikc_while_identity"xedyzswikc/while/Identity:output:0*(
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

c
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_1733002

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


í
while_cond_1736235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1736235___redundant_placeholder05
1while_while_cond_1736235___redundant_placeholder15
1while_while_cond_1736235___redundant_placeholder25
1while_while_cond_1736235___redundant_placeholder35
1while_while_cond_1736235___redundant_placeholder45
1while_while_cond_1736235___redundant_placeholder55
1while_while_cond_1736235___redundant_placeholder6
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
àh

G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735369
inputs_0<
)jgtgtymybc_matmul_readvariableop_resource:	>
+jgtgtymybc_matmul_1_readvariableop_resource:	 9
*jgtgtymybc_biasadd_readvariableop_resource:	0
"jgtgtymybc_readvariableop_resource: 2
$jgtgtymybc_readvariableop_1_resource: 2
$jgtgtymybc_readvariableop_2_resource: 
identity¢!jgtgtymybc/BiasAdd/ReadVariableOp¢ jgtgtymybc/MatMul/ReadVariableOp¢"jgtgtymybc/MatMul_1/ReadVariableOp¢jgtgtymybc/ReadVariableOp¢jgtgtymybc/ReadVariableOp_1¢jgtgtymybc/ReadVariableOp_2¢whileF
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
strided_slice_2¯
 jgtgtymybc/MatMul/ReadVariableOpReadVariableOp)jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jgtgtymybc/MatMul/ReadVariableOp§
jgtgtymybc/MatMulMatMulstrided_slice_2:output:0(jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMulµ
"jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp+jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jgtgtymybc/MatMul_1/ReadVariableOp£
jgtgtymybc/MatMul_1MatMulzeros:output:0*jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMul_1
jgtgtymybc/addAddV2jgtgtymybc/MatMul:product:0jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/add®
!jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp*jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jgtgtymybc/BiasAdd/ReadVariableOp¥
jgtgtymybc/BiasAddBiasAddjgtgtymybc/add:z:0)jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/BiasAddz
jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jgtgtymybc/split/split_dimë
jgtgtymybc/splitSplit#jgtgtymybc/split/split_dim:output:0jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jgtgtymybc/split
jgtgtymybc/ReadVariableOpReadVariableOp"jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp
jgtgtymybc/mulMul!jgtgtymybc/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul
jgtgtymybc/add_1AddV2jgtgtymybc/split:output:0jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_1{
jgtgtymybc/SigmoidSigmoidjgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid
jgtgtymybc/ReadVariableOp_1ReadVariableOp$jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_1
jgtgtymybc/mul_1Mul#jgtgtymybc/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_1
jgtgtymybc/add_2AddV2jgtgtymybc/split:output:1jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_2
jgtgtymybc/Sigmoid_1Sigmoidjgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_1
jgtgtymybc/mul_2Muljgtgtymybc/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_2w
jgtgtymybc/TanhTanhjgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh
jgtgtymybc/mul_3Muljgtgtymybc/Sigmoid:y:0jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_3
jgtgtymybc/add_3AddV2jgtgtymybc/mul_2:z:0jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_3
jgtgtymybc/ReadVariableOp_2ReadVariableOp$jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_2
jgtgtymybc/mul_4Mul#jgtgtymybc/ReadVariableOp_2:value:0jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_4
jgtgtymybc/add_4AddV2jgtgtymybc/split:output:3jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_4
jgtgtymybc/Sigmoid_2Sigmoidjgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_2v
jgtgtymybc/Tanh_1Tanhjgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh_1
jgtgtymybc/mul_5Muljgtgtymybc/Sigmoid_2:y:0jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jgtgtymybc_matmul_readvariableop_resource+jgtgtymybc_matmul_1_readvariableop_resource*jgtgtymybc_biasadd_readvariableop_resource"jgtgtymybc_readvariableop_resource$jgtgtymybc_readvariableop_1_resource$jgtgtymybc_readvariableop_2_resource*
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
while_body_1735268*
condR
while_cond_1735267*Q
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
transpose_1¼
IdentityIdentitytranspose_1:y:0"^jgtgtymybc/BiasAdd/ReadVariableOp!^jgtgtymybc/MatMul/ReadVariableOp#^jgtgtymybc/MatMul_1/ReadVariableOp^jgtgtymybc/ReadVariableOp^jgtgtymybc/ReadVariableOp_1^jgtgtymybc/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jgtgtymybc/BiasAdd/ReadVariableOp!jgtgtymybc/BiasAdd/ReadVariableOp2D
 jgtgtymybc/MatMul/ReadVariableOp jgtgtymybc/MatMul/ReadVariableOp2H
"jgtgtymybc/MatMul_1/ReadVariableOp"jgtgtymybc/MatMul_1/ReadVariableOp26
jgtgtymybc/ReadVariableOpjgtgtymybc/ReadVariableOp2:
jgtgtymybc/ReadVariableOp_1jgtgtymybc/ReadVariableOp_12:
jgtgtymybc/ReadVariableOp_2jgtgtymybc/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ü

(sequential_xedyzswikc_while_body_1731140H
Dsequential_xedyzswikc_while_sequential_xedyzswikc_while_loop_counterN
Jsequential_xedyzswikc_while_sequential_xedyzswikc_while_maximum_iterations+
'sequential_xedyzswikc_while_placeholder-
)sequential_xedyzswikc_while_placeholder_1-
)sequential_xedyzswikc_while_placeholder_2-
)sequential_xedyzswikc_while_placeholder_3G
Csequential_xedyzswikc_while_sequential_xedyzswikc_strided_slice_1_0
sequential_xedyzswikc_while_tensorarrayv2read_tensorlistgetitem_sequential_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0:	\
Isequential_xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 W
Hsequential_xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0:	N
@sequential_xedyzswikc_while_jgtgtymybc_readvariableop_resource_0: P
Bsequential_xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0: P
Bsequential_xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0: (
$sequential_xedyzswikc_while_identity*
&sequential_xedyzswikc_while_identity_1*
&sequential_xedyzswikc_while_identity_2*
&sequential_xedyzswikc_while_identity_3*
&sequential_xedyzswikc_while_identity_4*
&sequential_xedyzswikc_while_identity_5E
Asequential_xedyzswikc_while_sequential_xedyzswikc_strided_slice_1
}sequential_xedyzswikc_while_tensorarrayv2read_tensorlistgetitem_sequential_xedyzswikc_tensorarrayunstack_tensorlistfromtensorX
Esequential_xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource:	Z
Gsequential_xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource:	 U
Fsequential_xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource:	L
>sequential_xedyzswikc_while_jgtgtymybc_readvariableop_resource: N
@sequential_xedyzswikc_while_jgtgtymybc_readvariableop_1_resource: N
@sequential_xedyzswikc_while_jgtgtymybc_readvariableop_2_resource: ¢=sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp¢<sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp¢>sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp¢5sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp¢7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_1¢7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2ï
Msequential/xedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/xedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/xedyzswikc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_xedyzswikc_while_tensorarrayv2read_tensorlistgetitem_sequential_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0'sequential_xedyzswikc_while_placeholderVsequential/xedyzswikc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/xedyzswikc/while/TensorArrayV2Read/TensorListGetItem
<sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOpGsequential_xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp©
-sequential/xedyzswikc/while/jgtgtymybc/MatMulMatMulFsequential/xedyzswikc/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/xedyzswikc/while/jgtgtymybc/MatMul
>sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOpIsequential_xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp
/sequential/xedyzswikc/while/jgtgtymybc/MatMul_1MatMul)sequential_xedyzswikc_while_placeholder_2Fsequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/xedyzswikc/while/jgtgtymybc/MatMul_1
*sequential/xedyzswikc/while/jgtgtymybc/addAddV27sequential/xedyzswikc/while/jgtgtymybc/MatMul:product:09sequential/xedyzswikc/while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/xedyzswikc/while/jgtgtymybc/add
=sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOpHsequential_xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp
.sequential/xedyzswikc/while/jgtgtymybc/BiasAddBiasAdd.sequential/xedyzswikc/while/jgtgtymybc/add:z:0Esequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/xedyzswikc/while/jgtgtymybc/BiasAdd²
6sequential/xedyzswikc/while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/xedyzswikc/while/jgtgtymybc/split/split_dimÛ
,sequential/xedyzswikc/while/jgtgtymybc/splitSplit?sequential/xedyzswikc/while/jgtgtymybc/split/split_dim:output:07sequential/xedyzswikc/while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/xedyzswikc/while/jgtgtymybc/splitë
5sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOpReadVariableOp@sequential_xedyzswikc_while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOpû
*sequential/xedyzswikc/while/jgtgtymybc/mulMul=sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp:value:0)sequential_xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/xedyzswikc/while/jgtgtymybc/mulþ
,sequential/xedyzswikc/while/jgtgtymybc/add_1AddV25sequential/xedyzswikc/while/jgtgtymybc/split:output:0.sequential/xedyzswikc/while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/add_1Ï
.sequential/xedyzswikc/while/jgtgtymybc/SigmoidSigmoid0sequential/xedyzswikc/while/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/xedyzswikc/while/jgtgtymybc/Sigmoidñ
7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_1ReadVariableOpBsequential_xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_1
,sequential/xedyzswikc/while/jgtgtymybc/mul_1Mul?sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_1:value:0)sequential_xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/mul_1
,sequential/xedyzswikc/while/jgtgtymybc/add_2AddV25sequential/xedyzswikc/while/jgtgtymybc/split:output:10sequential/xedyzswikc/while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/add_2Ó
0sequential/xedyzswikc/while/jgtgtymybc/Sigmoid_1Sigmoid0sequential/xedyzswikc/while/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/xedyzswikc/while/jgtgtymybc/Sigmoid_1ö
,sequential/xedyzswikc/while/jgtgtymybc/mul_2Mul4sequential/xedyzswikc/while/jgtgtymybc/Sigmoid_1:y:0)sequential_xedyzswikc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/mul_2Ë
+sequential/xedyzswikc/while/jgtgtymybc/TanhTanh5sequential/xedyzswikc/while/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/xedyzswikc/while/jgtgtymybc/Tanhú
,sequential/xedyzswikc/while/jgtgtymybc/mul_3Mul2sequential/xedyzswikc/while/jgtgtymybc/Sigmoid:y:0/sequential/xedyzswikc/while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/mul_3û
,sequential/xedyzswikc/while/jgtgtymybc/add_3AddV20sequential/xedyzswikc/while/jgtgtymybc/mul_2:z:00sequential/xedyzswikc/while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/add_3ñ
7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2ReadVariableOpBsequential_xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2
,sequential/xedyzswikc/while/jgtgtymybc/mul_4Mul?sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2:value:00sequential/xedyzswikc/while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/mul_4
,sequential/xedyzswikc/while/jgtgtymybc/add_4AddV25sequential/xedyzswikc/while/jgtgtymybc/split:output:30sequential/xedyzswikc/while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/add_4Ó
0sequential/xedyzswikc/while/jgtgtymybc/Sigmoid_2Sigmoid0sequential/xedyzswikc/while/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/xedyzswikc/while/jgtgtymybc/Sigmoid_2Ê
-sequential/xedyzswikc/while/jgtgtymybc/Tanh_1Tanh0sequential/xedyzswikc/while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/xedyzswikc/while/jgtgtymybc/Tanh_1þ
,sequential/xedyzswikc/while/jgtgtymybc/mul_5Mul4sequential/xedyzswikc/while/jgtgtymybc/Sigmoid_2:y:01sequential/xedyzswikc/while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/xedyzswikc/while/jgtgtymybc/mul_5Ì
@sequential/xedyzswikc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_xedyzswikc_while_placeholder_1'sequential_xedyzswikc_while_placeholder0sequential/xedyzswikc/while/jgtgtymybc/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/xedyzswikc/while/TensorArrayV2Write/TensorListSetItem
!sequential/xedyzswikc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/xedyzswikc/while/add/yÁ
sequential/xedyzswikc/while/addAddV2'sequential_xedyzswikc_while_placeholder*sequential/xedyzswikc/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/xedyzswikc/while/add
#sequential/xedyzswikc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/xedyzswikc/while/add_1/yä
!sequential/xedyzswikc/while/add_1AddV2Dsequential_xedyzswikc_while_sequential_xedyzswikc_while_loop_counter,sequential/xedyzswikc/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/xedyzswikc/while/add_1
$sequential/xedyzswikc/while/IdentityIdentity%sequential/xedyzswikc/while/add_1:z:0>^sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp=^sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp?^sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp6^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp8^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_18^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/xedyzswikc/while/Identityµ
&sequential/xedyzswikc/while/Identity_1IdentityJsequential_xedyzswikc_while_sequential_xedyzswikc_while_maximum_iterations>^sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp=^sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp?^sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp6^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp8^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_18^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/xedyzswikc/while/Identity_1
&sequential/xedyzswikc/while/Identity_2Identity#sequential/xedyzswikc/while/add:z:0>^sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp=^sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp?^sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp6^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp8^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_18^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/xedyzswikc/while/Identity_2»
&sequential/xedyzswikc/while/Identity_3IdentityPsequential/xedyzswikc/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp=^sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp?^sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp6^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp8^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_18^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/xedyzswikc/while/Identity_3¬
&sequential/xedyzswikc/while/Identity_4Identity0sequential/xedyzswikc/while/jgtgtymybc/mul_5:z:0>^sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp=^sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp?^sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp6^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp8^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_18^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/while/Identity_4¬
&sequential/xedyzswikc/while/Identity_5Identity0sequential/xedyzswikc/while/jgtgtymybc/add_3:z:0>^sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp=^sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp?^sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp6^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp8^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_18^sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/xedyzswikc/while/Identity_5"U
$sequential_xedyzswikc_while_identity-sequential/xedyzswikc/while/Identity:output:0"Y
&sequential_xedyzswikc_while_identity_1/sequential/xedyzswikc/while/Identity_1:output:0"Y
&sequential_xedyzswikc_while_identity_2/sequential/xedyzswikc/while/Identity_2:output:0"Y
&sequential_xedyzswikc_while_identity_3/sequential/xedyzswikc/while/Identity_3:output:0"Y
&sequential_xedyzswikc_while_identity_4/sequential/xedyzswikc/while/Identity_4:output:0"Y
&sequential_xedyzswikc_while_identity_5/sequential/xedyzswikc/while/Identity_5:output:0"
Fsequential_xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resourceHsequential_xedyzswikc_while_jgtgtymybc_biasadd_readvariableop_resource_0"
Gsequential_xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resourceIsequential_xedyzswikc_while_jgtgtymybc_matmul_1_readvariableop_resource_0"
Esequential_xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resourceGsequential_xedyzswikc_while_jgtgtymybc_matmul_readvariableop_resource_0"
@sequential_xedyzswikc_while_jgtgtymybc_readvariableop_1_resourceBsequential_xedyzswikc_while_jgtgtymybc_readvariableop_1_resource_0"
@sequential_xedyzswikc_while_jgtgtymybc_readvariableop_2_resourceBsequential_xedyzswikc_while_jgtgtymybc_readvariableop_2_resource_0"
>sequential_xedyzswikc_while_jgtgtymybc_readvariableop_resource@sequential_xedyzswikc_while_jgtgtymybc_readvariableop_resource_0"
Asequential_xedyzswikc_while_sequential_xedyzswikc_strided_slice_1Csequential_xedyzswikc_while_sequential_xedyzswikc_strided_slice_1_0"
}sequential_xedyzswikc_while_tensorarrayv2read_tensorlistgetitem_sequential_xedyzswikc_tensorarrayunstack_tensorlistfromtensorsequential_xedyzswikc_while_tensorarrayv2read_tensorlistgetitem_sequential_xedyzswikc_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp=sequential/xedyzswikc/while/jgtgtymybc/BiasAdd/ReadVariableOp2|
<sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp<sequential/xedyzswikc/while/jgtgtymybc/MatMul/ReadVariableOp2
>sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp>sequential/xedyzswikc/while/jgtgtymybc/MatMul_1/ReadVariableOp2n
5sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp5sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp2r
7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_17sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_12r
7sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_27sequential/xedyzswikc/while/jgtgtymybc/ReadVariableOp_2: 
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
àY

while_body_1736596
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 F
3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 A
2while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	8
*while_qjjkgjcvpf_readvariableop_resource_0: :
,while_qjjkgjcvpf_readvariableop_1_resource_0: :
,while_qjjkgjcvpf_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_qjjkgjcvpf_matmul_readvariableop_resource:	 D
1while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 ?
0while_qjjkgjcvpf_biasadd_readvariableop_resource:	6
(while_qjjkgjcvpf_readvariableop_resource: 8
*while_qjjkgjcvpf_readvariableop_1_resource: 8
*while_qjjkgjcvpf_readvariableop_2_resource: ¢'while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢&while/qjjkgjcvpf/MatMul/ReadVariableOp¢(while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢while/qjjkgjcvpf/ReadVariableOp¢!while/qjjkgjcvpf/ReadVariableOp_1¢!while/qjjkgjcvpf/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp1while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/qjjkgjcvpf/MatMul/ReadVariableOpÑ
while/qjjkgjcvpf/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMulÉ
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/qjjkgjcvpf/MatMul_1/ReadVariableOpº
while/qjjkgjcvpf/MatMul_1MatMulwhile_placeholder_20while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/MatMul_1°
while/qjjkgjcvpf/addAddV2!while/qjjkgjcvpf/MatMul:product:0#while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/addÂ
'while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp2while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp½
while/qjjkgjcvpf/BiasAddBiasAddwhile/qjjkgjcvpf/add:z:0/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/qjjkgjcvpf/BiasAdd
 while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/qjjkgjcvpf/split/split_dim
while/qjjkgjcvpf/splitSplit)while/qjjkgjcvpf/split/split_dim:output:0!while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/qjjkgjcvpf/split©
while/qjjkgjcvpf/ReadVariableOpReadVariableOp*while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/qjjkgjcvpf/ReadVariableOp£
while/qjjkgjcvpf/mulMul'while/qjjkgjcvpf/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul¦
while/qjjkgjcvpf/add_1AddV2while/qjjkgjcvpf/split:output:0while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_1
while/qjjkgjcvpf/SigmoidSigmoidwhile/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid¯
!while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp,while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_1©
while/qjjkgjcvpf/mul_1Mul)while/qjjkgjcvpf/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_1¨
while/qjjkgjcvpf/add_2AddV2while/qjjkgjcvpf/split:output:1while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_2
while/qjjkgjcvpf/Sigmoid_1Sigmoidwhile/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_1
while/qjjkgjcvpf/mul_2Mulwhile/qjjkgjcvpf/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_2
while/qjjkgjcvpf/TanhTanhwhile/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh¢
while/qjjkgjcvpf/mul_3Mulwhile/qjjkgjcvpf/Sigmoid:y:0while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_3£
while/qjjkgjcvpf/add_3AddV2while/qjjkgjcvpf/mul_2:z:0while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_3¯
!while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp,while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/qjjkgjcvpf/ReadVariableOp_2°
while/qjjkgjcvpf/mul_4Mul)while/qjjkgjcvpf/ReadVariableOp_2:value:0while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_4¨
while/qjjkgjcvpf/add_4AddV2while/qjjkgjcvpf/split:output:3while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/add_4
while/qjjkgjcvpf/Sigmoid_2Sigmoidwhile/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Sigmoid_2
while/qjjkgjcvpf/Tanh_1Tanhwhile/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/Tanh_1¦
while/qjjkgjcvpf/mul_5Mulwhile/qjjkgjcvpf/Sigmoid_2:y:0while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/qjjkgjcvpf/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/qjjkgjcvpf/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/qjjkgjcvpf/mul_5:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/qjjkgjcvpf/add_3:z:0(^while/qjjkgjcvpf/BiasAdd/ReadVariableOp'^while/qjjkgjcvpf/MatMul/ReadVariableOp)^while/qjjkgjcvpf/MatMul_1/ReadVariableOp ^while/qjjkgjcvpf/ReadVariableOp"^while/qjjkgjcvpf/ReadVariableOp_1"^while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_qjjkgjcvpf_biasadd_readvariableop_resource2while_qjjkgjcvpf_biasadd_readvariableop_resource_0"h
1while_qjjkgjcvpf_matmul_1_readvariableop_resource3while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"d
/while_qjjkgjcvpf_matmul_readvariableop_resource1while_qjjkgjcvpf_matmul_readvariableop_resource_0"Z
*while_qjjkgjcvpf_readvariableop_1_resource,while_qjjkgjcvpf_readvariableop_1_resource_0"Z
*while_qjjkgjcvpf_readvariableop_2_resource,while_qjjkgjcvpf_readvariableop_2_resource_0"V
(while_qjjkgjcvpf_readvariableop_resource*while_qjjkgjcvpf_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/qjjkgjcvpf/BiasAdd/ReadVariableOp'while/qjjkgjcvpf/BiasAdd/ReadVariableOp2P
&while/qjjkgjcvpf/MatMul/ReadVariableOp&while/qjjkgjcvpf/MatMul/ReadVariableOp2T
(while/qjjkgjcvpf/MatMul_1/ReadVariableOp(while/qjjkgjcvpf/MatMul_1/ReadVariableOp2B
while/qjjkgjcvpf/ReadVariableOpwhile/qjjkgjcvpf/ReadVariableOp2F
!while/qjjkgjcvpf/ReadVariableOp_1!while/qjjkgjcvpf/ReadVariableOp_12F
!while/qjjkgjcvpf/ReadVariableOp_2!while/qjjkgjcvpf/ReadVariableOp_2: 
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
Üh

G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736337
inputs_0<
)qjjkgjcvpf_matmul_readvariableop_resource:	 >
+qjjkgjcvpf_matmul_1_readvariableop_resource:	 9
*qjjkgjcvpf_biasadd_readvariableop_resource:	0
"qjjkgjcvpf_readvariableop_resource: 2
$qjjkgjcvpf_readvariableop_1_resource: 2
$qjjkgjcvpf_readvariableop_2_resource: 
identity¢!qjjkgjcvpf/BiasAdd/ReadVariableOp¢ qjjkgjcvpf/MatMul/ReadVariableOp¢"qjjkgjcvpf/MatMul_1/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp_1¢qjjkgjcvpf/ReadVariableOp_2¢whileF
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
strided_slice_2¯
 qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp)qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 qjjkgjcvpf/MatMul/ReadVariableOp§
qjjkgjcvpf/MatMulMatMulstrided_slice_2:output:0(qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMulµ
"qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp+qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"qjjkgjcvpf/MatMul_1/ReadVariableOp£
qjjkgjcvpf/MatMul_1MatMulzeros:output:0*qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMul_1
qjjkgjcvpf/addAddV2qjjkgjcvpf/MatMul:product:0qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/add®
!qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp*qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!qjjkgjcvpf/BiasAdd/ReadVariableOp¥
qjjkgjcvpf/BiasAddBiasAddqjjkgjcvpf/add:z:0)qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/BiasAddz
qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
qjjkgjcvpf/split/split_dimë
qjjkgjcvpf/splitSplit#qjjkgjcvpf/split/split_dim:output:0qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qjjkgjcvpf/split
qjjkgjcvpf/ReadVariableOpReadVariableOp"qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp
qjjkgjcvpf/mulMul!qjjkgjcvpf/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul
qjjkgjcvpf/add_1AddV2qjjkgjcvpf/split:output:0qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_1{
qjjkgjcvpf/SigmoidSigmoidqjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid
qjjkgjcvpf/ReadVariableOp_1ReadVariableOp$qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_1
qjjkgjcvpf/mul_1Mul#qjjkgjcvpf/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_1
qjjkgjcvpf/add_2AddV2qjjkgjcvpf/split:output:1qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_2
qjjkgjcvpf/Sigmoid_1Sigmoidqjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_1
qjjkgjcvpf/mul_2Mulqjjkgjcvpf/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_2w
qjjkgjcvpf/TanhTanhqjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh
qjjkgjcvpf/mul_3Mulqjjkgjcvpf/Sigmoid:y:0qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_3
qjjkgjcvpf/add_3AddV2qjjkgjcvpf/mul_2:z:0qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_3
qjjkgjcvpf/ReadVariableOp_2ReadVariableOp$qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_2
qjjkgjcvpf/mul_4Mul#qjjkgjcvpf/ReadVariableOp_2:value:0qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_4
qjjkgjcvpf/add_4AddV2qjjkgjcvpf/split:output:3qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_4
qjjkgjcvpf/Sigmoid_2Sigmoidqjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_2v
qjjkgjcvpf/Tanh_1Tanhqjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh_1
qjjkgjcvpf/mul_5Mulqjjkgjcvpf/Sigmoid_2:y:0qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)qjjkgjcvpf_matmul_readvariableop_resource+qjjkgjcvpf_matmul_1_readvariableop_resource*qjjkgjcvpf_biasadd_readvariableop_resource"qjjkgjcvpf_readvariableop_resource$qjjkgjcvpf_readvariableop_1_resource$qjjkgjcvpf_readvariableop_2_resource*
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
while_body_1736236*
condR
while_cond_1736235*Q
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
transpose_1¸
IdentityIdentitystrided_slice_3:output:0"^qjjkgjcvpf/BiasAdd/ReadVariableOp!^qjjkgjcvpf/MatMul/ReadVariableOp#^qjjkgjcvpf/MatMul_1/ReadVariableOp^qjjkgjcvpf/ReadVariableOp^qjjkgjcvpf/ReadVariableOp_1^qjjkgjcvpf/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!qjjkgjcvpf/BiasAdd/ReadVariableOp!qjjkgjcvpf/BiasAdd/ReadVariableOp2D
 qjjkgjcvpf/MatMul/ReadVariableOp qjjkgjcvpf/MatMul/ReadVariableOp2H
"qjjkgjcvpf/MatMul_1/ReadVariableOp"qjjkgjcvpf/MatMul_1/ReadVariableOp26
qjjkgjcvpf/ReadVariableOpqjjkgjcvpf/ReadVariableOp2:
qjjkgjcvpf/ReadVariableOp_1qjjkgjcvpf/ReadVariableOp_12:
qjjkgjcvpf/ReadVariableOp_2qjjkgjcvpf/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
	

,__inference_xedyzswikc_layer_call_fn_1735138
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall½
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17316102
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
¤

,__inference_vfmtgawzzo_layer_call_fn_1736706

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallú
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
GPU2*0J 8 *P
fKRI
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_17334002
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
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_1734089

qtjxeibmnq(
zdelabjare_1734051: 
zdelabjare_1734053:%
xedyzswikc_1734057:	%
xedyzswikc_1734059:	 !
xedyzswikc_1734061:	 
xedyzswikc_1734063:  
xedyzswikc_1734065:  
xedyzswikc_1734067: %
ksthobzafc_1734070:	 %
ksthobzafc_1734072:	 !
ksthobzafc_1734074:	 
ksthobzafc_1734076:  
ksthobzafc_1734078:  
ksthobzafc_1734080: $
vfmtgawzzo_1734083:  
vfmtgawzzo_1734085:
identity¢"ksthobzafc/StatefulPartitionedCall¢"vfmtgawzzo/StatefulPartitionedCall¢"xedyzswikc/StatefulPartitionedCall¢"zdelabjare/StatefulPartitionedCall°
"zdelabjare/StatefulPartitionedCallStatefulPartitionedCall
qtjxeibmnqzdelabjare_1734051zdelabjare_1734053*
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
GPU2*0J 8 *P
fKRI
G__inference_zdelabjare_layer_call_and_return_conditional_losses_17329832$
"zdelabjare/StatefulPartitionedCall
ojrtxmspqi/PartitionedCallPartitionedCall+zdelabjare/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_17330022
ojrtxmspqi/PartitionedCall
"xedyzswikc/StatefulPartitionedCallStatefulPartitionedCall#ojrtxmspqi/PartitionedCall:output:0xedyzswikc_1734057xedyzswikc_1734059xedyzswikc_1734061xedyzswikc_1734063xedyzswikc_1734065xedyzswikc_1734067*
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17331832$
"xedyzswikc/StatefulPartitionedCall¡
"ksthobzafc/StatefulPartitionedCallStatefulPartitionedCall+xedyzswikc/StatefulPartitionedCall:output:0ksthobzafc_1734070ksthobzafc_1734072ksthobzafc_1734074ksthobzafc_1734076ksthobzafc_1734078ksthobzafc_1734080*
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17333762$
"ksthobzafc/StatefulPartitionedCallÉ
"vfmtgawzzo/StatefulPartitionedCallStatefulPartitionedCall+ksthobzafc/StatefulPartitionedCall:output:0vfmtgawzzo_1734083vfmtgawzzo_1734085*
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
GPU2*0J 8 *P
fKRI
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_17334002$
"vfmtgawzzo/StatefulPartitionedCall
IdentityIdentity+vfmtgawzzo/StatefulPartitionedCall:output:0#^ksthobzafc/StatefulPartitionedCall#^vfmtgawzzo/StatefulPartitionedCall#^xedyzswikc/StatefulPartitionedCall#^zdelabjare/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ksthobzafc/StatefulPartitionedCall"ksthobzafc/StatefulPartitionedCall2H
"vfmtgawzzo/StatefulPartitionedCall"vfmtgawzzo/StatefulPartitionedCall2H
"xedyzswikc/StatefulPartitionedCall"xedyzswikc/StatefulPartitionedCall2H
"zdelabjare/StatefulPartitionedCall"zdelabjare/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
qtjxeibmnq
¦h

G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736697

inputs<
)qjjkgjcvpf_matmul_readvariableop_resource:	 >
+qjjkgjcvpf_matmul_1_readvariableop_resource:	 9
*qjjkgjcvpf_biasadd_readvariableop_resource:	0
"qjjkgjcvpf_readvariableop_resource: 2
$qjjkgjcvpf_readvariableop_1_resource: 2
$qjjkgjcvpf_readvariableop_2_resource: 
identity¢!qjjkgjcvpf/BiasAdd/ReadVariableOp¢ qjjkgjcvpf/MatMul/ReadVariableOp¢"qjjkgjcvpf/MatMul_1/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp¢qjjkgjcvpf/ReadVariableOp_1¢qjjkgjcvpf/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp)qjjkgjcvpf_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 qjjkgjcvpf/MatMul/ReadVariableOp§
qjjkgjcvpf/MatMulMatMulstrided_slice_2:output:0(qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMulµ
"qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp+qjjkgjcvpf_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"qjjkgjcvpf/MatMul_1/ReadVariableOp£
qjjkgjcvpf/MatMul_1MatMulzeros:output:0*qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/MatMul_1
qjjkgjcvpf/addAddV2qjjkgjcvpf/MatMul:product:0qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/add®
!qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp*qjjkgjcvpf_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!qjjkgjcvpf/BiasAdd/ReadVariableOp¥
qjjkgjcvpf/BiasAddBiasAddqjjkgjcvpf/add:z:0)qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qjjkgjcvpf/BiasAddz
qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
qjjkgjcvpf/split/split_dimë
qjjkgjcvpf/splitSplit#qjjkgjcvpf/split/split_dim:output:0qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qjjkgjcvpf/split
qjjkgjcvpf/ReadVariableOpReadVariableOp"qjjkgjcvpf_readvariableop_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp
qjjkgjcvpf/mulMul!qjjkgjcvpf/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul
qjjkgjcvpf/add_1AddV2qjjkgjcvpf/split:output:0qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_1{
qjjkgjcvpf/SigmoidSigmoidqjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid
qjjkgjcvpf/ReadVariableOp_1ReadVariableOp$qjjkgjcvpf_readvariableop_1_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_1
qjjkgjcvpf/mul_1Mul#qjjkgjcvpf/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_1
qjjkgjcvpf/add_2AddV2qjjkgjcvpf/split:output:1qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_2
qjjkgjcvpf/Sigmoid_1Sigmoidqjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_1
qjjkgjcvpf/mul_2Mulqjjkgjcvpf/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_2w
qjjkgjcvpf/TanhTanhqjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh
qjjkgjcvpf/mul_3Mulqjjkgjcvpf/Sigmoid:y:0qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_3
qjjkgjcvpf/add_3AddV2qjjkgjcvpf/mul_2:z:0qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_3
qjjkgjcvpf/ReadVariableOp_2ReadVariableOp$qjjkgjcvpf_readvariableop_2_resource*
_output_shapes
: *
dtype02
qjjkgjcvpf/ReadVariableOp_2
qjjkgjcvpf/mul_4Mul#qjjkgjcvpf/ReadVariableOp_2:value:0qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_4
qjjkgjcvpf/add_4AddV2qjjkgjcvpf/split:output:3qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/add_4
qjjkgjcvpf/Sigmoid_2Sigmoidqjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Sigmoid_2v
qjjkgjcvpf/Tanh_1Tanhqjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/Tanh_1
qjjkgjcvpf/mul_5Mulqjjkgjcvpf/Sigmoid_2:y:0qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qjjkgjcvpf/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)qjjkgjcvpf_matmul_readvariableop_resource+qjjkgjcvpf_matmul_1_readvariableop_resource*qjjkgjcvpf_biasadd_readvariableop_resource"qjjkgjcvpf_readvariableop_resource$qjjkgjcvpf_readvariableop_1_resource$qjjkgjcvpf_readvariableop_2_resource*
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
while_body_1736596*
condR
while_cond_1736595*Q
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
transpose_1¸
IdentityIdentitystrided_slice_3:output:0"^qjjkgjcvpf/BiasAdd/ReadVariableOp!^qjjkgjcvpf/MatMul/ReadVariableOp#^qjjkgjcvpf/MatMul_1/ReadVariableOp^qjjkgjcvpf/ReadVariableOp^qjjkgjcvpf/ReadVariableOp_1^qjjkgjcvpf/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!qjjkgjcvpf/BiasAdd/ReadVariableOp!qjjkgjcvpf/BiasAdd/ReadVariableOp2D
 qjjkgjcvpf/MatMul/ReadVariableOp qjjkgjcvpf/MatMul/ReadVariableOp2H
"qjjkgjcvpf/MatMul_1/ReadVariableOp"qjjkgjcvpf/MatMul_1/ReadVariableOp26
qjjkgjcvpf/ReadVariableOpqjjkgjcvpf/ReadVariableOp2:
qjjkgjcvpf/ReadVariableOp_1qjjkgjcvpf/ReadVariableOp_12:
qjjkgjcvpf/ReadVariableOp_2qjjkgjcvpf/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
p
Ê
ksthobzafc_while_body_17349502
.ksthobzafc_while_ksthobzafc_while_loop_counter8
4ksthobzafc_while_ksthobzafc_while_maximum_iterations 
ksthobzafc_while_placeholder"
ksthobzafc_while_placeholder_1"
ksthobzafc_while_placeholder_2"
ksthobzafc_while_placeholder_31
-ksthobzafc_while_ksthobzafc_strided_slice_1_0m
iksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0O
<ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0:	 Q
>ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0:	 L
=ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0:	C
5ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0: E
7ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0: E
7ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0: 
ksthobzafc_while_identity
ksthobzafc_while_identity_1
ksthobzafc_while_identity_2
ksthobzafc_while_identity_3
ksthobzafc_while_identity_4
ksthobzafc_while_identity_5/
+ksthobzafc_while_ksthobzafc_strided_slice_1k
gksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensorM
:ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource:	 O
<ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource:	 J
;ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource:	A
3ksthobzafc_while_qjjkgjcvpf_readvariableop_resource: C
5ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource: C
5ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource: ¢2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp¢1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp¢3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp¢*ksthobzafc/while/qjjkgjcvpf/ReadVariableOp¢,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1¢,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2Ù
Bksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4ksthobzafc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0ksthobzafc_while_placeholderKksthobzafc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4ksthobzafc/while/TensorArrayV2Read/TensorListGetItemä
1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOpReadVariableOp<ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOpý
"ksthobzafc/while/qjjkgjcvpf/MatMulMatMul;ksthobzafc/while/TensorArrayV2Read/TensorListGetItem:item:09ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"ksthobzafc/while/qjjkgjcvpf/MatMulê
3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOpReadVariableOp>ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOpæ
$ksthobzafc/while/qjjkgjcvpf/MatMul_1MatMulksthobzafc_while_placeholder_2;ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$ksthobzafc/while/qjjkgjcvpf/MatMul_1Ü
ksthobzafc/while/qjjkgjcvpf/addAddV2,ksthobzafc/while/qjjkgjcvpf/MatMul:product:0.ksthobzafc/while/qjjkgjcvpf/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
ksthobzafc/while/qjjkgjcvpf/addã
2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOpReadVariableOp=ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOpé
#ksthobzafc/while/qjjkgjcvpf/BiasAddBiasAdd#ksthobzafc/while/qjjkgjcvpf/add:z:0:ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ksthobzafc/while/qjjkgjcvpf/BiasAdd
+ksthobzafc/while/qjjkgjcvpf/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+ksthobzafc/while/qjjkgjcvpf/split/split_dim¯
!ksthobzafc/while/qjjkgjcvpf/splitSplit4ksthobzafc/while/qjjkgjcvpf/split/split_dim:output:0,ksthobzafc/while/qjjkgjcvpf/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!ksthobzafc/while/qjjkgjcvpf/splitÊ
*ksthobzafc/while/qjjkgjcvpf/ReadVariableOpReadVariableOp5ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*ksthobzafc/while/qjjkgjcvpf/ReadVariableOpÏ
ksthobzafc/while/qjjkgjcvpf/mulMul2ksthobzafc/while/qjjkgjcvpf/ReadVariableOp:value:0ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ksthobzafc/while/qjjkgjcvpf/mulÒ
!ksthobzafc/while/qjjkgjcvpf/add_1AddV2*ksthobzafc/while/qjjkgjcvpf/split:output:0#ksthobzafc/while/qjjkgjcvpf/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_1®
#ksthobzafc/while/qjjkgjcvpf/SigmoidSigmoid%ksthobzafc/while/qjjkgjcvpf/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#ksthobzafc/while/qjjkgjcvpf/SigmoidÐ
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1ReadVariableOp7ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1Õ
!ksthobzafc/while/qjjkgjcvpf/mul_1Mul4ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1:value:0ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_1Ô
!ksthobzafc/while/qjjkgjcvpf/add_2AddV2*ksthobzafc/while/qjjkgjcvpf/split:output:1%ksthobzafc/while/qjjkgjcvpf/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_2²
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_1Sigmoid%ksthobzafc/while/qjjkgjcvpf/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_1Ê
!ksthobzafc/while/qjjkgjcvpf/mul_2Mul)ksthobzafc/while/qjjkgjcvpf/Sigmoid_1:y:0ksthobzafc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_2ª
 ksthobzafc/while/qjjkgjcvpf/TanhTanh*ksthobzafc/while/qjjkgjcvpf/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 ksthobzafc/while/qjjkgjcvpf/TanhÎ
!ksthobzafc/while/qjjkgjcvpf/mul_3Mul'ksthobzafc/while/qjjkgjcvpf/Sigmoid:y:0$ksthobzafc/while/qjjkgjcvpf/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_3Ï
!ksthobzafc/while/qjjkgjcvpf/add_3AddV2%ksthobzafc/while/qjjkgjcvpf/mul_2:z:0%ksthobzafc/while/qjjkgjcvpf/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_3Ð
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2ReadVariableOp7ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2Ü
!ksthobzafc/while/qjjkgjcvpf/mul_4Mul4ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2:value:0%ksthobzafc/while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_4Ô
!ksthobzafc/while/qjjkgjcvpf/add_4AddV2*ksthobzafc/while/qjjkgjcvpf/split:output:3%ksthobzafc/while/qjjkgjcvpf/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/add_4²
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_2Sigmoid%ksthobzafc/while/qjjkgjcvpf/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ksthobzafc/while/qjjkgjcvpf/Sigmoid_2©
"ksthobzafc/while/qjjkgjcvpf/Tanh_1Tanh%ksthobzafc/while/qjjkgjcvpf/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"ksthobzafc/while/qjjkgjcvpf/Tanh_1Ò
!ksthobzafc/while/qjjkgjcvpf/mul_5Mul)ksthobzafc/while/qjjkgjcvpf/Sigmoid_2:y:0&ksthobzafc/while/qjjkgjcvpf/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ksthobzafc/while/qjjkgjcvpf/mul_5
5ksthobzafc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemksthobzafc_while_placeholder_1ksthobzafc_while_placeholder%ksthobzafc/while/qjjkgjcvpf/mul_5:z:0*
_output_shapes
: *
element_dtype027
5ksthobzafc/while/TensorArrayV2Write/TensorListSetItemr
ksthobzafc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
ksthobzafc/while/add/y
ksthobzafc/while/addAddV2ksthobzafc_while_placeholderksthobzafc/while/add/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/while/addv
ksthobzafc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
ksthobzafc/while/add_1/y­
ksthobzafc/while/add_1AddV2.ksthobzafc_while_ksthobzafc_while_loop_counter!ksthobzafc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
ksthobzafc/while/add_1©
ksthobzafc/while/IdentityIdentityksthobzafc/while/add_1:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/IdentityÇ
ksthobzafc/while/Identity_1Identity4ksthobzafc_while_ksthobzafc_while_maximum_iterations3^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/Identity_1«
ksthobzafc/while/Identity_2Identityksthobzafc/while/add:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/Identity_2Ø
ksthobzafc/while/Identity_3IdentityEksthobzafc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*
_output_shapes
: 2
ksthobzafc/while/Identity_3É
ksthobzafc/while/Identity_4Identity%ksthobzafc/while/qjjkgjcvpf/mul_5:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/while/Identity_4É
ksthobzafc/while/Identity_5Identity%ksthobzafc/while/qjjkgjcvpf/add_3:z:03^ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2^ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp4^ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp+^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1-^ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ksthobzafc/while/Identity_5"?
ksthobzafc_while_identity"ksthobzafc/while/Identity:output:0"C
ksthobzafc_while_identity_1$ksthobzafc/while/Identity_1:output:0"C
ksthobzafc_while_identity_2$ksthobzafc/while/Identity_2:output:0"C
ksthobzafc_while_identity_3$ksthobzafc/while/Identity_3:output:0"C
ksthobzafc_while_identity_4$ksthobzafc/while/Identity_4:output:0"C
ksthobzafc_while_identity_5$ksthobzafc/while/Identity_5:output:0"\
+ksthobzafc_while_ksthobzafc_strided_slice_1-ksthobzafc_while_ksthobzafc_strided_slice_1_0"|
;ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource=ksthobzafc_while_qjjkgjcvpf_biasadd_readvariableop_resource_0"~
<ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource>ksthobzafc_while_qjjkgjcvpf_matmul_1_readvariableop_resource_0"z
:ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource<ksthobzafc_while_qjjkgjcvpf_matmul_readvariableop_resource_0"p
5ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource7ksthobzafc_while_qjjkgjcvpf_readvariableop_1_resource_0"p
5ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource7ksthobzafc_while_qjjkgjcvpf_readvariableop_2_resource_0"l
3ksthobzafc_while_qjjkgjcvpf_readvariableop_resource5ksthobzafc_while_qjjkgjcvpf_readvariableop_resource_0"Ô
gksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensoriksthobzafc_while_tensorarrayv2read_tensorlistgetitem_ksthobzafc_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2ksthobzafc/while/qjjkgjcvpf/BiasAdd/ReadVariableOp2f
1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp1ksthobzafc/while/qjjkgjcvpf/MatMul/ReadVariableOp2j
3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp3ksthobzafc/while/qjjkgjcvpf/MatMul_1/ReadVariableOp2X
*ksthobzafc/while/qjjkgjcvpf/ReadVariableOp*ksthobzafc/while/qjjkgjcvpf/ReadVariableOp2\
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_1,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_12\
,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2,ksthobzafc/while/qjjkgjcvpf/ReadVariableOp_2: 
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
¸
¼
G__inference_sequential_layer_call_and_return_conditional_losses_1733976

inputs(
zdelabjare_1733938: 
zdelabjare_1733940:%
xedyzswikc_1733944:	%
xedyzswikc_1733946:	 !
xedyzswikc_1733948:	 
xedyzswikc_1733950:  
xedyzswikc_1733952:  
xedyzswikc_1733954: %
ksthobzafc_1733957:	 %
ksthobzafc_1733959:	 !
ksthobzafc_1733961:	 
ksthobzafc_1733963:  
ksthobzafc_1733965:  
ksthobzafc_1733967: $
vfmtgawzzo_1733970:  
vfmtgawzzo_1733972:
identity¢"ksthobzafc/StatefulPartitionedCall¢"vfmtgawzzo/StatefulPartitionedCall¢"xedyzswikc/StatefulPartitionedCall¢"zdelabjare/StatefulPartitionedCall¬
"zdelabjare/StatefulPartitionedCallStatefulPartitionedCallinputszdelabjare_1733938zdelabjare_1733940*
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
GPU2*0J 8 *P
fKRI
G__inference_zdelabjare_layer_call_and_return_conditional_losses_17329832$
"zdelabjare/StatefulPartitionedCall
ojrtxmspqi/PartitionedCallPartitionedCall+zdelabjare/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_17330022
ojrtxmspqi/PartitionedCall
"xedyzswikc/StatefulPartitionedCallStatefulPartitionedCall#ojrtxmspqi/PartitionedCall:output:0xedyzswikc_1733944xedyzswikc_1733946xedyzswikc_1733948xedyzswikc_1733950xedyzswikc_1733952xedyzswikc_1733954*
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
GPU2*0J 8 *P
fKRI
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_17338652$
"xedyzswikc/StatefulPartitionedCall¡
"ksthobzafc/StatefulPartitionedCallStatefulPartitionedCall+xedyzswikc/StatefulPartitionedCall:output:0ksthobzafc_1733957ksthobzafc_1733959ksthobzafc_1733961ksthobzafc_1733963ksthobzafc_1733965ksthobzafc_1733967*
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17336512$
"ksthobzafc/StatefulPartitionedCallÉ
"vfmtgawzzo/StatefulPartitionedCallStatefulPartitionedCall+ksthobzafc/StatefulPartitionedCall:output:0vfmtgawzzo_1733970vfmtgawzzo_1733972*
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
GPU2*0J 8 *P
fKRI
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_17334002$
"vfmtgawzzo/StatefulPartitionedCall
IdentityIdentity+vfmtgawzzo/StatefulPartitionedCall:output:0#^ksthobzafc/StatefulPartitionedCall#^vfmtgawzzo/StatefulPartitionedCall#^xedyzswikc/StatefulPartitionedCall#^zdelabjare/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ksthobzafc/StatefulPartitionedCall"ksthobzafc/StatefulPartitionedCall2H
"vfmtgawzzo/StatefulPartitionedCall"vfmtgawzzo/StatefulPartitionedCall2H
"xedyzswikc/StatefulPartitionedCall"xedyzswikc/StatefulPartitionedCall2H
"zdelabjare/StatefulPartitionedCall"zdelabjare/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³F
ê
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1731610

inputs%
jgtgtymybc_1731511:	%
jgtgtymybc_1731513:	 !
jgtgtymybc_1731515:	 
jgtgtymybc_1731517:  
jgtgtymybc_1731519:  
jgtgtymybc_1731521: 
identity¢"jgtgtymybc/StatefulPartitionedCall¢whileD
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
strided_slice_2Ú
"jgtgtymybc/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0jgtgtymybc_1731511jgtgtymybc_1731513jgtgtymybc_1731515jgtgtymybc_1731517jgtgtymybc_1731519jgtgtymybc_1731521*
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
GPU2*0J 8 *P
fKRI
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_17315102$
"jgtgtymybc/StatefulPartitionedCall
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
while/loop_counterð
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0jgtgtymybc_1731511jgtgtymybc_1731513jgtgtymybc_1731515jgtgtymybc_1731517jgtgtymybc_1731519jgtgtymybc_1731521*
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
while_body_1731530*
condR
while_cond_1731529*Q
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
transpose_1
IdentityIdentitytranspose_1:y:0#^jgtgtymybc/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"jgtgtymybc/StatefulPartitionedCall"jgtgtymybc/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡h

G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735729

inputs<
)jgtgtymybc_matmul_readvariableop_resource:	>
+jgtgtymybc_matmul_1_readvariableop_resource:	 9
*jgtgtymybc_biasadd_readvariableop_resource:	0
"jgtgtymybc_readvariableop_resource: 2
$jgtgtymybc_readvariableop_1_resource: 2
$jgtgtymybc_readvariableop_2_resource: 
identity¢!jgtgtymybc/BiasAdd/ReadVariableOp¢ jgtgtymybc/MatMul/ReadVariableOp¢"jgtgtymybc/MatMul_1/ReadVariableOp¢jgtgtymybc/ReadVariableOp¢jgtgtymybc/ReadVariableOp_1¢jgtgtymybc/ReadVariableOp_2¢whileD
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
strided_slice_2¯
 jgtgtymybc/MatMul/ReadVariableOpReadVariableOp)jgtgtymybc_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jgtgtymybc/MatMul/ReadVariableOp§
jgtgtymybc/MatMulMatMulstrided_slice_2:output:0(jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMulµ
"jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp+jgtgtymybc_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jgtgtymybc/MatMul_1/ReadVariableOp£
jgtgtymybc/MatMul_1MatMulzeros:output:0*jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/MatMul_1
jgtgtymybc/addAddV2jgtgtymybc/MatMul:product:0jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/add®
!jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp*jgtgtymybc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jgtgtymybc/BiasAdd/ReadVariableOp¥
jgtgtymybc/BiasAddBiasAddjgtgtymybc/add:z:0)jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jgtgtymybc/BiasAddz
jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jgtgtymybc/split/split_dimë
jgtgtymybc/splitSplit#jgtgtymybc/split/split_dim:output:0jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jgtgtymybc/split
jgtgtymybc/ReadVariableOpReadVariableOp"jgtgtymybc_readvariableop_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp
jgtgtymybc/mulMul!jgtgtymybc/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul
jgtgtymybc/add_1AddV2jgtgtymybc/split:output:0jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_1{
jgtgtymybc/SigmoidSigmoidjgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid
jgtgtymybc/ReadVariableOp_1ReadVariableOp$jgtgtymybc_readvariableop_1_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_1
jgtgtymybc/mul_1Mul#jgtgtymybc/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_1
jgtgtymybc/add_2AddV2jgtgtymybc/split:output:1jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_2
jgtgtymybc/Sigmoid_1Sigmoidjgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_1
jgtgtymybc/mul_2Muljgtgtymybc/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_2w
jgtgtymybc/TanhTanhjgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh
jgtgtymybc/mul_3Muljgtgtymybc/Sigmoid:y:0jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_3
jgtgtymybc/add_3AddV2jgtgtymybc/mul_2:z:0jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_3
jgtgtymybc/ReadVariableOp_2ReadVariableOp$jgtgtymybc_readvariableop_2_resource*
_output_shapes
: *
dtype02
jgtgtymybc/ReadVariableOp_2
jgtgtymybc/mul_4Mul#jgtgtymybc/ReadVariableOp_2:value:0jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_4
jgtgtymybc/add_4AddV2jgtgtymybc/split:output:3jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/add_4
jgtgtymybc/Sigmoid_2Sigmoidjgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Sigmoid_2v
jgtgtymybc/Tanh_1Tanhjgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/Tanh_1
jgtgtymybc/mul_5Muljgtgtymybc/Sigmoid_2:y:0jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jgtgtymybc/mul_5
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
while/loop_counterì
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jgtgtymybc_matmul_readvariableop_resource+jgtgtymybc_matmul_1_readvariableop_resource*jgtgtymybc_biasadd_readvariableop_resource"jgtgtymybc_readvariableop_resource$jgtgtymybc_readvariableop_1_resource$jgtgtymybc_readvariableop_2_resource*
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
while_body_1735628*
condR
while_cond_1735627*Q
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
transpose_1³
IdentityIdentitytranspose_1:y:0"^jgtgtymybc/BiasAdd/ReadVariableOp!^jgtgtymybc/MatMul/ReadVariableOp#^jgtgtymybc/MatMul_1/ReadVariableOp^jgtgtymybc/ReadVariableOp^jgtgtymybc/ReadVariableOp_1^jgtgtymybc/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jgtgtymybc/BiasAdd/ReadVariableOp!jgtgtymybc/BiasAdd/ReadVariableOp2D
 jgtgtymybc/MatMul/ReadVariableOp jgtgtymybc/MatMul/ReadVariableOp2H
"jgtgtymybc/MatMul_1/ReadVariableOp"jgtgtymybc/MatMul_1/ReadVariableOp26
jgtgtymybc/ReadVariableOpjgtgtymybc/ReadVariableOp2:
jgtgtymybc/ReadVariableOp_1jgtgtymybc/ReadVariableOp_12:
jgtgtymybc/ReadVariableOp_2jgtgtymybc/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àY

while_body_1735808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jgtgtymybc_matmul_readvariableop_resource_0:	F
3while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 A
2while_jgtgtymybc_biasadd_readvariableop_resource_0:	8
*while_jgtgtymybc_readvariableop_resource_0: :
,while_jgtgtymybc_readvariableop_1_resource_0: :
,while_jgtgtymybc_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jgtgtymybc_matmul_readvariableop_resource:	D
1while_jgtgtymybc_matmul_1_readvariableop_resource:	 ?
0while_jgtgtymybc_biasadd_readvariableop_resource:	6
(while_jgtgtymybc_readvariableop_resource: 8
*while_jgtgtymybc_readvariableop_1_resource: 8
*while_jgtgtymybc_readvariableop_2_resource: ¢'while/jgtgtymybc/BiasAdd/ReadVariableOp¢&while/jgtgtymybc/MatMul/ReadVariableOp¢(while/jgtgtymybc/MatMul_1/ReadVariableOp¢while/jgtgtymybc/ReadVariableOp¢!while/jgtgtymybc/ReadVariableOp_1¢!while/jgtgtymybc/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp1while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jgtgtymybc/MatMul/ReadVariableOpÑ
while/jgtgtymybc/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMulÉ
(while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp3while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jgtgtymybc/MatMul_1/ReadVariableOpº
while/jgtgtymybc/MatMul_1MatMulwhile_placeholder_20while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMul_1°
while/jgtgtymybc/addAddV2!while/jgtgtymybc/MatMul:product:0#while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/addÂ
'while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp2while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jgtgtymybc/BiasAdd/ReadVariableOp½
while/jgtgtymybc/BiasAddBiasAddwhile/jgtgtymybc/add:z:0/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/BiasAdd
 while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jgtgtymybc/split/split_dim
while/jgtgtymybc/splitSplit)while/jgtgtymybc/split/split_dim:output:0!while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jgtgtymybc/split©
while/jgtgtymybc/ReadVariableOpReadVariableOp*while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jgtgtymybc/ReadVariableOp£
while/jgtgtymybc/mulMul'while/jgtgtymybc/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul¦
while/jgtgtymybc/add_1AddV2while/jgtgtymybc/split:output:0while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_1
while/jgtgtymybc/SigmoidSigmoidwhile/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid¯
!while/jgtgtymybc/ReadVariableOp_1ReadVariableOp,while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_1©
while/jgtgtymybc/mul_1Mul)while/jgtgtymybc/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_1¨
while/jgtgtymybc/add_2AddV2while/jgtgtymybc/split:output:1while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_2
while/jgtgtymybc/Sigmoid_1Sigmoidwhile/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_1
while/jgtgtymybc/mul_2Mulwhile/jgtgtymybc/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_2
while/jgtgtymybc/TanhTanhwhile/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh¢
while/jgtgtymybc/mul_3Mulwhile/jgtgtymybc/Sigmoid:y:0while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_3£
while/jgtgtymybc/add_3AddV2while/jgtgtymybc/mul_2:z:0while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_3¯
!while/jgtgtymybc/ReadVariableOp_2ReadVariableOp,while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_2°
while/jgtgtymybc/mul_4Mul)while/jgtgtymybc/ReadVariableOp_2:value:0while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_4¨
while/jgtgtymybc/add_4AddV2while/jgtgtymybc/split:output:3while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_4
while/jgtgtymybc/Sigmoid_2Sigmoidwhile/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_2
while/jgtgtymybc/Tanh_1Tanhwhile/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh_1¦
while/jgtgtymybc/mul_5Mulwhile/jgtgtymybc/Sigmoid_2:y:0while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jgtgtymybc/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jgtgtymybc/mul_5:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jgtgtymybc/add_3:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_jgtgtymybc_biasadd_readvariableop_resource2while_jgtgtymybc_biasadd_readvariableop_resource_0"h
1while_jgtgtymybc_matmul_1_readvariableop_resource3while_jgtgtymybc_matmul_1_readvariableop_resource_0"d
/while_jgtgtymybc_matmul_readvariableop_resource1while_jgtgtymybc_matmul_readvariableop_resource_0"Z
*while_jgtgtymybc_readvariableop_1_resource,while_jgtgtymybc_readvariableop_1_resource_0"Z
*while_jgtgtymybc_readvariableop_2_resource,while_jgtgtymybc_readvariableop_2_resource_0"V
(while_jgtgtymybc_readvariableop_resource*while_jgtgtymybc_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jgtgtymybc/BiasAdd/ReadVariableOp'while/jgtgtymybc/BiasAdd/ReadVariableOp2P
&while/jgtgtymybc/MatMul/ReadVariableOp&while/jgtgtymybc/MatMul/ReadVariableOp2T
(while/jgtgtymybc/MatMul_1/ReadVariableOp(while/jgtgtymybc/MatMul_1/ReadVariableOp2B
while/jgtgtymybc/ReadVariableOpwhile/jgtgtymybc/ReadVariableOp2F
!while/jgtgtymybc/ReadVariableOp_1!while/jgtgtymybc/ReadVariableOp_12F
!while/jgtgtymybc/ReadVariableOp_2!while/jgtgtymybc/ReadVariableOp_2: 
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
¯F
ê
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1732368

inputs%
qjjkgjcvpf_1732269:	 %
qjjkgjcvpf_1732271:	 !
qjjkgjcvpf_1732273:	 
qjjkgjcvpf_1732275:  
qjjkgjcvpf_1732277:  
qjjkgjcvpf_1732279: 
identity¢"qjjkgjcvpf/StatefulPartitionedCall¢whileD
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
strided_slice_2Ú
"qjjkgjcvpf/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0qjjkgjcvpf_1732269qjjkgjcvpf_1732271qjjkgjcvpf_1732273qjjkgjcvpf_1732275qjjkgjcvpf_1732277qjjkgjcvpf_1732279*
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
GPU2*0J 8 *P
fKRI
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_17322682$
"qjjkgjcvpf/StatefulPartitionedCall
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
while/loop_counterð
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0qjjkgjcvpf_1732269qjjkgjcvpf_1732271qjjkgjcvpf_1732273qjjkgjcvpf_1732275qjjkgjcvpf_1732277qjjkgjcvpf_1732279*
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
while_body_1732288*
condR
while_cond_1732287*Q
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
transpose_1
IdentityIdentitystrided_slice_3:output:0#^qjjkgjcvpf/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"qjjkgjcvpf/StatefulPartitionedCall"qjjkgjcvpf/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ë

,__inference_ksthobzafc_layer_call_fn_1735943
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall°
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
GPU2*0J 8 *P
fKRI
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_17326312
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
ç)
Ò
while_body_1732288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_qjjkgjcvpf_1732312_0:	 -
while_qjjkgjcvpf_1732314_0:	 )
while_qjjkgjcvpf_1732316_0:	(
while_qjjkgjcvpf_1732318_0: (
while_qjjkgjcvpf_1732320_0: (
while_qjjkgjcvpf_1732322_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_qjjkgjcvpf_1732312:	 +
while_qjjkgjcvpf_1732314:	 '
while_qjjkgjcvpf_1732316:	&
while_qjjkgjcvpf_1732318: &
while_qjjkgjcvpf_1732320: &
while_qjjkgjcvpf_1732322: ¢(while/qjjkgjcvpf/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¶
(while/qjjkgjcvpf/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_qjjkgjcvpf_1732312_0while_qjjkgjcvpf_1732314_0while_qjjkgjcvpf_1732316_0while_qjjkgjcvpf_1732318_0while_qjjkgjcvpf_1732320_0while_qjjkgjcvpf_1732322_0*
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
GPU2*0J 8 *P
fKRI
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_17322682*
(while/qjjkgjcvpf/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/qjjkgjcvpf/StatefulPartitionedCall:output:0*
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
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/qjjkgjcvpf/StatefulPartitionedCall:output:1)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/qjjkgjcvpf/StatefulPartitionedCall:output:2)^while/qjjkgjcvpf/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_qjjkgjcvpf_1732312while_qjjkgjcvpf_1732312_0"6
while_qjjkgjcvpf_1732314while_qjjkgjcvpf_1732314_0"6
while_qjjkgjcvpf_1732316while_qjjkgjcvpf_1732316_0"6
while_qjjkgjcvpf_1732318while_qjjkgjcvpf_1732318_0"6
while_qjjkgjcvpf_1732320while_qjjkgjcvpf_1732320_0"6
while_qjjkgjcvpf_1732322while_qjjkgjcvpf_1732322_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/qjjkgjcvpf/StatefulPartitionedCall(while/qjjkgjcvpf/StatefulPartitionedCall: 
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
àY

while_body_1733082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jgtgtymybc_matmul_readvariableop_resource_0:	F
3while_jgtgtymybc_matmul_1_readvariableop_resource_0:	 A
2while_jgtgtymybc_biasadd_readvariableop_resource_0:	8
*while_jgtgtymybc_readvariableop_resource_0: :
,while_jgtgtymybc_readvariableop_1_resource_0: :
,while_jgtgtymybc_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jgtgtymybc_matmul_readvariableop_resource:	D
1while_jgtgtymybc_matmul_1_readvariableop_resource:	 ?
0while_jgtgtymybc_biasadd_readvariableop_resource:	6
(while_jgtgtymybc_readvariableop_resource: 8
*while_jgtgtymybc_readvariableop_1_resource: 8
*while_jgtgtymybc_readvariableop_2_resource: ¢'while/jgtgtymybc/BiasAdd/ReadVariableOp¢&while/jgtgtymybc/MatMul/ReadVariableOp¢(while/jgtgtymybc/MatMul_1/ReadVariableOp¢while/jgtgtymybc/ReadVariableOp¢!while/jgtgtymybc/ReadVariableOp_1¢!while/jgtgtymybc/ReadVariableOp_2Ã
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
)while/TensorArrayV2Read/TensorListGetItemÃ
&while/jgtgtymybc/MatMul/ReadVariableOpReadVariableOp1while_jgtgtymybc_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jgtgtymybc/MatMul/ReadVariableOpÑ
while/jgtgtymybc/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jgtgtymybc/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMulÉ
(while/jgtgtymybc/MatMul_1/ReadVariableOpReadVariableOp3while_jgtgtymybc_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jgtgtymybc/MatMul_1/ReadVariableOpº
while/jgtgtymybc/MatMul_1MatMulwhile_placeholder_20while/jgtgtymybc/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/MatMul_1°
while/jgtgtymybc/addAddV2!while/jgtgtymybc/MatMul:product:0#while/jgtgtymybc/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/addÂ
'while/jgtgtymybc/BiasAdd/ReadVariableOpReadVariableOp2while_jgtgtymybc_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jgtgtymybc/BiasAdd/ReadVariableOp½
while/jgtgtymybc/BiasAddBiasAddwhile/jgtgtymybc/add:z:0/while/jgtgtymybc/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jgtgtymybc/BiasAdd
 while/jgtgtymybc/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jgtgtymybc/split/split_dim
while/jgtgtymybc/splitSplit)while/jgtgtymybc/split/split_dim:output:0!while/jgtgtymybc/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jgtgtymybc/split©
while/jgtgtymybc/ReadVariableOpReadVariableOp*while_jgtgtymybc_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jgtgtymybc/ReadVariableOp£
while/jgtgtymybc/mulMul'while/jgtgtymybc/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul¦
while/jgtgtymybc/add_1AddV2while/jgtgtymybc/split:output:0while/jgtgtymybc/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_1
while/jgtgtymybc/SigmoidSigmoidwhile/jgtgtymybc/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid¯
!while/jgtgtymybc/ReadVariableOp_1ReadVariableOp,while_jgtgtymybc_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_1©
while/jgtgtymybc/mul_1Mul)while/jgtgtymybc/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_1¨
while/jgtgtymybc/add_2AddV2while/jgtgtymybc/split:output:1while/jgtgtymybc/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_2
while/jgtgtymybc/Sigmoid_1Sigmoidwhile/jgtgtymybc/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_1
while/jgtgtymybc/mul_2Mulwhile/jgtgtymybc/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_2
while/jgtgtymybc/TanhTanhwhile/jgtgtymybc/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh¢
while/jgtgtymybc/mul_3Mulwhile/jgtgtymybc/Sigmoid:y:0while/jgtgtymybc/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_3£
while/jgtgtymybc/add_3AddV2while/jgtgtymybc/mul_2:z:0while/jgtgtymybc/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_3¯
!while/jgtgtymybc/ReadVariableOp_2ReadVariableOp,while_jgtgtymybc_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jgtgtymybc/ReadVariableOp_2°
while/jgtgtymybc/mul_4Mul)while/jgtgtymybc/ReadVariableOp_2:value:0while/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_4¨
while/jgtgtymybc/add_4AddV2while/jgtgtymybc/split:output:3while/jgtgtymybc/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/add_4
while/jgtgtymybc/Sigmoid_2Sigmoidwhile/jgtgtymybc/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Sigmoid_2
while/jgtgtymybc/Tanh_1Tanhwhile/jgtgtymybc/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/Tanh_1¦
while/jgtgtymybc/mul_5Mulwhile/jgtgtymybc/Sigmoid_2:y:0while/jgtgtymybc/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jgtgtymybc/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jgtgtymybc/mul_5:z:0*
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
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jgtgtymybc/mul_5:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jgtgtymybc/add_3:z:0(^while/jgtgtymybc/BiasAdd/ReadVariableOp'^while/jgtgtymybc/MatMul/ReadVariableOp)^while/jgtgtymybc/MatMul_1/ReadVariableOp ^while/jgtgtymybc/ReadVariableOp"^while/jgtgtymybc/ReadVariableOp_1"^while/jgtgtymybc/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"f
0while_jgtgtymybc_biasadd_readvariableop_resource2while_jgtgtymybc_biasadd_readvariableop_resource_0"h
1while_jgtgtymybc_matmul_1_readvariableop_resource3while_jgtgtymybc_matmul_1_readvariableop_resource_0"d
/while_jgtgtymybc_matmul_readvariableop_resource1while_jgtgtymybc_matmul_readvariableop_resource_0"Z
*while_jgtgtymybc_readvariableop_1_resource,while_jgtgtymybc_readvariableop_1_resource_0"Z
*while_jgtgtymybc_readvariableop_2_resource,while_jgtgtymybc_readvariableop_2_resource_0"V
(while_jgtgtymybc_readvariableop_resource*while_jgtgtymybc_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jgtgtymybc/BiasAdd/ReadVariableOp'while/jgtgtymybc/BiasAdd/ReadVariableOp2P
&while/jgtgtymybc/MatMul/ReadVariableOp&while/jgtgtymybc/MatMul/ReadVariableOp2T
(while/jgtgtymybc/MatMul_1/ReadVariableOp(while/jgtgtymybc/MatMul_1/ReadVariableOp2B
while/jgtgtymybc/ReadVariableOpwhile/jgtgtymybc/ReadVariableOp2F
!while/jgtgtymybc/ReadVariableOp_1!while/jgtgtymybc/ReadVariableOp_12F
!while/jgtgtymybc/ReadVariableOp_2!while/jgtgtymybc/ReadVariableOp_2: 
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
: "ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
I

qtjxeibmnq;
serving_default_qtjxeibmnq:0ÿÿÿÿÿÿÿÿÿ>

vfmtgawzzo0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ðº
¹D
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
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"ÂA
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "qtjxeibmnq"}}, {"class_name": "Conv1D", "config": {"name": "zdelabjare", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "ojrtxmspqi", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "xedyzswikc", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jgtgtymybc", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "ksthobzafc", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "qjjkgjcvpf", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "vfmtgawzzo", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "qtjxeibmnq"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "qtjxeibmnq"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "zdelabjare", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "ojrtxmspqi", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "xedyzswikc", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jgtgtymybc", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "ksthobzafc", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "qjjkgjcvpf", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "vfmtgawzzo", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
Ì

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¥

_tf_keras_layer
{"name": "zdelabjare", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "zdelabjare", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ÿ
_tf_keras_layerå{"name": "ojrtxmspqi", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "ojrtxmspqi", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
­
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_rnn_layerä{"name": "xedyzswikc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "xedyzswikc", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jgtgtymybc", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
³
cell

state_spec
regularization_losses
	variables
 trainable_variables
!	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_rnn_layerê{"name": "ksthobzafc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "ksthobzafc", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "qjjkgjcvpf", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
__call__
+&call_and_return_all_conditional_losses"²
_tf_keras_layer{"name": "vfmtgawzzo", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "vfmtgawzzo", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}

(iter
	)decay
*learning_rate
+momentum
,rho	rmsr	rmss	"rmst	#rmsu	-rmsv	.rmsw	/rmsx	0rmsy	1rmsz	2rms{	3rms|	4rms}	5rms~	6rms
7rms
8rms"
	optimizer
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
regularization_losses
	variables
9layer_regularization_losses
:non_trainable_variables
;layer_metrics
	trainable_variables
<metrics

=layers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
':%2zdelabjare/kernel
:2zdelabjare/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
>layer_regularization_losses
?non_trainable_variables
trainable_variables
@layer_metrics
	variables
Ametrics

Blayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
trainable_variables
Elayer_metrics
	variables
Fmetrics

Glayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


H
state_size

-kernel
.recurrent_kernel
/bias
0input_gate_peephole_weights
 1forget_gate_peephole_weights
 2output_gate_peephole_weights
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"name": "jgtgtymybc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jgtgtymybc", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
J
-0
.1
/2
03
14
25"
trackable_list_wrapper
¼
regularization_losses
	variables
Mlayer_regularization_losses

Nstates
Onon_trainable_variables
Player_metrics
trainable_variables
Qmetrics

Rlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


S
state_size

3kernel
4recurrent_kernel
5bias
6input_gate_peephole_weights
 7forget_gate_peephole_weights
 8output_gate_peephole_weights
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
__call__
+&call_and_return_all_conditional_losses"Ú
_tf_keras_layerÀ{"name": "qjjkgjcvpf", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "qjjkgjcvpf", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
J
30
41
52
63
74
85"
trackable_list_wrapper
¼
regularization_losses
	variables
Xlayer_regularization_losses

Ystates
Znon_trainable_variables
[layer_metrics
 trainable_variables
\metrics

]layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:! 2vfmtgawzzo/kernel
:2vfmtgawzzo/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
°
$regularization_losses
^layer_regularization_losses
_non_trainable_variables
%trainable_variables
`layer_metrics
&	variables
ametrics

blayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
/:-	2xedyzswikc/jgtgtymybc/kernel
9:7	 2&xedyzswikc/jgtgtymybc/recurrent_kernel
):'2xedyzswikc/jgtgtymybc/bias
?:= 21xedyzswikc/jgtgtymybc/input_gate_peephole_weights
@:> 22xedyzswikc/jgtgtymybc/forget_gate_peephole_weights
@:> 22xedyzswikc/jgtgtymybc/output_gate_peephole_weights
/:-	 2ksthobzafc/qjjkgjcvpf/kernel
9:7	 2&ksthobzafc/qjjkgjcvpf/recurrent_kernel
):'2ksthobzafc/qjjkgjcvpf/bias
?:= 21ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights
@:> 22ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights
@:> 22ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
c0"
trackable_list_wrapper
C
0
1
2
3
4"
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
J
-0
.1
/2
03
14
25"
trackable_list_wrapper
°
Iregularization_losses
dlayer_regularization_losses
enon_trainable_variables
Jtrainable_variables
flayer_metrics
K	variables
gmetrics

hlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
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
'
0"
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
J
30
41
52
63
74
85"
trackable_list_wrapper
°
Tregularization_losses
ilayer_regularization_losses
jnon_trainable_variables
Utrainable_variables
klayer_metrics
V	variables
lmetrics

mlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
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
'
0"
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
Ô
	ntotal
	ocount
p	variables
q	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}
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
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
1:/2RMSprop/zdelabjare/kernel/rms
':%2RMSprop/zdelabjare/bias/rms
-:+ 2RMSprop/vfmtgawzzo/kernel/rms
':%2RMSprop/vfmtgawzzo/bias/rms
9:7	2(RMSprop/xedyzswikc/jgtgtymybc/kernel/rms
C:A	 22RMSprop/xedyzswikc/jgtgtymybc/recurrent_kernel/rms
3:12&RMSprop/xedyzswikc/jgtgtymybc/bias/rms
I:G 2=RMSprop/xedyzswikc/jgtgtymybc/input_gate_peephole_weights/rms
J:H 2>RMSprop/xedyzswikc/jgtgtymybc/forget_gate_peephole_weights/rms
J:H 2>RMSprop/xedyzswikc/jgtgtymybc/output_gate_peephole_weights/rms
9:7	 2(RMSprop/ksthobzafc/qjjkgjcvpf/kernel/rms
C:A	 22RMSprop/ksthobzafc/qjjkgjcvpf/recurrent_kernel/rms
3:12&RMSprop/ksthobzafc/qjjkgjcvpf/bias/rms
I:G 2=RMSprop/ksthobzafc/qjjkgjcvpf/input_gate_peephole_weights/rms
J:H 2>RMSprop/ksthobzafc/qjjkgjcvpf/forget_gate_peephole_weights/rms
J:H 2>RMSprop/ksthobzafc/qjjkgjcvpf/output_gate_peephole_weights/rms
þ2û
,__inference_sequential_layer_call_fn_1733442
,__inference_sequential_layer_call_fn_1734212
,__inference_sequential_layer_call_fn_1734249
,__inference_sequential_layer_call_fn_1734048À
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
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_1734653
G__inference_sequential_layer_call_and_return_conditional_losses_1735057
G__inference_sequential_layer_call_and_return_conditional_losses_1734089
G__inference_sequential_layer_call_and_return_conditional_losses_1734130À
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
ë2è
"__inference__wrapped_model_1731423Á
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
annotationsª *1¢.
,)

qtjxeibmnqÿÿÿÿÿÿÿÿÿ
Ö2Ó
,__inference_zdelabjare_layer_call_fn_1735066¢
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
ñ2î
G__inference_zdelabjare_layer_call_and_return_conditional_losses_1735103¢
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
Ö2Ó
,__inference_ojrtxmspqi_layer_call_fn_1735108¢
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
ñ2î
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_1735121¢
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
¤2¡
,__inference_xedyzswikc_layer_call_fn_1735138
,__inference_xedyzswikc_layer_call_fn_1735155
,__inference_xedyzswikc_layer_call_fn_1735172
,__inference_xedyzswikc_layer_call_fn_1735189æ
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
2
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735369
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735549
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735729
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735909æ
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
¤2¡
,__inference_ksthobzafc_layer_call_fn_1735926
,__inference_ksthobzafc_layer_call_fn_1735943
,__inference_ksthobzafc_layer_call_fn_1735960
,__inference_ksthobzafc_layer_call_fn_1735977æ
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
2
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736157
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736337
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736517
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736697æ
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
Ö2Ó
,__inference_vfmtgawzzo_layer_call_fn_1736706¢
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
ñ2î
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_1736716¢
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
ÏBÌ
%__inference_signature_wrapper_1734175
qtjxeibmnq"
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
 2
,__inference_jgtgtymybc_layer_call_fn_1736739
,__inference_jgtgtymybc_layer_call_fn_1736762¾
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
Ö2Ó
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1736806
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1736850¾
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
 2
,__inference_qjjkgjcvpf_layer_call_fn_1736873
,__inference_qjjkgjcvpf_layer_call_fn_1736896¾
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
Ö2Ó
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1736940
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1736984¾
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
 ¯
"__inference__wrapped_model_1731423-./012345678"#;¢8
1¢.
,)

qtjxeibmnqÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

vfmtgawzzo$!

vfmtgawzzoÿÿÿÿÿÿÿÿÿÌ
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1736806-./012¢}
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
 Ì
G__inference_jgtgtymybc_layer_call_and_return_conditional_losses_1736850-./012¢}
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
 ¡
,__inference_jgtgtymybc_layer_call_fn_1736739ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ ¡
,__inference_jgtgtymybc_layer_call_fn_1736762ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ð
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736157345678S¢P
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
 Ð
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736337345678S¢P
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
 ¿
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736517t345678C¢@
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
 ¿
G__inference_ksthobzafc_layer_call_and_return_conditional_losses_1736697t345678C¢@
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
 §
,__inference_ksthobzafc_layer_call_fn_1735926w345678S¢P
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
ª "ÿÿÿÿÿÿÿÿÿ §
,__inference_ksthobzafc_layer_call_fn_1735943w345678S¢P
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
ª "ÿÿÿÿÿÿÿÿÿ 
,__inference_ksthobzafc_layer_call_fn_1735960g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ 
,__inference_ksthobzafc_layer_call_fn_1735977g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ ¯
G__inference_ojrtxmspqi_layer_call_and_return_conditional_losses_1735121d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_ojrtxmspqi_layer_call_fn_1735108W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÌ
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1736940345678¢}
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
 Ì
G__inference_qjjkgjcvpf_layer_call_and_return_conditional_losses_1736984345678¢}
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
 ¡
,__inference_qjjkgjcvpf_layer_call_fn_1736873ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ ¡
,__inference_qjjkgjcvpf_layer_call_fn_1736896ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ É
G__inference_sequential_layer_call_and_return_conditional_losses_1734089~-./012345678"#C¢@
9¢6
,)

qtjxeibmnqÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_layer_call_and_return_conditional_losses_1734130~-./012345678"#C¢@
9¢6
,)

qtjxeibmnqÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_1734653z-./012345678"#?¢<
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
G__inference_sequential_layer_call_and_return_conditional_losses_1735057z-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
,__inference_sequential_layer_call_fn_1733442q-./012345678"#C¢@
9¢6
,)

qtjxeibmnqÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_layer_call_fn_1734048q-./012345678"#C¢@
9¢6
,)

qtjxeibmnqÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1734212m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1734249m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
%__inference_signature_wrapper_1734175-./012345678"#I¢F
¢ 
?ª<
:

qtjxeibmnq,)

qtjxeibmnqÿÿÿÿÿÿÿÿÿ"7ª4
2

vfmtgawzzo$!

vfmtgawzzoÿÿÿÿÿÿÿÿÿ§
G__inference_vfmtgawzzo_layer_call_and_return_conditional_losses_1736716\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_vfmtgawzzo_layer_call_fn_1736706O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÝ
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735369-./012S¢P
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
 Ý
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735549-./012S¢P
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
 Ã
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735729x-./012C¢@
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
 Ã
G__inference_xedyzswikc_layer_call_and_return_conditional_losses_1735909x-./012C¢@
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
 µ
,__inference_xedyzswikc_layer_call_fn_1735138-./012S¢P
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
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ µ
,__inference_xedyzswikc_layer_call_fn_1735155-./012S¢P
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
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
,__inference_xedyzswikc_layer_call_fn_1735172k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ 
,__inference_xedyzswikc_layer_call_fn_1735189k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ ·
G__inference_zdelabjare_layer_call_and_return_conditional_losses_1735103l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_zdelabjare_layer_call_fn_1735066_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ