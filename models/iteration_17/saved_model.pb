ÿ5
ä
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
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
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
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a7182

mopvqfaljf/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namemopvqfaljf/kernel
{
%mopvqfaljf/kernel/Read/ReadVariableOpReadVariableOpmopvqfaljf/kernel*"
_output_shapes
:*
dtype0
v
mopvqfaljf/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namemopvqfaljf/bias
o
#mopvqfaljf/bias/Read/ReadVariableOpReadVariableOpmopvqfaljf/bias*
_output_shapes
:*
dtype0

ehhqjcwuju/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameehhqjcwuju/kernel
{
%ehhqjcwuju/kernel/Read/ReadVariableOpReadVariableOpehhqjcwuju/kernel*"
_output_shapes
:*
dtype0
v
ehhqjcwuju/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameehhqjcwuju/bias
o
#ehhqjcwuju/bias/Read/ReadVariableOpReadVariableOpehhqjcwuju/bias*
_output_shapes
:*
dtype0
~
pbmomrqadp/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namepbmomrqadp/kernel
w
%pbmomrqadp/kernel/Read/ReadVariableOpReadVariableOppbmomrqadp/kernel*
_output_shapes

: *
dtype0
v
pbmomrqadp/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namepbmomrqadp/bias
o
#pbmomrqadp/bias/Read/ReadVariableOpReadVariableOppbmomrqadp/bias*
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
cxhqebqpjz/rvncypflgq/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namecxhqebqpjz/rvncypflgq/kernel

0cxhqebqpjz/rvncypflgq/kernel/Read/ReadVariableOpReadVariableOpcxhqebqpjz/rvncypflgq/kernel*
_output_shapes
:	*
dtype0
©
&cxhqebqpjz/rvncypflgq/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&cxhqebqpjz/rvncypflgq/recurrent_kernel
¢
:cxhqebqpjz/rvncypflgq/recurrent_kernel/Read/ReadVariableOpReadVariableOp&cxhqebqpjz/rvncypflgq/recurrent_kernel*
_output_shapes
:	 *
dtype0

cxhqebqpjz/rvncypflgq/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecxhqebqpjz/rvncypflgq/bias

.cxhqebqpjz/rvncypflgq/bias/Read/ReadVariableOpReadVariableOpcxhqebqpjz/rvncypflgq/bias*
_output_shapes	
:*
dtype0
º
1cxhqebqpjz/rvncypflgq/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31cxhqebqpjz/rvncypflgq/input_gate_peephole_weights
³
Ecxhqebqpjz/rvncypflgq/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1cxhqebqpjz/rvncypflgq/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2cxhqebqpjz/rvncypflgq/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights
µ
Fcxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2cxhqebqpjz/rvncypflgq/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42cxhqebqpjz/rvncypflgq/output_gate_peephole_weights
µ
Fcxhqebqpjz/rvncypflgq/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2cxhqebqpjz/rvncypflgq/output_gate_peephole_weights*
_output_shapes
: *
dtype0

qzuziqqdld/aiccbsgdoo/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_nameqzuziqqdld/aiccbsgdoo/kernel

0qzuziqqdld/aiccbsgdoo/kernel/Read/ReadVariableOpReadVariableOpqzuziqqdld/aiccbsgdoo/kernel*
_output_shapes
:	 *
dtype0
©
&qzuziqqdld/aiccbsgdoo/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&qzuziqqdld/aiccbsgdoo/recurrent_kernel
¢
:qzuziqqdld/aiccbsgdoo/recurrent_kernel/Read/ReadVariableOpReadVariableOp&qzuziqqdld/aiccbsgdoo/recurrent_kernel*
_output_shapes
:	 *
dtype0

qzuziqqdld/aiccbsgdoo/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameqzuziqqdld/aiccbsgdoo/bias

.qzuziqqdld/aiccbsgdoo/bias/Read/ReadVariableOpReadVariableOpqzuziqqdld/aiccbsgdoo/bias*
_output_shapes	
:*
dtype0
º
1qzuziqqdld/aiccbsgdoo/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights
³
Eqzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights
µ
Fqzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2qzuziqqdld/aiccbsgdoo/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights
µ
Fqzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights*
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
RMSprop/mopvqfaljf/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/mopvqfaljf/kernel/rms

1RMSprop/mopvqfaljf/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/mopvqfaljf/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/mopvqfaljf/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/mopvqfaljf/bias/rms

/RMSprop/mopvqfaljf/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/mopvqfaljf/bias/rms*
_output_shapes
:*
dtype0

RMSprop/ehhqjcwuju/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/ehhqjcwuju/kernel/rms

1RMSprop/ehhqjcwuju/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/ehhqjcwuju/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/ehhqjcwuju/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/ehhqjcwuju/bias/rms

/RMSprop/ehhqjcwuju/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/ehhqjcwuju/bias/rms*
_output_shapes
:*
dtype0

RMSprop/pbmomrqadp/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/pbmomrqadp/kernel/rms

1RMSprop/pbmomrqadp/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/pbmomrqadp/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/pbmomrqadp/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/pbmomrqadp/bias/rms

/RMSprop/pbmomrqadp/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/pbmomrqadp/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/cxhqebqpjz/rvncypflgq/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/cxhqebqpjz/rvncypflgq/kernel/rms
¦
<RMSprop/cxhqebqpjz/rvncypflgq/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/cxhqebqpjz/rvncypflgq/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rms
º
FRMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/cxhqebqpjz/rvncypflgq/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/cxhqebqpjz/rvncypflgq/bias/rms

:RMSprop/cxhqebqpjz/rvncypflgq/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/cxhqebqpjz/rvncypflgq/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rms
Ë
QRMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rms
Í
RRMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rms
Í
RRMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rms
¦
<RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rms
º
FRMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/qzuziqqdld/aiccbsgdoo/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/qzuziqqdld/aiccbsgdoo/bias/rms

:RMSprop/qzuziqqdld/aiccbsgdoo/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/qzuziqqdld/aiccbsgdoo/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rms
Ë
QRMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rms
Í
RRMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rms
Í
RRMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0

NoOpNoOp
I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÇH
value½HBºH B³H
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
 regularization_losses
!trainable_variables
"	keras_api
l
#cell
$
state_spec
%	variables
&regularization_losses
'trainable_variables
(	keras_api
h

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api

/iter
	0decay
1learning_rate
2momentum
3rho	rms~	rms
rms
rms
)rms
*rms
4rms
5rms
6rms
7rms
8rms
9rms
:rms
;rms
<rms
=rms
>rms
?rms

0
1
2
3
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
)16
*17
 

0
1
2
3
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
)16
*17
­
@non_trainable_variables
	variables

Alayers
	regularization_losses
Blayer_metrics
Cmetrics
Dlayer_regularization_losses

trainable_variables
 
][
VARIABLE_VALUEmopvqfaljf/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmopvqfaljf/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Enon_trainable_variables
Flayer_metrics

Glayers
	variables
regularization_losses
Hmetrics
Ilayer_regularization_losses
trainable_variables
][
VARIABLE_VALUEehhqjcwuju/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEehhqjcwuju/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Jnon_trainable_variables
Klayer_metrics

Llayers
	variables
regularization_losses
Mmetrics
Nlayer_regularization_losses
trainable_variables
 
 
 
­
Onon_trainable_variables
Player_metrics

Qlayers
	variables
regularization_losses
Rmetrics
Slayer_regularization_losses
trainable_variables
ó
T
state_size

4kernel
5recurrent_kernel
6bias
7input_gate_peephole_weights
 8forget_gate_peephole_weights
 9output_gate_peephole_weights
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
 
*
40
51
62
73
84
95
 
*
40
51
62
73
84
95
¹
Ynon_trainable_variables
	variables

Zlayers
[layer_metrics
 regularization_losses

\states
]metrics
^layer_regularization_losses
!trainable_variables
ó
_
state_size

:kernel
;recurrent_kernel
<bias
=input_gate_peephole_weights
 >forget_gate_peephole_weights
 ?output_gate_peephole_weights
`	variables
aregularization_losses
btrainable_variables
c	keras_api
 
*
:0
;1
<2
=3
>4
?5
 
*
:0
;1
<2
=3
>4
?5
¹
dnon_trainable_variables
%	variables

elayers
flayer_metrics
&regularization_losses

gstates
hmetrics
ilayer_regularization_losses
'trainable_variables
][
VARIABLE_VALUEpbmomrqadp/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEpbmomrqadp/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
­
jnon_trainable_variables
klayer_metrics

llayers
+	variables
,regularization_losses
mmetrics
nlayer_regularization_losses
-trainable_variables
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
VARIABLE_VALUEcxhqebqpjz/rvncypflgq/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&cxhqebqpjz/rvncypflgq/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEcxhqebqpjz/rvncypflgq/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1cxhqebqpjz/rvncypflgq/input_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights&variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2cxhqebqpjz/rvncypflgq/output_gate_peephole_weights&variables/9/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEqzuziqqdld/aiccbsgdoo/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&qzuziqqdld/aiccbsgdoo/recurrent_kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEqzuziqqdld/aiccbsgdoo/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights'variables/14/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights'variables/15/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
3
4
5
 

o0
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
 
 
 
 
 
*
40
51
62
73
84
95
 
*
40
51
62
73
84
95
­
pnon_trainable_variables
qlayer_metrics

rlayers
U	variables
Vregularization_losses
smetrics
tlayer_regularization_losses
Wtrainable_variables
 

0
 
 
 
 
 
*
:0
;1
<2
=3
>4
?5
 
*
:0
;1
<2
=3
>4
?5
­
unon_trainable_variables
vlayer_metrics

wlayers
`	variables
aregularization_losses
xmetrics
ylayer_regularization_losses
btrainable_variables
 

#0
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
	ztotal
	{count
|	variables
}	keras_api
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
z0
{1

|	variables

VARIABLE_VALUERMSprop/mopvqfaljf/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/mopvqfaljf/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/ehhqjcwuju/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/ehhqjcwuju/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/pbmomrqadp/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/pbmomrqadp/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/cxhqebqpjz/rvncypflgq/kernel/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/cxhqebqpjz/rvncypflgq/bias/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/qzuziqqdld/aiccbsgdoo/bias/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rmsEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rmsEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_gvxdqcynanPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_gvxdqcynanmopvqfaljf/kernelmopvqfaljf/biasehhqjcwuju/kernelehhqjcwuju/biascxhqebqpjz/rvncypflgq/kernel&cxhqebqpjz/rvncypflgq/recurrent_kernelcxhqebqpjz/rvncypflgq/bias1cxhqebqpjz/rvncypflgq/input_gate_peephole_weights2cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights2cxhqebqpjz/rvncypflgq/output_gate_peephole_weightsqzuziqqdld/aiccbsgdoo/kernel&qzuziqqdld/aiccbsgdoo/recurrent_kernelqzuziqqdld/aiccbsgdoo/bias1qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights2qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights2qzuziqqdld/aiccbsgdoo/output_gate_peephole_weightspbmomrqadp/kernelpbmomrqadp/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1719690
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
«
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%mopvqfaljf/kernel/Read/ReadVariableOp#mopvqfaljf/bias/Read/ReadVariableOp%ehhqjcwuju/kernel/Read/ReadVariableOp#ehhqjcwuju/bias/Read/ReadVariableOp%pbmomrqadp/kernel/Read/ReadVariableOp#pbmomrqadp/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0cxhqebqpjz/rvncypflgq/kernel/Read/ReadVariableOp:cxhqebqpjz/rvncypflgq/recurrent_kernel/Read/ReadVariableOp.cxhqebqpjz/rvncypflgq/bias/Read/ReadVariableOpEcxhqebqpjz/rvncypflgq/input_gate_peephole_weights/Read/ReadVariableOpFcxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/Read/ReadVariableOpFcxhqebqpjz/rvncypflgq/output_gate_peephole_weights/Read/ReadVariableOp0qzuziqqdld/aiccbsgdoo/kernel/Read/ReadVariableOp:qzuziqqdld/aiccbsgdoo/recurrent_kernel/Read/ReadVariableOp.qzuziqqdld/aiccbsgdoo/bias/Read/ReadVariableOpEqzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/Read/ReadVariableOpFqzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/Read/ReadVariableOpFqzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/mopvqfaljf/kernel/rms/Read/ReadVariableOp/RMSprop/mopvqfaljf/bias/rms/Read/ReadVariableOp1RMSprop/ehhqjcwuju/kernel/rms/Read/ReadVariableOp/RMSprop/ehhqjcwuju/bias/rms/Read/ReadVariableOp1RMSprop/pbmomrqadp/kernel/rms/Read/ReadVariableOp/RMSprop/pbmomrqadp/bias/rms/Read/ReadVariableOp<RMSprop/cxhqebqpjz/rvncypflgq/kernel/rms/Read/ReadVariableOpFRMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/cxhqebqpjz/rvncypflgq/bias/rms/Read/ReadVariableOpQRMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rms/Read/ReadVariableOpFRMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/qzuziqqdld/aiccbsgdoo/bias/rms/Read/ReadVariableOpQRMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*8
Tin1
/2-	*
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
 __inference__traced_save_1722777
Ê
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemopvqfaljf/kernelmopvqfaljf/biasehhqjcwuju/kernelehhqjcwuju/biaspbmomrqadp/kernelpbmomrqadp/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhocxhqebqpjz/rvncypflgq/kernel&cxhqebqpjz/rvncypflgq/recurrent_kernelcxhqebqpjz/rvncypflgq/bias1cxhqebqpjz/rvncypflgq/input_gate_peephole_weights2cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights2cxhqebqpjz/rvncypflgq/output_gate_peephole_weightsqzuziqqdld/aiccbsgdoo/kernel&qzuziqqdld/aiccbsgdoo/recurrent_kernelqzuziqqdld/aiccbsgdoo/bias1qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights2qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights2qzuziqqdld/aiccbsgdoo/output_gate_peephole_weightstotalcountRMSprop/mopvqfaljf/kernel/rmsRMSprop/mopvqfaljf/bias/rmsRMSprop/ehhqjcwuju/kernel/rmsRMSprop/ehhqjcwuju/bias/rmsRMSprop/pbmomrqadp/kernel/rmsRMSprop/pbmomrqadp/bias/rms(RMSprop/cxhqebqpjz/rvncypflgq/kernel/rms2RMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rms&RMSprop/cxhqebqpjz/rvncypflgq/bias/rms=RMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rms>RMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rms>RMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rms(RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rms2RMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rms&RMSprop/qzuziqqdld/aiccbsgdoo/bias/rms=RMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rms>RMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rms>RMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rms*7
Tin0
.2,*
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
#__inference__traced_restore_1722916É²0
p
Ê
qzuziqqdld_while_body_17200222
.qzuziqqdld_while_qzuziqqdld_while_loop_counter8
4qzuziqqdld_while_qzuziqqdld_while_maximum_iterations 
qzuziqqdld_while_placeholder"
qzuziqqdld_while_placeholder_1"
qzuziqqdld_while_placeholder_2"
qzuziqqdld_while_placeholder_31
-qzuziqqdld_while_qzuziqqdld_strided_slice_1_0m
iqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0O
<qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0:	 Q
>qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 L
=qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0:	C
5qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0: E
7qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0: E
7qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0: 
qzuziqqdld_while_identity
qzuziqqdld_while_identity_1
qzuziqqdld_while_identity_2
qzuziqqdld_while_identity_3
qzuziqqdld_while_identity_4
qzuziqqdld_while_identity_5/
+qzuziqqdld_while_qzuziqqdld_strided_slice_1k
gqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensorM
:qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource:	 O
<qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource:	 J
;qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource:	A
3qzuziqqdld_while_aiccbsgdoo_readvariableop_resource: C
5qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource: C
5qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource: ¢2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp¢1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp¢3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp¢*qzuziqqdld/while/aiccbsgdoo/ReadVariableOp¢,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1¢,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2Ù
Bqzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bqzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shape
4qzuziqqdld/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0qzuziqqdld_while_placeholderKqzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4qzuziqqdld/while/TensorArrayV2Read/TensorListGetItemä
1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp<qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOpý
"qzuziqqdld/while/aiccbsgdoo/MatMulMatMul;qzuziqqdld/while/TensorArrayV2Read/TensorListGetItem:item:09qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"qzuziqqdld/while/aiccbsgdoo/MatMulê
3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp>qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOpæ
$qzuziqqdld/while/aiccbsgdoo/MatMul_1MatMulqzuziqqdld_while_placeholder_2;qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$qzuziqqdld/while/aiccbsgdoo/MatMul_1Ü
qzuziqqdld/while/aiccbsgdoo/addAddV2,qzuziqqdld/while/aiccbsgdoo/MatMul:product:0.qzuziqqdld/while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
qzuziqqdld/while/aiccbsgdoo/addã
2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp=qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOpé
#qzuziqqdld/while/aiccbsgdoo/BiasAddBiasAdd#qzuziqqdld/while/aiccbsgdoo/add:z:0:qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#qzuziqqdld/while/aiccbsgdoo/BiasAdd
+qzuziqqdld/while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+qzuziqqdld/while/aiccbsgdoo/split/split_dim¯
!qzuziqqdld/while/aiccbsgdoo/splitSplit4qzuziqqdld/while/aiccbsgdoo/split/split_dim:output:0,qzuziqqdld/while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!qzuziqqdld/while/aiccbsgdoo/splitÊ
*qzuziqqdld/while/aiccbsgdoo/ReadVariableOpReadVariableOp5qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*qzuziqqdld/while/aiccbsgdoo/ReadVariableOpÏ
qzuziqqdld/while/aiccbsgdoo/mulMul2qzuziqqdld/while/aiccbsgdoo/ReadVariableOp:value:0qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
qzuziqqdld/while/aiccbsgdoo/mulÒ
!qzuziqqdld/while/aiccbsgdoo/add_1AddV2*qzuziqqdld/while/aiccbsgdoo/split:output:0#qzuziqqdld/while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_1®
#qzuziqqdld/while/aiccbsgdoo/SigmoidSigmoid%qzuziqqdld/while/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#qzuziqqdld/while/aiccbsgdoo/SigmoidÐ
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp7qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1Õ
!qzuziqqdld/while/aiccbsgdoo/mul_1Mul4qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1:value:0qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_1Ô
!qzuziqqdld/while/aiccbsgdoo/add_2AddV2*qzuziqqdld/while/aiccbsgdoo/split:output:1%qzuziqqdld/while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_2²
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_1Sigmoid%qzuziqqdld/while/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_1Ê
!qzuziqqdld/while/aiccbsgdoo/mul_2Mul)qzuziqqdld/while/aiccbsgdoo/Sigmoid_1:y:0qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_2ª
 qzuziqqdld/while/aiccbsgdoo/TanhTanh*qzuziqqdld/while/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 qzuziqqdld/while/aiccbsgdoo/TanhÎ
!qzuziqqdld/while/aiccbsgdoo/mul_3Mul'qzuziqqdld/while/aiccbsgdoo/Sigmoid:y:0$qzuziqqdld/while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_3Ï
!qzuziqqdld/while/aiccbsgdoo/add_3AddV2%qzuziqqdld/while/aiccbsgdoo/mul_2:z:0%qzuziqqdld/while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_3Ð
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp7qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2Ü
!qzuziqqdld/while/aiccbsgdoo/mul_4Mul4qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2:value:0%qzuziqqdld/while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_4Ô
!qzuziqqdld/while/aiccbsgdoo/add_4AddV2*qzuziqqdld/while/aiccbsgdoo/split:output:3%qzuziqqdld/while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_4²
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_2Sigmoid%qzuziqqdld/while/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_2©
"qzuziqqdld/while/aiccbsgdoo/Tanh_1Tanh%qzuziqqdld/while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"qzuziqqdld/while/aiccbsgdoo/Tanh_1Ò
!qzuziqqdld/while/aiccbsgdoo/mul_5Mul)qzuziqqdld/while/aiccbsgdoo/Sigmoid_2:y:0&qzuziqqdld/while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_5
5qzuziqqdld/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqzuziqqdld_while_placeholder_1qzuziqqdld_while_placeholder%qzuziqqdld/while/aiccbsgdoo/mul_5:z:0*
_output_shapes
: *
element_dtype027
5qzuziqqdld/while/TensorArrayV2Write/TensorListSetItemr
qzuziqqdld/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
qzuziqqdld/while/add/y
qzuziqqdld/while/addAddV2qzuziqqdld_while_placeholderqzuziqqdld/while/add/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/while/addv
qzuziqqdld/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
qzuziqqdld/while/add_1/y­
qzuziqqdld/while/add_1AddV2.qzuziqqdld_while_qzuziqqdld_while_loop_counter!qzuziqqdld/while/add_1/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/while/add_1©
qzuziqqdld/while/IdentityIdentityqzuziqqdld/while/add_1:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/IdentityÇ
qzuziqqdld/while/Identity_1Identity4qzuziqqdld_while_qzuziqqdld_while_maximum_iterations3^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/Identity_1«
qzuziqqdld/while/Identity_2Identityqzuziqqdld/while/add:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/Identity_2Ø
qzuziqqdld/while/Identity_3IdentityEqzuziqqdld/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/Identity_3É
qzuziqqdld/while/Identity_4Identity%qzuziqqdld/while/aiccbsgdoo/mul_5:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/while/Identity_4É
qzuziqqdld/while/Identity_5Identity%qzuziqqdld/while/aiccbsgdoo/add_3:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/while/Identity_5"|
;qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource=qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0"~
<qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource>qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0"z
:qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource<qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0"p
5qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource7qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0"p
5qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource7qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0"l
3qzuziqqdld_while_aiccbsgdoo_readvariableop_resource5qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0"?
qzuziqqdld_while_identity"qzuziqqdld/while/Identity:output:0"C
qzuziqqdld_while_identity_1$qzuziqqdld/while/Identity_1:output:0"C
qzuziqqdld_while_identity_2$qzuziqqdld/while/Identity_2:output:0"C
qzuziqqdld_while_identity_3$qzuziqqdld/while/Identity_3:output:0"C
qzuziqqdld_while_identity_4$qzuziqqdld/while/Identity_4:output:0"C
qzuziqqdld_while_identity_5$qzuziqqdld/while/Identity_5:output:0"\
+qzuziqqdld_while_qzuziqqdld_strided_slice_1-qzuziqqdld_while_qzuziqqdld_strided_slice_1_0"Ô
gqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensoriqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2f
1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp2j
3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp2X
*qzuziqqdld/while/aiccbsgdoo/ReadVariableOp*qzuziqqdld/while/aiccbsgdoo/ReadVariableOp2\
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_12\
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2: 
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
Ê

G__inference_sequential_layer_call_and_return_conditional_losses_1720129

inputsL
6mopvqfaljf_conv1d_expanddims_1_readvariableop_resource:K
=mopvqfaljf_squeeze_batch_dims_biasadd_readvariableop_resource:L
6ehhqjcwuju_conv1d_expanddims_1_readvariableop_resource:K
=ehhqjcwuju_squeeze_batch_dims_biasadd_readvariableop_resource:G
4cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource:	I
6cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource:	 D
5cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource:	;
-cxhqebqpjz_rvncypflgq_readvariableop_resource: =
/cxhqebqpjz_rvncypflgq_readvariableop_1_resource: =
/cxhqebqpjz_rvncypflgq_readvariableop_2_resource: G
4qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource:	 I
6qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource:	 D
5qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource:	;
-qzuziqqdld_aiccbsgdoo_readvariableop_resource: =
/qzuziqqdld_aiccbsgdoo_readvariableop_1_resource: =
/qzuziqqdld_aiccbsgdoo_readvariableop_2_resource: ;
)pbmomrqadp_matmul_readvariableop_resource: 8
*pbmomrqadp_biasadd_readvariableop_resource:
identity¢,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp¢+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp¢-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp¢$cxhqebqpjz/rvncypflgq/ReadVariableOp¢&cxhqebqpjz/rvncypflgq/ReadVariableOp_1¢&cxhqebqpjz/rvncypflgq/ReadVariableOp_2¢cxhqebqpjz/while¢-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp¢4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp¢-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp¢4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!pbmomrqadp/BiasAdd/ReadVariableOp¢ pbmomrqadp/MatMul/ReadVariableOp¢,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp¢+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp¢-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp¢$qzuziqqdld/aiccbsgdoo/ReadVariableOp¢&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1¢&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2¢qzuziqqdld/while
 mopvqfaljf/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 mopvqfaljf/conv1d/ExpandDims/dim»
mopvqfaljf/conv1d/ExpandDims
ExpandDimsinputs)mopvqfaljf/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/ExpandDimsÙ
-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mopvqfaljf_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp
"mopvqfaljf/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"mopvqfaljf/conv1d/ExpandDims_1/dimã
mopvqfaljf/conv1d/ExpandDims_1
ExpandDims5mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp:value:0+mopvqfaljf/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
mopvqfaljf/conv1d/ExpandDims_1
mopvqfaljf/conv1d/ShapeShape%mopvqfaljf/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
mopvqfaljf/conv1d/Shape
%mopvqfaljf/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%mopvqfaljf/conv1d/strided_slice/stack¥
'mopvqfaljf/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'mopvqfaljf/conv1d/strided_slice/stack_1
'mopvqfaljf/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'mopvqfaljf/conv1d/strided_slice/stack_2Ì
mopvqfaljf/conv1d/strided_sliceStridedSlice mopvqfaljf/conv1d/Shape:output:0.mopvqfaljf/conv1d/strided_slice/stack:output:00mopvqfaljf/conv1d/strided_slice/stack_1:output:00mopvqfaljf/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
mopvqfaljf/conv1d/strided_slice
mopvqfaljf/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
mopvqfaljf/conv1d/Reshape/shapeÌ
mopvqfaljf/conv1d/ReshapeReshape%mopvqfaljf/conv1d/ExpandDims:output:0(mopvqfaljf/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/Reshapeî
mopvqfaljf/conv1d/Conv2DConv2D"mopvqfaljf/conv1d/Reshape:output:0'mopvqfaljf/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
mopvqfaljf/conv1d/Conv2D
!mopvqfaljf/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!mopvqfaljf/conv1d/concat/values_1
mopvqfaljf/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/concat/axisì
mopvqfaljf/conv1d/concatConcatV2(mopvqfaljf/conv1d/strided_slice:output:0*mopvqfaljf/conv1d/concat/values_1:output:0&mopvqfaljf/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
mopvqfaljf/conv1d/concatÉ
mopvqfaljf/conv1d/Reshape_1Reshape!mopvqfaljf/conv1d/Conv2D:output:0!mopvqfaljf/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/Reshape_1Á
mopvqfaljf/conv1d/SqueezeSqueeze$mopvqfaljf/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/Squeeze
#mopvqfaljf/squeeze_batch_dims/ShapeShape"mopvqfaljf/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#mopvqfaljf/squeeze_batch_dims/Shape°
1mopvqfaljf/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1mopvqfaljf/squeeze_batch_dims/strided_slice/stack½
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1´
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2
+mopvqfaljf/squeeze_batch_dims/strided_sliceStridedSlice,mopvqfaljf/squeeze_batch_dims/Shape:output:0:mopvqfaljf/squeeze_batch_dims/strided_slice/stack:output:0<mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1:output:0<mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+mopvqfaljf/squeeze_batch_dims/strided_slice¯
+mopvqfaljf/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+mopvqfaljf/squeeze_batch_dims/Reshape/shapeé
%mopvqfaljf/squeeze_batch_dims/ReshapeReshape"mopvqfaljf/conv1d/Squeeze:output:04mopvqfaljf/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%mopvqfaljf/squeeze_batch_dims/Reshapeæ
4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=mopvqfaljf_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%mopvqfaljf/squeeze_batch_dims/BiasAddBiasAdd.mopvqfaljf/squeeze_batch_dims/Reshape:output:0<mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%mopvqfaljf/squeeze_batch_dims/BiasAdd¯
-mopvqfaljf/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-mopvqfaljf/squeeze_batch_dims/concat/values_1¡
)mopvqfaljf/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)mopvqfaljf/squeeze_batch_dims/concat/axis¨
$mopvqfaljf/squeeze_batch_dims/concatConcatV24mopvqfaljf/squeeze_batch_dims/strided_slice:output:06mopvqfaljf/squeeze_batch_dims/concat/values_1:output:02mopvqfaljf/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$mopvqfaljf/squeeze_batch_dims/concatö
'mopvqfaljf/squeeze_batch_dims/Reshape_1Reshape.mopvqfaljf/squeeze_batch_dims/BiasAdd:output:0-mopvqfaljf/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'mopvqfaljf/squeeze_batch_dims/Reshape_1£
ehhqjcwuju/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
ehhqjcwuju/Pad/paddingsµ
ehhqjcwuju/PadPad0mopvqfaljf/squeeze_batch_dims/Reshape_1:output:0 ehhqjcwuju/Pad/paddings:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/Pad
 ehhqjcwuju/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 ehhqjcwuju/conv1d/ExpandDims/dimÌ
ehhqjcwuju/conv1d/ExpandDims
ExpandDimsehhqjcwuju/Pad:output:0)ehhqjcwuju/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/ExpandDimsÙ
-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6ehhqjcwuju_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp
"ehhqjcwuju/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ehhqjcwuju/conv1d/ExpandDims_1/dimã
ehhqjcwuju/conv1d/ExpandDims_1
ExpandDims5ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp:value:0+ehhqjcwuju/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
ehhqjcwuju/conv1d/ExpandDims_1
ehhqjcwuju/conv1d/ShapeShape%ehhqjcwuju/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
ehhqjcwuju/conv1d/Shape
%ehhqjcwuju/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%ehhqjcwuju/conv1d/strided_slice/stack¥
'ehhqjcwuju/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'ehhqjcwuju/conv1d/strided_slice/stack_1
'ehhqjcwuju/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'ehhqjcwuju/conv1d/strided_slice/stack_2Ì
ehhqjcwuju/conv1d/strided_sliceStridedSlice ehhqjcwuju/conv1d/Shape:output:0.ehhqjcwuju/conv1d/strided_slice/stack:output:00ehhqjcwuju/conv1d/strided_slice/stack_1:output:00ehhqjcwuju/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
ehhqjcwuju/conv1d/strided_slice
ehhqjcwuju/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
ehhqjcwuju/conv1d/Reshape/shapeÌ
ehhqjcwuju/conv1d/ReshapeReshape%ehhqjcwuju/conv1d/ExpandDims:output:0(ehhqjcwuju/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/Reshapeî
ehhqjcwuju/conv1d/Conv2DConv2D"ehhqjcwuju/conv1d/Reshape:output:0'ehhqjcwuju/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ehhqjcwuju/conv1d/Conv2D
!ehhqjcwuju/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!ehhqjcwuju/conv1d/concat/values_1
ehhqjcwuju/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/concat/axisì
ehhqjcwuju/conv1d/concatConcatV2(ehhqjcwuju/conv1d/strided_slice:output:0*ehhqjcwuju/conv1d/concat/values_1:output:0&ehhqjcwuju/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ehhqjcwuju/conv1d/concatÉ
ehhqjcwuju/conv1d/Reshape_1Reshape!ehhqjcwuju/conv1d/Conv2D:output:0!ehhqjcwuju/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/Reshape_1Á
ehhqjcwuju/conv1d/SqueezeSqueeze$ehhqjcwuju/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/Squeeze
#ehhqjcwuju/squeeze_batch_dims/ShapeShape"ehhqjcwuju/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#ehhqjcwuju/squeeze_batch_dims/Shape°
1ehhqjcwuju/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1ehhqjcwuju/squeeze_batch_dims/strided_slice/stack½
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1´
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2
+ehhqjcwuju/squeeze_batch_dims/strided_sliceStridedSlice,ehhqjcwuju/squeeze_batch_dims/Shape:output:0:ehhqjcwuju/squeeze_batch_dims/strided_slice/stack:output:0<ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1:output:0<ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+ehhqjcwuju/squeeze_batch_dims/strided_slice¯
+ehhqjcwuju/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+ehhqjcwuju/squeeze_batch_dims/Reshape/shapeé
%ehhqjcwuju/squeeze_batch_dims/ReshapeReshape"ehhqjcwuju/conv1d/Squeeze:output:04ehhqjcwuju/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ehhqjcwuju/squeeze_batch_dims/Reshapeæ
4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=ehhqjcwuju_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%ehhqjcwuju/squeeze_batch_dims/BiasAddBiasAdd.ehhqjcwuju/squeeze_batch_dims/Reshape:output:0<ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ehhqjcwuju/squeeze_batch_dims/BiasAdd¯
-ehhqjcwuju/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-ehhqjcwuju/squeeze_batch_dims/concat/values_1¡
)ehhqjcwuju/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)ehhqjcwuju/squeeze_batch_dims/concat/axis¨
$ehhqjcwuju/squeeze_batch_dims/concatConcatV24ehhqjcwuju/squeeze_batch_dims/strided_slice:output:06ehhqjcwuju/squeeze_batch_dims/concat/values_1:output:02ehhqjcwuju/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$ehhqjcwuju/squeeze_batch_dims/concatö
'ehhqjcwuju/squeeze_batch_dims/Reshape_1Reshape.ehhqjcwuju/squeeze_batch_dims/BiasAdd:output:0-ehhqjcwuju/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'ehhqjcwuju/squeeze_batch_dims/Reshape_1
abbthhzbau/ShapeShape0ehhqjcwuju/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
abbthhzbau/Shape
abbthhzbau/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
abbthhzbau/strided_slice/stack
 abbthhzbau/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 abbthhzbau/strided_slice/stack_1
 abbthhzbau/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 abbthhzbau/strided_slice/stack_2¤
abbthhzbau/strided_sliceStridedSliceabbthhzbau/Shape:output:0'abbthhzbau/strided_slice/stack:output:0)abbthhzbau/strided_slice/stack_1:output:0)abbthhzbau/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
abbthhzbau/strided_slicez
abbthhzbau/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
abbthhzbau/Reshape/shape/1z
abbthhzbau/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
abbthhzbau/Reshape/shape/2×
abbthhzbau/Reshape/shapePack!abbthhzbau/strided_slice:output:0#abbthhzbau/Reshape/shape/1:output:0#abbthhzbau/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
abbthhzbau/Reshape/shape¾
abbthhzbau/ReshapeReshape0ehhqjcwuju/squeeze_batch_dims/Reshape_1:output:0!abbthhzbau/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
abbthhzbau/Reshapeo
cxhqebqpjz/ShapeShapeabbthhzbau/Reshape:output:0*
T0*
_output_shapes
:2
cxhqebqpjz/Shape
cxhqebqpjz/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
cxhqebqpjz/strided_slice/stack
 cxhqebqpjz/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 cxhqebqpjz/strided_slice/stack_1
 cxhqebqpjz/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 cxhqebqpjz/strided_slice/stack_2¤
cxhqebqpjz/strided_sliceStridedSlicecxhqebqpjz/Shape:output:0'cxhqebqpjz/strided_slice/stack:output:0)cxhqebqpjz/strided_slice/stack_1:output:0)cxhqebqpjz/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cxhqebqpjz/strided_slicer
cxhqebqpjz/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros/mul/y
cxhqebqpjz/zeros/mulMul!cxhqebqpjz/strided_slice:output:0cxhqebqpjz/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros/mulu
cxhqebqpjz/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
cxhqebqpjz/zeros/Less/y
cxhqebqpjz/zeros/LessLesscxhqebqpjz/zeros/mul:z:0 cxhqebqpjz/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros/Lessx
cxhqebqpjz/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros/packed/1¯
cxhqebqpjz/zeros/packedPack!cxhqebqpjz/strided_slice:output:0"cxhqebqpjz/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
cxhqebqpjz/zeros/packedu
cxhqebqpjz/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
cxhqebqpjz/zeros/Const¡
cxhqebqpjz/zerosFill cxhqebqpjz/zeros/packed:output:0cxhqebqpjz/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/zerosv
cxhqebqpjz/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros_1/mul/y
cxhqebqpjz/zeros_1/mulMul!cxhqebqpjz/strided_slice:output:0!cxhqebqpjz/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros_1/muly
cxhqebqpjz/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
cxhqebqpjz/zeros_1/Less/y
cxhqebqpjz/zeros_1/LessLesscxhqebqpjz/zeros_1/mul:z:0"cxhqebqpjz/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros_1/Less|
cxhqebqpjz/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros_1/packed/1µ
cxhqebqpjz/zeros_1/packedPack!cxhqebqpjz/strided_slice:output:0$cxhqebqpjz/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
cxhqebqpjz/zeros_1/packedy
cxhqebqpjz/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
cxhqebqpjz/zeros_1/Const©
cxhqebqpjz/zeros_1Fill"cxhqebqpjz/zeros_1/packed:output:0!cxhqebqpjz/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/zeros_1
cxhqebqpjz/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cxhqebqpjz/transpose/perm°
cxhqebqpjz/transpose	Transposeabbthhzbau/Reshape:output:0"cxhqebqpjz/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/transposep
cxhqebqpjz/Shape_1Shapecxhqebqpjz/transpose:y:0*
T0*
_output_shapes
:2
cxhqebqpjz/Shape_1
 cxhqebqpjz/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 cxhqebqpjz/strided_slice_1/stack
"cxhqebqpjz/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_1/stack_1
"cxhqebqpjz/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_1/stack_2°
cxhqebqpjz/strided_slice_1StridedSlicecxhqebqpjz/Shape_1:output:0)cxhqebqpjz/strided_slice_1/stack:output:0+cxhqebqpjz/strided_slice_1/stack_1:output:0+cxhqebqpjz/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cxhqebqpjz/strided_slice_1
&cxhqebqpjz/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&cxhqebqpjz/TensorArrayV2/element_shapeÞ
cxhqebqpjz/TensorArrayV2TensorListReserve/cxhqebqpjz/TensorArrayV2/element_shape:output:0#cxhqebqpjz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cxhqebqpjz/TensorArrayV2Õ
@cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2cxhqebqpjz/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcxhqebqpjz/transpose:y:0Icxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor
 cxhqebqpjz/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 cxhqebqpjz/strided_slice_2/stack
"cxhqebqpjz/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_2/stack_1
"cxhqebqpjz/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_2/stack_2¾
cxhqebqpjz/strided_slice_2StridedSlicecxhqebqpjz/transpose:y:0)cxhqebqpjz/strided_slice_2/stack:output:0+cxhqebqpjz/strided_slice_2/stack_1:output:0+cxhqebqpjz/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
cxhqebqpjz/strided_slice_2Ð
+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOpReadVariableOp4cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOpÓ
cxhqebqpjz/rvncypflgq/MatMulMatMul#cxhqebqpjz/strided_slice_2:output:03cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/rvncypflgq/MatMulÖ
-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp6cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOpÏ
cxhqebqpjz/rvncypflgq/MatMul_1MatMulcxhqebqpjz/zeros:output:05cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
cxhqebqpjz/rvncypflgq/MatMul_1Ä
cxhqebqpjz/rvncypflgq/addAddV2&cxhqebqpjz/rvncypflgq/MatMul:product:0(cxhqebqpjz/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/rvncypflgq/addÏ
,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp5cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOpÑ
cxhqebqpjz/rvncypflgq/BiasAddBiasAddcxhqebqpjz/rvncypflgq/add:z:04cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/rvncypflgq/BiasAdd
%cxhqebqpjz/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%cxhqebqpjz/rvncypflgq/split/split_dim
cxhqebqpjz/rvncypflgq/splitSplit.cxhqebqpjz/rvncypflgq/split/split_dim:output:0&cxhqebqpjz/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
cxhqebqpjz/rvncypflgq/split¶
$cxhqebqpjz/rvncypflgq/ReadVariableOpReadVariableOp-cxhqebqpjz_rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02&
$cxhqebqpjz/rvncypflgq/ReadVariableOpº
cxhqebqpjz/rvncypflgq/mulMul,cxhqebqpjz/rvncypflgq/ReadVariableOp:value:0cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mulº
cxhqebqpjz/rvncypflgq/add_1AddV2$cxhqebqpjz/rvncypflgq/split:output:0cxhqebqpjz/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_1
cxhqebqpjz/rvncypflgq/SigmoidSigmoidcxhqebqpjz/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/Sigmoid¼
&cxhqebqpjz/rvncypflgq/ReadVariableOp_1ReadVariableOp/cxhqebqpjz_rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&cxhqebqpjz/rvncypflgq/ReadVariableOp_1À
cxhqebqpjz/rvncypflgq/mul_1Mul.cxhqebqpjz/rvncypflgq/ReadVariableOp_1:value:0cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_1¼
cxhqebqpjz/rvncypflgq/add_2AddV2$cxhqebqpjz/rvncypflgq/split:output:1cxhqebqpjz/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_2 
cxhqebqpjz/rvncypflgq/Sigmoid_1Sigmoidcxhqebqpjz/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
cxhqebqpjz/rvncypflgq/Sigmoid_1µ
cxhqebqpjz/rvncypflgq/mul_2Mul#cxhqebqpjz/rvncypflgq/Sigmoid_1:y:0cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_2
cxhqebqpjz/rvncypflgq/TanhTanh$cxhqebqpjz/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/Tanh¶
cxhqebqpjz/rvncypflgq/mul_3Mul!cxhqebqpjz/rvncypflgq/Sigmoid:y:0cxhqebqpjz/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_3·
cxhqebqpjz/rvncypflgq/add_3AddV2cxhqebqpjz/rvncypflgq/mul_2:z:0cxhqebqpjz/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_3¼
&cxhqebqpjz/rvncypflgq/ReadVariableOp_2ReadVariableOp/cxhqebqpjz_rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&cxhqebqpjz/rvncypflgq/ReadVariableOp_2Ä
cxhqebqpjz/rvncypflgq/mul_4Mul.cxhqebqpjz/rvncypflgq/ReadVariableOp_2:value:0cxhqebqpjz/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_4¼
cxhqebqpjz/rvncypflgq/add_4AddV2$cxhqebqpjz/rvncypflgq/split:output:3cxhqebqpjz/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_4 
cxhqebqpjz/rvncypflgq/Sigmoid_2Sigmoidcxhqebqpjz/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
cxhqebqpjz/rvncypflgq/Sigmoid_2
cxhqebqpjz/rvncypflgq/Tanh_1Tanhcxhqebqpjz/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/Tanh_1º
cxhqebqpjz/rvncypflgq/mul_5Mul#cxhqebqpjz/rvncypflgq/Sigmoid_2:y:0 cxhqebqpjz/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_5¥
(cxhqebqpjz/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(cxhqebqpjz/TensorArrayV2_1/element_shapeä
cxhqebqpjz/TensorArrayV2_1TensorListReserve1cxhqebqpjz/TensorArrayV2_1/element_shape:output:0#cxhqebqpjz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cxhqebqpjz/TensorArrayV2_1d
cxhqebqpjz/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/time
#cxhqebqpjz/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#cxhqebqpjz/while/maximum_iterations
cxhqebqpjz/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/while/loop_counter²
cxhqebqpjz/whileWhile&cxhqebqpjz/while/loop_counter:output:0,cxhqebqpjz/while/maximum_iterations:output:0cxhqebqpjz/time:output:0#cxhqebqpjz/TensorArrayV2_1:handle:0cxhqebqpjz/zeros:output:0cxhqebqpjz/zeros_1:output:0#cxhqebqpjz/strided_slice_1:output:0Bcxhqebqpjz/TensorArrayUnstack/TensorListFromTensor:output_handle:04cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource6cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource5cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource-cxhqebqpjz_rvncypflgq_readvariableop_resource/cxhqebqpjz_rvncypflgq_readvariableop_1_resource/cxhqebqpjz_rvncypflgq_readvariableop_2_resource*
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
cxhqebqpjz_while_body_1719846*)
cond!R
cxhqebqpjz_while_cond_1719845*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
cxhqebqpjz/whileË
;cxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;cxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shape
-cxhqebqpjz/TensorArrayV2Stack/TensorListStackTensorListStackcxhqebqpjz/while:output:3Dcxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-cxhqebqpjz/TensorArrayV2Stack/TensorListStack
 cxhqebqpjz/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 cxhqebqpjz/strided_slice_3/stack
"cxhqebqpjz/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"cxhqebqpjz/strided_slice_3/stack_1
"cxhqebqpjz/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_3/stack_2Ü
cxhqebqpjz/strided_slice_3StridedSlice6cxhqebqpjz/TensorArrayV2Stack/TensorListStack:tensor:0)cxhqebqpjz/strided_slice_3/stack:output:0+cxhqebqpjz/strided_slice_3/stack_1:output:0+cxhqebqpjz/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
cxhqebqpjz/strided_slice_3
cxhqebqpjz/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cxhqebqpjz/transpose_1/permÑ
cxhqebqpjz/transpose_1	Transpose6cxhqebqpjz/TensorArrayV2Stack/TensorListStack:tensor:0$cxhqebqpjz/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/transpose_1n
qzuziqqdld/ShapeShapecxhqebqpjz/transpose_1:y:0*
T0*
_output_shapes
:2
qzuziqqdld/Shape
qzuziqqdld/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
qzuziqqdld/strided_slice/stack
 qzuziqqdld/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 qzuziqqdld/strided_slice/stack_1
 qzuziqqdld/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 qzuziqqdld/strided_slice/stack_2¤
qzuziqqdld/strided_sliceStridedSliceqzuziqqdld/Shape:output:0'qzuziqqdld/strided_slice/stack:output:0)qzuziqqdld/strided_slice/stack_1:output:0)qzuziqqdld/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
qzuziqqdld/strided_slicer
qzuziqqdld/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros/mul/y
qzuziqqdld/zeros/mulMul!qzuziqqdld/strided_slice:output:0qzuziqqdld/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros/mulu
qzuziqqdld/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
qzuziqqdld/zeros/Less/y
qzuziqqdld/zeros/LessLessqzuziqqdld/zeros/mul:z:0 qzuziqqdld/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros/Lessx
qzuziqqdld/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros/packed/1¯
qzuziqqdld/zeros/packedPack!qzuziqqdld/strided_slice:output:0"qzuziqqdld/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
qzuziqqdld/zeros/packedu
qzuziqqdld/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
qzuziqqdld/zeros/Const¡
qzuziqqdld/zerosFill qzuziqqdld/zeros/packed:output:0qzuziqqdld/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/zerosv
qzuziqqdld/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros_1/mul/y
qzuziqqdld/zeros_1/mulMul!qzuziqqdld/strided_slice:output:0!qzuziqqdld/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros_1/muly
qzuziqqdld/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
qzuziqqdld/zeros_1/Less/y
qzuziqqdld/zeros_1/LessLessqzuziqqdld/zeros_1/mul:z:0"qzuziqqdld/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros_1/Less|
qzuziqqdld/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros_1/packed/1µ
qzuziqqdld/zeros_1/packedPack!qzuziqqdld/strided_slice:output:0$qzuziqqdld/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
qzuziqqdld/zeros_1/packedy
qzuziqqdld/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
qzuziqqdld/zeros_1/Const©
qzuziqqdld/zeros_1Fill"qzuziqqdld/zeros_1/packed:output:0!qzuziqqdld/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/zeros_1
qzuziqqdld/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
qzuziqqdld/transpose/perm¯
qzuziqqdld/transpose	Transposecxhqebqpjz/transpose_1:y:0"qzuziqqdld/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/transposep
qzuziqqdld/Shape_1Shapeqzuziqqdld/transpose:y:0*
T0*
_output_shapes
:2
qzuziqqdld/Shape_1
 qzuziqqdld/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 qzuziqqdld/strided_slice_1/stack
"qzuziqqdld/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_1/stack_1
"qzuziqqdld/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_1/stack_2°
qzuziqqdld/strided_slice_1StridedSliceqzuziqqdld/Shape_1:output:0)qzuziqqdld/strided_slice_1/stack:output:0+qzuziqqdld/strided_slice_1/stack_1:output:0+qzuziqqdld/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
qzuziqqdld/strided_slice_1
&qzuziqqdld/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&qzuziqqdld/TensorArrayV2/element_shapeÞ
qzuziqqdld/TensorArrayV2TensorListReserve/qzuziqqdld/TensorArrayV2/element_shape:output:0#qzuziqqdld/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
qzuziqqdld/TensorArrayV2Õ
@qzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@qzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2qzuziqqdld/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqzuziqqdld/transpose:y:0Iqzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2qzuziqqdld/TensorArrayUnstack/TensorListFromTensor
 qzuziqqdld/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 qzuziqqdld/strided_slice_2/stack
"qzuziqqdld/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_2/stack_1
"qzuziqqdld/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_2/stack_2¾
qzuziqqdld/strided_slice_2StridedSliceqzuziqqdld/transpose:y:0)qzuziqqdld/strided_slice_2/stack:output:0+qzuziqqdld/strided_slice_2/stack_1:output:0+qzuziqqdld/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
qzuziqqdld/strided_slice_2Ð
+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp4qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOpÓ
qzuziqqdld/aiccbsgdoo/MatMulMatMul#qzuziqqdld/strided_slice_2:output:03qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qzuziqqdld/aiccbsgdoo/MatMulÖ
-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp6qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOpÏ
qzuziqqdld/aiccbsgdoo/MatMul_1MatMulqzuziqqdld/zeros:output:05qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
qzuziqqdld/aiccbsgdoo/MatMul_1Ä
qzuziqqdld/aiccbsgdoo/addAddV2&qzuziqqdld/aiccbsgdoo/MatMul:product:0(qzuziqqdld/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qzuziqqdld/aiccbsgdoo/addÏ
,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp5qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOpÑ
qzuziqqdld/aiccbsgdoo/BiasAddBiasAddqzuziqqdld/aiccbsgdoo/add:z:04qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qzuziqqdld/aiccbsgdoo/BiasAdd
%qzuziqqdld/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%qzuziqqdld/aiccbsgdoo/split/split_dim
qzuziqqdld/aiccbsgdoo/splitSplit.qzuziqqdld/aiccbsgdoo/split/split_dim:output:0&qzuziqqdld/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qzuziqqdld/aiccbsgdoo/split¶
$qzuziqqdld/aiccbsgdoo/ReadVariableOpReadVariableOp-qzuziqqdld_aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02&
$qzuziqqdld/aiccbsgdoo/ReadVariableOpº
qzuziqqdld/aiccbsgdoo/mulMul,qzuziqqdld/aiccbsgdoo/ReadVariableOp:value:0qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mulº
qzuziqqdld/aiccbsgdoo/add_1AddV2$qzuziqqdld/aiccbsgdoo/split:output:0qzuziqqdld/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_1
qzuziqqdld/aiccbsgdoo/SigmoidSigmoidqzuziqqdld/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/Sigmoid¼
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1ReadVariableOp/qzuziqqdld_aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1À
qzuziqqdld/aiccbsgdoo/mul_1Mul.qzuziqqdld/aiccbsgdoo/ReadVariableOp_1:value:0qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_1¼
qzuziqqdld/aiccbsgdoo/add_2AddV2$qzuziqqdld/aiccbsgdoo/split:output:1qzuziqqdld/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_2 
qzuziqqdld/aiccbsgdoo/Sigmoid_1Sigmoidqzuziqqdld/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
qzuziqqdld/aiccbsgdoo/Sigmoid_1µ
qzuziqqdld/aiccbsgdoo/mul_2Mul#qzuziqqdld/aiccbsgdoo/Sigmoid_1:y:0qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_2
qzuziqqdld/aiccbsgdoo/TanhTanh$qzuziqqdld/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/Tanh¶
qzuziqqdld/aiccbsgdoo/mul_3Mul!qzuziqqdld/aiccbsgdoo/Sigmoid:y:0qzuziqqdld/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_3·
qzuziqqdld/aiccbsgdoo/add_3AddV2qzuziqqdld/aiccbsgdoo/mul_2:z:0qzuziqqdld/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_3¼
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2ReadVariableOp/qzuziqqdld_aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2Ä
qzuziqqdld/aiccbsgdoo/mul_4Mul.qzuziqqdld/aiccbsgdoo/ReadVariableOp_2:value:0qzuziqqdld/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_4¼
qzuziqqdld/aiccbsgdoo/add_4AddV2$qzuziqqdld/aiccbsgdoo/split:output:3qzuziqqdld/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_4 
qzuziqqdld/aiccbsgdoo/Sigmoid_2Sigmoidqzuziqqdld/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
qzuziqqdld/aiccbsgdoo/Sigmoid_2
qzuziqqdld/aiccbsgdoo/Tanh_1Tanhqzuziqqdld/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/Tanh_1º
qzuziqqdld/aiccbsgdoo/mul_5Mul#qzuziqqdld/aiccbsgdoo/Sigmoid_2:y:0 qzuziqqdld/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_5¥
(qzuziqqdld/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(qzuziqqdld/TensorArrayV2_1/element_shapeä
qzuziqqdld/TensorArrayV2_1TensorListReserve1qzuziqqdld/TensorArrayV2_1/element_shape:output:0#qzuziqqdld/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
qzuziqqdld/TensorArrayV2_1d
qzuziqqdld/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/time
#qzuziqqdld/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#qzuziqqdld/while/maximum_iterations
qzuziqqdld/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/while/loop_counter²
qzuziqqdld/whileWhile&qzuziqqdld/while/loop_counter:output:0,qzuziqqdld/while/maximum_iterations:output:0qzuziqqdld/time:output:0#qzuziqqdld/TensorArrayV2_1:handle:0qzuziqqdld/zeros:output:0qzuziqqdld/zeros_1:output:0#qzuziqqdld/strided_slice_1:output:0Bqzuziqqdld/TensorArrayUnstack/TensorListFromTensor:output_handle:04qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource6qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource5qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource-qzuziqqdld_aiccbsgdoo_readvariableop_resource/qzuziqqdld_aiccbsgdoo_readvariableop_1_resource/qzuziqqdld_aiccbsgdoo_readvariableop_2_resource*
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
qzuziqqdld_while_body_1720022*)
cond!R
qzuziqqdld_while_cond_1720021*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
qzuziqqdld/whileË
;qzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;qzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shape
-qzuziqqdld/TensorArrayV2Stack/TensorListStackTensorListStackqzuziqqdld/while:output:3Dqzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-qzuziqqdld/TensorArrayV2Stack/TensorListStack
 qzuziqqdld/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 qzuziqqdld/strided_slice_3/stack
"qzuziqqdld/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"qzuziqqdld/strided_slice_3/stack_1
"qzuziqqdld/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_3/stack_2Ü
qzuziqqdld/strided_slice_3StridedSlice6qzuziqqdld/TensorArrayV2Stack/TensorListStack:tensor:0)qzuziqqdld/strided_slice_3/stack:output:0+qzuziqqdld/strided_slice_3/stack_1:output:0+qzuziqqdld/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
qzuziqqdld/strided_slice_3
qzuziqqdld/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
qzuziqqdld/transpose_1/permÑ
qzuziqqdld/transpose_1	Transpose6qzuziqqdld/TensorArrayV2Stack/TensorListStack:tensor:0$qzuziqqdld/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/transpose_1®
 pbmomrqadp/MatMul/ReadVariableOpReadVariableOp)pbmomrqadp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 pbmomrqadp/MatMul/ReadVariableOp±
pbmomrqadp/MatMulMatMul#qzuziqqdld/strided_slice_3:output:0(pbmomrqadp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pbmomrqadp/MatMul­
!pbmomrqadp/BiasAdd/ReadVariableOpReadVariableOp*pbmomrqadp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!pbmomrqadp/BiasAdd/ReadVariableOp­
pbmomrqadp/BiasAddBiasAddpbmomrqadp/MatMul:product:0)pbmomrqadp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pbmomrqadp/BiasAdd¶
IdentityIdentitypbmomrqadp/BiasAdd:output:0-^cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp,^cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp.^cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp%^cxhqebqpjz/rvncypflgq/ReadVariableOp'^cxhqebqpjz/rvncypflgq/ReadVariableOp_1'^cxhqebqpjz/rvncypflgq/ReadVariableOp_2^cxhqebqpjz/while.^ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp5^ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp.^mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp5^mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp"^pbmomrqadp/BiasAdd/ReadVariableOp!^pbmomrqadp/MatMul/ReadVariableOp-^qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp,^qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp.^qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp%^qzuziqqdld/aiccbsgdoo/ReadVariableOp'^qzuziqqdld/aiccbsgdoo/ReadVariableOp_1'^qzuziqqdld/aiccbsgdoo/ReadVariableOp_2^qzuziqqdld/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2\
,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp2Z
+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp2^
-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp2L
$cxhqebqpjz/rvncypflgq/ReadVariableOp$cxhqebqpjz/rvncypflgq/ReadVariableOp2P
&cxhqebqpjz/rvncypflgq/ReadVariableOp_1&cxhqebqpjz/rvncypflgq/ReadVariableOp_12P
&cxhqebqpjz/rvncypflgq/ReadVariableOp_2&cxhqebqpjz/rvncypflgq/ReadVariableOp_22$
cxhqebqpjz/whilecxhqebqpjz/while2^
-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp2l
4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp2^
-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp2l
4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!pbmomrqadp/BiasAdd/ReadVariableOp!pbmomrqadp/BiasAdd/ReadVariableOp2D
 pbmomrqadp/MatMul/ReadVariableOp pbmomrqadp/MatMul/ReadVariableOp2\
,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp2Z
+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp2^
-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp2L
$qzuziqqdld/aiccbsgdoo/ReadVariableOp$qzuziqqdld/aiccbsgdoo/ReadVariableOp2P
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1&qzuziqqdld/aiccbsgdoo/ReadVariableOp_12P
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2&qzuziqqdld/aiccbsgdoo/ReadVariableOp_22$
qzuziqqdld/whileqzuziqqdld/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡h

G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721302

inputs<
)rvncypflgq_matmul_readvariableop_resource:	>
+rvncypflgq_matmul_1_readvariableop_resource:	 9
*rvncypflgq_biasadd_readvariableop_resource:	0
"rvncypflgq_readvariableop_resource: 2
$rvncypflgq_readvariableop_1_resource: 2
$rvncypflgq_readvariableop_2_resource: 
identity¢!rvncypflgq/BiasAdd/ReadVariableOp¢ rvncypflgq/MatMul/ReadVariableOp¢"rvncypflgq/MatMul_1/ReadVariableOp¢rvncypflgq/ReadVariableOp¢rvncypflgq/ReadVariableOp_1¢rvncypflgq/ReadVariableOp_2¢whileD
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
 rvncypflgq/MatMul/ReadVariableOpReadVariableOp)rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 rvncypflgq/MatMul/ReadVariableOp§
rvncypflgq/MatMulMatMulstrided_slice_2:output:0(rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMulµ
"rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp+rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"rvncypflgq/MatMul_1/ReadVariableOp£
rvncypflgq/MatMul_1MatMulzeros:output:0*rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMul_1
rvncypflgq/addAddV2rvncypflgq/MatMul:product:0rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/add®
!rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp*rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!rvncypflgq/BiasAdd/ReadVariableOp¥
rvncypflgq/BiasAddBiasAddrvncypflgq/add:z:0)rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/BiasAddz
rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
rvncypflgq/split/split_dimë
rvncypflgq/splitSplit#rvncypflgq/split/split_dim:output:0rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rvncypflgq/split
rvncypflgq/ReadVariableOpReadVariableOp"rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp
rvncypflgq/mulMul!rvncypflgq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul
rvncypflgq/add_1AddV2rvncypflgq/split:output:0rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_1{
rvncypflgq/SigmoidSigmoidrvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid
rvncypflgq/ReadVariableOp_1ReadVariableOp$rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_1
rvncypflgq/mul_1Mul#rvncypflgq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_1
rvncypflgq/add_2AddV2rvncypflgq/split:output:1rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_2
rvncypflgq/Sigmoid_1Sigmoidrvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_1
rvncypflgq/mul_2Mulrvncypflgq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_2w
rvncypflgq/TanhTanhrvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh
rvncypflgq/mul_3Mulrvncypflgq/Sigmoid:y:0rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_3
rvncypflgq/add_3AddV2rvncypflgq/mul_2:z:0rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_3
rvncypflgq/ReadVariableOp_2ReadVariableOp$rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_2
rvncypflgq/mul_4Mul#rvncypflgq/ReadVariableOp_2:value:0rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_4
rvncypflgq/add_4AddV2rvncypflgq/split:output:3rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_4
rvncypflgq/Sigmoid_2Sigmoidrvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_2v
rvncypflgq/Tanh_1Tanhrvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh_1
rvncypflgq/mul_5Mulrvncypflgq/Sigmoid_2:y:0rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)rvncypflgq_matmul_readvariableop_resource+rvncypflgq_matmul_1_readvariableop_resource*rvncypflgq_biasadd_readvariableop_resource"rvncypflgq_readvariableop_resource$rvncypflgq_readvariableop_1_resource$rvncypflgq_readvariableop_2_resource*
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
while_body_1721201*
condR
while_cond_1721200*Q
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
IdentityIdentitytranspose_1:y:0"^rvncypflgq/BiasAdd/ReadVariableOp!^rvncypflgq/MatMul/ReadVariableOp#^rvncypflgq/MatMul_1/ReadVariableOp^rvncypflgq/ReadVariableOp^rvncypflgq/ReadVariableOp_1^rvncypflgq/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!rvncypflgq/BiasAdd/ReadVariableOp!rvncypflgq/BiasAdd/ReadVariableOp2D
 rvncypflgq/MatMul/ReadVariableOp rvncypflgq/MatMul/ReadVariableOp2H
"rvncypflgq/MatMul_1/ReadVariableOp"rvncypflgq/MatMul_1/ReadVariableOp26
rvncypflgq/ReadVariableOprvncypflgq/ReadVariableOp2:
rvncypflgq/ReadVariableOp_1rvncypflgq/ReadVariableOp_12:
rvncypflgq/ReadVariableOp_2rvncypflgq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³F
ê
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1717298

inputs%
rvncypflgq_1717199:	%
rvncypflgq_1717201:	 !
rvncypflgq_1717203:	 
rvncypflgq_1717205:  
rvncypflgq_1717207:  
rvncypflgq_1717209: 
identity¢"rvncypflgq/StatefulPartitionedCall¢whileD
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
"rvncypflgq/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0rvncypflgq_1717199rvncypflgq_1717201rvncypflgq_1717203rvncypflgq_1717205rvncypflgq_1717207rvncypflgq_1717209*
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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_17171222$
"rvncypflgq/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0rvncypflgq_1717199rvncypflgq_1717201rvncypflgq_1717203rvncypflgq_1717205rvncypflgq_1717207rvncypflgq_1717209*
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
while_body_1717218*
condR
while_cond_1717217*Q
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
IdentityIdentitytranspose_1:y:0#^rvncypflgq/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"rvncypflgq/StatefulPartitionedCall"rvncypflgq/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


cxhqebqpjz_while_cond_17202842
.cxhqebqpjz_while_cxhqebqpjz_while_loop_counter8
4cxhqebqpjz_while_cxhqebqpjz_while_maximum_iterations 
cxhqebqpjz_while_placeholder"
cxhqebqpjz_while_placeholder_1"
cxhqebqpjz_while_placeholder_2"
cxhqebqpjz_while_placeholder_34
0cxhqebqpjz_while_less_cxhqebqpjz_strided_slice_1K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1720284___redundant_placeholder0K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1720284___redundant_placeholder1K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1720284___redundant_placeholder2K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1720284___redundant_placeholder3K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1720284___redundant_placeholder4K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1720284___redundant_placeholder5K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1720284___redundant_placeholder6
cxhqebqpjz_while_identity
§
cxhqebqpjz/while/LessLesscxhqebqpjz_while_placeholder0cxhqebqpjz_while_less_cxhqebqpjz_strided_slice_1*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Less~
cxhqebqpjz/while/IdentityIdentitycxhqebqpjz/while/Less:z:0*
T0
*
_output_shapes
: 2
cxhqebqpjz/while/Identity"?
cxhqebqpjz_while_identity"cxhqebqpjz/while/Identity:output:0*(
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
while_body_1719024
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aiccbsgdoo_matmul_readvariableop_resource_0:	 F
3while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 A
2while_aiccbsgdoo_biasadd_readvariableop_resource_0:	8
*while_aiccbsgdoo_readvariableop_resource_0: :
,while_aiccbsgdoo_readvariableop_1_resource_0: :
,while_aiccbsgdoo_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aiccbsgdoo_matmul_readvariableop_resource:	 D
1while_aiccbsgdoo_matmul_1_readvariableop_resource:	 ?
0while_aiccbsgdoo_biasadd_readvariableop_resource:	6
(while_aiccbsgdoo_readvariableop_resource: 8
*while_aiccbsgdoo_readvariableop_1_resource: 8
*while_aiccbsgdoo_readvariableop_2_resource: ¢'while/aiccbsgdoo/BiasAdd/ReadVariableOp¢&while/aiccbsgdoo/MatMul/ReadVariableOp¢(while/aiccbsgdoo/MatMul_1/ReadVariableOp¢while/aiccbsgdoo/ReadVariableOp¢!while/aiccbsgdoo/ReadVariableOp_1¢!while/aiccbsgdoo/ReadVariableOp_2Ã
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
&while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp1while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aiccbsgdoo/MatMul/ReadVariableOpÑ
while/aiccbsgdoo/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMulÉ
(while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp3while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aiccbsgdoo/MatMul_1/ReadVariableOpº
while/aiccbsgdoo/MatMul_1MatMulwhile_placeholder_20while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMul_1°
while/aiccbsgdoo/addAddV2!while/aiccbsgdoo/MatMul:product:0#while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/addÂ
'while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp2while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aiccbsgdoo/BiasAdd/ReadVariableOp½
while/aiccbsgdoo/BiasAddBiasAddwhile/aiccbsgdoo/add:z:0/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/BiasAdd
 while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aiccbsgdoo/split/split_dim
while/aiccbsgdoo/splitSplit)while/aiccbsgdoo/split/split_dim:output:0!while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aiccbsgdoo/split©
while/aiccbsgdoo/ReadVariableOpReadVariableOp*while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aiccbsgdoo/ReadVariableOp£
while/aiccbsgdoo/mulMul'while/aiccbsgdoo/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul¦
while/aiccbsgdoo/add_1AddV2while/aiccbsgdoo/split:output:0while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_1
while/aiccbsgdoo/SigmoidSigmoidwhile/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid¯
!while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp,while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_1©
while/aiccbsgdoo/mul_1Mul)while/aiccbsgdoo/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_1¨
while/aiccbsgdoo/add_2AddV2while/aiccbsgdoo/split:output:1while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_2
while/aiccbsgdoo/Sigmoid_1Sigmoidwhile/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_1
while/aiccbsgdoo/mul_2Mulwhile/aiccbsgdoo/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_2
while/aiccbsgdoo/TanhTanhwhile/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh¢
while/aiccbsgdoo/mul_3Mulwhile/aiccbsgdoo/Sigmoid:y:0while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_3£
while/aiccbsgdoo/add_3AddV2while/aiccbsgdoo/mul_2:z:0while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_3¯
!while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp,while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_2°
while/aiccbsgdoo/mul_4Mul)while/aiccbsgdoo/ReadVariableOp_2:value:0while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_4¨
while/aiccbsgdoo/add_4AddV2while/aiccbsgdoo/split:output:3while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_4
while/aiccbsgdoo/Sigmoid_2Sigmoidwhile/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_2
while/aiccbsgdoo/Tanh_1Tanhwhile/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh_1¦
while/aiccbsgdoo/mul_5Mulwhile/aiccbsgdoo/Sigmoid_2:y:0while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aiccbsgdoo/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aiccbsgdoo/mul_5:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aiccbsgdoo/add_3:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aiccbsgdoo_biasadd_readvariableop_resource2while_aiccbsgdoo_biasadd_readvariableop_resource_0"h
1while_aiccbsgdoo_matmul_1_readvariableop_resource3while_aiccbsgdoo_matmul_1_readvariableop_resource_0"d
/while_aiccbsgdoo_matmul_readvariableop_resource1while_aiccbsgdoo_matmul_readvariableop_resource_0"Z
*while_aiccbsgdoo_readvariableop_1_resource,while_aiccbsgdoo_readvariableop_1_resource_0"Z
*while_aiccbsgdoo_readvariableop_2_resource,while_aiccbsgdoo_readvariableop_2_resource_0"V
(while_aiccbsgdoo_readvariableop_resource*while_aiccbsgdoo_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/aiccbsgdoo/BiasAdd/ReadVariableOp'while/aiccbsgdoo/BiasAdd/ReadVariableOp2P
&while/aiccbsgdoo/MatMul/ReadVariableOp&while/aiccbsgdoo/MatMul/ReadVariableOp2T
(while/aiccbsgdoo/MatMul_1/ReadVariableOp(while/aiccbsgdoo/MatMul_1/ReadVariableOp2B
while/aiccbsgdoo/ReadVariableOpwhile/aiccbsgdoo/ReadVariableOp2F
!while/aiccbsgdoo/ReadVariableOp_1!while/aiccbsgdoo/ReadVariableOp_12F
!while/aiccbsgdoo/ReadVariableOp_2!while/aiccbsgdoo/ReadVariableOp_2: 
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
while_cond_1721628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1721628___redundant_placeholder05
1while_while_cond_1721628___redundant_placeholder15
1while_while_cond_1721628___redundant_placeholder25
1while_while_cond_1721628___redundant_placeholder35
1while_while_cond_1721628___redundant_placeholder45
1while_while_cond_1721628___redundant_placeholder55
1while_while_cond_1721628___redundant_placeholder6
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
,__inference_cxhqebqpjz_layer_call_fn_1721516
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17172982
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


í
while_cond_1718551
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1718551___redundant_placeholder05
1while_while_cond_1718551___redundant_placeholder15
1while_while_cond_1718551___redundant_placeholder25
1while_while_cond_1718551___redundant_placeholder35
1while_while_cond_1718551___redundant_placeholder45
1while_while_cond_1718551___redundant_placeholder55
1while_while_cond_1718551___redundant_placeholder6
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
,__inference_cxhqebqpjz_layer_call_fn_1721499
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17170352
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


í
while_cond_1717975
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1717975___redundant_placeholder05
1while_while_cond_1717975___redundant_placeholder15
1while_while_cond_1717975___redundant_placeholder25
1while_while_cond_1717975___redundant_placeholder35
1while_while_cond_1717975___redundant_placeholder45
1while_while_cond_1717975___redundant_placeholder55
1while_while_cond_1717975___redundant_placeholder6
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
while_body_1722169
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aiccbsgdoo_matmul_readvariableop_resource_0:	 F
3while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 A
2while_aiccbsgdoo_biasadd_readvariableop_resource_0:	8
*while_aiccbsgdoo_readvariableop_resource_0: :
,while_aiccbsgdoo_readvariableop_1_resource_0: :
,while_aiccbsgdoo_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aiccbsgdoo_matmul_readvariableop_resource:	 D
1while_aiccbsgdoo_matmul_1_readvariableop_resource:	 ?
0while_aiccbsgdoo_biasadd_readvariableop_resource:	6
(while_aiccbsgdoo_readvariableop_resource: 8
*while_aiccbsgdoo_readvariableop_1_resource: 8
*while_aiccbsgdoo_readvariableop_2_resource: ¢'while/aiccbsgdoo/BiasAdd/ReadVariableOp¢&while/aiccbsgdoo/MatMul/ReadVariableOp¢(while/aiccbsgdoo/MatMul_1/ReadVariableOp¢while/aiccbsgdoo/ReadVariableOp¢!while/aiccbsgdoo/ReadVariableOp_1¢!while/aiccbsgdoo/ReadVariableOp_2Ã
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
&while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp1while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aiccbsgdoo/MatMul/ReadVariableOpÑ
while/aiccbsgdoo/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMulÉ
(while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp3while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aiccbsgdoo/MatMul_1/ReadVariableOpº
while/aiccbsgdoo/MatMul_1MatMulwhile_placeholder_20while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMul_1°
while/aiccbsgdoo/addAddV2!while/aiccbsgdoo/MatMul:product:0#while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/addÂ
'while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp2while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aiccbsgdoo/BiasAdd/ReadVariableOp½
while/aiccbsgdoo/BiasAddBiasAddwhile/aiccbsgdoo/add:z:0/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/BiasAdd
 while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aiccbsgdoo/split/split_dim
while/aiccbsgdoo/splitSplit)while/aiccbsgdoo/split/split_dim:output:0!while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aiccbsgdoo/split©
while/aiccbsgdoo/ReadVariableOpReadVariableOp*while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aiccbsgdoo/ReadVariableOp£
while/aiccbsgdoo/mulMul'while/aiccbsgdoo/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul¦
while/aiccbsgdoo/add_1AddV2while/aiccbsgdoo/split:output:0while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_1
while/aiccbsgdoo/SigmoidSigmoidwhile/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid¯
!while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp,while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_1©
while/aiccbsgdoo/mul_1Mul)while/aiccbsgdoo/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_1¨
while/aiccbsgdoo/add_2AddV2while/aiccbsgdoo/split:output:1while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_2
while/aiccbsgdoo/Sigmoid_1Sigmoidwhile/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_1
while/aiccbsgdoo/mul_2Mulwhile/aiccbsgdoo/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_2
while/aiccbsgdoo/TanhTanhwhile/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh¢
while/aiccbsgdoo/mul_3Mulwhile/aiccbsgdoo/Sigmoid:y:0while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_3£
while/aiccbsgdoo/add_3AddV2while/aiccbsgdoo/mul_2:z:0while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_3¯
!while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp,while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_2°
while/aiccbsgdoo/mul_4Mul)while/aiccbsgdoo/ReadVariableOp_2:value:0while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_4¨
while/aiccbsgdoo/add_4AddV2while/aiccbsgdoo/split:output:3while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_4
while/aiccbsgdoo/Sigmoid_2Sigmoidwhile/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_2
while/aiccbsgdoo/Tanh_1Tanhwhile/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh_1¦
while/aiccbsgdoo/mul_5Mulwhile/aiccbsgdoo/Sigmoid_2:y:0while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aiccbsgdoo/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aiccbsgdoo/mul_5:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aiccbsgdoo/add_3:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aiccbsgdoo_biasadd_readvariableop_resource2while_aiccbsgdoo_biasadd_readvariableop_resource_0"h
1while_aiccbsgdoo_matmul_1_readvariableop_resource3while_aiccbsgdoo_matmul_1_readvariableop_resource_0"d
/while_aiccbsgdoo_matmul_readvariableop_resource1while_aiccbsgdoo_matmul_readvariableop_resource_0"Z
*while_aiccbsgdoo_readvariableop_1_resource,while_aiccbsgdoo_readvariableop_1_resource_0"Z
*while_aiccbsgdoo_readvariableop_2_resource,while_aiccbsgdoo_readvariableop_2_resource_0"V
(while_aiccbsgdoo_readvariableop_resource*while_aiccbsgdoo_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/aiccbsgdoo/BiasAdd/ReadVariableOp'while/aiccbsgdoo/BiasAdd/ReadVariableOp2P
&while/aiccbsgdoo/MatMul/ReadVariableOp&while/aiccbsgdoo/MatMul/ReadVariableOp2T
(while/aiccbsgdoo/MatMul_1/ReadVariableOp(while/aiccbsgdoo/MatMul_1/ReadVariableOp2B
while/aiccbsgdoo/ReadVariableOpwhile/aiccbsgdoo/ReadVariableOp2F
!while/aiccbsgdoo/ReadVariableOp_1!while/aiccbsgdoo/ReadVariableOp_12F
!while/aiccbsgdoo/ReadVariableOp_2!while/aiccbsgdoo/ReadVariableOp_2: 
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
±'
³
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1717122

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
¤

,__inference_pbmomrqadp_layer_call_fn_1722357

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
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_17188702
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
¦h

G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1719125

inputs<
)aiccbsgdoo_matmul_readvariableop_resource:	 >
+aiccbsgdoo_matmul_1_readvariableop_resource:	 9
*aiccbsgdoo_biasadd_readvariableop_resource:	0
"aiccbsgdoo_readvariableop_resource: 2
$aiccbsgdoo_readvariableop_1_resource: 2
$aiccbsgdoo_readvariableop_2_resource: 
identity¢!aiccbsgdoo/BiasAdd/ReadVariableOp¢ aiccbsgdoo/MatMul/ReadVariableOp¢"aiccbsgdoo/MatMul_1/ReadVariableOp¢aiccbsgdoo/ReadVariableOp¢aiccbsgdoo/ReadVariableOp_1¢aiccbsgdoo/ReadVariableOp_2¢whileD
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
 aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp)aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aiccbsgdoo/MatMul/ReadVariableOp§
aiccbsgdoo/MatMulMatMulstrided_slice_2:output:0(aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMulµ
"aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp+aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aiccbsgdoo/MatMul_1/ReadVariableOp£
aiccbsgdoo/MatMul_1MatMulzeros:output:0*aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMul_1
aiccbsgdoo/addAddV2aiccbsgdoo/MatMul:product:0aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/add®
!aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp*aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aiccbsgdoo/BiasAdd/ReadVariableOp¥
aiccbsgdoo/BiasAddBiasAddaiccbsgdoo/add:z:0)aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/BiasAddz
aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aiccbsgdoo/split/split_dimë
aiccbsgdoo/splitSplit#aiccbsgdoo/split/split_dim:output:0aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aiccbsgdoo/split
aiccbsgdoo/ReadVariableOpReadVariableOp"aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp
aiccbsgdoo/mulMul!aiccbsgdoo/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul
aiccbsgdoo/add_1AddV2aiccbsgdoo/split:output:0aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_1{
aiccbsgdoo/SigmoidSigmoidaiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid
aiccbsgdoo/ReadVariableOp_1ReadVariableOp$aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_1
aiccbsgdoo/mul_1Mul#aiccbsgdoo/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_1
aiccbsgdoo/add_2AddV2aiccbsgdoo/split:output:1aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_2
aiccbsgdoo/Sigmoid_1Sigmoidaiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_1
aiccbsgdoo/mul_2Mulaiccbsgdoo/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_2w
aiccbsgdoo/TanhTanhaiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh
aiccbsgdoo/mul_3Mulaiccbsgdoo/Sigmoid:y:0aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_3
aiccbsgdoo/add_3AddV2aiccbsgdoo/mul_2:z:0aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_3
aiccbsgdoo/ReadVariableOp_2ReadVariableOp$aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_2
aiccbsgdoo/mul_4Mul#aiccbsgdoo/ReadVariableOp_2:value:0aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_4
aiccbsgdoo/add_4AddV2aiccbsgdoo/split:output:3aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_4
aiccbsgdoo/Sigmoid_2Sigmoidaiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_2v
aiccbsgdoo/Tanh_1Tanhaiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh_1
aiccbsgdoo/mul_5Mulaiccbsgdoo/Sigmoid_2:y:0aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aiccbsgdoo_matmul_readvariableop_resource+aiccbsgdoo_matmul_1_readvariableop_resource*aiccbsgdoo_biasadd_readvariableop_resource"aiccbsgdoo_readvariableop_resource$aiccbsgdoo_readvariableop_1_resource$aiccbsgdoo_readvariableop_2_resource*
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
while_body_1719024*
condR
while_cond_1719023*Q
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
IdentityIdentitystrided_slice_3:output:0"^aiccbsgdoo/BiasAdd/ReadVariableOp!^aiccbsgdoo/MatMul/ReadVariableOp#^aiccbsgdoo/MatMul_1/ReadVariableOp^aiccbsgdoo/ReadVariableOp^aiccbsgdoo/ReadVariableOp_1^aiccbsgdoo/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aiccbsgdoo/BiasAdd/ReadVariableOp!aiccbsgdoo/BiasAdd/ReadVariableOp2D
 aiccbsgdoo/MatMul/ReadVariableOp aiccbsgdoo/MatMul/ReadVariableOp2H
"aiccbsgdoo/MatMul_1/ReadVariableOp"aiccbsgdoo/MatMul_1/ReadVariableOp26
aiccbsgdoo/ReadVariableOpaiccbsgdoo/ReadVariableOp2:
aiccbsgdoo/ReadVariableOp_1aiccbsgdoo/ReadVariableOp_12:
aiccbsgdoo/ReadVariableOp_2aiccbsgdoo/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ô
Ú
,__inference_sequential_layer_call_fn_1718916

gvxdqcynan
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	 
	unknown_5:	
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:	 

unknown_10:	 

unknown_11:	

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCall
gvxdqcynanunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17188772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
gvxdqcynan
¯F
ê
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1717793

inputs%
aiccbsgdoo_1717694:	 %
aiccbsgdoo_1717696:	 !
aiccbsgdoo_1717698:	 
aiccbsgdoo_1717700:  
aiccbsgdoo_1717702:  
aiccbsgdoo_1717704: 
identity¢"aiccbsgdoo/StatefulPartitionedCall¢whileD
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
"aiccbsgdoo/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0aiccbsgdoo_1717694aiccbsgdoo_1717696aiccbsgdoo_1717698aiccbsgdoo_1717700aiccbsgdoo_1717702aiccbsgdoo_1717704*
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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_17176932$
"aiccbsgdoo/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0aiccbsgdoo_1717694aiccbsgdoo_1717696aiccbsgdoo_1717698aiccbsgdoo_1717700aiccbsgdoo_1717702aiccbsgdoo_1717704*
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
while_body_1717713*
condR
while_cond_1717712*Q
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
IdentityIdentitystrided_slice_3:output:0#^aiccbsgdoo/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"aiccbsgdoo/StatefulPartitionedCall"aiccbsgdoo/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_1716954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1716954___redundant_placeholder05
1while_while_cond_1716954___redundant_placeholder15
1while_while_cond_1716954___redundant_placeholder25
1while_while_cond_1716954___redundant_placeholder35
1while_while_cond_1716954___redundant_placeholder45
1while_while_cond_1716954___redundant_placeholder55
1while_while_cond_1716954___redundant_placeholder6
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
ô
Ú
,__inference_sequential_layer_call_fn_1719549

gvxdqcynan
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	 
	unknown_5:	
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:	 

unknown_10:	 

unknown_11:	

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCall
gvxdqcynanunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17194692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
gvxdqcynan
Ü

(sequential_qzuziqqdld_while_body_1716741H
Dsequential_qzuziqqdld_while_sequential_qzuziqqdld_while_loop_counterN
Jsequential_qzuziqqdld_while_sequential_qzuziqqdld_while_maximum_iterations+
'sequential_qzuziqqdld_while_placeholder-
)sequential_qzuziqqdld_while_placeholder_1-
)sequential_qzuziqqdld_while_placeholder_2-
)sequential_qzuziqqdld_while_placeholder_3G
Csequential_qzuziqqdld_while_sequential_qzuziqqdld_strided_slice_1_0
sequential_qzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_sequential_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0:	 \
Isequential_qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 W
Hsequential_qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0:	N
@sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0: P
Bsequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0: P
Bsequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0: (
$sequential_qzuziqqdld_while_identity*
&sequential_qzuziqqdld_while_identity_1*
&sequential_qzuziqqdld_while_identity_2*
&sequential_qzuziqqdld_while_identity_3*
&sequential_qzuziqqdld_while_identity_4*
&sequential_qzuziqqdld_while_identity_5E
Asequential_qzuziqqdld_while_sequential_qzuziqqdld_strided_slice_1
}sequential_qzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_sequential_qzuziqqdld_tensorarrayunstack_tensorlistfromtensorX
Esequential_qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource:	 Z
Gsequential_qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource:	 U
Fsequential_qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource:	L
>sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_resource: N
@sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource: N
@sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource: ¢=sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp¢<sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp¢>sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp¢5sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp¢7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1¢7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2ï
Msequential/qzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/qzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/qzuziqqdld/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_qzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_sequential_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0'sequential_qzuziqqdld_while_placeholderVsequential/qzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/qzuziqqdld/while/TensorArrayV2Read/TensorListGetItem
<sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOpGsequential_qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp©
-sequential/qzuziqqdld/while/aiccbsgdoo/MatMulMatMulFsequential/qzuziqqdld/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/qzuziqqdld/while/aiccbsgdoo/MatMul
>sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOpIsequential_qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp
/sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1MatMul)sequential_qzuziqqdld_while_placeholder_2Fsequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1
*sequential/qzuziqqdld/while/aiccbsgdoo/addAddV27sequential/qzuziqqdld/while/aiccbsgdoo/MatMul:product:09sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/qzuziqqdld/while/aiccbsgdoo/add
=sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOpHsequential_qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp
.sequential/qzuziqqdld/while/aiccbsgdoo/BiasAddBiasAdd.sequential/qzuziqqdld/while/aiccbsgdoo/add:z:0Esequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd²
6sequential/qzuziqqdld/while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/qzuziqqdld/while/aiccbsgdoo/split/split_dimÛ
,sequential/qzuziqqdld/while/aiccbsgdoo/splitSplit?sequential/qzuziqqdld/while/aiccbsgdoo/split/split_dim:output:07sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/qzuziqqdld/while/aiccbsgdoo/splitë
5sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOpReadVariableOp@sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOpû
*sequential/qzuziqqdld/while/aiccbsgdoo/mulMul=sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp:value:0)sequential_qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/qzuziqqdld/while/aiccbsgdoo/mulþ
,sequential/qzuziqqdld/while/aiccbsgdoo/add_1AddV25sequential/qzuziqqdld/while/aiccbsgdoo/split:output:0.sequential/qzuziqqdld/while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/add_1Ï
.sequential/qzuziqqdld/while/aiccbsgdoo/SigmoidSigmoid0sequential/qzuziqqdld/while/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoidñ
7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1ReadVariableOpBsequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_1Mul?sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1:value:0)sequential_qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_1
,sequential/qzuziqqdld/while/aiccbsgdoo/add_2AddV25sequential/qzuziqqdld/while/aiccbsgdoo/split:output:10sequential/qzuziqqdld/while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/add_2Ó
0sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoid_1Sigmoid0sequential/qzuziqqdld/while/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoid_1ö
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_2Mul4sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoid_1:y:0)sequential_qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_2Ë
+sequential/qzuziqqdld/while/aiccbsgdoo/TanhTanh5sequential/qzuziqqdld/while/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/qzuziqqdld/while/aiccbsgdoo/Tanhú
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_3Mul2sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoid:y:0/sequential/qzuziqqdld/while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_3û
,sequential/qzuziqqdld/while/aiccbsgdoo/add_3AddV20sequential/qzuziqqdld/while/aiccbsgdoo/mul_2:z:00sequential/qzuziqqdld/while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/add_3ñ
7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2ReadVariableOpBsequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_4Mul?sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2:value:00sequential/qzuziqqdld/while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_4
,sequential/qzuziqqdld/while/aiccbsgdoo/add_4AddV25sequential/qzuziqqdld/while/aiccbsgdoo/split:output:30sequential/qzuziqqdld/while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/add_4Ó
0sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoid_2Sigmoid0sequential/qzuziqqdld/while/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoid_2Ê
-sequential/qzuziqqdld/while/aiccbsgdoo/Tanh_1Tanh0sequential/qzuziqqdld/while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/qzuziqqdld/while/aiccbsgdoo/Tanh_1þ
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_5Mul4sequential/qzuziqqdld/while/aiccbsgdoo/Sigmoid_2:y:01sequential/qzuziqqdld/while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/qzuziqqdld/while/aiccbsgdoo/mul_5Ì
@sequential/qzuziqqdld/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_qzuziqqdld_while_placeholder_1'sequential_qzuziqqdld_while_placeholder0sequential/qzuziqqdld/while/aiccbsgdoo/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/qzuziqqdld/while/TensorArrayV2Write/TensorListSetItem
!sequential/qzuziqqdld/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/qzuziqqdld/while/add/yÁ
sequential/qzuziqqdld/while/addAddV2'sequential_qzuziqqdld_while_placeholder*sequential/qzuziqqdld/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/qzuziqqdld/while/add
#sequential/qzuziqqdld/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/qzuziqqdld/while/add_1/yä
!sequential/qzuziqqdld/while/add_1AddV2Dsequential_qzuziqqdld_while_sequential_qzuziqqdld_while_loop_counter,sequential/qzuziqqdld/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/qzuziqqdld/while/add_1
$sequential/qzuziqqdld/while/IdentityIdentity%sequential/qzuziqqdld/while/add_1:z:0>^sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp=^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp?^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp6^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp8^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_18^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/qzuziqqdld/while/Identityµ
&sequential/qzuziqqdld/while/Identity_1IdentityJsequential_qzuziqqdld_while_sequential_qzuziqqdld_while_maximum_iterations>^sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp=^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp?^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp6^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp8^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_18^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/qzuziqqdld/while/Identity_1
&sequential/qzuziqqdld/while/Identity_2Identity#sequential/qzuziqqdld/while/add:z:0>^sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp=^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp?^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp6^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp8^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_18^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/qzuziqqdld/while/Identity_2»
&sequential/qzuziqqdld/while/Identity_3IdentityPsequential/qzuziqqdld/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp=^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp?^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp6^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp8^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_18^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/qzuziqqdld/while/Identity_3¬
&sequential/qzuziqqdld/while/Identity_4Identity0sequential/qzuziqqdld/while/aiccbsgdoo/mul_5:z:0>^sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp=^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp?^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp6^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp8^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_18^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/while/Identity_4¬
&sequential/qzuziqqdld/while/Identity_5Identity0sequential/qzuziqqdld/while/aiccbsgdoo/add_3:z:0>^sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp=^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp?^sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp6^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp8^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_18^sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/while/Identity_5"
Fsequential_qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resourceHsequential_qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0"
Gsequential_qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resourceIsequential_qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0"
Esequential_qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resourceGsequential_qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0"
@sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resourceBsequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0"
@sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resourceBsequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0"
>sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_resource@sequential_qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0"U
$sequential_qzuziqqdld_while_identity-sequential/qzuziqqdld/while/Identity:output:0"Y
&sequential_qzuziqqdld_while_identity_1/sequential/qzuziqqdld/while/Identity_1:output:0"Y
&sequential_qzuziqqdld_while_identity_2/sequential/qzuziqqdld/while/Identity_2:output:0"Y
&sequential_qzuziqqdld_while_identity_3/sequential/qzuziqqdld/while/Identity_3:output:0"Y
&sequential_qzuziqqdld_while_identity_4/sequential/qzuziqqdld/while/Identity_4:output:0"Y
&sequential_qzuziqqdld_while_identity_5/sequential/qzuziqqdld/while/Identity_5:output:0"
Asequential_qzuziqqdld_while_sequential_qzuziqqdld_strided_slice_1Csequential_qzuziqqdld_while_sequential_qzuziqqdld_strided_slice_1_0"
}sequential_qzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_sequential_qzuziqqdld_tensorarrayunstack_tensorlistfromtensorsequential_qzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_sequential_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp=sequential/qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2|
<sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp<sequential/qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp2
>sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp>sequential/qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp2n
5sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp5sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp2r
7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_17sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_12r
7sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_27sequential/qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2: 
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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1722535

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
ð$
­
G__inference_sequential_layer_call_and_return_conditional_losses_1719469

inputs(
mopvqfaljf_1719426: 
mopvqfaljf_1719428:(
ehhqjcwuju_1719431: 
ehhqjcwuju_1719433:%
cxhqebqpjz_1719437:	%
cxhqebqpjz_1719439:	 !
cxhqebqpjz_1719441:	 
cxhqebqpjz_1719443:  
cxhqebqpjz_1719445:  
cxhqebqpjz_1719447: %
qzuziqqdld_1719450:	 %
qzuziqqdld_1719452:	 !
qzuziqqdld_1719454:	 
qzuziqqdld_1719456:  
qzuziqqdld_1719458:  
qzuziqqdld_1719460: $
pbmomrqadp_1719463:  
pbmomrqadp_1719465:
identity¢"cxhqebqpjz/StatefulPartitionedCall¢"ehhqjcwuju/StatefulPartitionedCall¢"mopvqfaljf/StatefulPartitionedCall¢"pbmomrqadp/StatefulPartitionedCall¢"qzuziqqdld/StatefulPartitionedCall¬
"mopvqfaljf/StatefulPartitionedCallStatefulPartitionedCallinputsmopvqfaljf_1719426mopvqfaljf_1719428*
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
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_17184082$
"mopvqfaljf/StatefulPartitionedCallÑ
"ehhqjcwuju/StatefulPartitionedCallStatefulPartitionedCall+mopvqfaljf/StatefulPartitionedCall:output:0ehhqjcwuju_1719431ehhqjcwuju_1719433*
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
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_17184532$
"ehhqjcwuju/StatefulPartitionedCall
abbthhzbau/PartitionedCallPartitionedCall+ehhqjcwuju/StatefulPartitionedCall:output:0*
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
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_17184722
abbthhzbau/PartitionedCall
"cxhqebqpjz/StatefulPartitionedCallStatefulPartitionedCall#abbthhzbau/PartitionedCall:output:0cxhqebqpjz_1719437cxhqebqpjz_1719439cxhqebqpjz_1719441cxhqebqpjz_1719443cxhqebqpjz_1719445cxhqebqpjz_1719447*
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17193392$
"cxhqebqpjz/StatefulPartitionedCall¡
"qzuziqqdld/StatefulPartitionedCallStatefulPartitionedCall+cxhqebqpjz/StatefulPartitionedCall:output:0qzuziqqdld_1719450qzuziqqdld_1719452qzuziqqdld_1719454qzuziqqdld_1719456qzuziqqdld_1719458qzuziqqdld_1719460*
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17191252$
"qzuziqqdld/StatefulPartitionedCallÉ
"pbmomrqadp/StatefulPartitionedCallStatefulPartitionedCall+qzuziqqdld/StatefulPartitionedCall:output:0pbmomrqadp_1719463pbmomrqadp_1719465*
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
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_17188702$
"pbmomrqadp/StatefulPartitionedCall¸
IdentityIdentity+pbmomrqadp/StatefulPartitionedCall:output:0#^cxhqebqpjz/StatefulPartitionedCall#^ehhqjcwuju/StatefulPartitionedCall#^mopvqfaljf/StatefulPartitionedCall#^pbmomrqadp/StatefulPartitionedCall#^qzuziqqdld/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2H
"cxhqebqpjz/StatefulPartitionedCall"cxhqebqpjz/StatefulPartitionedCall2H
"ehhqjcwuju/StatefulPartitionedCall"ehhqjcwuju/StatefulPartitionedCall2H
"mopvqfaljf/StatefulPartitionedCall"mopvqfaljf/StatefulPartitionedCall2H
"pbmomrqadp/StatefulPartitionedCall"pbmomrqadp/StatefulPartitionedCall2H
"qzuziqqdld/StatefulPartitionedCall"qzuziqqdld/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
©	
(sequential_qzuziqqdld_while_cond_1716740H
Dsequential_qzuziqqdld_while_sequential_qzuziqqdld_while_loop_counterN
Jsequential_qzuziqqdld_while_sequential_qzuziqqdld_while_maximum_iterations+
'sequential_qzuziqqdld_while_placeholder-
)sequential_qzuziqqdld_while_placeholder_1-
)sequential_qzuziqqdld_while_placeholder_2-
)sequential_qzuziqqdld_while_placeholder_3J
Fsequential_qzuziqqdld_while_less_sequential_qzuziqqdld_strided_slice_1a
]sequential_qzuziqqdld_while_sequential_qzuziqqdld_while_cond_1716740___redundant_placeholder0a
]sequential_qzuziqqdld_while_sequential_qzuziqqdld_while_cond_1716740___redundant_placeholder1a
]sequential_qzuziqqdld_while_sequential_qzuziqqdld_while_cond_1716740___redundant_placeholder2a
]sequential_qzuziqqdld_while_sequential_qzuziqqdld_while_cond_1716740___redundant_placeholder3a
]sequential_qzuziqqdld_while_sequential_qzuziqqdld_while_cond_1716740___redundant_placeholder4a
]sequential_qzuziqqdld_while_sequential_qzuziqqdld_while_cond_1716740___redundant_placeholder5a
]sequential_qzuziqqdld_while_sequential_qzuziqqdld_while_cond_1716740___redundant_placeholder6(
$sequential_qzuziqqdld_while_identity
Þ
 sequential/qzuziqqdld/while/LessLess'sequential_qzuziqqdld_while_placeholderFsequential_qzuziqqdld_while_less_sequential_qzuziqqdld_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/qzuziqqdld/while/Less
$sequential/qzuziqqdld/while/IdentityIdentity$sequential/qzuziqqdld/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/qzuziqqdld/while/Identity"U
$sequential_qzuziqqdld_while_identity-sequential/qzuziqqdld/while/Identity:output:0*(
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721122
inputs_0<
)rvncypflgq_matmul_readvariableop_resource:	>
+rvncypflgq_matmul_1_readvariableop_resource:	 9
*rvncypflgq_biasadd_readvariableop_resource:	0
"rvncypflgq_readvariableop_resource: 2
$rvncypflgq_readvariableop_1_resource: 2
$rvncypflgq_readvariableop_2_resource: 
identity¢!rvncypflgq/BiasAdd/ReadVariableOp¢ rvncypflgq/MatMul/ReadVariableOp¢"rvncypflgq/MatMul_1/ReadVariableOp¢rvncypflgq/ReadVariableOp¢rvncypflgq/ReadVariableOp_1¢rvncypflgq/ReadVariableOp_2¢whileF
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
 rvncypflgq/MatMul/ReadVariableOpReadVariableOp)rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 rvncypflgq/MatMul/ReadVariableOp§
rvncypflgq/MatMulMatMulstrided_slice_2:output:0(rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMulµ
"rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp+rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"rvncypflgq/MatMul_1/ReadVariableOp£
rvncypflgq/MatMul_1MatMulzeros:output:0*rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMul_1
rvncypflgq/addAddV2rvncypflgq/MatMul:product:0rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/add®
!rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp*rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!rvncypflgq/BiasAdd/ReadVariableOp¥
rvncypflgq/BiasAddBiasAddrvncypflgq/add:z:0)rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/BiasAddz
rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
rvncypflgq/split/split_dimë
rvncypflgq/splitSplit#rvncypflgq/split/split_dim:output:0rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rvncypflgq/split
rvncypflgq/ReadVariableOpReadVariableOp"rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp
rvncypflgq/mulMul!rvncypflgq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul
rvncypflgq/add_1AddV2rvncypflgq/split:output:0rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_1{
rvncypflgq/SigmoidSigmoidrvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid
rvncypflgq/ReadVariableOp_1ReadVariableOp$rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_1
rvncypflgq/mul_1Mul#rvncypflgq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_1
rvncypflgq/add_2AddV2rvncypflgq/split:output:1rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_2
rvncypflgq/Sigmoid_1Sigmoidrvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_1
rvncypflgq/mul_2Mulrvncypflgq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_2w
rvncypflgq/TanhTanhrvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh
rvncypflgq/mul_3Mulrvncypflgq/Sigmoid:y:0rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_3
rvncypflgq/add_3AddV2rvncypflgq/mul_2:z:0rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_3
rvncypflgq/ReadVariableOp_2ReadVariableOp$rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_2
rvncypflgq/mul_4Mul#rvncypflgq/ReadVariableOp_2:value:0rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_4
rvncypflgq/add_4AddV2rvncypflgq/split:output:3rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_4
rvncypflgq/Sigmoid_2Sigmoidrvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_2v
rvncypflgq/Tanh_1Tanhrvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh_1
rvncypflgq/mul_5Mulrvncypflgq/Sigmoid_2:y:0rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)rvncypflgq_matmul_readvariableop_resource+rvncypflgq_matmul_1_readvariableop_resource*rvncypflgq_biasadd_readvariableop_resource"rvncypflgq_readvariableop_resource$rvncypflgq_readvariableop_1_resource$rvncypflgq_readvariableop_2_resource*
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
while_body_1721021*
condR
while_cond_1721020*Q
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
IdentityIdentitytranspose_1:y:0"^rvncypflgq/BiasAdd/ReadVariableOp!^rvncypflgq/MatMul/ReadVariableOp#^rvncypflgq/MatMul_1/ReadVariableOp^rvncypflgq/ReadVariableOp^rvncypflgq/ReadVariableOp_1^rvncypflgq/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!rvncypflgq/BiasAdd/ReadVariableOp!rvncypflgq/BiasAdd/ReadVariableOp2D
 rvncypflgq/MatMul/ReadVariableOp rvncypflgq/MatMul/ReadVariableOp2H
"rvncypflgq/MatMul_1/ReadVariableOp"rvncypflgq/MatMul_1/ReadVariableOp26
rvncypflgq/ReadVariableOprvncypflgq/ReadVariableOp2:
rvncypflgq/ReadVariableOp_1rvncypflgq/ReadVariableOp_12:
rvncypflgq/ReadVariableOp_2rvncypflgq/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
àY

while_body_1721989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aiccbsgdoo_matmul_readvariableop_resource_0:	 F
3while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 A
2while_aiccbsgdoo_biasadd_readvariableop_resource_0:	8
*while_aiccbsgdoo_readvariableop_resource_0: :
,while_aiccbsgdoo_readvariableop_1_resource_0: :
,while_aiccbsgdoo_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aiccbsgdoo_matmul_readvariableop_resource:	 D
1while_aiccbsgdoo_matmul_1_readvariableop_resource:	 ?
0while_aiccbsgdoo_biasadd_readvariableop_resource:	6
(while_aiccbsgdoo_readvariableop_resource: 8
*while_aiccbsgdoo_readvariableop_1_resource: 8
*while_aiccbsgdoo_readvariableop_2_resource: ¢'while/aiccbsgdoo/BiasAdd/ReadVariableOp¢&while/aiccbsgdoo/MatMul/ReadVariableOp¢(while/aiccbsgdoo/MatMul_1/ReadVariableOp¢while/aiccbsgdoo/ReadVariableOp¢!while/aiccbsgdoo/ReadVariableOp_1¢!while/aiccbsgdoo/ReadVariableOp_2Ã
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
&while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp1while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aiccbsgdoo/MatMul/ReadVariableOpÑ
while/aiccbsgdoo/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMulÉ
(while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp3while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aiccbsgdoo/MatMul_1/ReadVariableOpº
while/aiccbsgdoo/MatMul_1MatMulwhile_placeholder_20while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMul_1°
while/aiccbsgdoo/addAddV2!while/aiccbsgdoo/MatMul:product:0#while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/addÂ
'while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp2while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aiccbsgdoo/BiasAdd/ReadVariableOp½
while/aiccbsgdoo/BiasAddBiasAddwhile/aiccbsgdoo/add:z:0/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/BiasAdd
 while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aiccbsgdoo/split/split_dim
while/aiccbsgdoo/splitSplit)while/aiccbsgdoo/split/split_dim:output:0!while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aiccbsgdoo/split©
while/aiccbsgdoo/ReadVariableOpReadVariableOp*while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aiccbsgdoo/ReadVariableOp£
while/aiccbsgdoo/mulMul'while/aiccbsgdoo/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul¦
while/aiccbsgdoo/add_1AddV2while/aiccbsgdoo/split:output:0while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_1
while/aiccbsgdoo/SigmoidSigmoidwhile/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid¯
!while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp,while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_1©
while/aiccbsgdoo/mul_1Mul)while/aiccbsgdoo/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_1¨
while/aiccbsgdoo/add_2AddV2while/aiccbsgdoo/split:output:1while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_2
while/aiccbsgdoo/Sigmoid_1Sigmoidwhile/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_1
while/aiccbsgdoo/mul_2Mulwhile/aiccbsgdoo/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_2
while/aiccbsgdoo/TanhTanhwhile/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh¢
while/aiccbsgdoo/mul_3Mulwhile/aiccbsgdoo/Sigmoid:y:0while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_3£
while/aiccbsgdoo/add_3AddV2while/aiccbsgdoo/mul_2:z:0while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_3¯
!while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp,while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_2°
while/aiccbsgdoo/mul_4Mul)while/aiccbsgdoo/ReadVariableOp_2:value:0while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_4¨
while/aiccbsgdoo/add_4AddV2while/aiccbsgdoo/split:output:3while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_4
while/aiccbsgdoo/Sigmoid_2Sigmoidwhile/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_2
while/aiccbsgdoo/Tanh_1Tanhwhile/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh_1¦
while/aiccbsgdoo/mul_5Mulwhile/aiccbsgdoo/Sigmoid_2:y:0while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aiccbsgdoo/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aiccbsgdoo/mul_5:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aiccbsgdoo/add_3:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aiccbsgdoo_biasadd_readvariableop_resource2while_aiccbsgdoo_biasadd_readvariableop_resource_0"h
1while_aiccbsgdoo_matmul_1_readvariableop_resource3while_aiccbsgdoo_matmul_1_readvariableop_resource_0"d
/while_aiccbsgdoo_matmul_readvariableop_resource1while_aiccbsgdoo_matmul_readvariableop_resource_0"Z
*while_aiccbsgdoo_readvariableop_1_resource,while_aiccbsgdoo_readvariableop_1_resource_0"Z
*while_aiccbsgdoo_readvariableop_2_resource,while_aiccbsgdoo_readvariableop_2_resource_0"V
(while_aiccbsgdoo_readvariableop_resource*while_aiccbsgdoo_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/aiccbsgdoo/BiasAdd/ReadVariableOp'while/aiccbsgdoo/BiasAdd/ReadVariableOp2P
&while/aiccbsgdoo/MatMul/ReadVariableOp&while/aiccbsgdoo/MatMul/ReadVariableOp2T
(while/aiccbsgdoo/MatMul_1/ReadVariableOp(while/aiccbsgdoo/MatMul_1/ReadVariableOp2B
while/aiccbsgdoo/ReadVariableOpwhile/aiccbsgdoo/ReadVariableOp2F
!while/aiccbsgdoo/ReadVariableOp_1!while/aiccbsgdoo/ReadVariableOp_12F
!while/aiccbsgdoo/ReadVariableOp_2!while/aiccbsgdoo/ReadVariableOp_2: 
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
ü$
±
G__inference_sequential_layer_call_and_return_conditional_losses_1719641

gvxdqcynan(
mopvqfaljf_1719598: 
mopvqfaljf_1719600:(
ehhqjcwuju_1719603: 
ehhqjcwuju_1719605:%
cxhqebqpjz_1719609:	%
cxhqebqpjz_1719611:	 !
cxhqebqpjz_1719613:	 
cxhqebqpjz_1719615:  
cxhqebqpjz_1719617:  
cxhqebqpjz_1719619: %
qzuziqqdld_1719622:	 %
qzuziqqdld_1719624:	 !
qzuziqqdld_1719626:	 
qzuziqqdld_1719628:  
qzuziqqdld_1719630:  
qzuziqqdld_1719632: $
pbmomrqadp_1719635:  
pbmomrqadp_1719637:
identity¢"cxhqebqpjz/StatefulPartitionedCall¢"ehhqjcwuju/StatefulPartitionedCall¢"mopvqfaljf/StatefulPartitionedCall¢"pbmomrqadp/StatefulPartitionedCall¢"qzuziqqdld/StatefulPartitionedCall°
"mopvqfaljf/StatefulPartitionedCallStatefulPartitionedCall
gvxdqcynanmopvqfaljf_1719598mopvqfaljf_1719600*
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
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_17184082$
"mopvqfaljf/StatefulPartitionedCallÑ
"ehhqjcwuju/StatefulPartitionedCallStatefulPartitionedCall+mopvqfaljf/StatefulPartitionedCall:output:0ehhqjcwuju_1719603ehhqjcwuju_1719605*
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
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_17184532$
"ehhqjcwuju/StatefulPartitionedCall
abbthhzbau/PartitionedCallPartitionedCall+ehhqjcwuju/StatefulPartitionedCall:output:0*
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
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_17184722
abbthhzbau/PartitionedCall
"cxhqebqpjz/StatefulPartitionedCallStatefulPartitionedCall#abbthhzbau/PartitionedCall:output:0cxhqebqpjz_1719609cxhqebqpjz_1719611cxhqebqpjz_1719613cxhqebqpjz_1719615cxhqebqpjz_1719617cxhqebqpjz_1719619*
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17193392$
"cxhqebqpjz/StatefulPartitionedCall¡
"qzuziqqdld/StatefulPartitionedCallStatefulPartitionedCall+cxhqebqpjz/StatefulPartitionedCall:output:0qzuziqqdld_1719622qzuziqqdld_1719624qzuziqqdld_1719626qzuziqqdld_1719628qzuziqqdld_1719630qzuziqqdld_1719632*
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17191252$
"qzuziqqdld/StatefulPartitionedCallÉ
"pbmomrqadp/StatefulPartitionedCallStatefulPartitionedCall+qzuziqqdld/StatefulPartitionedCall:output:0pbmomrqadp_1719635pbmomrqadp_1719637*
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
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_17188702$
"pbmomrqadp/StatefulPartitionedCall¸
IdentityIdentity+pbmomrqadp/StatefulPartitionedCall:output:0#^cxhqebqpjz/StatefulPartitionedCall#^ehhqjcwuju/StatefulPartitionedCall#^mopvqfaljf/StatefulPartitionedCall#^pbmomrqadp/StatefulPartitionedCall#^qzuziqqdld/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2H
"cxhqebqpjz/StatefulPartitionedCall"cxhqebqpjz/StatefulPartitionedCall2H
"ehhqjcwuju/StatefulPartitionedCall"ehhqjcwuju/StatefulPartitionedCall2H
"mopvqfaljf/StatefulPartitionedCall"mopvqfaljf/StatefulPartitionedCall2H
"pbmomrqadp/StatefulPartitionedCall"pbmomrqadp/StatefulPartitionedCall2H
"qzuziqqdld/StatefulPartitionedCall"qzuziqqdld/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
gvxdqcynan


í
while_cond_1722168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1722168___redundant_placeholder05
1while_while_cond_1722168___redundant_placeholder15
1while_while_cond_1722168___redundant_placeholder25
1while_while_cond_1722168___redundant_placeholder35
1while_while_cond_1722168___redundant_placeholder45
1while_while_cond_1722168___redundant_placeholder55
1while_while_cond_1722168___redundant_placeholder6
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
µ
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1722579

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
ç)
Ò
while_body_1717713
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_aiccbsgdoo_1717737_0:	 -
while_aiccbsgdoo_1717739_0:	 )
while_aiccbsgdoo_1717741_0:	(
while_aiccbsgdoo_1717743_0: (
while_aiccbsgdoo_1717745_0: (
while_aiccbsgdoo_1717747_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_aiccbsgdoo_1717737:	 +
while_aiccbsgdoo_1717739:	 '
while_aiccbsgdoo_1717741:	&
while_aiccbsgdoo_1717743: &
while_aiccbsgdoo_1717745: &
while_aiccbsgdoo_1717747: ¢(while/aiccbsgdoo/StatefulPartitionedCallÃ
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
(while/aiccbsgdoo/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_aiccbsgdoo_1717737_0while_aiccbsgdoo_1717739_0while_aiccbsgdoo_1717741_0while_aiccbsgdoo_1717743_0while_aiccbsgdoo_1717745_0while_aiccbsgdoo_1717747_0*
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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_17176932*
(while/aiccbsgdoo/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/aiccbsgdoo/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/aiccbsgdoo/StatefulPartitionedCall:output:1)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/aiccbsgdoo/StatefulPartitionedCall:output:2)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_aiccbsgdoo_1717737while_aiccbsgdoo_1717737_0"6
while_aiccbsgdoo_1717739while_aiccbsgdoo_1717739_0"6
while_aiccbsgdoo_1717741while_aiccbsgdoo_1717741_0"6
while_aiccbsgdoo_1717743while_aiccbsgdoo_1717743_0"6
while_aiccbsgdoo_1717745while_aiccbsgdoo_1717745_0"6
while_aiccbsgdoo_1717747while_aiccbsgdoo_1717747_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/aiccbsgdoo/StatefulPartitionedCall(while/aiccbsgdoo/StatefulPartitionedCall: 
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
,__inference_ehhqjcwuju_layer_call_fn_1720744

inputs
unknown:
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
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_17184532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1720840
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1720840___redundant_placeholder05
1while_while_cond_1720840___redundant_placeholder15
1while_while_cond_1720840___redundant_placeholder25
1while_while_cond_1720840___redundant_placeholder35
1while_while_cond_1720840___redundant_placeholder45
1while_while_cond_1720840___redundant_placeholder55
1while_while_cond_1720840___redundant_placeholder6
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
while_cond_1721200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1721200___redundant_placeholder05
1while_while_cond_1721200___redundant_placeholder15
1while_while_cond_1721200___redundant_placeholder25
1while_while_cond_1721200___redundant_placeholder35
1while_while_cond_1721200___redundant_placeholder45
1while_while_cond_1721200___redundant_placeholder55
1while_while_cond_1721200___redundant_placeholder6
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
,__inference_rvncypflgq_layer_call_fn_1722491

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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_17171222
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

À
,__inference_aiccbsgdoo_layer_call_fn_1722625

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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_17178802
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
«2
¼
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_1718453

inputsA
+conv1d_expanddims_1_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢"conv1d/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
Pad/paddingsj
PadPadinputsPad/paddings:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
valueB"ÿÿÿÿ         2
conv1d/Reshape/shape 
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
p
Ê
cxhqebqpjz_while_body_17198462
.cxhqebqpjz_while_cxhqebqpjz_while_loop_counter8
4cxhqebqpjz_while_cxhqebqpjz_while_maximum_iterations 
cxhqebqpjz_while_placeholder"
cxhqebqpjz_while_placeholder_1"
cxhqebqpjz_while_placeholder_2"
cxhqebqpjz_while_placeholder_31
-cxhqebqpjz_while_cxhqebqpjz_strided_slice_1_0m
icxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0O
<cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0:	Q
>cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0:	 L
=cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0:	C
5cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0: E
7cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0: E
7cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0: 
cxhqebqpjz_while_identity
cxhqebqpjz_while_identity_1
cxhqebqpjz_while_identity_2
cxhqebqpjz_while_identity_3
cxhqebqpjz_while_identity_4
cxhqebqpjz_while_identity_5/
+cxhqebqpjz_while_cxhqebqpjz_strided_slice_1k
gcxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensorM
:cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource:	O
<cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource:	 J
;cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource:	A
3cxhqebqpjz_while_rvncypflgq_readvariableop_resource: C
5cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource: C
5cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource: ¢2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp¢1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp¢3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp¢*cxhqebqpjz/while/rvncypflgq/ReadVariableOp¢,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1¢,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2Ù
Bcxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bcxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shape
4cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemicxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0cxhqebqpjz_while_placeholderKcxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItemä
1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp<cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOpý
"cxhqebqpjz/while/rvncypflgq/MatMulMatMul;cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem:item:09cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"cxhqebqpjz/while/rvncypflgq/MatMulê
3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp>cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOpæ
$cxhqebqpjz/while/rvncypflgq/MatMul_1MatMulcxhqebqpjz_while_placeholder_2;cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$cxhqebqpjz/while/rvncypflgq/MatMul_1Ü
cxhqebqpjz/while/rvncypflgq/addAddV2,cxhqebqpjz/while/rvncypflgq/MatMul:product:0.cxhqebqpjz/while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
cxhqebqpjz/while/rvncypflgq/addã
2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp=cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOpé
#cxhqebqpjz/while/rvncypflgq/BiasAddBiasAdd#cxhqebqpjz/while/rvncypflgq/add:z:0:cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#cxhqebqpjz/while/rvncypflgq/BiasAdd
+cxhqebqpjz/while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+cxhqebqpjz/while/rvncypflgq/split/split_dim¯
!cxhqebqpjz/while/rvncypflgq/splitSplit4cxhqebqpjz/while/rvncypflgq/split/split_dim:output:0,cxhqebqpjz/while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!cxhqebqpjz/while/rvncypflgq/splitÊ
*cxhqebqpjz/while/rvncypflgq/ReadVariableOpReadVariableOp5cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*cxhqebqpjz/while/rvncypflgq/ReadVariableOpÏ
cxhqebqpjz/while/rvncypflgq/mulMul2cxhqebqpjz/while/rvncypflgq/ReadVariableOp:value:0cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
cxhqebqpjz/while/rvncypflgq/mulÒ
!cxhqebqpjz/while/rvncypflgq/add_1AddV2*cxhqebqpjz/while/rvncypflgq/split:output:0#cxhqebqpjz/while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_1®
#cxhqebqpjz/while/rvncypflgq/SigmoidSigmoid%cxhqebqpjz/while/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#cxhqebqpjz/while/rvncypflgq/SigmoidÐ
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1ReadVariableOp7cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1Õ
!cxhqebqpjz/while/rvncypflgq/mul_1Mul4cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1:value:0cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_1Ô
!cxhqebqpjz/while/rvncypflgq/add_2AddV2*cxhqebqpjz/while/rvncypflgq/split:output:1%cxhqebqpjz/while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_2²
%cxhqebqpjz/while/rvncypflgq/Sigmoid_1Sigmoid%cxhqebqpjz/while/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%cxhqebqpjz/while/rvncypflgq/Sigmoid_1Ê
!cxhqebqpjz/while/rvncypflgq/mul_2Mul)cxhqebqpjz/while/rvncypflgq/Sigmoid_1:y:0cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_2ª
 cxhqebqpjz/while/rvncypflgq/TanhTanh*cxhqebqpjz/while/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 cxhqebqpjz/while/rvncypflgq/TanhÎ
!cxhqebqpjz/while/rvncypflgq/mul_3Mul'cxhqebqpjz/while/rvncypflgq/Sigmoid:y:0$cxhqebqpjz/while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_3Ï
!cxhqebqpjz/while/rvncypflgq/add_3AddV2%cxhqebqpjz/while/rvncypflgq/mul_2:z:0%cxhqebqpjz/while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_3Ð
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2ReadVariableOp7cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2Ü
!cxhqebqpjz/while/rvncypflgq/mul_4Mul4cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2:value:0%cxhqebqpjz/while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_4Ô
!cxhqebqpjz/while/rvncypflgq/add_4AddV2*cxhqebqpjz/while/rvncypflgq/split:output:3%cxhqebqpjz/while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_4²
%cxhqebqpjz/while/rvncypflgq/Sigmoid_2Sigmoid%cxhqebqpjz/while/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%cxhqebqpjz/while/rvncypflgq/Sigmoid_2©
"cxhqebqpjz/while/rvncypflgq/Tanh_1Tanh%cxhqebqpjz/while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"cxhqebqpjz/while/rvncypflgq/Tanh_1Ò
!cxhqebqpjz/while/rvncypflgq/mul_5Mul)cxhqebqpjz/while/rvncypflgq/Sigmoid_2:y:0&cxhqebqpjz/while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_5
5cxhqebqpjz/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcxhqebqpjz_while_placeholder_1cxhqebqpjz_while_placeholder%cxhqebqpjz/while/rvncypflgq/mul_5:z:0*
_output_shapes
: *
element_dtype027
5cxhqebqpjz/while/TensorArrayV2Write/TensorListSetItemr
cxhqebqpjz/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cxhqebqpjz/while/add/y
cxhqebqpjz/while/addAddV2cxhqebqpjz_while_placeholdercxhqebqpjz/while/add/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/while/addv
cxhqebqpjz/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cxhqebqpjz/while/add_1/y­
cxhqebqpjz/while/add_1AddV2.cxhqebqpjz_while_cxhqebqpjz_while_loop_counter!cxhqebqpjz/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/while/add_1©
cxhqebqpjz/while/IdentityIdentitycxhqebqpjz/while/add_1:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/IdentityÇ
cxhqebqpjz/while/Identity_1Identity4cxhqebqpjz_while_cxhqebqpjz_while_maximum_iterations3^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Identity_1«
cxhqebqpjz/while/Identity_2Identitycxhqebqpjz/while/add:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Identity_2Ø
cxhqebqpjz/while/Identity_3IdentityEcxhqebqpjz/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Identity_3É
cxhqebqpjz/while/Identity_4Identity%cxhqebqpjz/while/rvncypflgq/mul_5:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/while/Identity_4É
cxhqebqpjz/while/Identity_5Identity%cxhqebqpjz/while/rvncypflgq/add_3:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/while/Identity_5"\
+cxhqebqpjz_while_cxhqebqpjz_strided_slice_1-cxhqebqpjz_while_cxhqebqpjz_strided_slice_1_0"?
cxhqebqpjz_while_identity"cxhqebqpjz/while/Identity:output:0"C
cxhqebqpjz_while_identity_1$cxhqebqpjz/while/Identity_1:output:0"C
cxhqebqpjz_while_identity_2$cxhqebqpjz/while/Identity_2:output:0"C
cxhqebqpjz_while_identity_3$cxhqebqpjz/while/Identity_3:output:0"C
cxhqebqpjz_while_identity_4$cxhqebqpjz/while/Identity_4:output:0"C
cxhqebqpjz_while_identity_5$cxhqebqpjz/while/Identity_5:output:0"|
;cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource=cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0"~
<cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource>cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0"z
:cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource<cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0"p
5cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource7cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0"p
5cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource7cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0"l
3cxhqebqpjz_while_rvncypflgq_readvariableop_resource5cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0"Ô
gcxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensoricxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2f
1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp2j
3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp2X
*cxhqebqpjz/while/rvncypflgq/ReadVariableOp*cxhqebqpjz/while/rvncypflgq/ReadVariableOp2\
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_12\
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2: 
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
while_cond_1721808
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1721808___redundant_placeholder05
1while_while_cond_1721808___redundant_placeholder15
1while_while_cond_1721808___redundant_placeholder25
1while_while_cond_1721808___redundant_placeholder35
1while_while_cond_1721808___redundant_placeholder45
1while_while_cond_1721808___redundant_placeholder55
1while_while_cond_1721808___redundant_placeholder6
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
while_cond_1718744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1718744___redundant_placeholder05
1while_while_cond_1718744___redundant_placeholder15
1while_while_cond_1718744___redundant_placeholder25
1while_while_cond_1718744___redundant_placeholder35
1while_while_cond_1718744___redundant_placeholder45
1while_while_cond_1718744___redundant_placeholder55
1while_while_cond_1718744___redundant_placeholder6
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
µ
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1722445

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


qzuziqqdld_while_cond_17204602
.qzuziqqdld_while_qzuziqqdld_while_loop_counter8
4qzuziqqdld_while_qzuziqqdld_while_maximum_iterations 
qzuziqqdld_while_placeholder"
qzuziqqdld_while_placeholder_1"
qzuziqqdld_while_placeholder_2"
qzuziqqdld_while_placeholder_34
0qzuziqqdld_while_less_qzuziqqdld_strided_slice_1K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720460___redundant_placeholder0K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720460___redundant_placeholder1K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720460___redundant_placeholder2K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720460___redundant_placeholder3K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720460___redundant_placeholder4K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720460___redundant_placeholder5K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720460___redundant_placeholder6
qzuziqqdld_while_identity
§
qzuziqqdld/while/LessLessqzuziqqdld_while_placeholder0qzuziqqdld_while_less_qzuziqqdld_strided_slice_1*
T0*
_output_shapes
: 2
qzuziqqdld/while/Less~
qzuziqqdld/while/IdentityIdentityqzuziqqdld/while/Less:z:0*
T0
*
_output_shapes
: 2
qzuziqqdld/while/Identity"?
qzuziqqdld_while_identity"qzuziqqdld/while/Identity:output:0*(
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
©0
¼
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_1718408

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
àY

while_body_1721021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_rvncypflgq_matmul_readvariableop_resource_0:	F
3while_rvncypflgq_matmul_1_readvariableop_resource_0:	 A
2while_rvncypflgq_biasadd_readvariableop_resource_0:	8
*while_rvncypflgq_readvariableop_resource_0: :
,while_rvncypflgq_readvariableop_1_resource_0: :
,while_rvncypflgq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_rvncypflgq_matmul_readvariableop_resource:	D
1while_rvncypflgq_matmul_1_readvariableop_resource:	 ?
0while_rvncypflgq_biasadd_readvariableop_resource:	6
(while_rvncypflgq_readvariableop_resource: 8
*while_rvncypflgq_readvariableop_1_resource: 8
*while_rvncypflgq_readvariableop_2_resource: ¢'while/rvncypflgq/BiasAdd/ReadVariableOp¢&while/rvncypflgq/MatMul/ReadVariableOp¢(while/rvncypflgq/MatMul_1/ReadVariableOp¢while/rvncypflgq/ReadVariableOp¢!while/rvncypflgq/ReadVariableOp_1¢!while/rvncypflgq/ReadVariableOp_2Ã
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
&while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp1while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/rvncypflgq/MatMul/ReadVariableOpÑ
while/rvncypflgq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMulÉ
(while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp3while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/rvncypflgq/MatMul_1/ReadVariableOpº
while/rvncypflgq/MatMul_1MatMulwhile_placeholder_20while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMul_1°
while/rvncypflgq/addAddV2!while/rvncypflgq/MatMul:product:0#while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/addÂ
'while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp2while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/rvncypflgq/BiasAdd/ReadVariableOp½
while/rvncypflgq/BiasAddBiasAddwhile/rvncypflgq/add:z:0/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/BiasAdd
 while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/rvncypflgq/split/split_dim
while/rvncypflgq/splitSplit)while/rvncypflgq/split/split_dim:output:0!while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/rvncypflgq/split©
while/rvncypflgq/ReadVariableOpReadVariableOp*while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/rvncypflgq/ReadVariableOp£
while/rvncypflgq/mulMul'while/rvncypflgq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul¦
while/rvncypflgq/add_1AddV2while/rvncypflgq/split:output:0while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_1
while/rvncypflgq/SigmoidSigmoidwhile/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid¯
!while/rvncypflgq/ReadVariableOp_1ReadVariableOp,while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_1©
while/rvncypflgq/mul_1Mul)while/rvncypflgq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_1¨
while/rvncypflgq/add_2AddV2while/rvncypflgq/split:output:1while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_2
while/rvncypflgq/Sigmoid_1Sigmoidwhile/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_1
while/rvncypflgq/mul_2Mulwhile/rvncypflgq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_2
while/rvncypflgq/TanhTanhwhile/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh¢
while/rvncypflgq/mul_3Mulwhile/rvncypflgq/Sigmoid:y:0while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_3£
while/rvncypflgq/add_3AddV2while/rvncypflgq/mul_2:z:0while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_3¯
!while/rvncypflgq/ReadVariableOp_2ReadVariableOp,while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_2°
while/rvncypflgq/mul_4Mul)while/rvncypflgq/ReadVariableOp_2:value:0while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_4¨
while/rvncypflgq/add_4AddV2while/rvncypflgq/split:output:3while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_4
while/rvncypflgq/Sigmoid_2Sigmoidwhile/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_2
while/rvncypflgq/Tanh_1Tanhwhile/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh_1¦
while/rvncypflgq/mul_5Mulwhile/rvncypflgq/Sigmoid_2:y:0while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/rvncypflgq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/rvncypflgq/mul_5:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/rvncypflgq/add_3:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
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
0while_rvncypflgq_biasadd_readvariableop_resource2while_rvncypflgq_biasadd_readvariableop_resource_0"h
1while_rvncypflgq_matmul_1_readvariableop_resource3while_rvncypflgq_matmul_1_readvariableop_resource_0"d
/while_rvncypflgq_matmul_readvariableop_resource1while_rvncypflgq_matmul_readvariableop_resource_0"Z
*while_rvncypflgq_readvariableop_1_resource,while_rvncypflgq_readvariableop_1_resource_0"Z
*while_rvncypflgq_readvariableop_2_resource,while_rvncypflgq_readvariableop_2_resource_0"V
(while_rvncypflgq_readvariableop_resource*while_rvncypflgq_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/rvncypflgq/BiasAdd/ReadVariableOp'while/rvncypflgq/BiasAdd/ReadVariableOp2P
&while/rvncypflgq/MatMul/ReadVariableOp&while/rvncypflgq/MatMul/ReadVariableOp2T
(while/rvncypflgq/MatMul_1/ReadVariableOp(while/rvncypflgq/MatMul_1/ReadVariableOp2B
while/rvncypflgq/ReadVariableOpwhile/rvncypflgq/ReadVariableOp2F
!while/rvncypflgq/ReadVariableOp_1!while/rvncypflgq/ReadVariableOp_12F
!while/rvncypflgq/ReadVariableOp_2!while/rvncypflgq/ReadVariableOp_2: 
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
a
§
 __inference__traced_save_1722777
file_prefix0
,savev2_mopvqfaljf_kernel_read_readvariableop.
*savev2_mopvqfaljf_bias_read_readvariableop0
,savev2_ehhqjcwuju_kernel_read_readvariableop.
*savev2_ehhqjcwuju_bias_read_readvariableop0
,savev2_pbmomrqadp_kernel_read_readvariableop.
*savev2_pbmomrqadp_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_cxhqebqpjz_rvncypflgq_kernel_read_readvariableopE
Asavev2_cxhqebqpjz_rvncypflgq_recurrent_kernel_read_readvariableop9
5savev2_cxhqebqpjz_rvncypflgq_bias_read_readvariableopP
Lsavev2_cxhqebqpjz_rvncypflgq_input_gate_peephole_weights_read_readvariableopQ
Msavev2_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_cxhqebqpjz_rvncypflgq_output_gate_peephole_weights_read_readvariableop;
7savev2_qzuziqqdld_aiccbsgdoo_kernel_read_readvariableopE
Asavev2_qzuziqqdld_aiccbsgdoo_recurrent_kernel_read_readvariableop9
5savev2_qzuziqqdld_aiccbsgdoo_bias_read_readvariableopP
Lsavev2_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weights_read_readvariableopQ
Msavev2_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_mopvqfaljf_kernel_rms_read_readvariableop:
6savev2_rmsprop_mopvqfaljf_bias_rms_read_readvariableop<
8savev2_rmsprop_ehhqjcwuju_kernel_rms_read_readvariableop:
6savev2_rmsprop_ehhqjcwuju_bias_rms_read_readvariableop<
8savev2_rmsprop_pbmomrqadp_kernel_rms_read_readvariableop:
6savev2_rmsprop_pbmomrqadp_bias_rms_read_readvariableopG
Csavev2_rmsprop_cxhqebqpjz_rvncypflgq_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_cxhqebqpjz_rvncypflgq_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_cxhqebqpjz_rvncypflgq_bias_rms_read_readvariableop\
Xsavev2_rmsprop_cxhqebqpjz_rvncypflgq_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_cxhqebqpjz_rvncypflgq_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_qzuziqqdld_aiccbsgdoo_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_qzuziqqdld_aiccbsgdoo_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_qzuziqqdld_aiccbsgdoo_bias_rms_read_readvariableop\
Xsavev2_rmsprop_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weights_rms_read_readvariableop
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
ShardedFilenameí
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*ÿ
valueõBò,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesà
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesó
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_mopvqfaljf_kernel_read_readvariableop*savev2_mopvqfaljf_bias_read_readvariableop,savev2_ehhqjcwuju_kernel_read_readvariableop*savev2_ehhqjcwuju_bias_read_readvariableop,savev2_pbmomrqadp_kernel_read_readvariableop*savev2_pbmomrqadp_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_cxhqebqpjz_rvncypflgq_kernel_read_readvariableopAsavev2_cxhqebqpjz_rvncypflgq_recurrent_kernel_read_readvariableop5savev2_cxhqebqpjz_rvncypflgq_bias_read_readvariableopLsavev2_cxhqebqpjz_rvncypflgq_input_gate_peephole_weights_read_readvariableopMsavev2_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weights_read_readvariableopMsavev2_cxhqebqpjz_rvncypflgq_output_gate_peephole_weights_read_readvariableop7savev2_qzuziqqdld_aiccbsgdoo_kernel_read_readvariableopAsavev2_qzuziqqdld_aiccbsgdoo_recurrent_kernel_read_readvariableop5savev2_qzuziqqdld_aiccbsgdoo_bias_read_readvariableopLsavev2_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weights_read_readvariableopMsavev2_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weights_read_readvariableopMsavev2_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_mopvqfaljf_kernel_rms_read_readvariableop6savev2_rmsprop_mopvqfaljf_bias_rms_read_readvariableop8savev2_rmsprop_ehhqjcwuju_kernel_rms_read_readvariableop6savev2_rmsprop_ehhqjcwuju_bias_rms_read_readvariableop8savev2_rmsprop_pbmomrqadp_kernel_rms_read_readvariableop6savev2_rmsprop_pbmomrqadp_bias_rms_read_readvariableopCsavev2_rmsprop_cxhqebqpjz_rvncypflgq_kernel_rms_read_readvariableopMsavev2_rmsprop_cxhqebqpjz_rvncypflgq_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_cxhqebqpjz_rvncypflgq_bias_rms_read_readvariableopXsavev2_rmsprop_cxhqebqpjz_rvncypflgq_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_cxhqebqpjz_rvncypflgq_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_qzuziqqdld_aiccbsgdoo_kernel_rms_read_readvariableopMsavev2_rmsprop_qzuziqqdld_aiccbsgdoo_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_qzuziqqdld_aiccbsgdoo_bias_rms_read_readvariableopXsavev2_rmsprop_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
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

identity_1Identity_1:output:0*Ó
_input_shapesÁ
¾: ::::: :: : : : : :	:	 :: : : :	 :	 :: : : : : ::::: ::	:	 :: : : :	 :	 :: : : : 2(
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
::($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:%!

_output_shapes
:	 :!

_output_shapes	
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	 :%!

_output_shapes
:	 :!

_output_shapes	
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::% !

_output_shapes
:	:%!!

_output_shapes
:	 :!"

_output_shapes	
:: #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: :%&!

_output_shapes
:	 :%'!

_output_shapes
:	 :!(

_output_shapes	
:: )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: :,

_output_shapes
: 


qzuziqqdld_while_cond_17200212
.qzuziqqdld_while_qzuziqqdld_while_loop_counter8
4qzuziqqdld_while_qzuziqqdld_while_maximum_iterations 
qzuziqqdld_while_placeholder"
qzuziqqdld_while_placeholder_1"
qzuziqqdld_while_placeholder_2"
qzuziqqdld_while_placeholder_34
0qzuziqqdld_while_less_qzuziqqdld_strided_slice_1K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720021___redundant_placeholder0K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720021___redundant_placeholder1K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720021___redundant_placeholder2K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720021___redundant_placeholder3K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720021___redundant_placeholder4K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720021___redundant_placeholder5K
Gqzuziqqdld_while_qzuziqqdld_while_cond_1720021___redundant_placeholder6
qzuziqqdld_while_identity
§
qzuziqqdld/while/LessLessqzuziqqdld_while_placeholder0qzuziqqdld_while_less_qzuziqqdld_strided_slice_1*
T0*
_output_shapes
: 2
qzuziqqdld/while/Less~
qzuziqqdld/while/IdentityIdentityqzuziqqdld/while/Less:z:0*
T0
*
_output_shapes
: 2
qzuziqqdld/while/Identity"?
qzuziqqdld_while_identity"qzuziqqdld/while/Identity:output:0*(
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
while_body_1721381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_rvncypflgq_matmul_readvariableop_resource_0:	F
3while_rvncypflgq_matmul_1_readvariableop_resource_0:	 A
2while_rvncypflgq_biasadd_readvariableop_resource_0:	8
*while_rvncypflgq_readvariableop_resource_0: :
,while_rvncypflgq_readvariableop_1_resource_0: :
,while_rvncypflgq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_rvncypflgq_matmul_readvariableop_resource:	D
1while_rvncypflgq_matmul_1_readvariableop_resource:	 ?
0while_rvncypflgq_biasadd_readvariableop_resource:	6
(while_rvncypflgq_readvariableop_resource: 8
*while_rvncypflgq_readvariableop_1_resource: 8
*while_rvncypflgq_readvariableop_2_resource: ¢'while/rvncypflgq/BiasAdd/ReadVariableOp¢&while/rvncypflgq/MatMul/ReadVariableOp¢(while/rvncypflgq/MatMul_1/ReadVariableOp¢while/rvncypflgq/ReadVariableOp¢!while/rvncypflgq/ReadVariableOp_1¢!while/rvncypflgq/ReadVariableOp_2Ã
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
&while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp1while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/rvncypflgq/MatMul/ReadVariableOpÑ
while/rvncypflgq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMulÉ
(while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp3while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/rvncypflgq/MatMul_1/ReadVariableOpº
while/rvncypflgq/MatMul_1MatMulwhile_placeholder_20while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMul_1°
while/rvncypflgq/addAddV2!while/rvncypflgq/MatMul:product:0#while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/addÂ
'while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp2while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/rvncypflgq/BiasAdd/ReadVariableOp½
while/rvncypflgq/BiasAddBiasAddwhile/rvncypflgq/add:z:0/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/BiasAdd
 while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/rvncypflgq/split/split_dim
while/rvncypflgq/splitSplit)while/rvncypflgq/split/split_dim:output:0!while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/rvncypflgq/split©
while/rvncypflgq/ReadVariableOpReadVariableOp*while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/rvncypflgq/ReadVariableOp£
while/rvncypflgq/mulMul'while/rvncypflgq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul¦
while/rvncypflgq/add_1AddV2while/rvncypflgq/split:output:0while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_1
while/rvncypflgq/SigmoidSigmoidwhile/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid¯
!while/rvncypflgq/ReadVariableOp_1ReadVariableOp,while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_1©
while/rvncypflgq/mul_1Mul)while/rvncypflgq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_1¨
while/rvncypflgq/add_2AddV2while/rvncypflgq/split:output:1while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_2
while/rvncypflgq/Sigmoid_1Sigmoidwhile/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_1
while/rvncypflgq/mul_2Mulwhile/rvncypflgq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_2
while/rvncypflgq/TanhTanhwhile/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh¢
while/rvncypflgq/mul_3Mulwhile/rvncypflgq/Sigmoid:y:0while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_3£
while/rvncypflgq/add_3AddV2while/rvncypflgq/mul_2:z:0while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_3¯
!while/rvncypflgq/ReadVariableOp_2ReadVariableOp,while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_2°
while/rvncypflgq/mul_4Mul)while/rvncypflgq/ReadVariableOp_2:value:0while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_4¨
while/rvncypflgq/add_4AddV2while/rvncypflgq/split:output:3while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_4
while/rvncypflgq/Sigmoid_2Sigmoidwhile/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_2
while/rvncypflgq/Tanh_1Tanhwhile/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh_1¦
while/rvncypflgq/mul_5Mulwhile/rvncypflgq/Sigmoid_2:y:0while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/rvncypflgq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/rvncypflgq/mul_5:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/rvncypflgq/add_3:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
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
0while_rvncypflgq_biasadd_readvariableop_resource2while_rvncypflgq_biasadd_readvariableop_resource_0"h
1while_rvncypflgq_matmul_1_readvariableop_resource3while_rvncypflgq_matmul_1_readvariableop_resource_0"d
/while_rvncypflgq_matmul_readvariableop_resource1while_rvncypflgq_matmul_readvariableop_resource_0"Z
*while_rvncypflgq_readvariableop_1_resource,while_rvncypflgq_readvariableop_1_resource_0"Z
*while_rvncypflgq_readvariableop_2_resource,while_rvncypflgq_readvariableop_2_resource_0"V
(while_rvncypflgq_readvariableop_resource*while_rvncypflgq_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/rvncypflgq/BiasAdd/ReadVariableOp'while/rvncypflgq/BiasAdd/ReadVariableOp2P
&while/rvncypflgq/MatMul/ReadVariableOp&while/rvncypflgq/MatMul/ReadVariableOp2T
(while/rvncypflgq/MatMul_1/ReadVariableOp(while/rvncypflgq/MatMul_1/ReadVariableOp2B
while/rvncypflgq/ReadVariableOpwhile/rvncypflgq/ReadVariableOp2F
!while/rvncypflgq/ReadVariableOp_1!while/rvncypflgq/ReadVariableOp_12F
!while/rvncypflgq/ReadVariableOp_2!while/rvncypflgq/ReadVariableOp_2: 
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
while_body_1721809
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aiccbsgdoo_matmul_readvariableop_resource_0:	 F
3while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 A
2while_aiccbsgdoo_biasadd_readvariableop_resource_0:	8
*while_aiccbsgdoo_readvariableop_resource_0: :
,while_aiccbsgdoo_readvariableop_1_resource_0: :
,while_aiccbsgdoo_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aiccbsgdoo_matmul_readvariableop_resource:	 D
1while_aiccbsgdoo_matmul_1_readvariableop_resource:	 ?
0while_aiccbsgdoo_biasadd_readvariableop_resource:	6
(while_aiccbsgdoo_readvariableop_resource: 8
*while_aiccbsgdoo_readvariableop_1_resource: 8
*while_aiccbsgdoo_readvariableop_2_resource: ¢'while/aiccbsgdoo/BiasAdd/ReadVariableOp¢&while/aiccbsgdoo/MatMul/ReadVariableOp¢(while/aiccbsgdoo/MatMul_1/ReadVariableOp¢while/aiccbsgdoo/ReadVariableOp¢!while/aiccbsgdoo/ReadVariableOp_1¢!while/aiccbsgdoo/ReadVariableOp_2Ã
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
&while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp1while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aiccbsgdoo/MatMul/ReadVariableOpÑ
while/aiccbsgdoo/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMulÉ
(while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp3while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aiccbsgdoo/MatMul_1/ReadVariableOpº
while/aiccbsgdoo/MatMul_1MatMulwhile_placeholder_20while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMul_1°
while/aiccbsgdoo/addAddV2!while/aiccbsgdoo/MatMul:product:0#while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/addÂ
'while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp2while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aiccbsgdoo/BiasAdd/ReadVariableOp½
while/aiccbsgdoo/BiasAddBiasAddwhile/aiccbsgdoo/add:z:0/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/BiasAdd
 while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aiccbsgdoo/split/split_dim
while/aiccbsgdoo/splitSplit)while/aiccbsgdoo/split/split_dim:output:0!while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aiccbsgdoo/split©
while/aiccbsgdoo/ReadVariableOpReadVariableOp*while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aiccbsgdoo/ReadVariableOp£
while/aiccbsgdoo/mulMul'while/aiccbsgdoo/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul¦
while/aiccbsgdoo/add_1AddV2while/aiccbsgdoo/split:output:0while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_1
while/aiccbsgdoo/SigmoidSigmoidwhile/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid¯
!while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp,while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_1©
while/aiccbsgdoo/mul_1Mul)while/aiccbsgdoo/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_1¨
while/aiccbsgdoo/add_2AddV2while/aiccbsgdoo/split:output:1while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_2
while/aiccbsgdoo/Sigmoid_1Sigmoidwhile/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_1
while/aiccbsgdoo/mul_2Mulwhile/aiccbsgdoo/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_2
while/aiccbsgdoo/TanhTanhwhile/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh¢
while/aiccbsgdoo/mul_3Mulwhile/aiccbsgdoo/Sigmoid:y:0while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_3£
while/aiccbsgdoo/add_3AddV2while/aiccbsgdoo/mul_2:z:0while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_3¯
!while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp,while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_2°
while/aiccbsgdoo/mul_4Mul)while/aiccbsgdoo/ReadVariableOp_2:value:0while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_4¨
while/aiccbsgdoo/add_4AddV2while/aiccbsgdoo/split:output:3while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_4
while/aiccbsgdoo/Sigmoid_2Sigmoidwhile/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_2
while/aiccbsgdoo/Tanh_1Tanhwhile/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh_1¦
while/aiccbsgdoo/mul_5Mulwhile/aiccbsgdoo/Sigmoid_2:y:0while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aiccbsgdoo/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aiccbsgdoo/mul_5:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aiccbsgdoo/add_3:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aiccbsgdoo_biasadd_readvariableop_resource2while_aiccbsgdoo_biasadd_readvariableop_resource_0"h
1while_aiccbsgdoo_matmul_1_readvariableop_resource3while_aiccbsgdoo_matmul_1_readvariableop_resource_0"d
/while_aiccbsgdoo_matmul_readvariableop_resource1while_aiccbsgdoo_matmul_readvariableop_resource_0"Z
*while_aiccbsgdoo_readvariableop_1_resource,while_aiccbsgdoo_readvariableop_1_resource_0"Z
*while_aiccbsgdoo_readvariableop_2_resource,while_aiccbsgdoo_readvariableop_2_resource_0"V
(while_aiccbsgdoo_readvariableop_resource*while_aiccbsgdoo_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/aiccbsgdoo/BiasAdd/ReadVariableOp'while/aiccbsgdoo/BiasAdd/ReadVariableOp2P
&while/aiccbsgdoo/MatMul/ReadVariableOp&while/aiccbsgdoo/MatMul/ReadVariableOp2T
(while/aiccbsgdoo/MatMul_1/ReadVariableOp(while/aiccbsgdoo/MatMul_1/ReadVariableOp2B
while/aiccbsgdoo/ReadVariableOpwhile/aiccbsgdoo/ReadVariableOp2F
!while/aiccbsgdoo/ReadVariableOp_1!while/aiccbsgdoo/ReadVariableOp_12F
!while/aiccbsgdoo/ReadVariableOp_2!while/aiccbsgdoo/ReadVariableOp_2: 
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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1722401

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
ð$
­
G__inference_sequential_layer_call_and_return_conditional_losses_1718877

inputs(
mopvqfaljf_1718409: 
mopvqfaljf_1718411:(
ehhqjcwuju_1718454: 
ehhqjcwuju_1718456:%
cxhqebqpjz_1718654:	%
cxhqebqpjz_1718656:	 !
cxhqebqpjz_1718658:	 
cxhqebqpjz_1718660:  
cxhqebqpjz_1718662:  
cxhqebqpjz_1718664: %
qzuziqqdld_1718847:	 %
qzuziqqdld_1718849:	 !
qzuziqqdld_1718851:	 
qzuziqqdld_1718853:  
qzuziqqdld_1718855:  
qzuziqqdld_1718857: $
pbmomrqadp_1718871:  
pbmomrqadp_1718873:
identity¢"cxhqebqpjz/StatefulPartitionedCall¢"ehhqjcwuju/StatefulPartitionedCall¢"mopvqfaljf/StatefulPartitionedCall¢"pbmomrqadp/StatefulPartitionedCall¢"qzuziqqdld/StatefulPartitionedCall¬
"mopvqfaljf/StatefulPartitionedCallStatefulPartitionedCallinputsmopvqfaljf_1718409mopvqfaljf_1718411*
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
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_17184082$
"mopvqfaljf/StatefulPartitionedCallÑ
"ehhqjcwuju/StatefulPartitionedCallStatefulPartitionedCall+mopvqfaljf/StatefulPartitionedCall:output:0ehhqjcwuju_1718454ehhqjcwuju_1718456*
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
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_17184532$
"ehhqjcwuju/StatefulPartitionedCall
abbthhzbau/PartitionedCallPartitionedCall+ehhqjcwuju/StatefulPartitionedCall:output:0*
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
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_17184722
abbthhzbau/PartitionedCall
"cxhqebqpjz/StatefulPartitionedCallStatefulPartitionedCall#abbthhzbau/PartitionedCall:output:0cxhqebqpjz_1718654cxhqebqpjz_1718656cxhqebqpjz_1718658cxhqebqpjz_1718660cxhqebqpjz_1718662cxhqebqpjz_1718664*
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17186532$
"cxhqebqpjz/StatefulPartitionedCall¡
"qzuziqqdld/StatefulPartitionedCallStatefulPartitionedCall+cxhqebqpjz/StatefulPartitionedCall:output:0qzuziqqdld_1718847qzuziqqdld_1718849qzuziqqdld_1718851qzuziqqdld_1718853qzuziqqdld_1718855qzuziqqdld_1718857*
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17188462$
"qzuziqqdld/StatefulPartitionedCallÉ
"pbmomrqadp/StatefulPartitionedCallStatefulPartitionedCall+qzuziqqdld/StatefulPartitionedCall:output:0pbmomrqadp_1718871pbmomrqadp_1718873*
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
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_17188702$
"pbmomrqadp/StatefulPartitionedCall¸
IdentityIdentity+pbmomrqadp/StatefulPartitionedCall:output:0#^cxhqebqpjz/StatefulPartitionedCall#^ehhqjcwuju/StatefulPartitionedCall#^mopvqfaljf/StatefulPartitionedCall#^pbmomrqadp/StatefulPartitionedCall#^qzuziqqdld/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2H
"cxhqebqpjz/StatefulPartitionedCall"cxhqebqpjz/StatefulPartitionedCall2H
"ehhqjcwuju/StatefulPartitionedCall"ehhqjcwuju/StatefulPartitionedCall2H
"mopvqfaljf/StatefulPartitionedCall"mopvqfaljf/StatefulPartitionedCall2H
"pbmomrqadp/StatefulPartitionedCall"pbmomrqadp/StatefulPartitionedCall2H
"qzuziqqdld/StatefulPartitionedCall"qzuziqqdld/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦h

G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1722270

inputs<
)aiccbsgdoo_matmul_readvariableop_resource:	 >
+aiccbsgdoo_matmul_1_readvariableop_resource:	 9
*aiccbsgdoo_biasadd_readvariableop_resource:	0
"aiccbsgdoo_readvariableop_resource: 2
$aiccbsgdoo_readvariableop_1_resource: 2
$aiccbsgdoo_readvariableop_2_resource: 
identity¢!aiccbsgdoo/BiasAdd/ReadVariableOp¢ aiccbsgdoo/MatMul/ReadVariableOp¢"aiccbsgdoo/MatMul_1/ReadVariableOp¢aiccbsgdoo/ReadVariableOp¢aiccbsgdoo/ReadVariableOp_1¢aiccbsgdoo/ReadVariableOp_2¢whileD
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
 aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp)aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aiccbsgdoo/MatMul/ReadVariableOp§
aiccbsgdoo/MatMulMatMulstrided_slice_2:output:0(aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMulµ
"aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp+aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aiccbsgdoo/MatMul_1/ReadVariableOp£
aiccbsgdoo/MatMul_1MatMulzeros:output:0*aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMul_1
aiccbsgdoo/addAddV2aiccbsgdoo/MatMul:product:0aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/add®
!aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp*aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aiccbsgdoo/BiasAdd/ReadVariableOp¥
aiccbsgdoo/BiasAddBiasAddaiccbsgdoo/add:z:0)aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/BiasAddz
aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aiccbsgdoo/split/split_dimë
aiccbsgdoo/splitSplit#aiccbsgdoo/split/split_dim:output:0aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aiccbsgdoo/split
aiccbsgdoo/ReadVariableOpReadVariableOp"aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp
aiccbsgdoo/mulMul!aiccbsgdoo/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul
aiccbsgdoo/add_1AddV2aiccbsgdoo/split:output:0aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_1{
aiccbsgdoo/SigmoidSigmoidaiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid
aiccbsgdoo/ReadVariableOp_1ReadVariableOp$aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_1
aiccbsgdoo/mul_1Mul#aiccbsgdoo/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_1
aiccbsgdoo/add_2AddV2aiccbsgdoo/split:output:1aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_2
aiccbsgdoo/Sigmoid_1Sigmoidaiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_1
aiccbsgdoo/mul_2Mulaiccbsgdoo/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_2w
aiccbsgdoo/TanhTanhaiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh
aiccbsgdoo/mul_3Mulaiccbsgdoo/Sigmoid:y:0aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_3
aiccbsgdoo/add_3AddV2aiccbsgdoo/mul_2:z:0aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_3
aiccbsgdoo/ReadVariableOp_2ReadVariableOp$aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_2
aiccbsgdoo/mul_4Mul#aiccbsgdoo/ReadVariableOp_2:value:0aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_4
aiccbsgdoo/add_4AddV2aiccbsgdoo/split:output:3aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_4
aiccbsgdoo/Sigmoid_2Sigmoidaiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_2v
aiccbsgdoo/Tanh_1Tanhaiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh_1
aiccbsgdoo/mul_5Mulaiccbsgdoo/Sigmoid_2:y:0aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aiccbsgdoo_matmul_readvariableop_resource+aiccbsgdoo_matmul_1_readvariableop_resource*aiccbsgdoo_biasadd_readvariableop_resource"aiccbsgdoo_readvariableop_resource$aiccbsgdoo_readvariableop_1_resource$aiccbsgdoo_readvariableop_2_resource*
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
while_body_1722169*
condR
while_cond_1722168*Q
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
IdentityIdentitystrided_slice_3:output:0"^aiccbsgdoo/BiasAdd/ReadVariableOp!^aiccbsgdoo/MatMul/ReadVariableOp#^aiccbsgdoo/MatMul_1/ReadVariableOp^aiccbsgdoo/ReadVariableOp^aiccbsgdoo/ReadVariableOp_1^aiccbsgdoo/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aiccbsgdoo/BiasAdd/ReadVariableOp!aiccbsgdoo/BiasAdd/ReadVariableOp2D
 aiccbsgdoo/MatMul/ReadVariableOp aiccbsgdoo/MatMul/ReadVariableOp2H
"aiccbsgdoo/MatMul_1/ReadVariableOp"aiccbsgdoo/MatMul_1/ReadVariableOp26
aiccbsgdoo/ReadVariableOpaiccbsgdoo/ReadVariableOp2:
aiccbsgdoo/ReadVariableOp_1aiccbsgdoo/ReadVariableOp_12:
aiccbsgdoo/ReadVariableOp_2aiccbsgdoo/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
Ó
%__inference_signature_wrapper_1719690

gvxdqcynan
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	 
	unknown_5:	
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:	 

unknown_10:	 

unknown_11:	

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCall
gvxdqcynanunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_17168482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
gvxdqcynan
Û

,__inference_cxhqebqpjz_layer_call_fn_1721533

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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17186532
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
ü$
±
G__inference_sequential_layer_call_and_return_conditional_losses_1719595

gvxdqcynan(
mopvqfaljf_1719552: 
mopvqfaljf_1719554:(
ehhqjcwuju_1719557: 
ehhqjcwuju_1719559:%
cxhqebqpjz_1719563:	%
cxhqebqpjz_1719565:	 !
cxhqebqpjz_1719567:	 
cxhqebqpjz_1719569:  
cxhqebqpjz_1719571:  
cxhqebqpjz_1719573: %
qzuziqqdld_1719576:	 %
qzuziqqdld_1719578:	 !
qzuziqqdld_1719580:	 
qzuziqqdld_1719582:  
qzuziqqdld_1719584:  
qzuziqqdld_1719586: $
pbmomrqadp_1719589:  
pbmomrqadp_1719591:
identity¢"cxhqebqpjz/StatefulPartitionedCall¢"ehhqjcwuju/StatefulPartitionedCall¢"mopvqfaljf/StatefulPartitionedCall¢"pbmomrqadp/StatefulPartitionedCall¢"qzuziqqdld/StatefulPartitionedCall°
"mopvqfaljf/StatefulPartitionedCallStatefulPartitionedCall
gvxdqcynanmopvqfaljf_1719552mopvqfaljf_1719554*
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
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_17184082$
"mopvqfaljf/StatefulPartitionedCallÑ
"ehhqjcwuju/StatefulPartitionedCallStatefulPartitionedCall+mopvqfaljf/StatefulPartitionedCall:output:0ehhqjcwuju_1719557ehhqjcwuju_1719559*
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
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_17184532$
"ehhqjcwuju/StatefulPartitionedCall
abbthhzbau/PartitionedCallPartitionedCall+ehhqjcwuju/StatefulPartitionedCall:output:0*
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
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_17184722
abbthhzbau/PartitionedCall
"cxhqebqpjz/StatefulPartitionedCallStatefulPartitionedCall#abbthhzbau/PartitionedCall:output:0cxhqebqpjz_1719563cxhqebqpjz_1719565cxhqebqpjz_1719567cxhqebqpjz_1719569cxhqebqpjz_1719571cxhqebqpjz_1719573*
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17186532$
"cxhqebqpjz/StatefulPartitionedCall¡
"qzuziqqdld/StatefulPartitionedCallStatefulPartitionedCall+cxhqebqpjz/StatefulPartitionedCall:output:0qzuziqqdld_1719576qzuziqqdld_1719578qzuziqqdld_1719580qzuziqqdld_1719582qzuziqqdld_1719584qzuziqqdld_1719586*
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17188462$
"qzuziqqdld/StatefulPartitionedCallÉ
"pbmomrqadp/StatefulPartitionedCallStatefulPartitionedCall+qzuziqqdld/StatefulPartitionedCall:output:0pbmomrqadp_1719589pbmomrqadp_1719591*
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
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_17188702$
"pbmomrqadp/StatefulPartitionedCall¸
IdentityIdentity+pbmomrqadp/StatefulPartitionedCall:output:0#^cxhqebqpjz/StatefulPartitionedCall#^ehhqjcwuju/StatefulPartitionedCall#^mopvqfaljf/StatefulPartitionedCall#^pbmomrqadp/StatefulPartitionedCall#^qzuziqqdld/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2H
"cxhqebqpjz/StatefulPartitionedCall"cxhqebqpjz/StatefulPartitionedCall2H
"ehhqjcwuju/StatefulPartitionedCall"ehhqjcwuju/StatefulPartitionedCall2H
"mopvqfaljf/StatefulPartitionedCall"mopvqfaljf/StatefulPartitionedCall2H
"pbmomrqadp/StatefulPartitionedCall"pbmomrqadp/StatefulPartitionedCall2H
"qzuziqqdld/StatefulPartitionedCall"qzuziqqdld/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
gvxdqcynan
ç)
Ò
while_body_1716955
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_rvncypflgq_1716979_0:	-
while_rvncypflgq_1716981_0:	 )
while_rvncypflgq_1716983_0:	(
while_rvncypflgq_1716985_0: (
while_rvncypflgq_1716987_0: (
while_rvncypflgq_1716989_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_rvncypflgq_1716979:	+
while_rvncypflgq_1716981:	 '
while_rvncypflgq_1716983:	&
while_rvncypflgq_1716985: &
while_rvncypflgq_1716987: &
while_rvncypflgq_1716989: ¢(while/rvncypflgq/StatefulPartitionedCallÃ
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
(while/rvncypflgq/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_rvncypflgq_1716979_0while_rvncypflgq_1716981_0while_rvncypflgq_1716983_0while_rvncypflgq_1716985_0while_rvncypflgq_1716987_0while_rvncypflgq_1716989_0*
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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_17169352*
(while/rvncypflgq/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/rvncypflgq/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/rvncypflgq/StatefulPartitionedCall:output:1)^while/rvncypflgq/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/rvncypflgq/StatefulPartitionedCall:output:2)^while/rvncypflgq/StatefulPartitionedCall*
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
while_rvncypflgq_1716979while_rvncypflgq_1716979_0"6
while_rvncypflgq_1716981while_rvncypflgq_1716981_0"6
while_rvncypflgq_1716983while_rvncypflgq_1716983_0"6
while_rvncypflgq_1716985while_rvncypflgq_1716985_0"6
while_rvncypflgq_1716987while_rvncypflgq_1716987_0"6
while_rvncypflgq_1716989while_rvncypflgq_1716989_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/rvncypflgq/StatefulPartitionedCall(while/rvncypflgq/StatefulPartitionedCall: 
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
©0
¼
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_1720687

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
àY

while_body_1721201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_rvncypflgq_matmul_readvariableop_resource_0:	F
3while_rvncypflgq_matmul_1_readvariableop_resource_0:	 A
2while_rvncypflgq_biasadd_readvariableop_resource_0:	8
*while_rvncypflgq_readvariableop_resource_0: :
,while_rvncypflgq_readvariableop_1_resource_0: :
,while_rvncypflgq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_rvncypflgq_matmul_readvariableop_resource:	D
1while_rvncypflgq_matmul_1_readvariableop_resource:	 ?
0while_rvncypflgq_biasadd_readvariableop_resource:	6
(while_rvncypflgq_readvariableop_resource: 8
*while_rvncypflgq_readvariableop_1_resource: 8
*while_rvncypflgq_readvariableop_2_resource: ¢'while/rvncypflgq/BiasAdd/ReadVariableOp¢&while/rvncypflgq/MatMul/ReadVariableOp¢(while/rvncypflgq/MatMul_1/ReadVariableOp¢while/rvncypflgq/ReadVariableOp¢!while/rvncypflgq/ReadVariableOp_1¢!while/rvncypflgq/ReadVariableOp_2Ã
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
&while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp1while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/rvncypflgq/MatMul/ReadVariableOpÑ
while/rvncypflgq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMulÉ
(while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp3while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/rvncypflgq/MatMul_1/ReadVariableOpº
while/rvncypflgq/MatMul_1MatMulwhile_placeholder_20while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMul_1°
while/rvncypflgq/addAddV2!while/rvncypflgq/MatMul:product:0#while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/addÂ
'while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp2while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/rvncypflgq/BiasAdd/ReadVariableOp½
while/rvncypflgq/BiasAddBiasAddwhile/rvncypflgq/add:z:0/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/BiasAdd
 while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/rvncypflgq/split/split_dim
while/rvncypflgq/splitSplit)while/rvncypflgq/split/split_dim:output:0!while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/rvncypflgq/split©
while/rvncypflgq/ReadVariableOpReadVariableOp*while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/rvncypflgq/ReadVariableOp£
while/rvncypflgq/mulMul'while/rvncypflgq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul¦
while/rvncypflgq/add_1AddV2while/rvncypflgq/split:output:0while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_1
while/rvncypflgq/SigmoidSigmoidwhile/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid¯
!while/rvncypflgq/ReadVariableOp_1ReadVariableOp,while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_1©
while/rvncypflgq/mul_1Mul)while/rvncypflgq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_1¨
while/rvncypflgq/add_2AddV2while/rvncypflgq/split:output:1while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_2
while/rvncypflgq/Sigmoid_1Sigmoidwhile/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_1
while/rvncypflgq/mul_2Mulwhile/rvncypflgq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_2
while/rvncypflgq/TanhTanhwhile/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh¢
while/rvncypflgq/mul_3Mulwhile/rvncypflgq/Sigmoid:y:0while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_3£
while/rvncypflgq/add_3AddV2while/rvncypflgq/mul_2:z:0while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_3¯
!while/rvncypflgq/ReadVariableOp_2ReadVariableOp,while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_2°
while/rvncypflgq/mul_4Mul)while/rvncypflgq/ReadVariableOp_2:value:0while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_4¨
while/rvncypflgq/add_4AddV2while/rvncypflgq/split:output:3while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_4
while/rvncypflgq/Sigmoid_2Sigmoidwhile/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_2
while/rvncypflgq/Tanh_1Tanhwhile/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh_1¦
while/rvncypflgq/mul_5Mulwhile/rvncypflgq/Sigmoid_2:y:0while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/rvncypflgq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/rvncypflgq/mul_5:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/rvncypflgq/add_3:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
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
0while_rvncypflgq_biasadd_readvariableop_resource2while_rvncypflgq_biasadd_readvariableop_resource_0"h
1while_rvncypflgq_matmul_1_readvariableop_resource3while_rvncypflgq_matmul_1_readvariableop_resource_0"d
/while_rvncypflgq_matmul_readvariableop_resource1while_rvncypflgq_matmul_readvariableop_resource_0"Z
*while_rvncypflgq_readvariableop_1_resource,while_rvncypflgq_readvariableop_1_resource_0"Z
*while_rvncypflgq_readvariableop_2_resource,while_rvncypflgq_readvariableop_2_resource_0"V
(while_rvncypflgq_readvariableop_resource*while_rvncypflgq_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/rvncypflgq/BiasAdd/ReadVariableOp'while/rvncypflgq/BiasAdd/ReadVariableOp2P
&while/rvncypflgq/MatMul/ReadVariableOp&while/rvncypflgq/MatMul/ReadVariableOp2T
(while/rvncypflgq/MatMul_1/ReadVariableOp(while/rvncypflgq/MatMul_1/ReadVariableOp2B
while/rvncypflgq/ReadVariableOpwhile/rvncypflgq/ReadVariableOp2F
!while/rvncypflgq/ReadVariableOp_1!while/rvncypflgq/ReadVariableOp_12F
!while/rvncypflgq/ReadVariableOp_2!while/rvncypflgq/ReadVariableOp_2: 
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
while_body_1718745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aiccbsgdoo_matmul_readvariableop_resource_0:	 F
3while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 A
2while_aiccbsgdoo_biasadd_readvariableop_resource_0:	8
*while_aiccbsgdoo_readvariableop_resource_0: :
,while_aiccbsgdoo_readvariableop_1_resource_0: :
,while_aiccbsgdoo_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aiccbsgdoo_matmul_readvariableop_resource:	 D
1while_aiccbsgdoo_matmul_1_readvariableop_resource:	 ?
0while_aiccbsgdoo_biasadd_readvariableop_resource:	6
(while_aiccbsgdoo_readvariableop_resource: 8
*while_aiccbsgdoo_readvariableop_1_resource: 8
*while_aiccbsgdoo_readvariableop_2_resource: ¢'while/aiccbsgdoo/BiasAdd/ReadVariableOp¢&while/aiccbsgdoo/MatMul/ReadVariableOp¢(while/aiccbsgdoo/MatMul_1/ReadVariableOp¢while/aiccbsgdoo/ReadVariableOp¢!while/aiccbsgdoo/ReadVariableOp_1¢!while/aiccbsgdoo/ReadVariableOp_2Ã
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
&while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp1while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aiccbsgdoo/MatMul/ReadVariableOpÑ
while/aiccbsgdoo/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMulÉ
(while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp3while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aiccbsgdoo/MatMul_1/ReadVariableOpº
while/aiccbsgdoo/MatMul_1MatMulwhile_placeholder_20while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMul_1°
while/aiccbsgdoo/addAddV2!while/aiccbsgdoo/MatMul:product:0#while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/addÂ
'while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp2while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aiccbsgdoo/BiasAdd/ReadVariableOp½
while/aiccbsgdoo/BiasAddBiasAddwhile/aiccbsgdoo/add:z:0/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/BiasAdd
 while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aiccbsgdoo/split/split_dim
while/aiccbsgdoo/splitSplit)while/aiccbsgdoo/split/split_dim:output:0!while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aiccbsgdoo/split©
while/aiccbsgdoo/ReadVariableOpReadVariableOp*while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aiccbsgdoo/ReadVariableOp£
while/aiccbsgdoo/mulMul'while/aiccbsgdoo/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul¦
while/aiccbsgdoo/add_1AddV2while/aiccbsgdoo/split:output:0while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_1
while/aiccbsgdoo/SigmoidSigmoidwhile/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid¯
!while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp,while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_1©
while/aiccbsgdoo/mul_1Mul)while/aiccbsgdoo/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_1¨
while/aiccbsgdoo/add_2AddV2while/aiccbsgdoo/split:output:1while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_2
while/aiccbsgdoo/Sigmoid_1Sigmoidwhile/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_1
while/aiccbsgdoo/mul_2Mulwhile/aiccbsgdoo/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_2
while/aiccbsgdoo/TanhTanhwhile/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh¢
while/aiccbsgdoo/mul_3Mulwhile/aiccbsgdoo/Sigmoid:y:0while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_3£
while/aiccbsgdoo/add_3AddV2while/aiccbsgdoo/mul_2:z:0while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_3¯
!while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp,while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_2°
while/aiccbsgdoo/mul_4Mul)while/aiccbsgdoo/ReadVariableOp_2:value:0while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_4¨
while/aiccbsgdoo/add_4AddV2while/aiccbsgdoo/split:output:3while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_4
while/aiccbsgdoo/Sigmoid_2Sigmoidwhile/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_2
while/aiccbsgdoo/Tanh_1Tanhwhile/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh_1¦
while/aiccbsgdoo/mul_5Mulwhile/aiccbsgdoo/Sigmoid_2:y:0while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aiccbsgdoo/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aiccbsgdoo/mul_5:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aiccbsgdoo/add_3:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aiccbsgdoo_biasadd_readvariableop_resource2while_aiccbsgdoo_biasadd_readvariableop_resource_0"h
1while_aiccbsgdoo_matmul_1_readvariableop_resource3while_aiccbsgdoo_matmul_1_readvariableop_resource_0"d
/while_aiccbsgdoo_matmul_readvariableop_resource1while_aiccbsgdoo_matmul_readvariableop_resource_0"Z
*while_aiccbsgdoo_readvariableop_1_resource,while_aiccbsgdoo_readvariableop_1_resource_0"Z
*while_aiccbsgdoo_readvariableop_2_resource,while_aiccbsgdoo_readvariableop_2_resource_0"V
(while_aiccbsgdoo_readvariableop_resource*while_aiccbsgdoo_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/aiccbsgdoo/BiasAdd/ReadVariableOp'while/aiccbsgdoo/BiasAdd/ReadVariableOp2P
&while/aiccbsgdoo/MatMul/ReadVariableOp&while/aiccbsgdoo/MatMul/ReadVariableOp2T
(while/aiccbsgdoo/MatMul_1/ReadVariableOp(while/aiccbsgdoo/MatMul_1/ReadVariableOp2B
while/aiccbsgdoo/ReadVariableOpwhile/aiccbsgdoo/ReadVariableOp2F
!while/aiccbsgdoo/ReadVariableOp_1!while/aiccbsgdoo/ReadVariableOp_12F
!while/aiccbsgdoo/ReadVariableOp_2!while/aiccbsgdoo/ReadVariableOp_2: 
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
«2
¼
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_1720735

inputsA
+conv1d_expanddims_1_readvariableop_resource:@
2squeeze_batch_dims_biasadd_readvariableop_resource:
identity¢"conv1d/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
Pad/paddingsj
PadPadinputsPad/paddings:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Pady
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim 
conv1d/ExpandDims
ExpandDimsPad:output:0conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
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
:2
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
valueB"ÿÿÿÿ         2
conv1d/Reshape/shape 
conv1d/ReshapeReshapeconv1d/ExpandDims:output:0conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1721380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1721380___redundant_placeholder05
1while_while_cond_1721380___redundant_placeholder15
1while_while_cond_1721380___redundant_placeholder25
1while_while_cond_1721380___redundant_placeholder35
1while_while_cond_1721380___redundant_placeholder45
1while_while_cond_1721380___redundant_placeholder55
1while_while_cond_1721380___redundant_placeholder6
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
while_body_1721629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aiccbsgdoo_matmul_readvariableop_resource_0:	 F
3while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 A
2while_aiccbsgdoo_biasadd_readvariableop_resource_0:	8
*while_aiccbsgdoo_readvariableop_resource_0: :
,while_aiccbsgdoo_readvariableop_1_resource_0: :
,while_aiccbsgdoo_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aiccbsgdoo_matmul_readvariableop_resource:	 D
1while_aiccbsgdoo_matmul_1_readvariableop_resource:	 ?
0while_aiccbsgdoo_biasadd_readvariableop_resource:	6
(while_aiccbsgdoo_readvariableop_resource: 8
*while_aiccbsgdoo_readvariableop_1_resource: 8
*while_aiccbsgdoo_readvariableop_2_resource: ¢'while/aiccbsgdoo/BiasAdd/ReadVariableOp¢&while/aiccbsgdoo/MatMul/ReadVariableOp¢(while/aiccbsgdoo/MatMul_1/ReadVariableOp¢while/aiccbsgdoo/ReadVariableOp¢!while/aiccbsgdoo/ReadVariableOp_1¢!while/aiccbsgdoo/ReadVariableOp_2Ã
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
&while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp1while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aiccbsgdoo/MatMul/ReadVariableOpÑ
while/aiccbsgdoo/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMulÉ
(while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp3while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aiccbsgdoo/MatMul_1/ReadVariableOpº
while/aiccbsgdoo/MatMul_1MatMulwhile_placeholder_20while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/MatMul_1°
while/aiccbsgdoo/addAddV2!while/aiccbsgdoo/MatMul:product:0#while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/addÂ
'while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp2while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aiccbsgdoo/BiasAdd/ReadVariableOp½
while/aiccbsgdoo/BiasAddBiasAddwhile/aiccbsgdoo/add:z:0/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aiccbsgdoo/BiasAdd
 while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aiccbsgdoo/split/split_dim
while/aiccbsgdoo/splitSplit)while/aiccbsgdoo/split/split_dim:output:0!while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aiccbsgdoo/split©
while/aiccbsgdoo/ReadVariableOpReadVariableOp*while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aiccbsgdoo/ReadVariableOp£
while/aiccbsgdoo/mulMul'while/aiccbsgdoo/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul¦
while/aiccbsgdoo/add_1AddV2while/aiccbsgdoo/split:output:0while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_1
while/aiccbsgdoo/SigmoidSigmoidwhile/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid¯
!while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp,while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_1©
while/aiccbsgdoo/mul_1Mul)while/aiccbsgdoo/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_1¨
while/aiccbsgdoo/add_2AddV2while/aiccbsgdoo/split:output:1while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_2
while/aiccbsgdoo/Sigmoid_1Sigmoidwhile/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_1
while/aiccbsgdoo/mul_2Mulwhile/aiccbsgdoo/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_2
while/aiccbsgdoo/TanhTanhwhile/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh¢
while/aiccbsgdoo/mul_3Mulwhile/aiccbsgdoo/Sigmoid:y:0while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_3£
while/aiccbsgdoo/add_3AddV2while/aiccbsgdoo/mul_2:z:0while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_3¯
!while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp,while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aiccbsgdoo/ReadVariableOp_2°
while/aiccbsgdoo/mul_4Mul)while/aiccbsgdoo/ReadVariableOp_2:value:0while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_4¨
while/aiccbsgdoo/add_4AddV2while/aiccbsgdoo/split:output:3while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/add_4
while/aiccbsgdoo/Sigmoid_2Sigmoidwhile/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Sigmoid_2
while/aiccbsgdoo/Tanh_1Tanhwhile/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/Tanh_1¦
while/aiccbsgdoo/mul_5Mulwhile/aiccbsgdoo/Sigmoid_2:y:0while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aiccbsgdoo/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aiccbsgdoo/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aiccbsgdoo/mul_5:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aiccbsgdoo/add_3:z:0(^while/aiccbsgdoo/BiasAdd/ReadVariableOp'^while/aiccbsgdoo/MatMul/ReadVariableOp)^while/aiccbsgdoo/MatMul_1/ReadVariableOp ^while/aiccbsgdoo/ReadVariableOp"^while/aiccbsgdoo/ReadVariableOp_1"^while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aiccbsgdoo_biasadd_readvariableop_resource2while_aiccbsgdoo_biasadd_readvariableop_resource_0"h
1while_aiccbsgdoo_matmul_1_readvariableop_resource3while_aiccbsgdoo_matmul_1_readvariableop_resource_0"d
/while_aiccbsgdoo_matmul_readvariableop_resource1while_aiccbsgdoo_matmul_readvariableop_resource_0"Z
*while_aiccbsgdoo_readvariableop_1_resource,while_aiccbsgdoo_readvariableop_1_resource_0"Z
*while_aiccbsgdoo_readvariableop_2_resource,while_aiccbsgdoo_readvariableop_2_resource_0"V
(while_aiccbsgdoo_readvariableop_resource*while_aiccbsgdoo_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/aiccbsgdoo/BiasAdd/ReadVariableOp'while/aiccbsgdoo/BiasAdd/ReadVariableOp2P
&while/aiccbsgdoo/MatMul/ReadVariableOp&while/aiccbsgdoo/MatMul/ReadVariableOp2T
(while/aiccbsgdoo/MatMul_1/ReadVariableOp(while/aiccbsgdoo/MatMul_1/ReadVariableOp2B
while/aiccbsgdoo/ReadVariableOpwhile/aiccbsgdoo/ReadVariableOp2F
!while/aiccbsgdoo/ReadVariableOp_1!while/aiccbsgdoo/ReadVariableOp_12F
!while/aiccbsgdoo/ReadVariableOp_2!while/aiccbsgdoo/ReadVariableOp_2: 
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
Ê

G__inference_sequential_layer_call_and_return_conditional_losses_1720568

inputsL
6mopvqfaljf_conv1d_expanddims_1_readvariableop_resource:K
=mopvqfaljf_squeeze_batch_dims_biasadd_readvariableop_resource:L
6ehhqjcwuju_conv1d_expanddims_1_readvariableop_resource:K
=ehhqjcwuju_squeeze_batch_dims_biasadd_readvariableop_resource:G
4cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource:	I
6cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource:	 D
5cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource:	;
-cxhqebqpjz_rvncypflgq_readvariableop_resource: =
/cxhqebqpjz_rvncypflgq_readvariableop_1_resource: =
/cxhqebqpjz_rvncypflgq_readvariableop_2_resource: G
4qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource:	 I
6qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource:	 D
5qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource:	;
-qzuziqqdld_aiccbsgdoo_readvariableop_resource: =
/qzuziqqdld_aiccbsgdoo_readvariableop_1_resource: =
/qzuziqqdld_aiccbsgdoo_readvariableop_2_resource: ;
)pbmomrqadp_matmul_readvariableop_resource: 8
*pbmomrqadp_biasadd_readvariableop_resource:
identity¢,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp¢+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp¢-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp¢$cxhqebqpjz/rvncypflgq/ReadVariableOp¢&cxhqebqpjz/rvncypflgq/ReadVariableOp_1¢&cxhqebqpjz/rvncypflgq/ReadVariableOp_2¢cxhqebqpjz/while¢-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp¢4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp¢-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp¢4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!pbmomrqadp/BiasAdd/ReadVariableOp¢ pbmomrqadp/MatMul/ReadVariableOp¢,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp¢+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp¢-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp¢$qzuziqqdld/aiccbsgdoo/ReadVariableOp¢&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1¢&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2¢qzuziqqdld/while
 mopvqfaljf/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 mopvqfaljf/conv1d/ExpandDims/dim»
mopvqfaljf/conv1d/ExpandDims
ExpandDimsinputs)mopvqfaljf/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/ExpandDimsÙ
-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6mopvqfaljf_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp
"mopvqfaljf/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"mopvqfaljf/conv1d/ExpandDims_1/dimã
mopvqfaljf/conv1d/ExpandDims_1
ExpandDims5mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp:value:0+mopvqfaljf/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
mopvqfaljf/conv1d/ExpandDims_1
mopvqfaljf/conv1d/ShapeShape%mopvqfaljf/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
mopvqfaljf/conv1d/Shape
%mopvqfaljf/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%mopvqfaljf/conv1d/strided_slice/stack¥
'mopvqfaljf/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'mopvqfaljf/conv1d/strided_slice/stack_1
'mopvqfaljf/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'mopvqfaljf/conv1d/strided_slice/stack_2Ì
mopvqfaljf/conv1d/strided_sliceStridedSlice mopvqfaljf/conv1d/Shape:output:0.mopvqfaljf/conv1d/strided_slice/stack:output:00mopvqfaljf/conv1d/strided_slice/stack_1:output:00mopvqfaljf/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
mopvqfaljf/conv1d/strided_slice
mopvqfaljf/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
mopvqfaljf/conv1d/Reshape/shapeÌ
mopvqfaljf/conv1d/ReshapeReshape%mopvqfaljf/conv1d/ExpandDims:output:0(mopvqfaljf/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/Reshapeî
mopvqfaljf/conv1d/Conv2DConv2D"mopvqfaljf/conv1d/Reshape:output:0'mopvqfaljf/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
mopvqfaljf/conv1d/Conv2D
!mopvqfaljf/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!mopvqfaljf/conv1d/concat/values_1
mopvqfaljf/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/concat/axisì
mopvqfaljf/conv1d/concatConcatV2(mopvqfaljf/conv1d/strided_slice:output:0*mopvqfaljf/conv1d/concat/values_1:output:0&mopvqfaljf/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
mopvqfaljf/conv1d/concatÉ
mopvqfaljf/conv1d/Reshape_1Reshape!mopvqfaljf/conv1d/Conv2D:output:0!mopvqfaljf/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/Reshape_1Á
mopvqfaljf/conv1d/SqueezeSqueeze$mopvqfaljf/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
mopvqfaljf/conv1d/Squeeze
#mopvqfaljf/squeeze_batch_dims/ShapeShape"mopvqfaljf/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#mopvqfaljf/squeeze_batch_dims/Shape°
1mopvqfaljf/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1mopvqfaljf/squeeze_batch_dims/strided_slice/stack½
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1´
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2
+mopvqfaljf/squeeze_batch_dims/strided_sliceStridedSlice,mopvqfaljf/squeeze_batch_dims/Shape:output:0:mopvqfaljf/squeeze_batch_dims/strided_slice/stack:output:0<mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1:output:0<mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+mopvqfaljf/squeeze_batch_dims/strided_slice¯
+mopvqfaljf/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+mopvqfaljf/squeeze_batch_dims/Reshape/shapeé
%mopvqfaljf/squeeze_batch_dims/ReshapeReshape"mopvqfaljf/conv1d/Squeeze:output:04mopvqfaljf/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%mopvqfaljf/squeeze_batch_dims/Reshapeæ
4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=mopvqfaljf_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%mopvqfaljf/squeeze_batch_dims/BiasAddBiasAdd.mopvqfaljf/squeeze_batch_dims/Reshape:output:0<mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%mopvqfaljf/squeeze_batch_dims/BiasAdd¯
-mopvqfaljf/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-mopvqfaljf/squeeze_batch_dims/concat/values_1¡
)mopvqfaljf/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)mopvqfaljf/squeeze_batch_dims/concat/axis¨
$mopvqfaljf/squeeze_batch_dims/concatConcatV24mopvqfaljf/squeeze_batch_dims/strided_slice:output:06mopvqfaljf/squeeze_batch_dims/concat/values_1:output:02mopvqfaljf/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$mopvqfaljf/squeeze_batch_dims/concatö
'mopvqfaljf/squeeze_batch_dims/Reshape_1Reshape.mopvqfaljf/squeeze_batch_dims/BiasAdd:output:0-mopvqfaljf/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'mopvqfaljf/squeeze_batch_dims/Reshape_1£
ehhqjcwuju/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2
ehhqjcwuju/Pad/paddingsµ
ehhqjcwuju/PadPad0mopvqfaljf/squeeze_batch_dims/Reshape_1:output:0 ehhqjcwuju/Pad/paddings:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/Pad
 ehhqjcwuju/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 ehhqjcwuju/conv1d/ExpandDims/dimÌ
ehhqjcwuju/conv1d/ExpandDims
ExpandDimsehhqjcwuju/Pad:output:0)ehhqjcwuju/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/ExpandDimsÙ
-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6ehhqjcwuju_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp
"ehhqjcwuju/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ehhqjcwuju/conv1d/ExpandDims_1/dimã
ehhqjcwuju/conv1d/ExpandDims_1
ExpandDims5ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp:value:0+ehhqjcwuju/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
ehhqjcwuju/conv1d/ExpandDims_1
ehhqjcwuju/conv1d/ShapeShape%ehhqjcwuju/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
ehhqjcwuju/conv1d/Shape
%ehhqjcwuju/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%ehhqjcwuju/conv1d/strided_slice/stack¥
'ehhqjcwuju/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'ehhqjcwuju/conv1d/strided_slice/stack_1
'ehhqjcwuju/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'ehhqjcwuju/conv1d/strided_slice/stack_2Ì
ehhqjcwuju/conv1d/strided_sliceStridedSlice ehhqjcwuju/conv1d/Shape:output:0.ehhqjcwuju/conv1d/strided_slice/stack:output:00ehhqjcwuju/conv1d/strided_slice/stack_1:output:00ehhqjcwuju/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
ehhqjcwuju/conv1d/strided_slice
ehhqjcwuju/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
ehhqjcwuju/conv1d/Reshape/shapeÌ
ehhqjcwuju/conv1d/ReshapeReshape%ehhqjcwuju/conv1d/ExpandDims:output:0(ehhqjcwuju/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/Reshapeî
ehhqjcwuju/conv1d/Conv2DConv2D"ehhqjcwuju/conv1d/Reshape:output:0'ehhqjcwuju/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ehhqjcwuju/conv1d/Conv2D
!ehhqjcwuju/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!ehhqjcwuju/conv1d/concat/values_1
ehhqjcwuju/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/concat/axisì
ehhqjcwuju/conv1d/concatConcatV2(ehhqjcwuju/conv1d/strided_slice:output:0*ehhqjcwuju/conv1d/concat/values_1:output:0&ehhqjcwuju/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ehhqjcwuju/conv1d/concatÉ
ehhqjcwuju/conv1d/Reshape_1Reshape!ehhqjcwuju/conv1d/Conv2D:output:0!ehhqjcwuju/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/Reshape_1Á
ehhqjcwuju/conv1d/SqueezeSqueeze$ehhqjcwuju/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
ehhqjcwuju/conv1d/Squeeze
#ehhqjcwuju/squeeze_batch_dims/ShapeShape"ehhqjcwuju/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#ehhqjcwuju/squeeze_batch_dims/Shape°
1ehhqjcwuju/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1ehhqjcwuju/squeeze_batch_dims/strided_slice/stack½
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1´
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2
+ehhqjcwuju/squeeze_batch_dims/strided_sliceStridedSlice,ehhqjcwuju/squeeze_batch_dims/Shape:output:0:ehhqjcwuju/squeeze_batch_dims/strided_slice/stack:output:0<ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1:output:0<ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+ehhqjcwuju/squeeze_batch_dims/strided_slice¯
+ehhqjcwuju/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+ehhqjcwuju/squeeze_batch_dims/Reshape/shapeé
%ehhqjcwuju/squeeze_batch_dims/ReshapeReshape"ehhqjcwuju/conv1d/Squeeze:output:04ehhqjcwuju/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ehhqjcwuju/squeeze_batch_dims/Reshapeæ
4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=ehhqjcwuju_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%ehhqjcwuju/squeeze_batch_dims/BiasAddBiasAdd.ehhqjcwuju/squeeze_batch_dims/Reshape:output:0<ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ehhqjcwuju/squeeze_batch_dims/BiasAdd¯
-ehhqjcwuju/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-ehhqjcwuju/squeeze_batch_dims/concat/values_1¡
)ehhqjcwuju/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)ehhqjcwuju/squeeze_batch_dims/concat/axis¨
$ehhqjcwuju/squeeze_batch_dims/concatConcatV24ehhqjcwuju/squeeze_batch_dims/strided_slice:output:06ehhqjcwuju/squeeze_batch_dims/concat/values_1:output:02ehhqjcwuju/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$ehhqjcwuju/squeeze_batch_dims/concatö
'ehhqjcwuju/squeeze_batch_dims/Reshape_1Reshape.ehhqjcwuju/squeeze_batch_dims/BiasAdd:output:0-ehhqjcwuju/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'ehhqjcwuju/squeeze_batch_dims/Reshape_1
abbthhzbau/ShapeShape0ehhqjcwuju/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
abbthhzbau/Shape
abbthhzbau/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
abbthhzbau/strided_slice/stack
 abbthhzbau/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 abbthhzbau/strided_slice/stack_1
 abbthhzbau/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 abbthhzbau/strided_slice/stack_2¤
abbthhzbau/strided_sliceStridedSliceabbthhzbau/Shape:output:0'abbthhzbau/strided_slice/stack:output:0)abbthhzbau/strided_slice/stack_1:output:0)abbthhzbau/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
abbthhzbau/strided_slicez
abbthhzbau/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
abbthhzbau/Reshape/shape/1z
abbthhzbau/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
abbthhzbau/Reshape/shape/2×
abbthhzbau/Reshape/shapePack!abbthhzbau/strided_slice:output:0#abbthhzbau/Reshape/shape/1:output:0#abbthhzbau/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
abbthhzbau/Reshape/shape¾
abbthhzbau/ReshapeReshape0ehhqjcwuju/squeeze_batch_dims/Reshape_1:output:0!abbthhzbau/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
abbthhzbau/Reshapeo
cxhqebqpjz/ShapeShapeabbthhzbau/Reshape:output:0*
T0*
_output_shapes
:2
cxhqebqpjz/Shape
cxhqebqpjz/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
cxhqebqpjz/strided_slice/stack
 cxhqebqpjz/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 cxhqebqpjz/strided_slice/stack_1
 cxhqebqpjz/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 cxhqebqpjz/strided_slice/stack_2¤
cxhqebqpjz/strided_sliceStridedSlicecxhqebqpjz/Shape:output:0'cxhqebqpjz/strided_slice/stack:output:0)cxhqebqpjz/strided_slice/stack_1:output:0)cxhqebqpjz/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cxhqebqpjz/strided_slicer
cxhqebqpjz/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros/mul/y
cxhqebqpjz/zeros/mulMul!cxhqebqpjz/strided_slice:output:0cxhqebqpjz/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros/mulu
cxhqebqpjz/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
cxhqebqpjz/zeros/Less/y
cxhqebqpjz/zeros/LessLesscxhqebqpjz/zeros/mul:z:0 cxhqebqpjz/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros/Lessx
cxhqebqpjz/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros/packed/1¯
cxhqebqpjz/zeros/packedPack!cxhqebqpjz/strided_slice:output:0"cxhqebqpjz/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
cxhqebqpjz/zeros/packedu
cxhqebqpjz/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
cxhqebqpjz/zeros/Const¡
cxhqebqpjz/zerosFill cxhqebqpjz/zeros/packed:output:0cxhqebqpjz/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/zerosv
cxhqebqpjz/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros_1/mul/y
cxhqebqpjz/zeros_1/mulMul!cxhqebqpjz/strided_slice:output:0!cxhqebqpjz/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros_1/muly
cxhqebqpjz/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
cxhqebqpjz/zeros_1/Less/y
cxhqebqpjz/zeros_1/LessLesscxhqebqpjz/zeros_1/mul:z:0"cxhqebqpjz/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/zeros_1/Less|
cxhqebqpjz/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/zeros_1/packed/1µ
cxhqebqpjz/zeros_1/packedPack!cxhqebqpjz/strided_slice:output:0$cxhqebqpjz/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
cxhqebqpjz/zeros_1/packedy
cxhqebqpjz/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
cxhqebqpjz/zeros_1/Const©
cxhqebqpjz/zeros_1Fill"cxhqebqpjz/zeros_1/packed:output:0!cxhqebqpjz/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/zeros_1
cxhqebqpjz/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cxhqebqpjz/transpose/perm°
cxhqebqpjz/transpose	Transposeabbthhzbau/Reshape:output:0"cxhqebqpjz/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/transposep
cxhqebqpjz/Shape_1Shapecxhqebqpjz/transpose:y:0*
T0*
_output_shapes
:2
cxhqebqpjz/Shape_1
 cxhqebqpjz/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 cxhqebqpjz/strided_slice_1/stack
"cxhqebqpjz/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_1/stack_1
"cxhqebqpjz/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_1/stack_2°
cxhqebqpjz/strided_slice_1StridedSlicecxhqebqpjz/Shape_1:output:0)cxhqebqpjz/strided_slice_1/stack:output:0+cxhqebqpjz/strided_slice_1/stack_1:output:0+cxhqebqpjz/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cxhqebqpjz/strided_slice_1
&cxhqebqpjz/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&cxhqebqpjz/TensorArrayV2/element_shapeÞ
cxhqebqpjz/TensorArrayV2TensorListReserve/cxhqebqpjz/TensorArrayV2/element_shape:output:0#cxhqebqpjz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cxhqebqpjz/TensorArrayV2Õ
@cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2cxhqebqpjz/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcxhqebqpjz/transpose:y:0Icxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor
 cxhqebqpjz/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 cxhqebqpjz/strided_slice_2/stack
"cxhqebqpjz/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_2/stack_1
"cxhqebqpjz/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_2/stack_2¾
cxhqebqpjz/strided_slice_2StridedSlicecxhqebqpjz/transpose:y:0)cxhqebqpjz/strided_slice_2/stack:output:0+cxhqebqpjz/strided_slice_2/stack_1:output:0+cxhqebqpjz/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
cxhqebqpjz/strided_slice_2Ð
+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOpReadVariableOp4cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOpÓ
cxhqebqpjz/rvncypflgq/MatMulMatMul#cxhqebqpjz/strided_slice_2:output:03cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/rvncypflgq/MatMulÖ
-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp6cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOpÏ
cxhqebqpjz/rvncypflgq/MatMul_1MatMulcxhqebqpjz/zeros:output:05cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
cxhqebqpjz/rvncypflgq/MatMul_1Ä
cxhqebqpjz/rvncypflgq/addAddV2&cxhqebqpjz/rvncypflgq/MatMul:product:0(cxhqebqpjz/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/rvncypflgq/addÏ
,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp5cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOpÑ
cxhqebqpjz/rvncypflgq/BiasAddBiasAddcxhqebqpjz/rvncypflgq/add:z:04cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
cxhqebqpjz/rvncypflgq/BiasAdd
%cxhqebqpjz/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%cxhqebqpjz/rvncypflgq/split/split_dim
cxhqebqpjz/rvncypflgq/splitSplit.cxhqebqpjz/rvncypflgq/split/split_dim:output:0&cxhqebqpjz/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
cxhqebqpjz/rvncypflgq/split¶
$cxhqebqpjz/rvncypflgq/ReadVariableOpReadVariableOp-cxhqebqpjz_rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02&
$cxhqebqpjz/rvncypflgq/ReadVariableOpº
cxhqebqpjz/rvncypflgq/mulMul,cxhqebqpjz/rvncypflgq/ReadVariableOp:value:0cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mulº
cxhqebqpjz/rvncypflgq/add_1AddV2$cxhqebqpjz/rvncypflgq/split:output:0cxhqebqpjz/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_1
cxhqebqpjz/rvncypflgq/SigmoidSigmoidcxhqebqpjz/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/Sigmoid¼
&cxhqebqpjz/rvncypflgq/ReadVariableOp_1ReadVariableOp/cxhqebqpjz_rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&cxhqebqpjz/rvncypflgq/ReadVariableOp_1À
cxhqebqpjz/rvncypflgq/mul_1Mul.cxhqebqpjz/rvncypflgq/ReadVariableOp_1:value:0cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_1¼
cxhqebqpjz/rvncypflgq/add_2AddV2$cxhqebqpjz/rvncypflgq/split:output:1cxhqebqpjz/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_2 
cxhqebqpjz/rvncypflgq/Sigmoid_1Sigmoidcxhqebqpjz/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
cxhqebqpjz/rvncypflgq/Sigmoid_1µ
cxhqebqpjz/rvncypflgq/mul_2Mul#cxhqebqpjz/rvncypflgq/Sigmoid_1:y:0cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_2
cxhqebqpjz/rvncypflgq/TanhTanh$cxhqebqpjz/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/Tanh¶
cxhqebqpjz/rvncypflgq/mul_3Mul!cxhqebqpjz/rvncypflgq/Sigmoid:y:0cxhqebqpjz/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_3·
cxhqebqpjz/rvncypflgq/add_3AddV2cxhqebqpjz/rvncypflgq/mul_2:z:0cxhqebqpjz/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_3¼
&cxhqebqpjz/rvncypflgq/ReadVariableOp_2ReadVariableOp/cxhqebqpjz_rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&cxhqebqpjz/rvncypflgq/ReadVariableOp_2Ä
cxhqebqpjz/rvncypflgq/mul_4Mul.cxhqebqpjz/rvncypflgq/ReadVariableOp_2:value:0cxhqebqpjz/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_4¼
cxhqebqpjz/rvncypflgq/add_4AddV2$cxhqebqpjz/rvncypflgq/split:output:3cxhqebqpjz/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/add_4 
cxhqebqpjz/rvncypflgq/Sigmoid_2Sigmoidcxhqebqpjz/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
cxhqebqpjz/rvncypflgq/Sigmoid_2
cxhqebqpjz/rvncypflgq/Tanh_1Tanhcxhqebqpjz/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/Tanh_1º
cxhqebqpjz/rvncypflgq/mul_5Mul#cxhqebqpjz/rvncypflgq/Sigmoid_2:y:0 cxhqebqpjz/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/rvncypflgq/mul_5¥
(cxhqebqpjz/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(cxhqebqpjz/TensorArrayV2_1/element_shapeä
cxhqebqpjz/TensorArrayV2_1TensorListReserve1cxhqebqpjz/TensorArrayV2_1/element_shape:output:0#cxhqebqpjz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cxhqebqpjz/TensorArrayV2_1d
cxhqebqpjz/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/time
#cxhqebqpjz/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#cxhqebqpjz/while/maximum_iterations
cxhqebqpjz/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cxhqebqpjz/while/loop_counter²
cxhqebqpjz/whileWhile&cxhqebqpjz/while/loop_counter:output:0,cxhqebqpjz/while/maximum_iterations:output:0cxhqebqpjz/time:output:0#cxhqebqpjz/TensorArrayV2_1:handle:0cxhqebqpjz/zeros:output:0cxhqebqpjz/zeros_1:output:0#cxhqebqpjz/strided_slice_1:output:0Bcxhqebqpjz/TensorArrayUnstack/TensorListFromTensor:output_handle:04cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource6cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource5cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource-cxhqebqpjz_rvncypflgq_readvariableop_resource/cxhqebqpjz_rvncypflgq_readvariableop_1_resource/cxhqebqpjz_rvncypflgq_readvariableop_2_resource*
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
cxhqebqpjz_while_body_1720285*)
cond!R
cxhqebqpjz_while_cond_1720284*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
cxhqebqpjz/whileË
;cxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;cxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shape
-cxhqebqpjz/TensorArrayV2Stack/TensorListStackTensorListStackcxhqebqpjz/while:output:3Dcxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-cxhqebqpjz/TensorArrayV2Stack/TensorListStack
 cxhqebqpjz/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 cxhqebqpjz/strided_slice_3/stack
"cxhqebqpjz/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"cxhqebqpjz/strided_slice_3/stack_1
"cxhqebqpjz/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"cxhqebqpjz/strided_slice_3/stack_2Ü
cxhqebqpjz/strided_slice_3StridedSlice6cxhqebqpjz/TensorArrayV2Stack/TensorListStack:tensor:0)cxhqebqpjz/strided_slice_3/stack:output:0+cxhqebqpjz/strided_slice_3/stack_1:output:0+cxhqebqpjz/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
cxhqebqpjz/strided_slice_3
cxhqebqpjz/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cxhqebqpjz/transpose_1/permÑ
cxhqebqpjz/transpose_1	Transpose6cxhqebqpjz/TensorArrayV2Stack/TensorListStack:tensor:0$cxhqebqpjz/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/transpose_1n
qzuziqqdld/ShapeShapecxhqebqpjz/transpose_1:y:0*
T0*
_output_shapes
:2
qzuziqqdld/Shape
qzuziqqdld/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
qzuziqqdld/strided_slice/stack
 qzuziqqdld/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 qzuziqqdld/strided_slice/stack_1
 qzuziqqdld/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 qzuziqqdld/strided_slice/stack_2¤
qzuziqqdld/strided_sliceStridedSliceqzuziqqdld/Shape:output:0'qzuziqqdld/strided_slice/stack:output:0)qzuziqqdld/strided_slice/stack_1:output:0)qzuziqqdld/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
qzuziqqdld/strided_slicer
qzuziqqdld/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros/mul/y
qzuziqqdld/zeros/mulMul!qzuziqqdld/strided_slice:output:0qzuziqqdld/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros/mulu
qzuziqqdld/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
qzuziqqdld/zeros/Less/y
qzuziqqdld/zeros/LessLessqzuziqqdld/zeros/mul:z:0 qzuziqqdld/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros/Lessx
qzuziqqdld/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros/packed/1¯
qzuziqqdld/zeros/packedPack!qzuziqqdld/strided_slice:output:0"qzuziqqdld/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
qzuziqqdld/zeros/packedu
qzuziqqdld/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
qzuziqqdld/zeros/Const¡
qzuziqqdld/zerosFill qzuziqqdld/zeros/packed:output:0qzuziqqdld/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/zerosv
qzuziqqdld/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros_1/mul/y
qzuziqqdld/zeros_1/mulMul!qzuziqqdld/strided_slice:output:0!qzuziqqdld/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros_1/muly
qzuziqqdld/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
qzuziqqdld/zeros_1/Less/y
qzuziqqdld/zeros_1/LessLessqzuziqqdld/zeros_1/mul:z:0"qzuziqqdld/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/zeros_1/Less|
qzuziqqdld/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/zeros_1/packed/1µ
qzuziqqdld/zeros_1/packedPack!qzuziqqdld/strided_slice:output:0$qzuziqqdld/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
qzuziqqdld/zeros_1/packedy
qzuziqqdld/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
qzuziqqdld/zeros_1/Const©
qzuziqqdld/zeros_1Fill"qzuziqqdld/zeros_1/packed:output:0!qzuziqqdld/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/zeros_1
qzuziqqdld/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
qzuziqqdld/transpose/perm¯
qzuziqqdld/transpose	Transposecxhqebqpjz/transpose_1:y:0"qzuziqqdld/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/transposep
qzuziqqdld/Shape_1Shapeqzuziqqdld/transpose:y:0*
T0*
_output_shapes
:2
qzuziqqdld/Shape_1
 qzuziqqdld/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 qzuziqqdld/strided_slice_1/stack
"qzuziqqdld/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_1/stack_1
"qzuziqqdld/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_1/stack_2°
qzuziqqdld/strided_slice_1StridedSliceqzuziqqdld/Shape_1:output:0)qzuziqqdld/strided_slice_1/stack:output:0+qzuziqqdld/strided_slice_1/stack_1:output:0+qzuziqqdld/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
qzuziqqdld/strided_slice_1
&qzuziqqdld/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&qzuziqqdld/TensorArrayV2/element_shapeÞ
qzuziqqdld/TensorArrayV2TensorListReserve/qzuziqqdld/TensorArrayV2/element_shape:output:0#qzuziqqdld/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
qzuziqqdld/TensorArrayV2Õ
@qzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@qzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2qzuziqqdld/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorqzuziqqdld/transpose:y:0Iqzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2qzuziqqdld/TensorArrayUnstack/TensorListFromTensor
 qzuziqqdld/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 qzuziqqdld/strided_slice_2/stack
"qzuziqqdld/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_2/stack_1
"qzuziqqdld/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_2/stack_2¾
qzuziqqdld/strided_slice_2StridedSliceqzuziqqdld/transpose:y:0)qzuziqqdld/strided_slice_2/stack:output:0+qzuziqqdld/strided_slice_2/stack_1:output:0+qzuziqqdld/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
qzuziqqdld/strided_slice_2Ð
+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp4qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOpÓ
qzuziqqdld/aiccbsgdoo/MatMulMatMul#qzuziqqdld/strided_slice_2:output:03qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qzuziqqdld/aiccbsgdoo/MatMulÖ
-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp6qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOpÏ
qzuziqqdld/aiccbsgdoo/MatMul_1MatMulqzuziqqdld/zeros:output:05qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
qzuziqqdld/aiccbsgdoo/MatMul_1Ä
qzuziqqdld/aiccbsgdoo/addAddV2&qzuziqqdld/aiccbsgdoo/MatMul:product:0(qzuziqqdld/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qzuziqqdld/aiccbsgdoo/addÏ
,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp5qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOpÑ
qzuziqqdld/aiccbsgdoo/BiasAddBiasAddqzuziqqdld/aiccbsgdoo/add:z:04qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
qzuziqqdld/aiccbsgdoo/BiasAdd
%qzuziqqdld/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%qzuziqqdld/aiccbsgdoo/split/split_dim
qzuziqqdld/aiccbsgdoo/splitSplit.qzuziqqdld/aiccbsgdoo/split/split_dim:output:0&qzuziqqdld/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
qzuziqqdld/aiccbsgdoo/split¶
$qzuziqqdld/aiccbsgdoo/ReadVariableOpReadVariableOp-qzuziqqdld_aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02&
$qzuziqqdld/aiccbsgdoo/ReadVariableOpº
qzuziqqdld/aiccbsgdoo/mulMul,qzuziqqdld/aiccbsgdoo/ReadVariableOp:value:0qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mulº
qzuziqqdld/aiccbsgdoo/add_1AddV2$qzuziqqdld/aiccbsgdoo/split:output:0qzuziqqdld/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_1
qzuziqqdld/aiccbsgdoo/SigmoidSigmoidqzuziqqdld/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/Sigmoid¼
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1ReadVariableOp/qzuziqqdld_aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1À
qzuziqqdld/aiccbsgdoo/mul_1Mul.qzuziqqdld/aiccbsgdoo/ReadVariableOp_1:value:0qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_1¼
qzuziqqdld/aiccbsgdoo/add_2AddV2$qzuziqqdld/aiccbsgdoo/split:output:1qzuziqqdld/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_2 
qzuziqqdld/aiccbsgdoo/Sigmoid_1Sigmoidqzuziqqdld/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
qzuziqqdld/aiccbsgdoo/Sigmoid_1µ
qzuziqqdld/aiccbsgdoo/mul_2Mul#qzuziqqdld/aiccbsgdoo/Sigmoid_1:y:0qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_2
qzuziqqdld/aiccbsgdoo/TanhTanh$qzuziqqdld/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/Tanh¶
qzuziqqdld/aiccbsgdoo/mul_3Mul!qzuziqqdld/aiccbsgdoo/Sigmoid:y:0qzuziqqdld/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_3·
qzuziqqdld/aiccbsgdoo/add_3AddV2qzuziqqdld/aiccbsgdoo/mul_2:z:0qzuziqqdld/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_3¼
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2ReadVariableOp/qzuziqqdld_aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2Ä
qzuziqqdld/aiccbsgdoo/mul_4Mul.qzuziqqdld/aiccbsgdoo/ReadVariableOp_2:value:0qzuziqqdld/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_4¼
qzuziqqdld/aiccbsgdoo/add_4AddV2$qzuziqqdld/aiccbsgdoo/split:output:3qzuziqqdld/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/add_4 
qzuziqqdld/aiccbsgdoo/Sigmoid_2Sigmoidqzuziqqdld/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
qzuziqqdld/aiccbsgdoo/Sigmoid_2
qzuziqqdld/aiccbsgdoo/Tanh_1Tanhqzuziqqdld/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/Tanh_1º
qzuziqqdld/aiccbsgdoo/mul_5Mul#qzuziqqdld/aiccbsgdoo/Sigmoid_2:y:0 qzuziqqdld/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/aiccbsgdoo/mul_5¥
(qzuziqqdld/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(qzuziqqdld/TensorArrayV2_1/element_shapeä
qzuziqqdld/TensorArrayV2_1TensorListReserve1qzuziqqdld/TensorArrayV2_1/element_shape:output:0#qzuziqqdld/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
qzuziqqdld/TensorArrayV2_1d
qzuziqqdld/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/time
#qzuziqqdld/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#qzuziqqdld/while/maximum_iterations
qzuziqqdld/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
qzuziqqdld/while/loop_counter²
qzuziqqdld/whileWhile&qzuziqqdld/while/loop_counter:output:0,qzuziqqdld/while/maximum_iterations:output:0qzuziqqdld/time:output:0#qzuziqqdld/TensorArrayV2_1:handle:0qzuziqqdld/zeros:output:0qzuziqqdld/zeros_1:output:0#qzuziqqdld/strided_slice_1:output:0Bqzuziqqdld/TensorArrayUnstack/TensorListFromTensor:output_handle:04qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource6qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource5qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource-qzuziqqdld_aiccbsgdoo_readvariableop_resource/qzuziqqdld_aiccbsgdoo_readvariableop_1_resource/qzuziqqdld_aiccbsgdoo_readvariableop_2_resource*
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
qzuziqqdld_while_body_1720461*)
cond!R
qzuziqqdld_while_cond_1720460*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
qzuziqqdld/whileË
;qzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;qzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shape
-qzuziqqdld/TensorArrayV2Stack/TensorListStackTensorListStackqzuziqqdld/while:output:3Dqzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-qzuziqqdld/TensorArrayV2Stack/TensorListStack
 qzuziqqdld/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 qzuziqqdld/strided_slice_3/stack
"qzuziqqdld/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"qzuziqqdld/strided_slice_3/stack_1
"qzuziqqdld/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"qzuziqqdld/strided_slice_3/stack_2Ü
qzuziqqdld/strided_slice_3StridedSlice6qzuziqqdld/TensorArrayV2Stack/TensorListStack:tensor:0)qzuziqqdld/strided_slice_3/stack:output:0+qzuziqqdld/strided_slice_3/stack_1:output:0+qzuziqqdld/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
qzuziqqdld/strided_slice_3
qzuziqqdld/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
qzuziqqdld/transpose_1/permÑ
qzuziqqdld/transpose_1	Transpose6qzuziqqdld/TensorArrayV2Stack/TensorListStack:tensor:0$qzuziqqdld/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/transpose_1®
 pbmomrqadp/MatMul/ReadVariableOpReadVariableOp)pbmomrqadp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 pbmomrqadp/MatMul/ReadVariableOp±
pbmomrqadp/MatMulMatMul#qzuziqqdld/strided_slice_3:output:0(pbmomrqadp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pbmomrqadp/MatMul­
!pbmomrqadp/BiasAdd/ReadVariableOpReadVariableOp*pbmomrqadp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!pbmomrqadp/BiasAdd/ReadVariableOp­
pbmomrqadp/BiasAddBiasAddpbmomrqadp/MatMul:product:0)pbmomrqadp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pbmomrqadp/BiasAdd¶
IdentityIdentitypbmomrqadp/BiasAdd:output:0-^cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp,^cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp.^cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp%^cxhqebqpjz/rvncypflgq/ReadVariableOp'^cxhqebqpjz/rvncypflgq/ReadVariableOp_1'^cxhqebqpjz/rvncypflgq/ReadVariableOp_2^cxhqebqpjz/while.^ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp5^ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp.^mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp5^mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp"^pbmomrqadp/BiasAdd/ReadVariableOp!^pbmomrqadp/MatMul/ReadVariableOp-^qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp,^qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp.^qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp%^qzuziqqdld/aiccbsgdoo/ReadVariableOp'^qzuziqqdld/aiccbsgdoo/ReadVariableOp_1'^qzuziqqdld/aiccbsgdoo/ReadVariableOp_2^qzuziqqdld/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2\
,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp,cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp2Z
+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp+cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp2^
-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp-cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp2L
$cxhqebqpjz/rvncypflgq/ReadVariableOp$cxhqebqpjz/rvncypflgq/ReadVariableOp2P
&cxhqebqpjz/rvncypflgq/ReadVariableOp_1&cxhqebqpjz/rvncypflgq/ReadVariableOp_12P
&cxhqebqpjz/rvncypflgq/ReadVariableOp_2&cxhqebqpjz/rvncypflgq/ReadVariableOp_22$
cxhqebqpjz/whilecxhqebqpjz/while2^
-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp-ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp2l
4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp4ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp2^
-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp-mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp2l
4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp4mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!pbmomrqadp/BiasAdd/ReadVariableOp!pbmomrqadp/BiasAdd/ReadVariableOp2D
 pbmomrqadp/MatMul/ReadVariableOp pbmomrqadp/MatMul/ReadVariableOp2\
,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp,qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp2Z
+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp+qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp2^
-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp-qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp2L
$qzuziqqdld/aiccbsgdoo/ReadVariableOp$qzuziqqdld/aiccbsgdoo/ReadVariableOp2P
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_1&qzuziqqdld/aiccbsgdoo/ReadVariableOp_12P
&qzuziqqdld/aiccbsgdoo/ReadVariableOp_2&qzuziqqdld/aiccbsgdoo/ReadVariableOp_22$
qzuziqqdld/whileqzuziqqdld/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
Ö
,__inference_sequential_layer_call_fn_1720609

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	 
	unknown_5:	
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:	 

unknown_10:	 

unknown_11:	

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17188772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±'
³
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1717880

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
while_cond_1719237
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1719237___redundant_placeholder05
1while_while_cond_1719237___redundant_placeholder15
1while_while_cond_1719237___redundant_placeholder25
1while_while_cond_1719237___redundant_placeholder35
1while_while_cond_1719237___redundant_placeholder45
1while_while_cond_1719237___redundant_placeholder55
1while_while_cond_1719237___redundant_placeholder6
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
,__inference_rvncypflgq_layer_call_fn_1722468

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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_17169352
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
p
Ê
qzuziqqdld_while_body_17204612
.qzuziqqdld_while_qzuziqqdld_while_loop_counter8
4qzuziqqdld_while_qzuziqqdld_while_maximum_iterations 
qzuziqqdld_while_placeholder"
qzuziqqdld_while_placeholder_1"
qzuziqqdld_while_placeholder_2"
qzuziqqdld_while_placeholder_31
-qzuziqqdld_while_qzuziqqdld_strided_slice_1_0m
iqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0O
<qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0:	 Q
>qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0:	 L
=qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0:	C
5qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0: E
7qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0: E
7qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0: 
qzuziqqdld_while_identity
qzuziqqdld_while_identity_1
qzuziqqdld_while_identity_2
qzuziqqdld_while_identity_3
qzuziqqdld_while_identity_4
qzuziqqdld_while_identity_5/
+qzuziqqdld_while_qzuziqqdld_strided_slice_1k
gqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensorM
:qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource:	 O
<qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource:	 J
;qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource:	A
3qzuziqqdld_while_aiccbsgdoo_readvariableop_resource: C
5qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource: C
5qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource: ¢2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp¢1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp¢3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp¢*qzuziqqdld/while/aiccbsgdoo/ReadVariableOp¢,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1¢,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2Ù
Bqzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bqzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shape
4qzuziqqdld/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0qzuziqqdld_while_placeholderKqzuziqqdld/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4qzuziqqdld/while/TensorArrayV2Read/TensorListGetItemä
1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp<qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOpý
"qzuziqqdld/while/aiccbsgdoo/MatMulMatMul;qzuziqqdld/while/TensorArrayV2Read/TensorListGetItem:item:09qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"qzuziqqdld/while/aiccbsgdoo/MatMulê
3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp>qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOpæ
$qzuziqqdld/while/aiccbsgdoo/MatMul_1MatMulqzuziqqdld_while_placeholder_2;qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$qzuziqqdld/while/aiccbsgdoo/MatMul_1Ü
qzuziqqdld/while/aiccbsgdoo/addAddV2,qzuziqqdld/while/aiccbsgdoo/MatMul:product:0.qzuziqqdld/while/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
qzuziqqdld/while/aiccbsgdoo/addã
2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp=qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOpé
#qzuziqqdld/while/aiccbsgdoo/BiasAddBiasAdd#qzuziqqdld/while/aiccbsgdoo/add:z:0:qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#qzuziqqdld/while/aiccbsgdoo/BiasAdd
+qzuziqqdld/while/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+qzuziqqdld/while/aiccbsgdoo/split/split_dim¯
!qzuziqqdld/while/aiccbsgdoo/splitSplit4qzuziqqdld/while/aiccbsgdoo/split/split_dim:output:0,qzuziqqdld/while/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!qzuziqqdld/while/aiccbsgdoo/splitÊ
*qzuziqqdld/while/aiccbsgdoo/ReadVariableOpReadVariableOp5qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*qzuziqqdld/while/aiccbsgdoo/ReadVariableOpÏ
qzuziqqdld/while/aiccbsgdoo/mulMul2qzuziqqdld/while/aiccbsgdoo/ReadVariableOp:value:0qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
qzuziqqdld/while/aiccbsgdoo/mulÒ
!qzuziqqdld/while/aiccbsgdoo/add_1AddV2*qzuziqqdld/while/aiccbsgdoo/split:output:0#qzuziqqdld/while/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_1®
#qzuziqqdld/while/aiccbsgdoo/SigmoidSigmoid%qzuziqqdld/while/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#qzuziqqdld/while/aiccbsgdoo/SigmoidÐ
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1ReadVariableOp7qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1Õ
!qzuziqqdld/while/aiccbsgdoo/mul_1Mul4qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1:value:0qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_1Ô
!qzuziqqdld/while/aiccbsgdoo/add_2AddV2*qzuziqqdld/while/aiccbsgdoo/split:output:1%qzuziqqdld/while/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_2²
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_1Sigmoid%qzuziqqdld/while/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_1Ê
!qzuziqqdld/while/aiccbsgdoo/mul_2Mul)qzuziqqdld/while/aiccbsgdoo/Sigmoid_1:y:0qzuziqqdld_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_2ª
 qzuziqqdld/while/aiccbsgdoo/TanhTanh*qzuziqqdld/while/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 qzuziqqdld/while/aiccbsgdoo/TanhÎ
!qzuziqqdld/while/aiccbsgdoo/mul_3Mul'qzuziqqdld/while/aiccbsgdoo/Sigmoid:y:0$qzuziqqdld/while/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_3Ï
!qzuziqqdld/while/aiccbsgdoo/add_3AddV2%qzuziqqdld/while/aiccbsgdoo/mul_2:z:0%qzuziqqdld/while/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_3Ð
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2ReadVariableOp7qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2Ü
!qzuziqqdld/while/aiccbsgdoo/mul_4Mul4qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2:value:0%qzuziqqdld/while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_4Ô
!qzuziqqdld/while/aiccbsgdoo/add_4AddV2*qzuziqqdld/while/aiccbsgdoo/split:output:3%qzuziqqdld/while/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/add_4²
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_2Sigmoid%qzuziqqdld/while/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%qzuziqqdld/while/aiccbsgdoo/Sigmoid_2©
"qzuziqqdld/while/aiccbsgdoo/Tanh_1Tanh%qzuziqqdld/while/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"qzuziqqdld/while/aiccbsgdoo/Tanh_1Ò
!qzuziqqdld/while/aiccbsgdoo/mul_5Mul)qzuziqqdld/while/aiccbsgdoo/Sigmoid_2:y:0&qzuziqqdld/while/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!qzuziqqdld/while/aiccbsgdoo/mul_5
5qzuziqqdld/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemqzuziqqdld_while_placeholder_1qzuziqqdld_while_placeholder%qzuziqqdld/while/aiccbsgdoo/mul_5:z:0*
_output_shapes
: *
element_dtype027
5qzuziqqdld/while/TensorArrayV2Write/TensorListSetItemr
qzuziqqdld/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
qzuziqqdld/while/add/y
qzuziqqdld/while/addAddV2qzuziqqdld_while_placeholderqzuziqqdld/while/add/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/while/addv
qzuziqqdld/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
qzuziqqdld/while/add_1/y­
qzuziqqdld/while/add_1AddV2.qzuziqqdld_while_qzuziqqdld_while_loop_counter!qzuziqqdld/while/add_1/y:output:0*
T0*
_output_shapes
: 2
qzuziqqdld/while/add_1©
qzuziqqdld/while/IdentityIdentityqzuziqqdld/while/add_1:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/IdentityÇ
qzuziqqdld/while/Identity_1Identity4qzuziqqdld_while_qzuziqqdld_while_maximum_iterations3^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/Identity_1«
qzuziqqdld/while/Identity_2Identityqzuziqqdld/while/add:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/Identity_2Ø
qzuziqqdld/while/Identity_3IdentityEqzuziqqdld/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*
_output_shapes
: 2
qzuziqqdld/while/Identity_3É
qzuziqqdld/while/Identity_4Identity%qzuziqqdld/while/aiccbsgdoo/mul_5:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/while/Identity_4É
qzuziqqdld/while/Identity_5Identity%qzuziqqdld/while/aiccbsgdoo/add_3:z:03^qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2^qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp4^qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp+^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1-^qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
qzuziqqdld/while/Identity_5"|
;qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource=qzuziqqdld_while_aiccbsgdoo_biasadd_readvariableop_resource_0"~
<qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource>qzuziqqdld_while_aiccbsgdoo_matmul_1_readvariableop_resource_0"z
:qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource<qzuziqqdld_while_aiccbsgdoo_matmul_readvariableop_resource_0"p
5qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource7qzuziqqdld_while_aiccbsgdoo_readvariableop_1_resource_0"p
5qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource7qzuziqqdld_while_aiccbsgdoo_readvariableop_2_resource_0"l
3qzuziqqdld_while_aiccbsgdoo_readvariableop_resource5qzuziqqdld_while_aiccbsgdoo_readvariableop_resource_0"?
qzuziqqdld_while_identity"qzuziqqdld/while/Identity:output:0"C
qzuziqqdld_while_identity_1$qzuziqqdld/while/Identity_1:output:0"C
qzuziqqdld_while_identity_2$qzuziqqdld/while/Identity_2:output:0"C
qzuziqqdld_while_identity_3$qzuziqqdld/while/Identity_3:output:0"C
qzuziqqdld_while_identity_4$qzuziqqdld/while/Identity_4:output:0"C
qzuziqqdld_while_identity_5$qzuziqqdld/while/Identity_5:output:0"\
+qzuziqqdld_while_qzuziqqdld_strided_slice_1-qzuziqqdld_while_qzuziqqdld_strided_slice_1_0"Ô
gqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensoriqzuziqqdld_while_tensorarrayv2read_tensorlistgetitem_qzuziqqdld_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2qzuziqqdld/while/aiccbsgdoo/BiasAdd/ReadVariableOp2f
1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp1qzuziqqdld/while/aiccbsgdoo/MatMul/ReadVariableOp2j
3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp3qzuziqqdld/while/aiccbsgdoo/MatMul_1/ReadVariableOp2X
*qzuziqqdld/while/aiccbsgdoo/ReadVariableOp*qzuziqqdld/while/aiccbsgdoo/ReadVariableOp2\
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_1,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_12\
,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2,qzuziqqdld/while/aiccbsgdoo/ReadVariableOp_2: 
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
while_body_1720841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_rvncypflgq_matmul_readvariableop_resource_0:	F
3while_rvncypflgq_matmul_1_readvariableop_resource_0:	 A
2while_rvncypflgq_biasadd_readvariableop_resource_0:	8
*while_rvncypflgq_readvariableop_resource_0: :
,while_rvncypflgq_readvariableop_1_resource_0: :
,while_rvncypflgq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_rvncypflgq_matmul_readvariableop_resource:	D
1while_rvncypflgq_matmul_1_readvariableop_resource:	 ?
0while_rvncypflgq_biasadd_readvariableop_resource:	6
(while_rvncypflgq_readvariableop_resource: 8
*while_rvncypflgq_readvariableop_1_resource: 8
*while_rvncypflgq_readvariableop_2_resource: ¢'while/rvncypflgq/BiasAdd/ReadVariableOp¢&while/rvncypflgq/MatMul/ReadVariableOp¢(while/rvncypflgq/MatMul_1/ReadVariableOp¢while/rvncypflgq/ReadVariableOp¢!while/rvncypflgq/ReadVariableOp_1¢!while/rvncypflgq/ReadVariableOp_2Ã
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
&while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp1while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/rvncypflgq/MatMul/ReadVariableOpÑ
while/rvncypflgq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMulÉ
(while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp3while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/rvncypflgq/MatMul_1/ReadVariableOpº
while/rvncypflgq/MatMul_1MatMulwhile_placeholder_20while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMul_1°
while/rvncypflgq/addAddV2!while/rvncypflgq/MatMul:product:0#while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/addÂ
'while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp2while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/rvncypflgq/BiasAdd/ReadVariableOp½
while/rvncypflgq/BiasAddBiasAddwhile/rvncypflgq/add:z:0/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/BiasAdd
 while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/rvncypflgq/split/split_dim
while/rvncypflgq/splitSplit)while/rvncypflgq/split/split_dim:output:0!while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/rvncypflgq/split©
while/rvncypflgq/ReadVariableOpReadVariableOp*while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/rvncypflgq/ReadVariableOp£
while/rvncypflgq/mulMul'while/rvncypflgq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul¦
while/rvncypflgq/add_1AddV2while/rvncypflgq/split:output:0while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_1
while/rvncypflgq/SigmoidSigmoidwhile/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid¯
!while/rvncypflgq/ReadVariableOp_1ReadVariableOp,while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_1©
while/rvncypflgq/mul_1Mul)while/rvncypflgq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_1¨
while/rvncypflgq/add_2AddV2while/rvncypflgq/split:output:1while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_2
while/rvncypflgq/Sigmoid_1Sigmoidwhile/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_1
while/rvncypflgq/mul_2Mulwhile/rvncypflgq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_2
while/rvncypflgq/TanhTanhwhile/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh¢
while/rvncypflgq/mul_3Mulwhile/rvncypflgq/Sigmoid:y:0while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_3£
while/rvncypflgq/add_3AddV2while/rvncypflgq/mul_2:z:0while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_3¯
!while/rvncypflgq/ReadVariableOp_2ReadVariableOp,while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_2°
while/rvncypflgq/mul_4Mul)while/rvncypflgq/ReadVariableOp_2:value:0while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_4¨
while/rvncypflgq/add_4AddV2while/rvncypflgq/split:output:3while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_4
while/rvncypflgq/Sigmoid_2Sigmoidwhile/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_2
while/rvncypflgq/Tanh_1Tanhwhile/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh_1¦
while/rvncypflgq/mul_5Mulwhile/rvncypflgq/Sigmoid_2:y:0while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/rvncypflgq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/rvncypflgq/mul_5:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/rvncypflgq/add_3:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
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
0while_rvncypflgq_biasadd_readvariableop_resource2while_rvncypflgq_biasadd_readvariableop_resource_0"h
1while_rvncypflgq_matmul_1_readvariableop_resource3while_rvncypflgq_matmul_1_readvariableop_resource_0"d
/while_rvncypflgq_matmul_readvariableop_resource1while_rvncypflgq_matmul_readvariableop_resource_0"Z
*while_rvncypflgq_readvariableop_1_resource,while_rvncypflgq_readvariableop_1_resource_0"Z
*while_rvncypflgq_readvariableop_2_resource,while_rvncypflgq_readvariableop_2_resource_0"V
(while_rvncypflgq_readvariableop_resource*while_rvncypflgq_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/rvncypflgq/BiasAdd/ReadVariableOp'while/rvncypflgq/BiasAdd/ReadVariableOp2P
&while/rvncypflgq/MatMul/ReadVariableOp&while/rvncypflgq/MatMul/ReadVariableOp2T
(while/rvncypflgq/MatMul_1/ReadVariableOp(while/rvncypflgq/MatMul_1/ReadVariableOp2B
while/rvncypflgq/ReadVariableOpwhile/rvncypflgq/ReadVariableOp2F
!while/rvncypflgq/ReadVariableOp_1!while/rvncypflgq/ReadVariableOp_12F
!while/rvncypflgq/ReadVariableOp_2!while/rvncypflgq/ReadVariableOp_2: 
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
Ó	
ø
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_1718870

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
¡h

G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1718653

inputs<
)rvncypflgq_matmul_readvariableop_resource:	>
+rvncypflgq_matmul_1_readvariableop_resource:	 9
*rvncypflgq_biasadd_readvariableop_resource:	0
"rvncypflgq_readvariableop_resource: 2
$rvncypflgq_readvariableop_1_resource: 2
$rvncypflgq_readvariableop_2_resource: 
identity¢!rvncypflgq/BiasAdd/ReadVariableOp¢ rvncypflgq/MatMul/ReadVariableOp¢"rvncypflgq/MatMul_1/ReadVariableOp¢rvncypflgq/ReadVariableOp¢rvncypflgq/ReadVariableOp_1¢rvncypflgq/ReadVariableOp_2¢whileD
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
 rvncypflgq/MatMul/ReadVariableOpReadVariableOp)rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 rvncypflgq/MatMul/ReadVariableOp§
rvncypflgq/MatMulMatMulstrided_slice_2:output:0(rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMulµ
"rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp+rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"rvncypflgq/MatMul_1/ReadVariableOp£
rvncypflgq/MatMul_1MatMulzeros:output:0*rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMul_1
rvncypflgq/addAddV2rvncypflgq/MatMul:product:0rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/add®
!rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp*rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!rvncypflgq/BiasAdd/ReadVariableOp¥
rvncypflgq/BiasAddBiasAddrvncypflgq/add:z:0)rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/BiasAddz
rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
rvncypflgq/split/split_dimë
rvncypflgq/splitSplit#rvncypflgq/split/split_dim:output:0rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rvncypflgq/split
rvncypflgq/ReadVariableOpReadVariableOp"rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp
rvncypflgq/mulMul!rvncypflgq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul
rvncypflgq/add_1AddV2rvncypflgq/split:output:0rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_1{
rvncypflgq/SigmoidSigmoidrvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid
rvncypflgq/ReadVariableOp_1ReadVariableOp$rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_1
rvncypflgq/mul_1Mul#rvncypflgq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_1
rvncypflgq/add_2AddV2rvncypflgq/split:output:1rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_2
rvncypflgq/Sigmoid_1Sigmoidrvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_1
rvncypflgq/mul_2Mulrvncypflgq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_2w
rvncypflgq/TanhTanhrvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh
rvncypflgq/mul_3Mulrvncypflgq/Sigmoid:y:0rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_3
rvncypflgq/add_3AddV2rvncypflgq/mul_2:z:0rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_3
rvncypflgq/ReadVariableOp_2ReadVariableOp$rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_2
rvncypflgq/mul_4Mul#rvncypflgq/ReadVariableOp_2:value:0rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_4
rvncypflgq/add_4AddV2rvncypflgq/split:output:3rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_4
rvncypflgq/Sigmoid_2Sigmoidrvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_2v
rvncypflgq/Tanh_1Tanhrvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh_1
rvncypflgq/mul_5Mulrvncypflgq/Sigmoid_2:y:0rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)rvncypflgq_matmul_readvariableop_resource+rvncypflgq_matmul_1_readvariableop_resource*rvncypflgq_biasadd_readvariableop_resource"rvncypflgq_readvariableop_resource$rvncypflgq_readvariableop_1_resource$rvncypflgq_readvariableop_2_resource*
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
while_body_1718552*
condR
while_cond_1718551*Q
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
IdentityIdentitytranspose_1:y:0"^rvncypflgq/BiasAdd/ReadVariableOp!^rvncypflgq/MatMul/ReadVariableOp#^rvncypflgq/MatMul_1/ReadVariableOp^rvncypflgq/ReadVariableOp^rvncypflgq/ReadVariableOp_1^rvncypflgq/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!rvncypflgq/BiasAdd/ReadVariableOp!rvncypflgq/BiasAdd/ReadVariableOp2D
 rvncypflgq/MatMul/ReadVariableOp rvncypflgq/MatMul/ReadVariableOp2H
"rvncypflgq/MatMul_1/ReadVariableOp"rvncypflgq/MatMul_1/ReadVariableOp26
rvncypflgq/ReadVariableOprvncypflgq/ReadVariableOp2:
rvncypflgq/ReadVariableOp_1rvncypflgq/ReadVariableOp_12:
rvncypflgq/ReadVariableOp_2rvncypflgq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1717712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1717712___redundant_placeholder05
1while_while_cond_1717712___redundant_placeholder15
1while_while_cond_1717712___redundant_placeholder25
1while_while_cond_1717712___redundant_placeholder35
1while_while_cond_1717712___redundant_placeholder45
1while_while_cond_1717712___redundant_placeholder55
1while_while_cond_1717712___redundant_placeholder6
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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1716935

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
while_cond_1717217
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1717217___redundant_placeholder05
1while_while_cond_1717217___redundant_placeholder15
1while_while_cond_1717217___redundant_placeholder25
1while_while_cond_1717217___redundant_placeholder35
1while_while_cond_1717217___redundant_placeholder45
1while_while_cond_1717217___redundant_placeholder55
1while_while_cond_1717217___redundant_placeholder6
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
while_body_1718552
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_rvncypflgq_matmul_readvariableop_resource_0:	F
3while_rvncypflgq_matmul_1_readvariableop_resource_0:	 A
2while_rvncypflgq_biasadd_readvariableop_resource_0:	8
*while_rvncypflgq_readvariableop_resource_0: :
,while_rvncypflgq_readvariableop_1_resource_0: :
,while_rvncypflgq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_rvncypflgq_matmul_readvariableop_resource:	D
1while_rvncypflgq_matmul_1_readvariableop_resource:	 ?
0while_rvncypflgq_biasadd_readvariableop_resource:	6
(while_rvncypflgq_readvariableop_resource: 8
*while_rvncypflgq_readvariableop_1_resource: 8
*while_rvncypflgq_readvariableop_2_resource: ¢'while/rvncypflgq/BiasAdd/ReadVariableOp¢&while/rvncypflgq/MatMul/ReadVariableOp¢(while/rvncypflgq/MatMul_1/ReadVariableOp¢while/rvncypflgq/ReadVariableOp¢!while/rvncypflgq/ReadVariableOp_1¢!while/rvncypflgq/ReadVariableOp_2Ã
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
&while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp1while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/rvncypflgq/MatMul/ReadVariableOpÑ
while/rvncypflgq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMulÉ
(while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp3while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/rvncypflgq/MatMul_1/ReadVariableOpº
while/rvncypflgq/MatMul_1MatMulwhile_placeholder_20while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMul_1°
while/rvncypflgq/addAddV2!while/rvncypflgq/MatMul:product:0#while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/addÂ
'while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp2while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/rvncypflgq/BiasAdd/ReadVariableOp½
while/rvncypflgq/BiasAddBiasAddwhile/rvncypflgq/add:z:0/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/BiasAdd
 while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/rvncypflgq/split/split_dim
while/rvncypflgq/splitSplit)while/rvncypflgq/split/split_dim:output:0!while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/rvncypflgq/split©
while/rvncypflgq/ReadVariableOpReadVariableOp*while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/rvncypflgq/ReadVariableOp£
while/rvncypflgq/mulMul'while/rvncypflgq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul¦
while/rvncypflgq/add_1AddV2while/rvncypflgq/split:output:0while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_1
while/rvncypflgq/SigmoidSigmoidwhile/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid¯
!while/rvncypflgq/ReadVariableOp_1ReadVariableOp,while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_1©
while/rvncypflgq/mul_1Mul)while/rvncypflgq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_1¨
while/rvncypflgq/add_2AddV2while/rvncypflgq/split:output:1while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_2
while/rvncypflgq/Sigmoid_1Sigmoidwhile/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_1
while/rvncypflgq/mul_2Mulwhile/rvncypflgq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_2
while/rvncypflgq/TanhTanhwhile/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh¢
while/rvncypflgq/mul_3Mulwhile/rvncypflgq/Sigmoid:y:0while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_3£
while/rvncypflgq/add_3AddV2while/rvncypflgq/mul_2:z:0while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_3¯
!while/rvncypflgq/ReadVariableOp_2ReadVariableOp,while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_2°
while/rvncypflgq/mul_4Mul)while/rvncypflgq/ReadVariableOp_2:value:0while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_4¨
while/rvncypflgq/add_4AddV2while/rvncypflgq/split:output:3while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_4
while/rvncypflgq/Sigmoid_2Sigmoidwhile/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_2
while/rvncypflgq/Tanh_1Tanhwhile/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh_1¦
while/rvncypflgq/mul_5Mulwhile/rvncypflgq/Sigmoid_2:y:0while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/rvncypflgq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/rvncypflgq/mul_5:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/rvncypflgq/add_3:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
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
0while_rvncypflgq_biasadd_readvariableop_resource2while_rvncypflgq_biasadd_readvariableop_resource_0"h
1while_rvncypflgq_matmul_1_readvariableop_resource3while_rvncypflgq_matmul_1_readvariableop_resource_0"d
/while_rvncypflgq_matmul_readvariableop_resource1while_rvncypflgq_matmul_readvariableop_resource_0"Z
*while_rvncypflgq_readvariableop_1_resource,while_rvncypflgq_readvariableop_1_resource_0"Z
*while_rvncypflgq_readvariableop_2_resource,while_rvncypflgq_readvariableop_2_resource_0"V
(while_rvncypflgq_readvariableop_resource*while_rvncypflgq_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/rvncypflgq/BiasAdd/ReadVariableOp'while/rvncypflgq/BiasAdd/ReadVariableOp2P
&while/rvncypflgq/MatMul/ReadVariableOp&while/rvncypflgq/MatMul/ReadVariableOp2T
(while/rvncypflgq/MatMul_1/ReadVariableOp(while/rvncypflgq/MatMul_1/ReadVariableOp2B
while/rvncypflgq/ReadVariableOpwhile/rvncypflgq/ReadVariableOp2F
!while/rvncypflgq/ReadVariableOp_1!while/rvncypflgq/ReadVariableOp_12F
!while/rvncypflgq/ReadVariableOp_2!while/rvncypflgq/ReadVariableOp_2: 
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
ë

,__inference_qzuziqqdld_layer_call_fn_1722304
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17180562
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
Ó

,__inference_qzuziqqdld_layer_call_fn_1722338

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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17191252
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

À
,__inference_aiccbsgdoo_layer_call_fn_1722602

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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_17176932
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
è
Ö
,__inference_sequential_layer_call_fn_1720650

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:	
	unknown_4:	 
	unknown_5:	
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:	 

unknown_10:	 

unknown_11:	

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_17194692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ûÌ

"__inference__wrapped_model_1716848

gvxdqcynanW
Asequential_mopvqfaljf_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_mopvqfaljf_squeeze_batch_dims_biasadd_readvariableop_resource:W
Asequential_ehhqjcwuju_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_ehhqjcwuju_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource:	T
Asequential_cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource:	 O
@sequential_cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource:	F
8sequential_cxhqebqpjz_rvncypflgq_readvariableop_resource: H
:sequential_cxhqebqpjz_rvncypflgq_readvariableop_1_resource: H
:sequential_cxhqebqpjz_rvncypflgq_readvariableop_2_resource: R
?sequential_qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource:	 T
Asequential_qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource:	 O
@sequential_qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource:	F
8sequential_qzuziqqdld_aiccbsgdoo_readvariableop_resource: H
:sequential_qzuziqqdld_aiccbsgdoo_readvariableop_1_resource: H
:sequential_qzuziqqdld_aiccbsgdoo_readvariableop_2_resource: F
4sequential_pbmomrqadp_matmul_readvariableop_resource: C
5sequential_pbmomrqadp_biasadd_readvariableop_resource:
identity¢7sequential/cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp¢6sequential/cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp¢8sequential/cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp¢/sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp¢1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_1¢1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_2¢sequential/cxhqebqpjz/while¢8sequential/ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp¢8sequential/mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,sequential/pbmomrqadp/BiasAdd/ReadVariableOp¢+sequential/pbmomrqadp/MatMul/ReadVariableOp¢7sequential/qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp¢6sequential/qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp¢8sequential/qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp¢/sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp¢1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_1¢1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_2¢sequential/qzuziqqdld/while¥
+sequential/mopvqfaljf/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/mopvqfaljf/conv1d/ExpandDims/dimà
'sequential/mopvqfaljf/conv1d/ExpandDims
ExpandDims
gvxdqcynan4sequential/mopvqfaljf/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/mopvqfaljf/conv1d/ExpandDimsú
8sequential/mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_mopvqfaljf_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/mopvqfaljf/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/mopvqfaljf/conv1d/ExpandDims_1/dim
)sequential/mopvqfaljf/conv1d/ExpandDims_1
ExpandDims@sequential/mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/mopvqfaljf/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/mopvqfaljf/conv1d/ExpandDims_1¨
"sequential/mopvqfaljf/conv1d/ShapeShape0sequential/mopvqfaljf/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/mopvqfaljf/conv1d/Shape®
0sequential/mopvqfaljf/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/mopvqfaljf/conv1d/strided_slice/stack»
2sequential/mopvqfaljf/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/mopvqfaljf/conv1d/strided_slice/stack_1²
2sequential/mopvqfaljf/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/mopvqfaljf/conv1d/strided_slice/stack_2
*sequential/mopvqfaljf/conv1d/strided_sliceStridedSlice+sequential/mopvqfaljf/conv1d/Shape:output:09sequential/mopvqfaljf/conv1d/strided_slice/stack:output:0;sequential/mopvqfaljf/conv1d/strided_slice/stack_1:output:0;sequential/mopvqfaljf/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/mopvqfaljf/conv1d/strided_slice±
*sequential/mopvqfaljf/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/mopvqfaljf/conv1d/Reshape/shapeø
$sequential/mopvqfaljf/conv1d/ReshapeReshape0sequential/mopvqfaljf/conv1d/ExpandDims:output:03sequential/mopvqfaljf/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/mopvqfaljf/conv1d/Reshape
#sequential/mopvqfaljf/conv1d/Conv2DConv2D-sequential/mopvqfaljf/conv1d/Reshape:output:02sequential/mopvqfaljf/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/mopvqfaljf/conv1d/Conv2D±
,sequential/mopvqfaljf/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/mopvqfaljf/conv1d/concat/values_1
(sequential/mopvqfaljf/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/mopvqfaljf/conv1d/concat/axis£
#sequential/mopvqfaljf/conv1d/concatConcatV23sequential/mopvqfaljf/conv1d/strided_slice:output:05sequential/mopvqfaljf/conv1d/concat/values_1:output:01sequential/mopvqfaljf/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/mopvqfaljf/conv1d/concatõ
&sequential/mopvqfaljf/conv1d/Reshape_1Reshape,sequential/mopvqfaljf/conv1d/Conv2D:output:0,sequential/mopvqfaljf/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/mopvqfaljf/conv1d/Reshape_1â
$sequential/mopvqfaljf/conv1d/SqueezeSqueeze/sequential/mopvqfaljf/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/mopvqfaljf/conv1d/Squeeze½
.sequential/mopvqfaljf/squeeze_batch_dims/ShapeShape-sequential/mopvqfaljf/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/mopvqfaljf/squeeze_batch_dims/ShapeÆ
<sequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stackÓ
>sequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/mopvqfaljf/squeeze_batch_dims/strided_sliceStridedSlice7sequential/mopvqfaljf/squeeze_batch_dims/Shape:output:0Esequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/mopvqfaljf/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/mopvqfaljf/squeeze_batch_dims/strided_sliceÅ
6sequential/mopvqfaljf/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/mopvqfaljf/squeeze_batch_dims/Reshape/shape
0sequential/mopvqfaljf/squeeze_batch_dims/ReshapeReshape-sequential/mopvqfaljf/conv1d/Squeeze:output:0?sequential/mopvqfaljf/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/mopvqfaljf/squeeze_batch_dims/Reshape
?sequential/mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_mopvqfaljf_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/mopvqfaljf/squeeze_batch_dims/BiasAddBiasAdd9sequential/mopvqfaljf/squeeze_batch_dims/Reshape:output:0Gsequential/mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/mopvqfaljf/squeeze_batch_dims/BiasAddÅ
8sequential/mopvqfaljf/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/mopvqfaljf/squeeze_batch_dims/concat/values_1·
4sequential/mopvqfaljf/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/mopvqfaljf/squeeze_batch_dims/concat/axisß
/sequential/mopvqfaljf/squeeze_batch_dims/concatConcatV2?sequential/mopvqfaljf/squeeze_batch_dims/strided_slice:output:0Asequential/mopvqfaljf/squeeze_batch_dims/concat/values_1:output:0=sequential/mopvqfaljf/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/mopvqfaljf/squeeze_batch_dims/concat¢
2sequential/mopvqfaljf/squeeze_batch_dims/Reshape_1Reshape9sequential/mopvqfaljf/squeeze_batch_dims/BiasAdd:output:08sequential/mopvqfaljf/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/mopvqfaljf/squeeze_batch_dims/Reshape_1¹
"sequential/ehhqjcwuju/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                2$
"sequential/ehhqjcwuju/Pad/paddingsá
sequential/ehhqjcwuju/PadPad;sequential/mopvqfaljf/squeeze_batch_dims/Reshape_1:output:0+sequential/ehhqjcwuju/Pad/paddings:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/ehhqjcwuju/Pad¥
+sequential/ehhqjcwuju/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/ehhqjcwuju/conv1d/ExpandDims/dimø
'sequential/ehhqjcwuju/conv1d/ExpandDims
ExpandDims"sequential/ehhqjcwuju/Pad:output:04sequential/ehhqjcwuju/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/ehhqjcwuju/conv1d/ExpandDimsú
8sequential/ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_ehhqjcwuju_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/ehhqjcwuju/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/ehhqjcwuju/conv1d/ExpandDims_1/dim
)sequential/ehhqjcwuju/conv1d/ExpandDims_1
ExpandDims@sequential/ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/ehhqjcwuju/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/ehhqjcwuju/conv1d/ExpandDims_1¨
"sequential/ehhqjcwuju/conv1d/ShapeShape0sequential/ehhqjcwuju/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/ehhqjcwuju/conv1d/Shape®
0sequential/ehhqjcwuju/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/ehhqjcwuju/conv1d/strided_slice/stack»
2sequential/ehhqjcwuju/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/ehhqjcwuju/conv1d/strided_slice/stack_1²
2sequential/ehhqjcwuju/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/ehhqjcwuju/conv1d/strided_slice/stack_2
*sequential/ehhqjcwuju/conv1d/strided_sliceStridedSlice+sequential/ehhqjcwuju/conv1d/Shape:output:09sequential/ehhqjcwuju/conv1d/strided_slice/stack:output:0;sequential/ehhqjcwuju/conv1d/strided_slice/stack_1:output:0;sequential/ehhqjcwuju/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/ehhqjcwuju/conv1d/strided_slice±
*sequential/ehhqjcwuju/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/ehhqjcwuju/conv1d/Reshape/shapeø
$sequential/ehhqjcwuju/conv1d/ReshapeReshape0sequential/ehhqjcwuju/conv1d/ExpandDims:output:03sequential/ehhqjcwuju/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/ehhqjcwuju/conv1d/Reshape
#sequential/ehhqjcwuju/conv1d/Conv2DConv2D-sequential/ehhqjcwuju/conv1d/Reshape:output:02sequential/ehhqjcwuju/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/ehhqjcwuju/conv1d/Conv2D±
,sequential/ehhqjcwuju/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/ehhqjcwuju/conv1d/concat/values_1
(sequential/ehhqjcwuju/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/ehhqjcwuju/conv1d/concat/axis£
#sequential/ehhqjcwuju/conv1d/concatConcatV23sequential/ehhqjcwuju/conv1d/strided_slice:output:05sequential/ehhqjcwuju/conv1d/concat/values_1:output:01sequential/ehhqjcwuju/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/ehhqjcwuju/conv1d/concatõ
&sequential/ehhqjcwuju/conv1d/Reshape_1Reshape,sequential/ehhqjcwuju/conv1d/Conv2D:output:0,sequential/ehhqjcwuju/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/ehhqjcwuju/conv1d/Reshape_1â
$sequential/ehhqjcwuju/conv1d/SqueezeSqueeze/sequential/ehhqjcwuju/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/ehhqjcwuju/conv1d/Squeeze½
.sequential/ehhqjcwuju/squeeze_batch_dims/ShapeShape-sequential/ehhqjcwuju/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/ehhqjcwuju/squeeze_batch_dims/ShapeÆ
<sequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stackÓ
>sequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/ehhqjcwuju/squeeze_batch_dims/strided_sliceStridedSlice7sequential/ehhqjcwuju/squeeze_batch_dims/Shape:output:0Esequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/ehhqjcwuju/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/ehhqjcwuju/squeeze_batch_dims/strided_sliceÅ
6sequential/ehhqjcwuju/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/ehhqjcwuju/squeeze_batch_dims/Reshape/shape
0sequential/ehhqjcwuju/squeeze_batch_dims/ReshapeReshape-sequential/ehhqjcwuju/conv1d/Squeeze:output:0?sequential/ehhqjcwuju/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/ehhqjcwuju/squeeze_batch_dims/Reshape
?sequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_ehhqjcwuju_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/ehhqjcwuju/squeeze_batch_dims/BiasAddBiasAdd9sequential/ehhqjcwuju/squeeze_batch_dims/Reshape:output:0Gsequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/ehhqjcwuju/squeeze_batch_dims/BiasAddÅ
8sequential/ehhqjcwuju/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/ehhqjcwuju/squeeze_batch_dims/concat/values_1·
4sequential/ehhqjcwuju/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/ehhqjcwuju/squeeze_batch_dims/concat/axisß
/sequential/ehhqjcwuju/squeeze_batch_dims/concatConcatV2?sequential/ehhqjcwuju/squeeze_batch_dims/strided_slice:output:0Asequential/ehhqjcwuju/squeeze_batch_dims/concat/values_1:output:0=sequential/ehhqjcwuju/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/ehhqjcwuju/squeeze_batch_dims/concat¢
2sequential/ehhqjcwuju/squeeze_batch_dims/Reshape_1Reshape9sequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd:output:08sequential/ehhqjcwuju/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/ehhqjcwuju/squeeze_batch_dims/Reshape_1¥
sequential/abbthhzbau/ShapeShape;sequential/ehhqjcwuju/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/abbthhzbau/Shape 
)sequential/abbthhzbau/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/abbthhzbau/strided_slice/stack¤
+sequential/abbthhzbau/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/abbthhzbau/strided_slice/stack_1¤
+sequential/abbthhzbau/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/abbthhzbau/strided_slice/stack_2æ
#sequential/abbthhzbau/strided_sliceStridedSlice$sequential/abbthhzbau/Shape:output:02sequential/abbthhzbau/strided_slice/stack:output:04sequential/abbthhzbau/strided_slice/stack_1:output:04sequential/abbthhzbau/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/abbthhzbau/strided_slice
%sequential/abbthhzbau/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/abbthhzbau/Reshape/shape/1
%sequential/abbthhzbau/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/abbthhzbau/Reshape/shape/2
#sequential/abbthhzbau/Reshape/shapePack,sequential/abbthhzbau/strided_slice:output:0.sequential/abbthhzbau/Reshape/shape/1:output:0.sequential/abbthhzbau/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/abbthhzbau/Reshape/shapeê
sequential/abbthhzbau/ReshapeReshape;sequential/ehhqjcwuju/squeeze_batch_dims/Reshape_1:output:0,sequential/abbthhzbau/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/abbthhzbau/Reshape
sequential/cxhqebqpjz/ShapeShape&sequential/abbthhzbau/Reshape:output:0*
T0*
_output_shapes
:2
sequential/cxhqebqpjz/Shape 
)sequential/cxhqebqpjz/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/cxhqebqpjz/strided_slice/stack¤
+sequential/cxhqebqpjz/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/cxhqebqpjz/strided_slice/stack_1¤
+sequential/cxhqebqpjz/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/cxhqebqpjz/strided_slice/stack_2æ
#sequential/cxhqebqpjz/strided_sliceStridedSlice$sequential/cxhqebqpjz/Shape:output:02sequential/cxhqebqpjz/strided_slice/stack:output:04sequential/cxhqebqpjz/strided_slice/stack_1:output:04sequential/cxhqebqpjz/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/cxhqebqpjz/strided_slice
!sequential/cxhqebqpjz/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/cxhqebqpjz/zeros/mul/yÄ
sequential/cxhqebqpjz/zeros/mulMul,sequential/cxhqebqpjz/strided_slice:output:0*sequential/cxhqebqpjz/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/cxhqebqpjz/zeros/mul
"sequential/cxhqebqpjz/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/cxhqebqpjz/zeros/Less/y¿
 sequential/cxhqebqpjz/zeros/LessLess#sequential/cxhqebqpjz/zeros/mul:z:0+sequential/cxhqebqpjz/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/cxhqebqpjz/zeros/Less
$sequential/cxhqebqpjz/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/cxhqebqpjz/zeros/packed/1Û
"sequential/cxhqebqpjz/zeros/packedPack,sequential/cxhqebqpjz/strided_slice:output:0-sequential/cxhqebqpjz/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/cxhqebqpjz/zeros/packed
!sequential/cxhqebqpjz/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/cxhqebqpjz/zeros/ConstÍ
sequential/cxhqebqpjz/zerosFill+sequential/cxhqebqpjz/zeros/packed:output:0*sequential/cxhqebqpjz/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/cxhqebqpjz/zeros
#sequential/cxhqebqpjz/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/cxhqebqpjz/zeros_1/mul/yÊ
!sequential/cxhqebqpjz/zeros_1/mulMul,sequential/cxhqebqpjz/strided_slice:output:0,sequential/cxhqebqpjz/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/cxhqebqpjz/zeros_1/mul
$sequential/cxhqebqpjz/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/cxhqebqpjz/zeros_1/Less/yÇ
"sequential/cxhqebqpjz/zeros_1/LessLess%sequential/cxhqebqpjz/zeros_1/mul:z:0-sequential/cxhqebqpjz/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/cxhqebqpjz/zeros_1/Less
&sequential/cxhqebqpjz/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/cxhqebqpjz/zeros_1/packed/1á
$sequential/cxhqebqpjz/zeros_1/packedPack,sequential/cxhqebqpjz/strided_slice:output:0/sequential/cxhqebqpjz/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/cxhqebqpjz/zeros_1/packed
#sequential/cxhqebqpjz/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/cxhqebqpjz/zeros_1/ConstÕ
sequential/cxhqebqpjz/zeros_1Fill-sequential/cxhqebqpjz/zeros_1/packed:output:0,sequential/cxhqebqpjz/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/cxhqebqpjz/zeros_1¡
$sequential/cxhqebqpjz/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/cxhqebqpjz/transpose/permÜ
sequential/cxhqebqpjz/transpose	Transpose&sequential/abbthhzbau/Reshape:output:0-sequential/cxhqebqpjz/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/cxhqebqpjz/transpose
sequential/cxhqebqpjz/Shape_1Shape#sequential/cxhqebqpjz/transpose:y:0*
T0*
_output_shapes
:2
sequential/cxhqebqpjz/Shape_1¤
+sequential/cxhqebqpjz/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/cxhqebqpjz/strided_slice_1/stack¨
-sequential/cxhqebqpjz/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/cxhqebqpjz/strided_slice_1/stack_1¨
-sequential/cxhqebqpjz/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/cxhqebqpjz/strided_slice_1/stack_2ò
%sequential/cxhqebqpjz/strided_slice_1StridedSlice&sequential/cxhqebqpjz/Shape_1:output:04sequential/cxhqebqpjz/strided_slice_1/stack:output:06sequential/cxhqebqpjz/strided_slice_1/stack_1:output:06sequential/cxhqebqpjz/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/cxhqebqpjz/strided_slice_1±
1sequential/cxhqebqpjz/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/cxhqebqpjz/TensorArrayV2/element_shape
#sequential/cxhqebqpjz/TensorArrayV2TensorListReserve:sequential/cxhqebqpjz/TensorArrayV2/element_shape:output:0.sequential/cxhqebqpjz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/cxhqebqpjz/TensorArrayV2ë
Ksequential/cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/cxhqebqpjz/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/cxhqebqpjz/transpose:y:0Tsequential/cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor¤
+sequential/cxhqebqpjz/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/cxhqebqpjz/strided_slice_2/stack¨
-sequential/cxhqebqpjz/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/cxhqebqpjz/strided_slice_2/stack_1¨
-sequential/cxhqebqpjz/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/cxhqebqpjz/strided_slice_2/stack_2
%sequential/cxhqebqpjz/strided_slice_2StridedSlice#sequential/cxhqebqpjz/transpose:y:04sequential/cxhqebqpjz/strided_slice_2/stack:output:06sequential/cxhqebqpjz/strided_slice_2/stack_1:output:06sequential/cxhqebqpjz/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/cxhqebqpjz/strided_slice_2ñ
6sequential/cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOpReadVariableOp?sequential_cxhqebqpjz_rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOpÿ
'sequential/cxhqebqpjz/rvncypflgq/MatMulMatMul.sequential/cxhqebqpjz/strided_slice_2:output:0>sequential/cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/cxhqebqpjz/rvncypflgq/MatMul÷
8sequential/cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOpAsequential_cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOpû
)sequential/cxhqebqpjz/rvncypflgq/MatMul_1MatMul$sequential/cxhqebqpjz/zeros:output:0@sequential/cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/cxhqebqpjz/rvncypflgq/MatMul_1ð
$sequential/cxhqebqpjz/rvncypflgq/addAddV21sequential/cxhqebqpjz/rvncypflgq/MatMul:product:03sequential/cxhqebqpjz/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/cxhqebqpjz/rvncypflgq/addð
7sequential/cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp@sequential_cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOpý
(sequential/cxhqebqpjz/rvncypflgq/BiasAddBiasAdd(sequential/cxhqebqpjz/rvncypflgq/add:z:0?sequential/cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/cxhqebqpjz/rvncypflgq/BiasAdd¦
0sequential/cxhqebqpjz/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/cxhqebqpjz/rvncypflgq/split/split_dimÃ
&sequential/cxhqebqpjz/rvncypflgq/splitSplit9sequential/cxhqebqpjz/rvncypflgq/split/split_dim:output:01sequential/cxhqebqpjz/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/cxhqebqpjz/rvncypflgq/split×
/sequential/cxhqebqpjz/rvncypflgq/ReadVariableOpReadVariableOp8sequential_cxhqebqpjz_rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/cxhqebqpjz/rvncypflgq/ReadVariableOpæ
$sequential/cxhqebqpjz/rvncypflgq/mulMul7sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp:value:0&sequential/cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/cxhqebqpjz/rvncypflgq/mulæ
&sequential/cxhqebqpjz/rvncypflgq/add_1AddV2/sequential/cxhqebqpjz/rvncypflgq/split:output:0(sequential/cxhqebqpjz/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/add_1½
(sequential/cxhqebqpjz/rvncypflgq/SigmoidSigmoid*sequential/cxhqebqpjz/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/cxhqebqpjz/rvncypflgq/SigmoidÝ
1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_1ReadVariableOp:sequential_cxhqebqpjz_rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_1ì
&sequential/cxhqebqpjz/rvncypflgq/mul_1Mul9sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_1:value:0&sequential/cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/mul_1è
&sequential/cxhqebqpjz/rvncypflgq/add_2AddV2/sequential/cxhqebqpjz/rvncypflgq/split:output:1*sequential/cxhqebqpjz/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/add_2Á
*sequential/cxhqebqpjz/rvncypflgq/Sigmoid_1Sigmoid*sequential/cxhqebqpjz/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/cxhqebqpjz/rvncypflgq/Sigmoid_1á
&sequential/cxhqebqpjz/rvncypflgq/mul_2Mul.sequential/cxhqebqpjz/rvncypflgq/Sigmoid_1:y:0&sequential/cxhqebqpjz/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/mul_2¹
%sequential/cxhqebqpjz/rvncypflgq/TanhTanh/sequential/cxhqebqpjz/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/cxhqebqpjz/rvncypflgq/Tanhâ
&sequential/cxhqebqpjz/rvncypflgq/mul_3Mul,sequential/cxhqebqpjz/rvncypflgq/Sigmoid:y:0)sequential/cxhqebqpjz/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/mul_3ã
&sequential/cxhqebqpjz/rvncypflgq/add_3AddV2*sequential/cxhqebqpjz/rvncypflgq/mul_2:z:0*sequential/cxhqebqpjz/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/add_3Ý
1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_2ReadVariableOp:sequential_cxhqebqpjz_rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_2ð
&sequential/cxhqebqpjz/rvncypflgq/mul_4Mul9sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_2:value:0*sequential/cxhqebqpjz/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/mul_4è
&sequential/cxhqebqpjz/rvncypflgq/add_4AddV2/sequential/cxhqebqpjz/rvncypflgq/split:output:3*sequential/cxhqebqpjz/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/add_4Á
*sequential/cxhqebqpjz/rvncypflgq/Sigmoid_2Sigmoid*sequential/cxhqebqpjz/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/cxhqebqpjz/rvncypflgq/Sigmoid_2¸
'sequential/cxhqebqpjz/rvncypflgq/Tanh_1Tanh*sequential/cxhqebqpjz/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/cxhqebqpjz/rvncypflgq/Tanh_1æ
&sequential/cxhqebqpjz/rvncypflgq/mul_5Mul.sequential/cxhqebqpjz/rvncypflgq/Sigmoid_2:y:0+sequential/cxhqebqpjz/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/rvncypflgq/mul_5»
3sequential/cxhqebqpjz/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/cxhqebqpjz/TensorArrayV2_1/element_shape
%sequential/cxhqebqpjz/TensorArrayV2_1TensorListReserve<sequential/cxhqebqpjz/TensorArrayV2_1/element_shape:output:0.sequential/cxhqebqpjz/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/cxhqebqpjz/TensorArrayV2_1z
sequential/cxhqebqpjz/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/cxhqebqpjz/time«
.sequential/cxhqebqpjz/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/cxhqebqpjz/while/maximum_iterations
(sequential/cxhqebqpjz/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/cxhqebqpjz/while/loop_counterø	
sequential/cxhqebqpjz/whileWhile1sequential/cxhqebqpjz/while/loop_counter:output:07sequential/cxhqebqpjz/while/maximum_iterations:output:0#sequential/cxhqebqpjz/time:output:0.sequential/cxhqebqpjz/TensorArrayV2_1:handle:0$sequential/cxhqebqpjz/zeros:output:0&sequential/cxhqebqpjz/zeros_1:output:0.sequential/cxhqebqpjz/strided_slice_1:output:0Msequential/cxhqebqpjz/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_cxhqebqpjz_rvncypflgq_matmul_readvariableop_resourceAsequential_cxhqebqpjz_rvncypflgq_matmul_1_readvariableop_resource@sequential_cxhqebqpjz_rvncypflgq_biasadd_readvariableop_resource8sequential_cxhqebqpjz_rvncypflgq_readvariableop_resource:sequential_cxhqebqpjz_rvncypflgq_readvariableop_1_resource:sequential_cxhqebqpjz_rvncypflgq_readvariableop_2_resource*
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
(sequential_cxhqebqpjz_while_body_1716565*4
cond,R*
(sequential_cxhqebqpjz_while_cond_1716564*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/cxhqebqpjz/whileá
Fsequential/cxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/cxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/cxhqebqpjz/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/cxhqebqpjz/while:output:3Osequential/cxhqebqpjz/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/cxhqebqpjz/TensorArrayV2Stack/TensorListStack­
+sequential/cxhqebqpjz/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/cxhqebqpjz/strided_slice_3/stack¨
-sequential/cxhqebqpjz/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/cxhqebqpjz/strided_slice_3/stack_1¨
-sequential/cxhqebqpjz/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/cxhqebqpjz/strided_slice_3/stack_2
%sequential/cxhqebqpjz/strided_slice_3StridedSliceAsequential/cxhqebqpjz/TensorArrayV2Stack/TensorListStack:tensor:04sequential/cxhqebqpjz/strided_slice_3/stack:output:06sequential/cxhqebqpjz/strided_slice_3/stack_1:output:06sequential/cxhqebqpjz/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/cxhqebqpjz/strided_slice_3¥
&sequential/cxhqebqpjz/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/cxhqebqpjz/transpose_1/permý
!sequential/cxhqebqpjz/transpose_1	TransposeAsequential/cxhqebqpjz/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/cxhqebqpjz/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/cxhqebqpjz/transpose_1
sequential/qzuziqqdld/ShapeShape%sequential/cxhqebqpjz/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/qzuziqqdld/Shape 
)sequential/qzuziqqdld/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/qzuziqqdld/strided_slice/stack¤
+sequential/qzuziqqdld/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/qzuziqqdld/strided_slice/stack_1¤
+sequential/qzuziqqdld/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/qzuziqqdld/strided_slice/stack_2æ
#sequential/qzuziqqdld/strided_sliceStridedSlice$sequential/qzuziqqdld/Shape:output:02sequential/qzuziqqdld/strided_slice/stack:output:04sequential/qzuziqqdld/strided_slice/stack_1:output:04sequential/qzuziqqdld/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/qzuziqqdld/strided_slice
!sequential/qzuziqqdld/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/qzuziqqdld/zeros/mul/yÄ
sequential/qzuziqqdld/zeros/mulMul,sequential/qzuziqqdld/strided_slice:output:0*sequential/qzuziqqdld/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/qzuziqqdld/zeros/mul
"sequential/qzuziqqdld/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/qzuziqqdld/zeros/Less/y¿
 sequential/qzuziqqdld/zeros/LessLess#sequential/qzuziqqdld/zeros/mul:z:0+sequential/qzuziqqdld/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/qzuziqqdld/zeros/Less
$sequential/qzuziqqdld/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/qzuziqqdld/zeros/packed/1Û
"sequential/qzuziqqdld/zeros/packedPack,sequential/qzuziqqdld/strided_slice:output:0-sequential/qzuziqqdld/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/qzuziqqdld/zeros/packed
!sequential/qzuziqqdld/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/qzuziqqdld/zeros/ConstÍ
sequential/qzuziqqdld/zerosFill+sequential/qzuziqqdld/zeros/packed:output:0*sequential/qzuziqqdld/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/qzuziqqdld/zeros
#sequential/qzuziqqdld/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/qzuziqqdld/zeros_1/mul/yÊ
!sequential/qzuziqqdld/zeros_1/mulMul,sequential/qzuziqqdld/strided_slice:output:0,sequential/qzuziqqdld/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/qzuziqqdld/zeros_1/mul
$sequential/qzuziqqdld/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/qzuziqqdld/zeros_1/Less/yÇ
"sequential/qzuziqqdld/zeros_1/LessLess%sequential/qzuziqqdld/zeros_1/mul:z:0-sequential/qzuziqqdld/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/qzuziqqdld/zeros_1/Less
&sequential/qzuziqqdld/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/qzuziqqdld/zeros_1/packed/1á
$sequential/qzuziqqdld/zeros_1/packedPack,sequential/qzuziqqdld/strided_slice:output:0/sequential/qzuziqqdld/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/qzuziqqdld/zeros_1/packed
#sequential/qzuziqqdld/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/qzuziqqdld/zeros_1/ConstÕ
sequential/qzuziqqdld/zeros_1Fill-sequential/qzuziqqdld/zeros_1/packed:output:0,sequential/qzuziqqdld/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/qzuziqqdld/zeros_1¡
$sequential/qzuziqqdld/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/qzuziqqdld/transpose/permÛ
sequential/qzuziqqdld/transpose	Transpose%sequential/cxhqebqpjz/transpose_1:y:0-sequential/qzuziqqdld/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/qzuziqqdld/transpose
sequential/qzuziqqdld/Shape_1Shape#sequential/qzuziqqdld/transpose:y:0*
T0*
_output_shapes
:2
sequential/qzuziqqdld/Shape_1¤
+sequential/qzuziqqdld/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/qzuziqqdld/strided_slice_1/stack¨
-sequential/qzuziqqdld/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/qzuziqqdld/strided_slice_1/stack_1¨
-sequential/qzuziqqdld/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/qzuziqqdld/strided_slice_1/stack_2ò
%sequential/qzuziqqdld/strided_slice_1StridedSlice&sequential/qzuziqqdld/Shape_1:output:04sequential/qzuziqqdld/strided_slice_1/stack:output:06sequential/qzuziqqdld/strided_slice_1/stack_1:output:06sequential/qzuziqqdld/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/qzuziqqdld/strided_slice_1±
1sequential/qzuziqqdld/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/qzuziqqdld/TensorArrayV2/element_shape
#sequential/qzuziqqdld/TensorArrayV2TensorListReserve:sequential/qzuziqqdld/TensorArrayV2/element_shape:output:0.sequential/qzuziqqdld/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/qzuziqqdld/TensorArrayV2ë
Ksequential/qzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/qzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/qzuziqqdld/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/qzuziqqdld/transpose:y:0Tsequential/qzuziqqdld/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/qzuziqqdld/TensorArrayUnstack/TensorListFromTensor¤
+sequential/qzuziqqdld/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/qzuziqqdld/strided_slice_2/stack¨
-sequential/qzuziqqdld/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/qzuziqqdld/strided_slice_2/stack_1¨
-sequential/qzuziqqdld/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/qzuziqqdld/strided_slice_2/stack_2
%sequential/qzuziqqdld/strided_slice_2StridedSlice#sequential/qzuziqqdld/transpose:y:04sequential/qzuziqqdld/strided_slice_2/stack:output:06sequential/qzuziqqdld/strided_slice_2/stack_1:output:06sequential/qzuziqqdld/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/qzuziqqdld/strided_slice_2ñ
6sequential/qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp?sequential_qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOpÿ
'sequential/qzuziqqdld/aiccbsgdoo/MatMulMatMul.sequential/qzuziqqdld/strided_slice_2:output:0>sequential/qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/qzuziqqdld/aiccbsgdoo/MatMul÷
8sequential/qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOpAsequential_qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOpû
)sequential/qzuziqqdld/aiccbsgdoo/MatMul_1MatMul$sequential/qzuziqqdld/zeros:output:0@sequential/qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/qzuziqqdld/aiccbsgdoo/MatMul_1ð
$sequential/qzuziqqdld/aiccbsgdoo/addAddV21sequential/qzuziqqdld/aiccbsgdoo/MatMul:product:03sequential/qzuziqqdld/aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/qzuziqqdld/aiccbsgdoo/addð
7sequential/qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp@sequential_qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOpý
(sequential/qzuziqqdld/aiccbsgdoo/BiasAddBiasAdd(sequential/qzuziqqdld/aiccbsgdoo/add:z:0?sequential/qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/qzuziqqdld/aiccbsgdoo/BiasAdd¦
0sequential/qzuziqqdld/aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/qzuziqqdld/aiccbsgdoo/split/split_dimÃ
&sequential/qzuziqqdld/aiccbsgdoo/splitSplit9sequential/qzuziqqdld/aiccbsgdoo/split/split_dim:output:01sequential/qzuziqqdld/aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/qzuziqqdld/aiccbsgdoo/split×
/sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOpReadVariableOp8sequential_qzuziqqdld_aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOpæ
$sequential/qzuziqqdld/aiccbsgdoo/mulMul7sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp:value:0&sequential/qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/qzuziqqdld/aiccbsgdoo/mulæ
&sequential/qzuziqqdld/aiccbsgdoo/add_1AddV2/sequential/qzuziqqdld/aiccbsgdoo/split:output:0(sequential/qzuziqqdld/aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/add_1½
(sequential/qzuziqqdld/aiccbsgdoo/SigmoidSigmoid*sequential/qzuziqqdld/aiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/qzuziqqdld/aiccbsgdoo/SigmoidÝ
1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_1ReadVariableOp:sequential_qzuziqqdld_aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_1ì
&sequential/qzuziqqdld/aiccbsgdoo/mul_1Mul9sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_1:value:0&sequential/qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/mul_1è
&sequential/qzuziqqdld/aiccbsgdoo/add_2AddV2/sequential/qzuziqqdld/aiccbsgdoo/split:output:1*sequential/qzuziqqdld/aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/add_2Á
*sequential/qzuziqqdld/aiccbsgdoo/Sigmoid_1Sigmoid*sequential/qzuziqqdld/aiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/qzuziqqdld/aiccbsgdoo/Sigmoid_1á
&sequential/qzuziqqdld/aiccbsgdoo/mul_2Mul.sequential/qzuziqqdld/aiccbsgdoo/Sigmoid_1:y:0&sequential/qzuziqqdld/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/mul_2¹
%sequential/qzuziqqdld/aiccbsgdoo/TanhTanh/sequential/qzuziqqdld/aiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/qzuziqqdld/aiccbsgdoo/Tanhâ
&sequential/qzuziqqdld/aiccbsgdoo/mul_3Mul,sequential/qzuziqqdld/aiccbsgdoo/Sigmoid:y:0)sequential/qzuziqqdld/aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/mul_3ã
&sequential/qzuziqqdld/aiccbsgdoo/add_3AddV2*sequential/qzuziqqdld/aiccbsgdoo/mul_2:z:0*sequential/qzuziqqdld/aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/add_3Ý
1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_2ReadVariableOp:sequential_qzuziqqdld_aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_2ð
&sequential/qzuziqqdld/aiccbsgdoo/mul_4Mul9sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_2:value:0*sequential/qzuziqqdld/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/mul_4è
&sequential/qzuziqqdld/aiccbsgdoo/add_4AddV2/sequential/qzuziqqdld/aiccbsgdoo/split:output:3*sequential/qzuziqqdld/aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/add_4Á
*sequential/qzuziqqdld/aiccbsgdoo/Sigmoid_2Sigmoid*sequential/qzuziqqdld/aiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/qzuziqqdld/aiccbsgdoo/Sigmoid_2¸
'sequential/qzuziqqdld/aiccbsgdoo/Tanh_1Tanh*sequential/qzuziqqdld/aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/qzuziqqdld/aiccbsgdoo/Tanh_1æ
&sequential/qzuziqqdld/aiccbsgdoo/mul_5Mul.sequential/qzuziqqdld/aiccbsgdoo/Sigmoid_2:y:0+sequential/qzuziqqdld/aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/qzuziqqdld/aiccbsgdoo/mul_5»
3sequential/qzuziqqdld/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/qzuziqqdld/TensorArrayV2_1/element_shape
%sequential/qzuziqqdld/TensorArrayV2_1TensorListReserve<sequential/qzuziqqdld/TensorArrayV2_1/element_shape:output:0.sequential/qzuziqqdld/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/qzuziqqdld/TensorArrayV2_1z
sequential/qzuziqqdld/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/qzuziqqdld/time«
.sequential/qzuziqqdld/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/qzuziqqdld/while/maximum_iterations
(sequential/qzuziqqdld/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/qzuziqqdld/while/loop_counterø	
sequential/qzuziqqdld/whileWhile1sequential/qzuziqqdld/while/loop_counter:output:07sequential/qzuziqqdld/while/maximum_iterations:output:0#sequential/qzuziqqdld/time:output:0.sequential/qzuziqqdld/TensorArrayV2_1:handle:0$sequential/qzuziqqdld/zeros:output:0&sequential/qzuziqqdld/zeros_1:output:0.sequential/qzuziqqdld/strided_slice_1:output:0Msequential/qzuziqqdld/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_qzuziqqdld_aiccbsgdoo_matmul_readvariableop_resourceAsequential_qzuziqqdld_aiccbsgdoo_matmul_1_readvariableop_resource@sequential_qzuziqqdld_aiccbsgdoo_biasadd_readvariableop_resource8sequential_qzuziqqdld_aiccbsgdoo_readvariableop_resource:sequential_qzuziqqdld_aiccbsgdoo_readvariableop_1_resource:sequential_qzuziqqdld_aiccbsgdoo_readvariableop_2_resource*
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
(sequential_qzuziqqdld_while_body_1716741*4
cond,R*
(sequential_qzuziqqdld_while_cond_1716740*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/qzuziqqdld/whileá
Fsequential/qzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/qzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/qzuziqqdld/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/qzuziqqdld/while:output:3Osequential/qzuziqqdld/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/qzuziqqdld/TensorArrayV2Stack/TensorListStack­
+sequential/qzuziqqdld/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/qzuziqqdld/strided_slice_3/stack¨
-sequential/qzuziqqdld/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/qzuziqqdld/strided_slice_3/stack_1¨
-sequential/qzuziqqdld/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/qzuziqqdld/strided_slice_3/stack_2
%sequential/qzuziqqdld/strided_slice_3StridedSliceAsequential/qzuziqqdld/TensorArrayV2Stack/TensorListStack:tensor:04sequential/qzuziqqdld/strided_slice_3/stack:output:06sequential/qzuziqqdld/strided_slice_3/stack_1:output:06sequential/qzuziqqdld/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/qzuziqqdld/strided_slice_3¥
&sequential/qzuziqqdld/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/qzuziqqdld/transpose_1/permý
!sequential/qzuziqqdld/transpose_1	TransposeAsequential/qzuziqqdld/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/qzuziqqdld/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/qzuziqqdld/transpose_1Ï
+sequential/pbmomrqadp/MatMul/ReadVariableOpReadVariableOp4sequential_pbmomrqadp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/pbmomrqadp/MatMul/ReadVariableOpÝ
sequential/pbmomrqadp/MatMulMatMul.sequential/qzuziqqdld/strided_slice_3:output:03sequential/pbmomrqadp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/pbmomrqadp/MatMulÎ
,sequential/pbmomrqadp/BiasAdd/ReadVariableOpReadVariableOp5sequential_pbmomrqadp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/pbmomrqadp/BiasAdd/ReadVariableOpÙ
sequential/pbmomrqadp/BiasAddBiasAdd&sequential/pbmomrqadp/MatMul:product:04sequential/pbmomrqadp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/pbmomrqadp/BiasAdd	
IdentityIdentity&sequential/pbmomrqadp/BiasAdd:output:08^sequential/cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp7^sequential/cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp9^sequential/cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp0^sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp2^sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_12^sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_2^sequential/cxhqebqpjz/while9^sequential/ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp@^sequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp9^sequential/mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp@^sequential/mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp-^sequential/pbmomrqadp/BiasAdd/ReadVariableOp,^sequential/pbmomrqadp/MatMul/ReadVariableOp8^sequential/qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp7^sequential/qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp9^sequential/qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp0^sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp2^sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_12^sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_2^sequential/qzuziqqdld/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2r
7sequential/cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp7sequential/cxhqebqpjz/rvncypflgq/BiasAdd/ReadVariableOp2p
6sequential/cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp6sequential/cxhqebqpjz/rvncypflgq/MatMul/ReadVariableOp2t
8sequential/cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp8sequential/cxhqebqpjz/rvncypflgq/MatMul_1/ReadVariableOp2b
/sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp/sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp2f
1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_11sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_12f
1sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_21sequential/cxhqebqpjz/rvncypflgq/ReadVariableOp_22:
sequential/cxhqebqpjz/whilesequential/cxhqebqpjz/while2t
8sequential/ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp8sequential/ehhqjcwuju/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/ehhqjcwuju/squeeze_batch_dims/BiasAdd/ReadVariableOp2t
8sequential/mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp8sequential/mopvqfaljf/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/mopvqfaljf/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,sequential/pbmomrqadp/BiasAdd/ReadVariableOp,sequential/pbmomrqadp/BiasAdd/ReadVariableOp2Z
+sequential/pbmomrqadp/MatMul/ReadVariableOp+sequential/pbmomrqadp/MatMul/ReadVariableOp2r
7sequential/qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp7sequential/qzuziqqdld/aiccbsgdoo/BiasAdd/ReadVariableOp2p
6sequential/qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp6sequential/qzuziqqdld/aiccbsgdoo/MatMul/ReadVariableOp2t
8sequential/qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp8sequential/qzuziqqdld/aiccbsgdoo/MatMul_1/ReadVariableOp2b
/sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp/sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp2f
1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_11sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_12f
1sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_21sequential/qzuziqqdld/aiccbsgdoo/ReadVariableOp_22:
sequential/qzuziqqdld/whilesequential/qzuziqqdld/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
gvxdqcynan
àY

while_body_1719238
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_rvncypflgq_matmul_readvariableop_resource_0:	F
3while_rvncypflgq_matmul_1_readvariableop_resource_0:	 A
2while_rvncypflgq_biasadd_readvariableop_resource_0:	8
*while_rvncypflgq_readvariableop_resource_0: :
,while_rvncypflgq_readvariableop_1_resource_0: :
,while_rvncypflgq_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_rvncypflgq_matmul_readvariableop_resource:	D
1while_rvncypflgq_matmul_1_readvariableop_resource:	 ?
0while_rvncypflgq_biasadd_readvariableop_resource:	6
(while_rvncypflgq_readvariableop_resource: 8
*while_rvncypflgq_readvariableop_1_resource: 8
*while_rvncypflgq_readvariableop_2_resource: ¢'while/rvncypflgq/BiasAdd/ReadVariableOp¢&while/rvncypflgq/MatMul/ReadVariableOp¢(while/rvncypflgq/MatMul_1/ReadVariableOp¢while/rvncypflgq/ReadVariableOp¢!while/rvncypflgq/ReadVariableOp_1¢!while/rvncypflgq/ReadVariableOp_2Ã
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
&while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp1while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/rvncypflgq/MatMul/ReadVariableOpÑ
while/rvncypflgq/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMulÉ
(while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp3while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/rvncypflgq/MatMul_1/ReadVariableOpº
while/rvncypflgq/MatMul_1MatMulwhile_placeholder_20while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/MatMul_1°
while/rvncypflgq/addAddV2!while/rvncypflgq/MatMul:product:0#while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/addÂ
'while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp2while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/rvncypflgq/BiasAdd/ReadVariableOp½
while/rvncypflgq/BiasAddBiasAddwhile/rvncypflgq/add:z:0/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/rvncypflgq/BiasAdd
 while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/rvncypflgq/split/split_dim
while/rvncypflgq/splitSplit)while/rvncypflgq/split/split_dim:output:0!while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/rvncypflgq/split©
while/rvncypflgq/ReadVariableOpReadVariableOp*while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/rvncypflgq/ReadVariableOp£
while/rvncypflgq/mulMul'while/rvncypflgq/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul¦
while/rvncypflgq/add_1AddV2while/rvncypflgq/split:output:0while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_1
while/rvncypflgq/SigmoidSigmoidwhile/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid¯
!while/rvncypflgq/ReadVariableOp_1ReadVariableOp,while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_1©
while/rvncypflgq/mul_1Mul)while/rvncypflgq/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_1¨
while/rvncypflgq/add_2AddV2while/rvncypflgq/split:output:1while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_2
while/rvncypflgq/Sigmoid_1Sigmoidwhile/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_1
while/rvncypflgq/mul_2Mulwhile/rvncypflgq/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_2
while/rvncypflgq/TanhTanhwhile/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh¢
while/rvncypflgq/mul_3Mulwhile/rvncypflgq/Sigmoid:y:0while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_3£
while/rvncypflgq/add_3AddV2while/rvncypflgq/mul_2:z:0while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_3¯
!while/rvncypflgq/ReadVariableOp_2ReadVariableOp,while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/rvncypflgq/ReadVariableOp_2°
while/rvncypflgq/mul_4Mul)while/rvncypflgq/ReadVariableOp_2:value:0while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_4¨
while/rvncypflgq/add_4AddV2while/rvncypflgq/split:output:3while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/add_4
while/rvncypflgq/Sigmoid_2Sigmoidwhile/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Sigmoid_2
while/rvncypflgq/Tanh_1Tanhwhile/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/Tanh_1¦
while/rvncypflgq/mul_5Mulwhile/rvncypflgq/Sigmoid_2:y:0while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/rvncypflgq/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/rvncypflgq/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/rvncypflgq/mul_5:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/rvncypflgq/add_3:z:0(^while/rvncypflgq/BiasAdd/ReadVariableOp'^while/rvncypflgq/MatMul/ReadVariableOp)^while/rvncypflgq/MatMul_1/ReadVariableOp ^while/rvncypflgq/ReadVariableOp"^while/rvncypflgq/ReadVariableOp_1"^while/rvncypflgq/ReadVariableOp_2*
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
0while_rvncypflgq_biasadd_readvariableop_resource2while_rvncypflgq_biasadd_readvariableop_resource_0"h
1while_rvncypflgq_matmul_1_readvariableop_resource3while_rvncypflgq_matmul_1_readvariableop_resource_0"d
/while_rvncypflgq_matmul_readvariableop_resource1while_rvncypflgq_matmul_readvariableop_resource_0"Z
*while_rvncypflgq_readvariableop_1_resource,while_rvncypflgq_readvariableop_1_resource_0"Z
*while_rvncypflgq_readvariableop_2_resource,while_rvncypflgq_readvariableop_2_resource_0"V
(while_rvncypflgq_readvariableop_resource*while_rvncypflgq_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/rvncypflgq/BiasAdd/ReadVariableOp'while/rvncypflgq/BiasAdd/ReadVariableOp2P
&while/rvncypflgq/MatMul/ReadVariableOp&while/rvncypflgq/MatMul/ReadVariableOp2T
(while/rvncypflgq/MatMul_1/ReadVariableOp(while/rvncypflgq/MatMul_1/ReadVariableOp2B
while/rvncypflgq/ReadVariableOpwhile/rvncypflgq/ReadVariableOp2F
!while/rvncypflgq/ReadVariableOp_1!while/rvncypflgq/ReadVariableOp_12F
!while/rvncypflgq/ReadVariableOp_2!while/rvncypflgq/ReadVariableOp_2: 
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
±'
³
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1717693

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
¡h

G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1719339

inputs<
)rvncypflgq_matmul_readvariableop_resource:	>
+rvncypflgq_matmul_1_readvariableop_resource:	 9
*rvncypflgq_biasadd_readvariableop_resource:	0
"rvncypflgq_readvariableop_resource: 2
$rvncypflgq_readvariableop_1_resource: 2
$rvncypflgq_readvariableop_2_resource: 
identity¢!rvncypflgq/BiasAdd/ReadVariableOp¢ rvncypflgq/MatMul/ReadVariableOp¢"rvncypflgq/MatMul_1/ReadVariableOp¢rvncypflgq/ReadVariableOp¢rvncypflgq/ReadVariableOp_1¢rvncypflgq/ReadVariableOp_2¢whileD
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
 rvncypflgq/MatMul/ReadVariableOpReadVariableOp)rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 rvncypflgq/MatMul/ReadVariableOp§
rvncypflgq/MatMulMatMulstrided_slice_2:output:0(rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMulµ
"rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp+rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"rvncypflgq/MatMul_1/ReadVariableOp£
rvncypflgq/MatMul_1MatMulzeros:output:0*rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMul_1
rvncypflgq/addAddV2rvncypflgq/MatMul:product:0rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/add®
!rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp*rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!rvncypflgq/BiasAdd/ReadVariableOp¥
rvncypflgq/BiasAddBiasAddrvncypflgq/add:z:0)rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/BiasAddz
rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
rvncypflgq/split/split_dimë
rvncypflgq/splitSplit#rvncypflgq/split/split_dim:output:0rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rvncypflgq/split
rvncypflgq/ReadVariableOpReadVariableOp"rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp
rvncypflgq/mulMul!rvncypflgq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul
rvncypflgq/add_1AddV2rvncypflgq/split:output:0rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_1{
rvncypflgq/SigmoidSigmoidrvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid
rvncypflgq/ReadVariableOp_1ReadVariableOp$rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_1
rvncypflgq/mul_1Mul#rvncypflgq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_1
rvncypflgq/add_2AddV2rvncypflgq/split:output:1rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_2
rvncypflgq/Sigmoid_1Sigmoidrvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_1
rvncypflgq/mul_2Mulrvncypflgq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_2w
rvncypflgq/TanhTanhrvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh
rvncypflgq/mul_3Mulrvncypflgq/Sigmoid:y:0rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_3
rvncypflgq/add_3AddV2rvncypflgq/mul_2:z:0rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_3
rvncypflgq/ReadVariableOp_2ReadVariableOp$rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_2
rvncypflgq/mul_4Mul#rvncypflgq/ReadVariableOp_2:value:0rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_4
rvncypflgq/add_4AddV2rvncypflgq/split:output:3rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_4
rvncypflgq/Sigmoid_2Sigmoidrvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_2v
rvncypflgq/Tanh_1Tanhrvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh_1
rvncypflgq/mul_5Mulrvncypflgq/Sigmoid_2:y:0rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)rvncypflgq_matmul_readvariableop_resource+rvncypflgq_matmul_1_readvariableop_resource*rvncypflgq_biasadd_readvariableop_resource"rvncypflgq_readvariableop_resource$rvncypflgq_readvariableop_1_resource$rvncypflgq_readvariableop_2_resource*
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
while_body_1719238*
condR
while_cond_1719237*Q
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
IdentityIdentitytranspose_1:y:0"^rvncypflgq/BiasAdd/ReadVariableOp!^rvncypflgq/MatMul/ReadVariableOp#^rvncypflgq/MatMul_1/ReadVariableOp^rvncypflgq/ReadVariableOp^rvncypflgq/ReadVariableOp_1^rvncypflgq/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!rvncypflgq/BiasAdd/ReadVariableOp!rvncypflgq/BiasAdd/ReadVariableOp2D
 rvncypflgq/MatMul/ReadVariableOp rvncypflgq/MatMul/ReadVariableOp2H
"rvncypflgq/MatMul_1/ReadVariableOp"rvncypflgq/MatMul_1/ReadVariableOp26
rvncypflgq/ReadVariableOprvncypflgq/ReadVariableOp2:
rvncypflgq/ReadVariableOp_1rvncypflgq/ReadVariableOp_12:
rvncypflgq/ReadVariableOp_2rvncypflgq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç)
Ò
while_body_1717976
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_aiccbsgdoo_1718000_0:	 -
while_aiccbsgdoo_1718002_0:	 )
while_aiccbsgdoo_1718004_0:	(
while_aiccbsgdoo_1718006_0: (
while_aiccbsgdoo_1718008_0: (
while_aiccbsgdoo_1718010_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_aiccbsgdoo_1718000:	 +
while_aiccbsgdoo_1718002:	 '
while_aiccbsgdoo_1718004:	&
while_aiccbsgdoo_1718006: &
while_aiccbsgdoo_1718008: &
while_aiccbsgdoo_1718010: ¢(while/aiccbsgdoo/StatefulPartitionedCallÃ
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
(while/aiccbsgdoo/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_aiccbsgdoo_1718000_0while_aiccbsgdoo_1718002_0while_aiccbsgdoo_1718004_0while_aiccbsgdoo_1718006_0while_aiccbsgdoo_1718008_0while_aiccbsgdoo_1718010_0*
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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_17178802*
(while/aiccbsgdoo/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/aiccbsgdoo/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/aiccbsgdoo/StatefulPartitionedCall:output:1)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/aiccbsgdoo/StatefulPartitionedCall:output:2)^while/aiccbsgdoo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_aiccbsgdoo_1718000while_aiccbsgdoo_1718000_0"6
while_aiccbsgdoo_1718002while_aiccbsgdoo_1718002_0"6
while_aiccbsgdoo_1718004while_aiccbsgdoo_1718004_0"6
while_aiccbsgdoo_1718006while_aiccbsgdoo_1718006_0"6
while_aiccbsgdoo_1718008while_aiccbsgdoo_1718008_0"6
while_aiccbsgdoo_1718010while_aiccbsgdoo_1718010_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/aiccbsgdoo/StatefulPartitionedCall(while/aiccbsgdoo/StatefulPartitionedCall: 
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
while_cond_1721988
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1721988___redundant_placeholder05
1while_while_cond_1721988___redundant_placeholder15
1while_while_cond_1721988___redundant_placeholder25
1while_while_cond_1721988___redundant_placeholder35
1while_while_cond_1721988___redundant_placeholder45
1while_while_cond_1721988___redundant_placeholder55
1while_while_cond_1721988___redundant_placeholder6
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
(sequential_cxhqebqpjz_while_cond_1716564H
Dsequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_loop_counterN
Jsequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_maximum_iterations+
'sequential_cxhqebqpjz_while_placeholder-
)sequential_cxhqebqpjz_while_placeholder_1-
)sequential_cxhqebqpjz_while_placeholder_2-
)sequential_cxhqebqpjz_while_placeholder_3J
Fsequential_cxhqebqpjz_while_less_sequential_cxhqebqpjz_strided_slice_1a
]sequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_cond_1716564___redundant_placeholder0a
]sequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_cond_1716564___redundant_placeholder1a
]sequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_cond_1716564___redundant_placeholder2a
]sequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_cond_1716564___redundant_placeholder3a
]sequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_cond_1716564___redundant_placeholder4a
]sequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_cond_1716564___redundant_placeholder5a
]sequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_cond_1716564___redundant_placeholder6(
$sequential_cxhqebqpjz_while_identity
Þ
 sequential/cxhqebqpjz/while/LessLess'sequential_cxhqebqpjz_while_placeholderFsequential_cxhqebqpjz_while_less_sequential_cxhqebqpjz_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/cxhqebqpjz/while/Less
$sequential/cxhqebqpjz/while/IdentityIdentity$sequential/cxhqebqpjz/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/cxhqebqpjz/while/Identity"U
$sequential_cxhqebqpjz_while_identity-sequential/cxhqebqpjz/while/Identity:output:0*(
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
Ü

(sequential_cxhqebqpjz_while_body_1716565H
Dsequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_loop_counterN
Jsequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_maximum_iterations+
'sequential_cxhqebqpjz_while_placeholder-
)sequential_cxhqebqpjz_while_placeholder_1-
)sequential_cxhqebqpjz_while_placeholder_2-
)sequential_cxhqebqpjz_while_placeholder_3G
Csequential_cxhqebqpjz_while_sequential_cxhqebqpjz_strided_slice_1_0
sequential_cxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_sequential_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0:	\
Isequential_cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0:	 W
Hsequential_cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0:	N
@sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0: P
Bsequential_cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0: P
Bsequential_cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0: (
$sequential_cxhqebqpjz_while_identity*
&sequential_cxhqebqpjz_while_identity_1*
&sequential_cxhqebqpjz_while_identity_2*
&sequential_cxhqebqpjz_while_identity_3*
&sequential_cxhqebqpjz_while_identity_4*
&sequential_cxhqebqpjz_while_identity_5E
Asequential_cxhqebqpjz_while_sequential_cxhqebqpjz_strided_slice_1
}sequential_cxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_sequential_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensorX
Esequential_cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource:	Z
Gsequential_cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource:	 U
Fsequential_cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource:	L
>sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_resource: N
@sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource: N
@sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource: ¢=sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp¢<sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp¢>sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp¢5sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp¢7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1¢7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2ï
Msequential/cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_cxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_sequential_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0'sequential_cxhqebqpjz_while_placeholderVsequential/cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem
<sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOpReadVariableOpGsequential_cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp©
-sequential/cxhqebqpjz/while/rvncypflgq/MatMulMatMulFsequential/cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/cxhqebqpjz/while/rvncypflgq/MatMul
>sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOpIsequential_cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp
/sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1MatMul)sequential_cxhqebqpjz_while_placeholder_2Fsequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1
*sequential/cxhqebqpjz/while/rvncypflgq/addAddV27sequential/cxhqebqpjz/while/rvncypflgq/MatMul:product:09sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/cxhqebqpjz/while/rvncypflgq/add
=sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOpHsequential_cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp
.sequential/cxhqebqpjz/while/rvncypflgq/BiasAddBiasAdd.sequential/cxhqebqpjz/while/rvncypflgq/add:z:0Esequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd²
6sequential/cxhqebqpjz/while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/cxhqebqpjz/while/rvncypflgq/split/split_dimÛ
,sequential/cxhqebqpjz/while/rvncypflgq/splitSplit?sequential/cxhqebqpjz/while/rvncypflgq/split/split_dim:output:07sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/cxhqebqpjz/while/rvncypflgq/splitë
5sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOpReadVariableOp@sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOpû
*sequential/cxhqebqpjz/while/rvncypflgq/mulMul=sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp:value:0)sequential_cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/cxhqebqpjz/while/rvncypflgq/mulþ
,sequential/cxhqebqpjz/while/rvncypflgq/add_1AddV25sequential/cxhqebqpjz/while/rvncypflgq/split:output:0.sequential/cxhqebqpjz/while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/add_1Ï
.sequential/cxhqebqpjz/while/rvncypflgq/SigmoidSigmoid0sequential/cxhqebqpjz/while/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/cxhqebqpjz/while/rvncypflgq/Sigmoidñ
7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1ReadVariableOpBsequential_cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1
,sequential/cxhqebqpjz/while/rvncypflgq/mul_1Mul?sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1:value:0)sequential_cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/mul_1
,sequential/cxhqebqpjz/while/rvncypflgq/add_2AddV25sequential/cxhqebqpjz/while/rvncypflgq/split:output:10sequential/cxhqebqpjz/while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/add_2Ó
0sequential/cxhqebqpjz/while/rvncypflgq/Sigmoid_1Sigmoid0sequential/cxhqebqpjz/while/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/cxhqebqpjz/while/rvncypflgq/Sigmoid_1ö
,sequential/cxhqebqpjz/while/rvncypflgq/mul_2Mul4sequential/cxhqebqpjz/while/rvncypflgq/Sigmoid_1:y:0)sequential_cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/mul_2Ë
+sequential/cxhqebqpjz/while/rvncypflgq/TanhTanh5sequential/cxhqebqpjz/while/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/cxhqebqpjz/while/rvncypflgq/Tanhú
,sequential/cxhqebqpjz/while/rvncypflgq/mul_3Mul2sequential/cxhqebqpjz/while/rvncypflgq/Sigmoid:y:0/sequential/cxhqebqpjz/while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/mul_3û
,sequential/cxhqebqpjz/while/rvncypflgq/add_3AddV20sequential/cxhqebqpjz/while/rvncypflgq/mul_2:z:00sequential/cxhqebqpjz/while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/add_3ñ
7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2ReadVariableOpBsequential_cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2
,sequential/cxhqebqpjz/while/rvncypflgq/mul_4Mul?sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2:value:00sequential/cxhqebqpjz/while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/mul_4
,sequential/cxhqebqpjz/while/rvncypflgq/add_4AddV25sequential/cxhqebqpjz/while/rvncypflgq/split:output:30sequential/cxhqebqpjz/while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/add_4Ó
0sequential/cxhqebqpjz/while/rvncypflgq/Sigmoid_2Sigmoid0sequential/cxhqebqpjz/while/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/cxhqebqpjz/while/rvncypflgq/Sigmoid_2Ê
-sequential/cxhqebqpjz/while/rvncypflgq/Tanh_1Tanh0sequential/cxhqebqpjz/while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/cxhqebqpjz/while/rvncypflgq/Tanh_1þ
,sequential/cxhqebqpjz/while/rvncypflgq/mul_5Mul4sequential/cxhqebqpjz/while/rvncypflgq/Sigmoid_2:y:01sequential/cxhqebqpjz/while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/cxhqebqpjz/while/rvncypflgq/mul_5Ì
@sequential/cxhqebqpjz/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_cxhqebqpjz_while_placeholder_1'sequential_cxhqebqpjz_while_placeholder0sequential/cxhqebqpjz/while/rvncypflgq/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/cxhqebqpjz/while/TensorArrayV2Write/TensorListSetItem
!sequential/cxhqebqpjz/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/cxhqebqpjz/while/add/yÁ
sequential/cxhqebqpjz/while/addAddV2'sequential_cxhqebqpjz_while_placeholder*sequential/cxhqebqpjz/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/cxhqebqpjz/while/add
#sequential/cxhqebqpjz/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/cxhqebqpjz/while/add_1/yä
!sequential/cxhqebqpjz/while/add_1AddV2Dsequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_loop_counter,sequential/cxhqebqpjz/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/cxhqebqpjz/while/add_1
$sequential/cxhqebqpjz/while/IdentityIdentity%sequential/cxhqebqpjz/while/add_1:z:0>^sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp=^sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp?^sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp6^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp8^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_18^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/cxhqebqpjz/while/Identityµ
&sequential/cxhqebqpjz/while/Identity_1IdentityJsequential_cxhqebqpjz_while_sequential_cxhqebqpjz_while_maximum_iterations>^sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp=^sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp?^sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp6^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp8^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_18^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/cxhqebqpjz/while/Identity_1
&sequential/cxhqebqpjz/while/Identity_2Identity#sequential/cxhqebqpjz/while/add:z:0>^sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp=^sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp?^sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp6^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp8^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_18^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/cxhqebqpjz/while/Identity_2»
&sequential/cxhqebqpjz/while/Identity_3IdentityPsequential/cxhqebqpjz/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp=^sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp?^sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp6^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp8^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_18^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/cxhqebqpjz/while/Identity_3¬
&sequential/cxhqebqpjz/while/Identity_4Identity0sequential/cxhqebqpjz/while/rvncypflgq/mul_5:z:0>^sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp=^sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp?^sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp6^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp8^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_18^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/while/Identity_4¬
&sequential/cxhqebqpjz/while/Identity_5Identity0sequential/cxhqebqpjz/while/rvncypflgq/add_3:z:0>^sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp=^sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp?^sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp6^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp8^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_18^sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/cxhqebqpjz/while/Identity_5"U
$sequential_cxhqebqpjz_while_identity-sequential/cxhqebqpjz/while/Identity:output:0"Y
&sequential_cxhqebqpjz_while_identity_1/sequential/cxhqebqpjz/while/Identity_1:output:0"Y
&sequential_cxhqebqpjz_while_identity_2/sequential/cxhqebqpjz/while/Identity_2:output:0"Y
&sequential_cxhqebqpjz_while_identity_3/sequential/cxhqebqpjz/while/Identity_3:output:0"Y
&sequential_cxhqebqpjz_while_identity_4/sequential/cxhqebqpjz/while/Identity_4:output:0"Y
&sequential_cxhqebqpjz_while_identity_5/sequential/cxhqebqpjz/while/Identity_5:output:0"
Fsequential_cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resourceHsequential_cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0"
Gsequential_cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resourceIsequential_cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0"
Esequential_cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resourceGsequential_cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0"
@sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_1_resourceBsequential_cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0"
@sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_2_resourceBsequential_cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0"
>sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_resource@sequential_cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0"
Asequential_cxhqebqpjz_while_sequential_cxhqebqpjz_strided_slice_1Csequential_cxhqebqpjz_while_sequential_cxhqebqpjz_strided_slice_1_0"
}sequential_cxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_sequential_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensorsequential_cxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_sequential_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp=sequential/cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2|
<sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp<sequential/cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp2
>sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp>sequential/cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp2n
5sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp5sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp2r
7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_17sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_12r
7sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_27sequential/cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2: 
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
Ó

,__inference_qzuziqqdld_layer_call_fn_1722321

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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17188462
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


cxhqebqpjz_while_cond_17198452
.cxhqebqpjz_while_cxhqebqpjz_while_loop_counter8
4cxhqebqpjz_while_cxhqebqpjz_while_maximum_iterations 
cxhqebqpjz_while_placeholder"
cxhqebqpjz_while_placeholder_1"
cxhqebqpjz_while_placeholder_2"
cxhqebqpjz_while_placeholder_34
0cxhqebqpjz_while_less_cxhqebqpjz_strided_slice_1K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1719845___redundant_placeholder0K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1719845___redundant_placeholder1K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1719845___redundant_placeholder2K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1719845___redundant_placeholder3K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1719845___redundant_placeholder4K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1719845___redundant_placeholder5K
Gcxhqebqpjz_while_cxhqebqpjz_while_cond_1719845___redundant_placeholder6
cxhqebqpjz_while_identity
§
cxhqebqpjz/while/LessLesscxhqebqpjz_while_placeholder0cxhqebqpjz_while_less_cxhqebqpjz_strided_slice_1*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Less~
cxhqebqpjz/while/IdentityIdentitycxhqebqpjz/while/Less:z:0*
T0
*
_output_shapes
: 2
cxhqebqpjz/while/Identity"?
cxhqebqpjz_while_identity"cxhqebqpjz/while/Identity:output:0*(
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
È

,__inference_mopvqfaljf_layer_call_fn_1720696

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
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_17184082
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
À
×
#__inference__traced_restore_1722916
file_prefix8
"assignvariableop_mopvqfaljf_kernel:0
"assignvariableop_1_mopvqfaljf_bias::
$assignvariableop_2_ehhqjcwuju_kernel:0
"assignvariableop_3_ehhqjcwuju_bias:6
$assignvariableop_4_pbmomrqadp_kernel: 0
"assignvariableop_5_pbmomrqadp_bias:)
assignvariableop_6_rmsprop_iter:	 *
 assignvariableop_7_rmsprop_decay: 2
(assignvariableop_8_rmsprop_learning_rate: -
#assignvariableop_9_rmsprop_momentum: )
assignvariableop_10_rmsprop_rho: C
0assignvariableop_11_cxhqebqpjz_rvncypflgq_kernel:	M
:assignvariableop_12_cxhqebqpjz_rvncypflgq_recurrent_kernel:	 =
.assignvariableop_13_cxhqebqpjz_rvncypflgq_bias:	S
Eassignvariableop_14_cxhqebqpjz_rvncypflgq_input_gate_peephole_weights: T
Fassignvariableop_15_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weights: T
Fassignvariableop_16_cxhqebqpjz_rvncypflgq_output_gate_peephole_weights: C
0assignvariableop_17_qzuziqqdld_aiccbsgdoo_kernel:	 M
:assignvariableop_18_qzuziqqdld_aiccbsgdoo_recurrent_kernel:	 =
.assignvariableop_19_qzuziqqdld_aiccbsgdoo_bias:	S
Eassignvariableop_20_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weights: T
Fassignvariableop_21_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weights: T
Fassignvariableop_22_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weights: #
assignvariableop_23_total: #
assignvariableop_24_count: G
1assignvariableop_25_rmsprop_mopvqfaljf_kernel_rms:=
/assignvariableop_26_rmsprop_mopvqfaljf_bias_rms:G
1assignvariableop_27_rmsprop_ehhqjcwuju_kernel_rms:=
/assignvariableop_28_rmsprop_ehhqjcwuju_bias_rms:C
1assignvariableop_29_rmsprop_pbmomrqadp_kernel_rms: =
/assignvariableop_30_rmsprop_pbmomrqadp_bias_rms:O
<assignvariableop_31_rmsprop_cxhqebqpjz_rvncypflgq_kernel_rms:	Y
Fassignvariableop_32_rmsprop_cxhqebqpjz_rvncypflgq_recurrent_kernel_rms:	 I
:assignvariableop_33_rmsprop_cxhqebqpjz_rvncypflgq_bias_rms:	_
Qassignvariableop_34_rmsprop_cxhqebqpjz_rvncypflgq_input_gate_peephole_weights_rms: `
Rassignvariableop_35_rmsprop_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weights_rms: `
Rassignvariableop_36_rmsprop_cxhqebqpjz_rvncypflgq_output_gate_peephole_weights_rms: O
<assignvariableop_37_rmsprop_qzuziqqdld_aiccbsgdoo_kernel_rms:	 Y
Fassignvariableop_38_rmsprop_qzuziqqdld_aiccbsgdoo_recurrent_kernel_rms:	 I
:assignvariableop_39_rmsprop_qzuziqqdld_aiccbsgdoo_bias_rms:	_
Qassignvariableop_40_rmsprop_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weights_rms: `
Rassignvariableop_41_rmsprop_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weights_rms: `
Rassignvariableop_42_rmsprop_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weights_rms: 
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ó
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*ÿ
valueõBò,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/14/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/15/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesæ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_mopvqfaljf_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_mopvqfaljf_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_ehhqjcwuju_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_ehhqjcwuju_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4©
AssignVariableOp_4AssignVariableOp$assignvariableop_4_pbmomrqadp_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_pbmomrqadp_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_rmsprop_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8­
AssignVariableOp_8AssignVariableOp(assignvariableop_8_rmsprop_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¨
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rmsprop_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10§
AssignVariableOp_10AssignVariableOpassignvariableop_10_rmsprop_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¸
AssignVariableOp_11AssignVariableOp0assignvariableop_11_cxhqebqpjz_rvncypflgq_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Â
AssignVariableOp_12AssignVariableOp:assignvariableop_12_cxhqebqpjz_rvncypflgq_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¶
AssignVariableOp_13AssignVariableOp.assignvariableop_13_cxhqebqpjz_rvncypflgq_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Í
AssignVariableOp_14AssignVariableOpEassignvariableop_14_cxhqebqpjz_rvncypflgq_input_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Î
AssignVariableOp_15AssignVariableOpFassignvariableop_15_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weightsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Î
AssignVariableOp_16AssignVariableOpFassignvariableop_16_cxhqebqpjz_rvncypflgq_output_gate_peephole_weightsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¸
AssignVariableOp_17AssignVariableOp0assignvariableop_17_qzuziqqdld_aiccbsgdoo_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Â
AssignVariableOp_18AssignVariableOp:assignvariableop_18_qzuziqqdld_aiccbsgdoo_recurrent_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¶
AssignVariableOp_19AssignVariableOp.assignvariableop_19_qzuziqqdld_aiccbsgdoo_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Í
AssignVariableOp_20AssignVariableOpEassignvariableop_20_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Î
AssignVariableOp_21AssignVariableOpFassignvariableop_21_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weightsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Î
AssignVariableOp_22AssignVariableOpFassignvariableop_22_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weightsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¡
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¡
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_mopvqfaljf_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_mopvqfaljf_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¹
AssignVariableOp_27AssignVariableOp1assignvariableop_27_rmsprop_ehhqjcwuju_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28·
AssignVariableOp_28AssignVariableOp/assignvariableop_28_rmsprop_ehhqjcwuju_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¹
AssignVariableOp_29AssignVariableOp1assignvariableop_29_rmsprop_pbmomrqadp_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30·
AssignVariableOp_30AssignVariableOp/assignvariableop_30_rmsprop_pbmomrqadp_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ä
AssignVariableOp_31AssignVariableOp<assignvariableop_31_rmsprop_cxhqebqpjz_rvncypflgq_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Î
AssignVariableOp_32AssignVariableOpFassignvariableop_32_rmsprop_cxhqebqpjz_rvncypflgq_recurrent_kernel_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Â
AssignVariableOp_33AssignVariableOp:assignvariableop_33_rmsprop_cxhqebqpjz_rvncypflgq_bias_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ù
AssignVariableOp_34AssignVariableOpQassignvariableop_34_rmsprop_cxhqebqpjz_rvncypflgq_input_gate_peephole_weights_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ú
AssignVariableOp_35AssignVariableOpRassignvariableop_35_rmsprop_cxhqebqpjz_rvncypflgq_forget_gate_peephole_weights_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ú
AssignVariableOp_36AssignVariableOpRassignvariableop_36_rmsprop_cxhqebqpjz_rvncypflgq_output_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ä
AssignVariableOp_37AssignVariableOp<assignvariableop_37_rmsprop_qzuziqqdld_aiccbsgdoo_kernel_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Î
AssignVariableOp_38AssignVariableOpFassignvariableop_38_rmsprop_qzuziqqdld_aiccbsgdoo_recurrent_kernel_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Â
AssignVariableOp_39AssignVariableOp:assignvariableop_39_rmsprop_qzuziqqdld_aiccbsgdoo_bias_rmsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ù
AssignVariableOp_40AssignVariableOpQassignvariableop_40_rmsprop_qzuziqqdld_aiccbsgdoo_input_gate_peephole_weights_rmsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ú
AssignVariableOp_41AssignVariableOpRassignvariableop_41_rmsprop_qzuziqqdld_aiccbsgdoo_forget_gate_peephole_weights_rmsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ú
AssignVariableOp_42AssignVariableOpRassignvariableop_42_rmsprop_qzuziqqdld_aiccbsgdoo_output_gate_peephole_weights_rmsIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
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
¦h

G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1722090

inputs<
)aiccbsgdoo_matmul_readvariableop_resource:	 >
+aiccbsgdoo_matmul_1_readvariableop_resource:	 9
*aiccbsgdoo_biasadd_readvariableop_resource:	0
"aiccbsgdoo_readvariableop_resource: 2
$aiccbsgdoo_readvariableop_1_resource: 2
$aiccbsgdoo_readvariableop_2_resource: 
identity¢!aiccbsgdoo/BiasAdd/ReadVariableOp¢ aiccbsgdoo/MatMul/ReadVariableOp¢"aiccbsgdoo/MatMul_1/ReadVariableOp¢aiccbsgdoo/ReadVariableOp¢aiccbsgdoo/ReadVariableOp_1¢aiccbsgdoo/ReadVariableOp_2¢whileD
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
 aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp)aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aiccbsgdoo/MatMul/ReadVariableOp§
aiccbsgdoo/MatMulMatMulstrided_slice_2:output:0(aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMulµ
"aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp+aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aiccbsgdoo/MatMul_1/ReadVariableOp£
aiccbsgdoo/MatMul_1MatMulzeros:output:0*aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMul_1
aiccbsgdoo/addAddV2aiccbsgdoo/MatMul:product:0aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/add®
!aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp*aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aiccbsgdoo/BiasAdd/ReadVariableOp¥
aiccbsgdoo/BiasAddBiasAddaiccbsgdoo/add:z:0)aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/BiasAddz
aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aiccbsgdoo/split/split_dimë
aiccbsgdoo/splitSplit#aiccbsgdoo/split/split_dim:output:0aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aiccbsgdoo/split
aiccbsgdoo/ReadVariableOpReadVariableOp"aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp
aiccbsgdoo/mulMul!aiccbsgdoo/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul
aiccbsgdoo/add_1AddV2aiccbsgdoo/split:output:0aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_1{
aiccbsgdoo/SigmoidSigmoidaiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid
aiccbsgdoo/ReadVariableOp_1ReadVariableOp$aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_1
aiccbsgdoo/mul_1Mul#aiccbsgdoo/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_1
aiccbsgdoo/add_2AddV2aiccbsgdoo/split:output:1aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_2
aiccbsgdoo/Sigmoid_1Sigmoidaiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_1
aiccbsgdoo/mul_2Mulaiccbsgdoo/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_2w
aiccbsgdoo/TanhTanhaiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh
aiccbsgdoo/mul_3Mulaiccbsgdoo/Sigmoid:y:0aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_3
aiccbsgdoo/add_3AddV2aiccbsgdoo/mul_2:z:0aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_3
aiccbsgdoo/ReadVariableOp_2ReadVariableOp$aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_2
aiccbsgdoo/mul_4Mul#aiccbsgdoo/ReadVariableOp_2:value:0aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_4
aiccbsgdoo/add_4AddV2aiccbsgdoo/split:output:3aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_4
aiccbsgdoo/Sigmoid_2Sigmoidaiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_2v
aiccbsgdoo/Tanh_1Tanhaiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh_1
aiccbsgdoo/mul_5Mulaiccbsgdoo/Sigmoid_2:y:0aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aiccbsgdoo_matmul_readvariableop_resource+aiccbsgdoo_matmul_1_readvariableop_resource*aiccbsgdoo_biasadd_readvariableop_resource"aiccbsgdoo_readvariableop_resource$aiccbsgdoo_readvariableop_1_resource$aiccbsgdoo_readvariableop_2_resource*
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
while_body_1721989*
condR
while_cond_1721988*Q
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
IdentityIdentitystrided_slice_3:output:0"^aiccbsgdoo/BiasAdd/ReadVariableOp!^aiccbsgdoo/MatMul/ReadVariableOp#^aiccbsgdoo/MatMul_1/ReadVariableOp^aiccbsgdoo/ReadVariableOp^aiccbsgdoo/ReadVariableOp_1^aiccbsgdoo/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aiccbsgdoo/BiasAdd/ReadVariableOp!aiccbsgdoo/BiasAdd/ReadVariableOp2D
 aiccbsgdoo/MatMul/ReadVariableOp aiccbsgdoo/MatMul/ReadVariableOp2H
"aiccbsgdoo/MatMul_1/ReadVariableOp"aiccbsgdoo/MatMul_1/ReadVariableOp26
aiccbsgdoo/ReadVariableOpaiccbsgdoo/ReadVariableOp2:
aiccbsgdoo/ReadVariableOp_1aiccbsgdoo/ReadVariableOp_12:
aiccbsgdoo/ReadVariableOp_2aiccbsgdoo/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç)
Ò
while_body_1717218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_rvncypflgq_1717242_0:	-
while_rvncypflgq_1717244_0:	 )
while_rvncypflgq_1717246_0:	(
while_rvncypflgq_1717248_0: (
while_rvncypflgq_1717250_0: (
while_rvncypflgq_1717252_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_rvncypflgq_1717242:	+
while_rvncypflgq_1717244:	 '
while_rvncypflgq_1717246:	&
while_rvncypflgq_1717248: &
while_rvncypflgq_1717250: &
while_rvncypflgq_1717252: ¢(while/rvncypflgq/StatefulPartitionedCallÃ
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
(while/rvncypflgq/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_rvncypflgq_1717242_0while_rvncypflgq_1717244_0while_rvncypflgq_1717246_0while_rvncypflgq_1717248_0while_rvncypflgq_1717250_0while_rvncypflgq_1717252_0*
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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_17171222*
(while/rvncypflgq/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/rvncypflgq/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/rvncypflgq/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/rvncypflgq/StatefulPartitionedCall:output:1)^while/rvncypflgq/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/rvncypflgq/StatefulPartitionedCall:output:2)^while/rvncypflgq/StatefulPartitionedCall*
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
while_rvncypflgq_1717242while_rvncypflgq_1717242_0"6
while_rvncypflgq_1717244while_rvncypflgq_1717244_0"6
while_rvncypflgq_1717246while_rvncypflgq_1717246_0"6
while_rvncypflgq_1717248while_rvncypflgq_1717248_0"6
while_rvncypflgq_1717250while_rvncypflgq_1717250_0"6
while_rvncypflgq_1717252while_rvncypflgq_1717252_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/rvncypflgq/StatefulPartitionedCall(while/rvncypflgq/StatefulPartitionedCall: 
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
while_cond_1719023
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1719023___redundant_placeholder05
1while_while_cond_1719023___redundant_placeholder15
1while_while_cond_1719023___redundant_placeholder25
1while_while_cond_1719023___redundant_placeholder35
1while_while_cond_1719023___redundant_placeholder45
1while_while_cond_1719023___redundant_placeholder55
1while_while_cond_1719023___redundant_placeholder6
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

c
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_1720757

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
¯F
ê
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1718056

inputs%
aiccbsgdoo_1717957:	 %
aiccbsgdoo_1717959:	 !
aiccbsgdoo_1717961:	 
aiccbsgdoo_1717963:  
aiccbsgdoo_1717965:  
aiccbsgdoo_1717967: 
identity¢"aiccbsgdoo/StatefulPartitionedCall¢whileD
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
"aiccbsgdoo/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0aiccbsgdoo_1717957aiccbsgdoo_1717959aiccbsgdoo_1717961aiccbsgdoo_1717963aiccbsgdoo_1717965aiccbsgdoo_1717967*
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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_17178802$
"aiccbsgdoo/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0aiccbsgdoo_1717957aiccbsgdoo_1717959aiccbsgdoo_1717961aiccbsgdoo_1717963aiccbsgdoo_1717965aiccbsgdoo_1717967*
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
while_body_1717976*
condR
while_cond_1717975*Q
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
IdentityIdentitystrided_slice_3:output:0#^aiccbsgdoo/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"aiccbsgdoo/StatefulPartitionedCall"aiccbsgdoo/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý
H
,__inference_abbthhzbau_layer_call_fn_1720762

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
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_17184722
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
p
Ê
cxhqebqpjz_while_body_17202852
.cxhqebqpjz_while_cxhqebqpjz_while_loop_counter8
4cxhqebqpjz_while_cxhqebqpjz_while_maximum_iterations 
cxhqebqpjz_while_placeholder"
cxhqebqpjz_while_placeholder_1"
cxhqebqpjz_while_placeholder_2"
cxhqebqpjz_while_placeholder_31
-cxhqebqpjz_while_cxhqebqpjz_strided_slice_1_0m
icxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0O
<cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0:	Q
>cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0:	 L
=cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0:	C
5cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0: E
7cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0: E
7cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0: 
cxhqebqpjz_while_identity
cxhqebqpjz_while_identity_1
cxhqebqpjz_while_identity_2
cxhqebqpjz_while_identity_3
cxhqebqpjz_while_identity_4
cxhqebqpjz_while_identity_5/
+cxhqebqpjz_while_cxhqebqpjz_strided_slice_1k
gcxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensorM
:cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource:	O
<cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource:	 J
;cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource:	A
3cxhqebqpjz_while_rvncypflgq_readvariableop_resource: C
5cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource: C
5cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource: ¢2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp¢1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp¢3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp¢*cxhqebqpjz/while/rvncypflgq/ReadVariableOp¢,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1¢,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2Ù
Bcxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bcxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shape
4cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemicxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0cxhqebqpjz_while_placeholderKcxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItemä
1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOpReadVariableOp<cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOpý
"cxhqebqpjz/while/rvncypflgq/MatMulMatMul;cxhqebqpjz/while/TensorArrayV2Read/TensorListGetItem:item:09cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"cxhqebqpjz/while/rvncypflgq/MatMulê
3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp>cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOpæ
$cxhqebqpjz/while/rvncypflgq/MatMul_1MatMulcxhqebqpjz_while_placeholder_2;cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$cxhqebqpjz/while/rvncypflgq/MatMul_1Ü
cxhqebqpjz/while/rvncypflgq/addAddV2,cxhqebqpjz/while/rvncypflgq/MatMul:product:0.cxhqebqpjz/while/rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
cxhqebqpjz/while/rvncypflgq/addã
2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp=cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOpé
#cxhqebqpjz/while/rvncypflgq/BiasAddBiasAdd#cxhqebqpjz/while/rvncypflgq/add:z:0:cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#cxhqebqpjz/while/rvncypflgq/BiasAdd
+cxhqebqpjz/while/rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+cxhqebqpjz/while/rvncypflgq/split/split_dim¯
!cxhqebqpjz/while/rvncypflgq/splitSplit4cxhqebqpjz/while/rvncypflgq/split/split_dim:output:0,cxhqebqpjz/while/rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!cxhqebqpjz/while/rvncypflgq/splitÊ
*cxhqebqpjz/while/rvncypflgq/ReadVariableOpReadVariableOp5cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*cxhqebqpjz/while/rvncypflgq/ReadVariableOpÏ
cxhqebqpjz/while/rvncypflgq/mulMul2cxhqebqpjz/while/rvncypflgq/ReadVariableOp:value:0cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
cxhqebqpjz/while/rvncypflgq/mulÒ
!cxhqebqpjz/while/rvncypflgq/add_1AddV2*cxhqebqpjz/while/rvncypflgq/split:output:0#cxhqebqpjz/while/rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_1®
#cxhqebqpjz/while/rvncypflgq/SigmoidSigmoid%cxhqebqpjz/while/rvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#cxhqebqpjz/while/rvncypflgq/SigmoidÐ
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1ReadVariableOp7cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1Õ
!cxhqebqpjz/while/rvncypflgq/mul_1Mul4cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1:value:0cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_1Ô
!cxhqebqpjz/while/rvncypflgq/add_2AddV2*cxhqebqpjz/while/rvncypflgq/split:output:1%cxhqebqpjz/while/rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_2²
%cxhqebqpjz/while/rvncypflgq/Sigmoid_1Sigmoid%cxhqebqpjz/while/rvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%cxhqebqpjz/while/rvncypflgq/Sigmoid_1Ê
!cxhqebqpjz/while/rvncypflgq/mul_2Mul)cxhqebqpjz/while/rvncypflgq/Sigmoid_1:y:0cxhqebqpjz_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_2ª
 cxhqebqpjz/while/rvncypflgq/TanhTanh*cxhqebqpjz/while/rvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 cxhqebqpjz/while/rvncypflgq/TanhÎ
!cxhqebqpjz/while/rvncypflgq/mul_3Mul'cxhqebqpjz/while/rvncypflgq/Sigmoid:y:0$cxhqebqpjz/while/rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_3Ï
!cxhqebqpjz/while/rvncypflgq/add_3AddV2%cxhqebqpjz/while/rvncypflgq/mul_2:z:0%cxhqebqpjz/while/rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_3Ð
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2ReadVariableOp7cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2Ü
!cxhqebqpjz/while/rvncypflgq/mul_4Mul4cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2:value:0%cxhqebqpjz/while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_4Ô
!cxhqebqpjz/while/rvncypflgq/add_4AddV2*cxhqebqpjz/while/rvncypflgq/split:output:3%cxhqebqpjz/while/rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/add_4²
%cxhqebqpjz/while/rvncypflgq/Sigmoid_2Sigmoid%cxhqebqpjz/while/rvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%cxhqebqpjz/while/rvncypflgq/Sigmoid_2©
"cxhqebqpjz/while/rvncypflgq/Tanh_1Tanh%cxhqebqpjz/while/rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"cxhqebqpjz/while/rvncypflgq/Tanh_1Ò
!cxhqebqpjz/while/rvncypflgq/mul_5Mul)cxhqebqpjz/while/rvncypflgq/Sigmoid_2:y:0&cxhqebqpjz/while/rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!cxhqebqpjz/while/rvncypflgq/mul_5
5cxhqebqpjz/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcxhqebqpjz_while_placeholder_1cxhqebqpjz_while_placeholder%cxhqebqpjz/while/rvncypflgq/mul_5:z:0*
_output_shapes
: *
element_dtype027
5cxhqebqpjz/while/TensorArrayV2Write/TensorListSetItemr
cxhqebqpjz/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cxhqebqpjz/while/add/y
cxhqebqpjz/while/addAddV2cxhqebqpjz_while_placeholdercxhqebqpjz/while/add/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/while/addv
cxhqebqpjz/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cxhqebqpjz/while/add_1/y­
cxhqebqpjz/while/add_1AddV2.cxhqebqpjz_while_cxhqebqpjz_while_loop_counter!cxhqebqpjz/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cxhqebqpjz/while/add_1©
cxhqebqpjz/while/IdentityIdentitycxhqebqpjz/while/add_1:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/IdentityÇ
cxhqebqpjz/while/Identity_1Identity4cxhqebqpjz_while_cxhqebqpjz_while_maximum_iterations3^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Identity_1«
cxhqebqpjz/while/Identity_2Identitycxhqebqpjz/while/add:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Identity_2Ø
cxhqebqpjz/while/Identity_3IdentityEcxhqebqpjz/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*
_output_shapes
: 2
cxhqebqpjz/while/Identity_3É
cxhqebqpjz/while/Identity_4Identity%cxhqebqpjz/while/rvncypflgq/mul_5:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/while/Identity_4É
cxhqebqpjz/while/Identity_5Identity%cxhqebqpjz/while/rvncypflgq/add_3:z:03^cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2^cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp4^cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp+^cxhqebqpjz/while/rvncypflgq/ReadVariableOp-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1-^cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
cxhqebqpjz/while/Identity_5"\
+cxhqebqpjz_while_cxhqebqpjz_strided_slice_1-cxhqebqpjz_while_cxhqebqpjz_strided_slice_1_0"?
cxhqebqpjz_while_identity"cxhqebqpjz/while/Identity:output:0"C
cxhqebqpjz_while_identity_1$cxhqebqpjz/while/Identity_1:output:0"C
cxhqebqpjz_while_identity_2$cxhqebqpjz/while/Identity_2:output:0"C
cxhqebqpjz_while_identity_3$cxhqebqpjz/while/Identity_3:output:0"C
cxhqebqpjz_while_identity_4$cxhqebqpjz/while/Identity_4:output:0"C
cxhqebqpjz_while_identity_5$cxhqebqpjz/while/Identity_5:output:0"|
;cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource=cxhqebqpjz_while_rvncypflgq_biasadd_readvariableop_resource_0"~
<cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource>cxhqebqpjz_while_rvncypflgq_matmul_1_readvariableop_resource_0"z
:cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource<cxhqebqpjz_while_rvncypflgq_matmul_readvariableop_resource_0"p
5cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource7cxhqebqpjz_while_rvncypflgq_readvariableop_1_resource_0"p
5cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource7cxhqebqpjz_while_rvncypflgq_readvariableop_2_resource_0"l
3cxhqebqpjz_while_rvncypflgq_readvariableop_resource5cxhqebqpjz_while_rvncypflgq_readvariableop_resource_0"Ô
gcxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensoricxhqebqpjz_while_tensorarrayv2read_tensorlistgetitem_cxhqebqpjz_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2cxhqebqpjz/while/rvncypflgq/BiasAdd/ReadVariableOp2f
1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp1cxhqebqpjz/while/rvncypflgq/MatMul/ReadVariableOp2j
3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp3cxhqebqpjz/while/rvncypflgq/MatMul_1/ReadVariableOp2X
*cxhqebqpjz/while/rvncypflgq/ReadVariableOp*cxhqebqpjz/while/rvncypflgq/ReadVariableOp2\
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_1,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_12\
,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2,cxhqebqpjz/while/rvncypflgq/ReadVariableOp_2: 
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
ë

,__inference_qzuziqqdld_layer_call_fn_1722287
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_17177932
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
Û

,__inference_cxhqebqpjz_layer_call_fn_1721550

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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_17193392
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
¦h

G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1718846

inputs<
)aiccbsgdoo_matmul_readvariableop_resource:	 >
+aiccbsgdoo_matmul_1_readvariableop_resource:	 9
*aiccbsgdoo_biasadd_readvariableop_resource:	0
"aiccbsgdoo_readvariableop_resource: 2
$aiccbsgdoo_readvariableop_1_resource: 2
$aiccbsgdoo_readvariableop_2_resource: 
identity¢!aiccbsgdoo/BiasAdd/ReadVariableOp¢ aiccbsgdoo/MatMul/ReadVariableOp¢"aiccbsgdoo/MatMul_1/ReadVariableOp¢aiccbsgdoo/ReadVariableOp¢aiccbsgdoo/ReadVariableOp_1¢aiccbsgdoo/ReadVariableOp_2¢whileD
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
 aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp)aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aiccbsgdoo/MatMul/ReadVariableOp§
aiccbsgdoo/MatMulMatMulstrided_slice_2:output:0(aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMulµ
"aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp+aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aiccbsgdoo/MatMul_1/ReadVariableOp£
aiccbsgdoo/MatMul_1MatMulzeros:output:0*aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMul_1
aiccbsgdoo/addAddV2aiccbsgdoo/MatMul:product:0aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/add®
!aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp*aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aiccbsgdoo/BiasAdd/ReadVariableOp¥
aiccbsgdoo/BiasAddBiasAddaiccbsgdoo/add:z:0)aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/BiasAddz
aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aiccbsgdoo/split/split_dimë
aiccbsgdoo/splitSplit#aiccbsgdoo/split/split_dim:output:0aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aiccbsgdoo/split
aiccbsgdoo/ReadVariableOpReadVariableOp"aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp
aiccbsgdoo/mulMul!aiccbsgdoo/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul
aiccbsgdoo/add_1AddV2aiccbsgdoo/split:output:0aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_1{
aiccbsgdoo/SigmoidSigmoidaiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid
aiccbsgdoo/ReadVariableOp_1ReadVariableOp$aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_1
aiccbsgdoo/mul_1Mul#aiccbsgdoo/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_1
aiccbsgdoo/add_2AddV2aiccbsgdoo/split:output:1aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_2
aiccbsgdoo/Sigmoid_1Sigmoidaiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_1
aiccbsgdoo/mul_2Mulaiccbsgdoo/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_2w
aiccbsgdoo/TanhTanhaiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh
aiccbsgdoo/mul_3Mulaiccbsgdoo/Sigmoid:y:0aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_3
aiccbsgdoo/add_3AddV2aiccbsgdoo/mul_2:z:0aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_3
aiccbsgdoo/ReadVariableOp_2ReadVariableOp$aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_2
aiccbsgdoo/mul_4Mul#aiccbsgdoo/ReadVariableOp_2:value:0aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_4
aiccbsgdoo/add_4AddV2aiccbsgdoo/split:output:3aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_4
aiccbsgdoo/Sigmoid_2Sigmoidaiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_2v
aiccbsgdoo/Tanh_1Tanhaiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh_1
aiccbsgdoo/mul_5Mulaiccbsgdoo/Sigmoid_2:y:0aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aiccbsgdoo_matmul_readvariableop_resource+aiccbsgdoo_matmul_1_readvariableop_resource*aiccbsgdoo_biasadd_readvariableop_resource"aiccbsgdoo_readvariableop_resource$aiccbsgdoo_readvariableop_1_resource$aiccbsgdoo_readvariableop_2_resource*
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
while_body_1718745*
condR
while_cond_1718744*Q
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
IdentityIdentitystrided_slice_3:output:0"^aiccbsgdoo/BiasAdd/ReadVariableOp!^aiccbsgdoo/MatMul/ReadVariableOp#^aiccbsgdoo/MatMul_1/ReadVariableOp^aiccbsgdoo/ReadVariableOp^aiccbsgdoo/ReadVariableOp_1^aiccbsgdoo/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aiccbsgdoo/BiasAdd/ReadVariableOp!aiccbsgdoo/BiasAdd/ReadVariableOp2D
 aiccbsgdoo/MatMul/ReadVariableOp aiccbsgdoo/MatMul/ReadVariableOp2H
"aiccbsgdoo/MatMul_1/ReadVariableOp"aiccbsgdoo/MatMul_1/ReadVariableOp26
aiccbsgdoo/ReadVariableOpaiccbsgdoo/ReadVariableOp2:
aiccbsgdoo/ReadVariableOp_1aiccbsgdoo/ReadVariableOp_12:
aiccbsgdoo/ReadVariableOp_2aiccbsgdoo/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó	
ø
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_1722348

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
¡h

G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721482

inputs<
)rvncypflgq_matmul_readvariableop_resource:	>
+rvncypflgq_matmul_1_readvariableop_resource:	 9
*rvncypflgq_biasadd_readvariableop_resource:	0
"rvncypflgq_readvariableop_resource: 2
$rvncypflgq_readvariableop_1_resource: 2
$rvncypflgq_readvariableop_2_resource: 
identity¢!rvncypflgq/BiasAdd/ReadVariableOp¢ rvncypflgq/MatMul/ReadVariableOp¢"rvncypflgq/MatMul_1/ReadVariableOp¢rvncypflgq/ReadVariableOp¢rvncypflgq/ReadVariableOp_1¢rvncypflgq/ReadVariableOp_2¢whileD
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
 rvncypflgq/MatMul/ReadVariableOpReadVariableOp)rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 rvncypflgq/MatMul/ReadVariableOp§
rvncypflgq/MatMulMatMulstrided_slice_2:output:0(rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMulµ
"rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp+rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"rvncypflgq/MatMul_1/ReadVariableOp£
rvncypflgq/MatMul_1MatMulzeros:output:0*rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMul_1
rvncypflgq/addAddV2rvncypflgq/MatMul:product:0rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/add®
!rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp*rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!rvncypflgq/BiasAdd/ReadVariableOp¥
rvncypflgq/BiasAddBiasAddrvncypflgq/add:z:0)rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/BiasAddz
rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
rvncypflgq/split/split_dimë
rvncypflgq/splitSplit#rvncypflgq/split/split_dim:output:0rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rvncypflgq/split
rvncypflgq/ReadVariableOpReadVariableOp"rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp
rvncypflgq/mulMul!rvncypflgq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul
rvncypflgq/add_1AddV2rvncypflgq/split:output:0rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_1{
rvncypflgq/SigmoidSigmoidrvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid
rvncypflgq/ReadVariableOp_1ReadVariableOp$rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_1
rvncypflgq/mul_1Mul#rvncypflgq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_1
rvncypflgq/add_2AddV2rvncypflgq/split:output:1rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_2
rvncypflgq/Sigmoid_1Sigmoidrvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_1
rvncypflgq/mul_2Mulrvncypflgq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_2w
rvncypflgq/TanhTanhrvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh
rvncypflgq/mul_3Mulrvncypflgq/Sigmoid:y:0rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_3
rvncypflgq/add_3AddV2rvncypflgq/mul_2:z:0rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_3
rvncypflgq/ReadVariableOp_2ReadVariableOp$rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_2
rvncypflgq/mul_4Mul#rvncypflgq/ReadVariableOp_2:value:0rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_4
rvncypflgq/add_4AddV2rvncypflgq/split:output:3rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_4
rvncypflgq/Sigmoid_2Sigmoidrvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_2v
rvncypflgq/Tanh_1Tanhrvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh_1
rvncypflgq/mul_5Mulrvncypflgq/Sigmoid_2:y:0rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)rvncypflgq_matmul_readvariableop_resource+rvncypflgq_matmul_1_readvariableop_resource*rvncypflgq_biasadd_readvariableop_resource"rvncypflgq_readvariableop_resource$rvncypflgq_readvariableop_1_resource$rvncypflgq_readvariableop_2_resource*
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
while_body_1721381*
condR
while_cond_1721380*Q
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
IdentityIdentitytranspose_1:y:0"^rvncypflgq/BiasAdd/ReadVariableOp!^rvncypflgq/MatMul/ReadVariableOp#^rvncypflgq/MatMul_1/ReadVariableOp^rvncypflgq/ReadVariableOp^rvncypflgq/ReadVariableOp_1^rvncypflgq/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!rvncypflgq/BiasAdd/ReadVariableOp!rvncypflgq/BiasAdd/ReadVariableOp2D
 rvncypflgq/MatMul/ReadVariableOp rvncypflgq/MatMul/ReadVariableOp2H
"rvncypflgq/MatMul_1/ReadVariableOp"rvncypflgq/MatMul_1/ReadVariableOp26
rvncypflgq/ReadVariableOprvncypflgq/ReadVariableOp2:
rvncypflgq/ReadVariableOp_1rvncypflgq/ReadVariableOp_12:
rvncypflgq/ReadVariableOp_2rvncypflgq/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_1718472

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
³F
ê
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1717035

inputs%
rvncypflgq_1716936:	%
rvncypflgq_1716938:	 !
rvncypflgq_1716940:	 
rvncypflgq_1716942:  
rvncypflgq_1716944:  
rvncypflgq_1716946: 
identity¢"rvncypflgq/StatefulPartitionedCall¢whileD
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
"rvncypflgq/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0rvncypflgq_1716936rvncypflgq_1716938rvncypflgq_1716940rvncypflgq_1716942rvncypflgq_1716944rvncypflgq_1716946*
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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_17169352$
"rvncypflgq/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0rvncypflgq_1716936rvncypflgq_1716938rvncypflgq_1716940rvncypflgq_1716942rvncypflgq_1716944rvncypflgq_1716946*
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
while_body_1716955*
condR
while_cond_1716954*Q
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
IdentityIdentitytranspose_1:y:0#^rvncypflgq/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"rvncypflgq/StatefulPartitionedCall"rvncypflgq/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Üh

G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1721730
inputs_0<
)aiccbsgdoo_matmul_readvariableop_resource:	 >
+aiccbsgdoo_matmul_1_readvariableop_resource:	 9
*aiccbsgdoo_biasadd_readvariableop_resource:	0
"aiccbsgdoo_readvariableop_resource: 2
$aiccbsgdoo_readvariableop_1_resource: 2
$aiccbsgdoo_readvariableop_2_resource: 
identity¢!aiccbsgdoo/BiasAdd/ReadVariableOp¢ aiccbsgdoo/MatMul/ReadVariableOp¢"aiccbsgdoo/MatMul_1/ReadVariableOp¢aiccbsgdoo/ReadVariableOp¢aiccbsgdoo/ReadVariableOp_1¢aiccbsgdoo/ReadVariableOp_2¢whileF
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
 aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp)aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aiccbsgdoo/MatMul/ReadVariableOp§
aiccbsgdoo/MatMulMatMulstrided_slice_2:output:0(aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMulµ
"aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp+aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aiccbsgdoo/MatMul_1/ReadVariableOp£
aiccbsgdoo/MatMul_1MatMulzeros:output:0*aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMul_1
aiccbsgdoo/addAddV2aiccbsgdoo/MatMul:product:0aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/add®
!aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp*aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aiccbsgdoo/BiasAdd/ReadVariableOp¥
aiccbsgdoo/BiasAddBiasAddaiccbsgdoo/add:z:0)aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/BiasAddz
aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aiccbsgdoo/split/split_dimë
aiccbsgdoo/splitSplit#aiccbsgdoo/split/split_dim:output:0aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aiccbsgdoo/split
aiccbsgdoo/ReadVariableOpReadVariableOp"aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp
aiccbsgdoo/mulMul!aiccbsgdoo/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul
aiccbsgdoo/add_1AddV2aiccbsgdoo/split:output:0aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_1{
aiccbsgdoo/SigmoidSigmoidaiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid
aiccbsgdoo/ReadVariableOp_1ReadVariableOp$aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_1
aiccbsgdoo/mul_1Mul#aiccbsgdoo/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_1
aiccbsgdoo/add_2AddV2aiccbsgdoo/split:output:1aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_2
aiccbsgdoo/Sigmoid_1Sigmoidaiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_1
aiccbsgdoo/mul_2Mulaiccbsgdoo/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_2w
aiccbsgdoo/TanhTanhaiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh
aiccbsgdoo/mul_3Mulaiccbsgdoo/Sigmoid:y:0aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_3
aiccbsgdoo/add_3AddV2aiccbsgdoo/mul_2:z:0aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_3
aiccbsgdoo/ReadVariableOp_2ReadVariableOp$aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_2
aiccbsgdoo/mul_4Mul#aiccbsgdoo/ReadVariableOp_2:value:0aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_4
aiccbsgdoo/add_4AddV2aiccbsgdoo/split:output:3aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_4
aiccbsgdoo/Sigmoid_2Sigmoidaiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_2v
aiccbsgdoo/Tanh_1Tanhaiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh_1
aiccbsgdoo/mul_5Mulaiccbsgdoo/Sigmoid_2:y:0aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aiccbsgdoo_matmul_readvariableop_resource+aiccbsgdoo_matmul_1_readvariableop_resource*aiccbsgdoo_biasadd_readvariableop_resource"aiccbsgdoo_readvariableop_resource$aiccbsgdoo_readvariableop_1_resource$aiccbsgdoo_readvariableop_2_resource*
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
while_body_1721629*
condR
while_cond_1721628*Q
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
IdentityIdentitystrided_slice_3:output:0"^aiccbsgdoo/BiasAdd/ReadVariableOp!^aiccbsgdoo/MatMul/ReadVariableOp#^aiccbsgdoo/MatMul_1/ReadVariableOp^aiccbsgdoo/ReadVariableOp^aiccbsgdoo/ReadVariableOp_1^aiccbsgdoo/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aiccbsgdoo/BiasAdd/ReadVariableOp!aiccbsgdoo/BiasAdd/ReadVariableOp2D
 aiccbsgdoo/MatMul/ReadVariableOp aiccbsgdoo/MatMul/ReadVariableOp2H
"aiccbsgdoo/MatMul_1/ReadVariableOp"aiccbsgdoo/MatMul_1/ReadVariableOp26
aiccbsgdoo/ReadVariableOpaiccbsgdoo/ReadVariableOp2:
aiccbsgdoo/ReadVariableOp_1aiccbsgdoo/ReadVariableOp_12:
aiccbsgdoo/ReadVariableOp_2aiccbsgdoo/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Üh

G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1721910
inputs_0<
)aiccbsgdoo_matmul_readvariableop_resource:	 >
+aiccbsgdoo_matmul_1_readvariableop_resource:	 9
*aiccbsgdoo_biasadd_readvariableop_resource:	0
"aiccbsgdoo_readvariableop_resource: 2
$aiccbsgdoo_readvariableop_1_resource: 2
$aiccbsgdoo_readvariableop_2_resource: 
identity¢!aiccbsgdoo/BiasAdd/ReadVariableOp¢ aiccbsgdoo/MatMul/ReadVariableOp¢"aiccbsgdoo/MatMul_1/ReadVariableOp¢aiccbsgdoo/ReadVariableOp¢aiccbsgdoo/ReadVariableOp_1¢aiccbsgdoo/ReadVariableOp_2¢whileF
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
 aiccbsgdoo/MatMul/ReadVariableOpReadVariableOp)aiccbsgdoo_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aiccbsgdoo/MatMul/ReadVariableOp§
aiccbsgdoo/MatMulMatMulstrided_slice_2:output:0(aiccbsgdoo/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMulµ
"aiccbsgdoo/MatMul_1/ReadVariableOpReadVariableOp+aiccbsgdoo_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aiccbsgdoo/MatMul_1/ReadVariableOp£
aiccbsgdoo/MatMul_1MatMulzeros:output:0*aiccbsgdoo/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/MatMul_1
aiccbsgdoo/addAddV2aiccbsgdoo/MatMul:product:0aiccbsgdoo/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/add®
!aiccbsgdoo/BiasAdd/ReadVariableOpReadVariableOp*aiccbsgdoo_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aiccbsgdoo/BiasAdd/ReadVariableOp¥
aiccbsgdoo/BiasAddBiasAddaiccbsgdoo/add:z:0)aiccbsgdoo/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aiccbsgdoo/BiasAddz
aiccbsgdoo/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aiccbsgdoo/split/split_dimë
aiccbsgdoo/splitSplit#aiccbsgdoo/split/split_dim:output:0aiccbsgdoo/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aiccbsgdoo/split
aiccbsgdoo/ReadVariableOpReadVariableOp"aiccbsgdoo_readvariableop_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp
aiccbsgdoo/mulMul!aiccbsgdoo/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul
aiccbsgdoo/add_1AddV2aiccbsgdoo/split:output:0aiccbsgdoo/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_1{
aiccbsgdoo/SigmoidSigmoidaiccbsgdoo/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid
aiccbsgdoo/ReadVariableOp_1ReadVariableOp$aiccbsgdoo_readvariableop_1_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_1
aiccbsgdoo/mul_1Mul#aiccbsgdoo/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_1
aiccbsgdoo/add_2AddV2aiccbsgdoo/split:output:1aiccbsgdoo/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_2
aiccbsgdoo/Sigmoid_1Sigmoidaiccbsgdoo/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_1
aiccbsgdoo/mul_2Mulaiccbsgdoo/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_2w
aiccbsgdoo/TanhTanhaiccbsgdoo/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh
aiccbsgdoo/mul_3Mulaiccbsgdoo/Sigmoid:y:0aiccbsgdoo/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_3
aiccbsgdoo/add_3AddV2aiccbsgdoo/mul_2:z:0aiccbsgdoo/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_3
aiccbsgdoo/ReadVariableOp_2ReadVariableOp$aiccbsgdoo_readvariableop_2_resource*
_output_shapes
: *
dtype02
aiccbsgdoo/ReadVariableOp_2
aiccbsgdoo/mul_4Mul#aiccbsgdoo/ReadVariableOp_2:value:0aiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_4
aiccbsgdoo/add_4AddV2aiccbsgdoo/split:output:3aiccbsgdoo/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/add_4
aiccbsgdoo/Sigmoid_2Sigmoidaiccbsgdoo/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Sigmoid_2v
aiccbsgdoo/Tanh_1Tanhaiccbsgdoo/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/Tanh_1
aiccbsgdoo/mul_5Mulaiccbsgdoo/Sigmoid_2:y:0aiccbsgdoo/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aiccbsgdoo/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aiccbsgdoo_matmul_readvariableop_resource+aiccbsgdoo_matmul_1_readvariableop_resource*aiccbsgdoo_biasadd_readvariableop_resource"aiccbsgdoo_readvariableop_resource$aiccbsgdoo_readvariableop_1_resource$aiccbsgdoo_readvariableop_2_resource*
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
while_body_1721809*
condR
while_cond_1721808*Q
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
IdentityIdentitystrided_slice_3:output:0"^aiccbsgdoo/BiasAdd/ReadVariableOp!^aiccbsgdoo/MatMul/ReadVariableOp#^aiccbsgdoo/MatMul_1/ReadVariableOp^aiccbsgdoo/ReadVariableOp^aiccbsgdoo/ReadVariableOp_1^aiccbsgdoo/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aiccbsgdoo/BiasAdd/ReadVariableOp!aiccbsgdoo/BiasAdd/ReadVariableOp2D
 aiccbsgdoo/MatMul/ReadVariableOp aiccbsgdoo/MatMul/ReadVariableOp2H
"aiccbsgdoo/MatMul_1/ReadVariableOp"aiccbsgdoo/MatMul_1/ReadVariableOp26
aiccbsgdoo/ReadVariableOpaiccbsgdoo/ReadVariableOp2:
aiccbsgdoo/ReadVariableOp_1aiccbsgdoo/ReadVariableOp_12:
aiccbsgdoo/ReadVariableOp_2aiccbsgdoo/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
àh

G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1720942
inputs_0<
)rvncypflgq_matmul_readvariableop_resource:	>
+rvncypflgq_matmul_1_readvariableop_resource:	 9
*rvncypflgq_biasadd_readvariableop_resource:	0
"rvncypflgq_readvariableop_resource: 2
$rvncypflgq_readvariableop_1_resource: 2
$rvncypflgq_readvariableop_2_resource: 
identity¢!rvncypflgq/BiasAdd/ReadVariableOp¢ rvncypflgq/MatMul/ReadVariableOp¢"rvncypflgq/MatMul_1/ReadVariableOp¢rvncypflgq/ReadVariableOp¢rvncypflgq/ReadVariableOp_1¢rvncypflgq/ReadVariableOp_2¢whileF
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
 rvncypflgq/MatMul/ReadVariableOpReadVariableOp)rvncypflgq_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 rvncypflgq/MatMul/ReadVariableOp§
rvncypflgq/MatMulMatMulstrided_slice_2:output:0(rvncypflgq/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMulµ
"rvncypflgq/MatMul_1/ReadVariableOpReadVariableOp+rvncypflgq_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"rvncypflgq/MatMul_1/ReadVariableOp£
rvncypflgq/MatMul_1MatMulzeros:output:0*rvncypflgq/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/MatMul_1
rvncypflgq/addAddV2rvncypflgq/MatMul:product:0rvncypflgq/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/add®
!rvncypflgq/BiasAdd/ReadVariableOpReadVariableOp*rvncypflgq_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!rvncypflgq/BiasAdd/ReadVariableOp¥
rvncypflgq/BiasAddBiasAddrvncypflgq/add:z:0)rvncypflgq/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rvncypflgq/BiasAddz
rvncypflgq/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
rvncypflgq/split/split_dimë
rvncypflgq/splitSplit#rvncypflgq/split/split_dim:output:0rvncypflgq/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rvncypflgq/split
rvncypflgq/ReadVariableOpReadVariableOp"rvncypflgq_readvariableop_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp
rvncypflgq/mulMul!rvncypflgq/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul
rvncypflgq/add_1AddV2rvncypflgq/split:output:0rvncypflgq/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_1{
rvncypflgq/SigmoidSigmoidrvncypflgq/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid
rvncypflgq/ReadVariableOp_1ReadVariableOp$rvncypflgq_readvariableop_1_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_1
rvncypflgq/mul_1Mul#rvncypflgq/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_1
rvncypflgq/add_2AddV2rvncypflgq/split:output:1rvncypflgq/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_2
rvncypflgq/Sigmoid_1Sigmoidrvncypflgq/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_1
rvncypflgq/mul_2Mulrvncypflgq/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_2w
rvncypflgq/TanhTanhrvncypflgq/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh
rvncypflgq/mul_3Mulrvncypflgq/Sigmoid:y:0rvncypflgq/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_3
rvncypflgq/add_3AddV2rvncypflgq/mul_2:z:0rvncypflgq/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_3
rvncypflgq/ReadVariableOp_2ReadVariableOp$rvncypflgq_readvariableop_2_resource*
_output_shapes
: *
dtype02
rvncypflgq/ReadVariableOp_2
rvncypflgq/mul_4Mul#rvncypflgq/ReadVariableOp_2:value:0rvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_4
rvncypflgq/add_4AddV2rvncypflgq/split:output:3rvncypflgq/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/add_4
rvncypflgq/Sigmoid_2Sigmoidrvncypflgq/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Sigmoid_2v
rvncypflgq/Tanh_1Tanhrvncypflgq/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/Tanh_1
rvncypflgq/mul_5Mulrvncypflgq/Sigmoid_2:y:0rvncypflgq/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rvncypflgq/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)rvncypflgq_matmul_readvariableop_resource+rvncypflgq_matmul_1_readvariableop_resource*rvncypflgq_biasadd_readvariableop_resource"rvncypflgq_readvariableop_resource$rvncypflgq_readvariableop_1_resource$rvncypflgq_readvariableop_2_resource*
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
while_body_1720841*
condR
while_cond_1720840*Q
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
IdentityIdentitytranspose_1:y:0"^rvncypflgq/BiasAdd/ReadVariableOp!^rvncypflgq/MatMul/ReadVariableOp#^rvncypflgq/MatMul_1/ReadVariableOp^rvncypflgq/ReadVariableOp^rvncypflgq/ReadVariableOp_1^rvncypflgq/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!rvncypflgq/BiasAdd/ReadVariableOp!rvncypflgq/BiasAdd/ReadVariableOp2D
 rvncypflgq/MatMul/ReadVariableOp rvncypflgq/MatMul/ReadVariableOp2H
"rvncypflgq/MatMul_1/ReadVariableOp"rvncypflgq/MatMul_1/ReadVariableOp26
rvncypflgq/ReadVariableOprvncypflgq/ReadVariableOp2:
rvncypflgq/ReadVariableOp_1rvncypflgq/ReadVariableOp_12:
rvncypflgq/ReadVariableOp_2rvncypflgq/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


í
while_cond_1721020
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1721020___redundant_placeholder05
1while_while_cond_1721020___redundant_placeholder15
1while_while_cond_1721020___redundant_placeholder25
1while_while_cond_1721020___redundant_placeholder35
1while_while_cond_1721020___redundant_placeholder45
1while_while_cond_1721020___redundant_placeholder55
1while_while_cond_1721020___redundant_placeholder6
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
:"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
I

gvxdqcynan;
serving_default_gvxdqcynan:0ÿÿÿÿÿÿÿÿÿ>

pbmomrqadp0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:çß
þP
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
_default_save_signature
+&call_and_return_all_conditional_losses
__call__"àM
_tf_keras_sequentialÁM{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gvxdqcynan"}}, {"class_name": "Conv1D", "config": {"name": "mopvqfaljf", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv1D", "config": {"name": "ehhqjcwuju", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "abbthhzbau", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "cxhqebqpjz", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "rvncypflgq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "qzuziqqdld", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aiccbsgdoo", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "pbmomrqadp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "gvxdqcynan"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gvxdqcynan"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "mopvqfaljf", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Conv1D", "config": {"name": "ehhqjcwuju", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "Reshape", "config": {"name": "abbthhzbau", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 7}, {"class_name": "RNN", "config": {"name": "cxhqebqpjz", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "rvncypflgq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 11}}, "shared_object_id": 12}, {"class_name": "RNN", "config": {"name": "qzuziqqdld", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aiccbsgdoo", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 16}}, "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "pbmomrqadp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
Ì

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"¥

_tf_keras_layer
{"name": "mopvqfaljf", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "mopvqfaljf", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}
Í

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"¦

_tf_keras_layer
{"name": "ehhqjcwuju", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "ehhqjcwuju", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 20}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 1, 20]}}

	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"name": "abbthhzbau", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "abbthhzbau", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 7}
°
cell

state_spec
	variables
 regularization_losses
!trainable_variables
"	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layerç{"name": "cxhqebqpjz", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "cxhqebqpjz", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "rvncypflgq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 11}}, "shared_object_id": 12, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 24}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
³
#cell
$
state_spec
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layerê{"name": "qzuziqqdld", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "qzuziqqdld", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aiccbsgdoo", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 16}}, "shared_object_id": 17, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 25}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "pbmomrqadp", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "pbmomrqadp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
¨
/iter
	0decay
1learning_rate
2momentum
3rho	rms~	rms
rms
rms
)rms
*rms
4rms
5rms
6rms
7rms
8rms
9rms
:rms
;rms
<rms
=rms
>rms
?rms"
	optimizer
¦
0
1
2
3
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
)16
*17"
trackable_list_wrapper
 "
trackable_list_wrapper
¦
0
1
2
3
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
)16
*17"
trackable_list_wrapper
Î
@non_trainable_variables
	variables

Alayers
	regularization_losses
Blayer_metrics
Cmetrics
Dlayer_regularization_losses

trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
':%2mopvqfaljf/kernel
:2mopvqfaljf/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
Enon_trainable_variables
Flayer_metrics

Glayers
	variables
regularization_losses
Hmetrics
Ilayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%2ehhqjcwuju/kernel
:2ehhqjcwuju/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
Jnon_trainable_variables
Klayer_metrics

Llayers
	variables
regularization_losses
Mmetrics
Nlayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Onon_trainable_variables
Player_metrics

Qlayers
	variables
regularization_losses
Rmetrics
Slayer_regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


T
state_size

4kernel
5recurrent_kernel
6bias
7input_gate_peephole_weights
 8forget_gate_peephole_weights
 9output_gate_peephole_weights
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"Ø
_tf_keras_layer¾{"name": "rvncypflgq", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "rvncypflgq", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 11}
 "
trackable_list_wrapper
J
40
51
62
73
84
95"
trackable_list_wrapper
 "
trackable_list_wrapper
J
40
51
62
73
84
95"
trackable_list_wrapper
¼
Ynon_trainable_variables
	variables

Zlayers
[layer_metrics
 regularization_losses

\states
]metrics
^layer_regularization_losses
!trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


_
state_size

:kernel
;recurrent_kernel
<bias
=input_gate_peephole_weights
 >forget_gate_peephole_weights
 ?output_gate_peephole_weights
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"Ú
_tf_keras_layerÀ{"name": "aiccbsgdoo", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aiccbsgdoo", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 16}
 "
trackable_list_wrapper
J
:0
;1
<2
=3
>4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
:0
;1
<2
=3
>4
?5"
trackable_list_wrapper
¼
dnon_trainable_variables
%	variables

elayers
flayer_metrics
&regularization_losses

gstates
hmetrics
ilayer_regularization_losses
'trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:! 2pbmomrqadp/kernel
:2pbmomrqadp/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
°
jnon_trainable_variables
klayer_metrics

llayers
+	variables
,regularization_losses
mmetrics
nlayer_regularization_losses
-trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
/:-	2cxhqebqpjz/rvncypflgq/kernel
9:7	 2&cxhqebqpjz/rvncypflgq/recurrent_kernel
):'2cxhqebqpjz/rvncypflgq/bias
?:= 21cxhqebqpjz/rvncypflgq/input_gate_peephole_weights
@:> 22cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights
@:> 22cxhqebqpjz/rvncypflgq/output_gate_peephole_weights
/:-	 2qzuziqqdld/aiccbsgdoo/kernel
9:7	 2&qzuziqqdld/aiccbsgdoo/recurrent_kernel
):'2qzuziqqdld/aiccbsgdoo/bias
?:= 21qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights
@:> 22qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights
@:> 22qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
o0"
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
40
51
62
73
84
95"
trackable_list_wrapper
 "
trackable_list_wrapper
J
40
51
62
73
84
95"
trackable_list_wrapper
°
pnon_trainable_variables
qlayer_metrics

rlayers
U	variables
Vregularization_losses
smetrics
tlayer_regularization_losses
Wtrainable_variables
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
:0
;1
<2
=3
>4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
:0
;1
<2
=3
>4
?5"
trackable_list_wrapper
°
unon_trainable_variables
vlayer_metrics

wlayers
`	variables
aregularization_losses
xmetrics
ylayer_regularization_losses
btrainable_variables
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
#0"
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
	ztotal
	{count
|	variables
}	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 27}
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
z0
{1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
1:/2RMSprop/mopvqfaljf/kernel/rms
':%2RMSprop/mopvqfaljf/bias/rms
1:/2RMSprop/ehhqjcwuju/kernel/rms
':%2RMSprop/ehhqjcwuju/bias/rms
-:+ 2RMSprop/pbmomrqadp/kernel/rms
':%2RMSprop/pbmomrqadp/bias/rms
9:7	2(RMSprop/cxhqebqpjz/rvncypflgq/kernel/rms
C:A	 22RMSprop/cxhqebqpjz/rvncypflgq/recurrent_kernel/rms
3:12&RMSprop/cxhqebqpjz/rvncypflgq/bias/rms
I:G 2=RMSprop/cxhqebqpjz/rvncypflgq/input_gate_peephole_weights/rms
J:H 2>RMSprop/cxhqebqpjz/rvncypflgq/forget_gate_peephole_weights/rms
J:H 2>RMSprop/cxhqebqpjz/rvncypflgq/output_gate_peephole_weights/rms
9:7	 2(RMSprop/qzuziqqdld/aiccbsgdoo/kernel/rms
C:A	 22RMSprop/qzuziqqdld/aiccbsgdoo/recurrent_kernel/rms
3:12&RMSprop/qzuziqqdld/aiccbsgdoo/bias/rms
I:G 2=RMSprop/qzuziqqdld/aiccbsgdoo/input_gate_peephole_weights/rms
J:H 2>RMSprop/qzuziqqdld/aiccbsgdoo/forget_gate_peephole_weights/rms
J:H 2>RMSprop/qzuziqqdld/aiccbsgdoo/output_gate_peephole_weights/rms
ë2è
"__inference__wrapped_model_1716848Á
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

gvxdqcynanÿÿÿÿÿÿÿÿÿ
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_1720129
G__inference_sequential_layer_call_and_return_conditional_losses_1720568
G__inference_sequential_layer_call_and_return_conditional_losses_1719595
G__inference_sequential_layer_call_and_return_conditional_losses_1719641À
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
,__inference_sequential_layer_call_fn_1718916
,__inference_sequential_layer_call_fn_1720609
,__inference_sequential_layer_call_fn_1720650
,__inference_sequential_layer_call_fn_1719549À
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
ñ2î
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_1720687¢
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
,__inference_mopvqfaljf_layer_call_fn_1720696¢
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
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_1720735¢
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
,__inference_ehhqjcwuju_layer_call_fn_1720744¢
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
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_1720757¢
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
,__inference_abbthhzbau_layer_call_fn_1720762¢
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
2
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1720942
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721122
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721302
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721482æ
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
,__inference_cxhqebqpjz_layer_call_fn_1721499
,__inference_cxhqebqpjz_layer_call_fn_1721516
,__inference_cxhqebqpjz_layer_call_fn_1721533
,__inference_cxhqebqpjz_layer_call_fn_1721550æ
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1721730
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1721910
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1722090
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1722270æ
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
,__inference_qzuziqqdld_layer_call_fn_1722287
,__inference_qzuziqqdld_layer_call_fn_1722304
,__inference_qzuziqqdld_layer_call_fn_1722321
,__inference_qzuziqqdld_layer_call_fn_1722338æ
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
ñ2î
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_1722348¢
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
,__inference_pbmomrqadp_layer_call_fn_1722357¢
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
%__inference_signature_wrapper_1719690
gvxdqcynan"
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
Ö2Ó
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1722401
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1722445¾
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
,__inference_rvncypflgq_layer_call_fn_1722468
,__inference_rvncypflgq_layer_call_fn_1722491¾
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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1722535
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1722579¾
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
,__inference_aiccbsgdoo_layer_call_fn_1722602
,__inference_aiccbsgdoo_layer_call_fn_1722625¾
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
 ±
"__inference__wrapped_model_1716848456789:;<=>?)*;¢8
1¢.
,)

gvxdqcynanÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

pbmomrqadp$!

pbmomrqadpÿÿÿÿÿÿÿÿÿ¯
G__inference_abbthhzbau_layer_call_and_return_conditional_losses_1720757d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_abbthhzbau_layer_call_fn_1720762W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÌ
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1722535:;<=>?¢}
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
G__inference_aiccbsgdoo_layer_call_and_return_conditional_losses_1722579:;<=>?¢}
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
,__inference_aiccbsgdoo_layer_call_fn_1722602ð:;<=>?¢}
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
,__inference_aiccbsgdoo_layer_call_fn_1722625ð:;<=>?¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ý
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1720942456789S¢P
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721122456789S¢P
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721302x456789C¢@
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
G__inference_cxhqebqpjz_layer_call_and_return_conditional_losses_1721482x456789C¢@
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
,__inference_cxhqebqpjz_layer_call_fn_1721499456789S¢P
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
,__inference_cxhqebqpjz_layer_call_fn_1721516456789S¢P
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
,__inference_cxhqebqpjz_layer_call_fn_1721533k456789C¢@
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
,__inference_cxhqebqpjz_layer_call_fn_1721550k456789C¢@
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
G__inference_ehhqjcwuju_layer_call_and_return_conditional_losses_1720735l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_ehhqjcwuju_layer_call_fn_1720744_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ·
G__inference_mopvqfaljf_layer_call_and_return_conditional_losses_1720687l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_mopvqfaljf_layer_call_fn_1720696_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ§
G__inference_pbmomrqadp_layer_call_and_return_conditional_losses_1722348\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_pbmomrqadp_layer_call_fn_1722357O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÐ
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1721730:;<=>?S¢P
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1721910:;<=>?S¢P
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1722090t:;<=>?C¢@
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
G__inference_qzuziqqdld_layer_call_and_return_conditional_losses_1722270t:;<=>?C¢@
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
,__inference_qzuziqqdld_layer_call_fn_1722287w:;<=>?S¢P
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
,__inference_qzuziqqdld_layer_call_fn_1722304w:;<=>?S¢P
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
,__inference_qzuziqqdld_layer_call_fn_1722321g:;<=>?C¢@
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
,__inference_qzuziqqdld_layer_call_fn_1722338g:;<=>?C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ Ì
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1722401456789¢}
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
G__inference_rvncypflgq_layer_call_and_return_conditional_losses_1722445456789¢}
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
,__inference_rvncypflgq_layer_call_fn_1722468ð456789¢}
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
,__inference_rvncypflgq_layer_call_fn_1722491ð456789¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ì
G__inference_sequential_layer_call_and_return_conditional_losses_1719595456789:;<=>?)*C¢@
9¢6
,)

gvxdqcynanÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
G__inference_sequential_layer_call_and_return_conditional_losses_1719641456789:;<=>?)*C¢@
9¢6
,)

gvxdqcynanÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
G__inference_sequential_layer_call_and_return_conditional_losses_1720129|456789:;<=>?)*?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
G__inference_sequential_layer_call_and_return_conditional_losses_1720568|456789:;<=>?)*?¢<
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
,__inference_sequential_layer_call_fn_1718916s456789:;<=>?)*C¢@
9¢6
,)

gvxdqcynanÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
,__inference_sequential_layer_call_fn_1719549s456789:;<=>?)*C¢@
9¢6
,)

gvxdqcynanÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1720609o456789:;<=>?)*?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1720650o456789:;<=>?)*?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
%__inference_signature_wrapper_1719690456789:;<=>?)*I¢F
¢ 
?ª<
:

gvxdqcynan,)

gvxdqcynanÿÿÿÿÿÿÿÿÿ"7ª4
2

pbmomrqadp$!

pbmomrqadpÿÿÿÿÿÿÿÿÿ