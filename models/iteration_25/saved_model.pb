Ï2
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
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ô³/

kjxhlaztnm/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namekjxhlaztnm/kernel
{
%kjxhlaztnm/kernel/Read/ReadVariableOpReadVariableOpkjxhlaztnm/kernel*"
_output_shapes
:*
dtype0
v
kjxhlaztnm/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namekjxhlaztnm/bias
o
#kjxhlaztnm/bias/Read/ReadVariableOpReadVariableOpkjxhlaztnm/bias*
_output_shapes
:*
dtype0
~
uilnjhxhrx/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameuilnjhxhrx/kernel
w
%uilnjhxhrx/kernel/Read/ReadVariableOpReadVariableOpuilnjhxhrx/kernel*
_output_shapes

: *
dtype0
v
uilnjhxhrx/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameuilnjhxhrx/bias
o
#uilnjhxhrx/bias/Read/ReadVariableOpReadVariableOpuilnjhxhrx/bias*
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
lctpanrywj/dhxpxqfhna/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namelctpanrywj/dhxpxqfhna/kernel

0lctpanrywj/dhxpxqfhna/kernel/Read/ReadVariableOpReadVariableOplctpanrywj/dhxpxqfhna/kernel*
_output_shapes
:	*
dtype0
©
&lctpanrywj/dhxpxqfhna/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&lctpanrywj/dhxpxqfhna/recurrent_kernel
¢
:lctpanrywj/dhxpxqfhna/recurrent_kernel/Read/ReadVariableOpReadVariableOp&lctpanrywj/dhxpxqfhna/recurrent_kernel*
_output_shapes
:	 *
dtype0

lctpanrywj/dhxpxqfhna/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelctpanrywj/dhxpxqfhna/bias

.lctpanrywj/dhxpxqfhna/bias/Read/ReadVariableOpReadVariableOplctpanrywj/dhxpxqfhna/bias*
_output_shapes	
:*
dtype0
º
1lctpanrywj/dhxpxqfhna/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31lctpanrywj/dhxpxqfhna/input_gate_peephole_weights
³
Elctpanrywj/dhxpxqfhna/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1lctpanrywj/dhxpxqfhna/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2lctpanrywj/dhxpxqfhna/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights
µ
Flctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2lctpanrywj/dhxpxqfhna/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42lctpanrywj/dhxpxqfhna/output_gate_peephole_weights
µ
Flctpanrywj/dhxpxqfhna/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2lctpanrywj/dhxpxqfhna/output_gate_peephole_weights*
_output_shapes
: *
dtype0

rienwrhgrh/kngiiuzftt/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namerienwrhgrh/kngiiuzftt/kernel

0rienwrhgrh/kngiiuzftt/kernel/Read/ReadVariableOpReadVariableOprienwrhgrh/kngiiuzftt/kernel*
_output_shapes
:	 *
dtype0
©
&rienwrhgrh/kngiiuzftt/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&rienwrhgrh/kngiiuzftt/recurrent_kernel
¢
:rienwrhgrh/kngiiuzftt/recurrent_kernel/Read/ReadVariableOpReadVariableOp&rienwrhgrh/kngiiuzftt/recurrent_kernel*
_output_shapes
:	 *
dtype0

rienwrhgrh/kngiiuzftt/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namerienwrhgrh/kngiiuzftt/bias

.rienwrhgrh/kngiiuzftt/bias/Read/ReadVariableOpReadVariableOprienwrhgrh/kngiiuzftt/bias*
_output_shapes	
:*
dtype0
º
1rienwrhgrh/kngiiuzftt/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31rienwrhgrh/kngiiuzftt/input_gate_peephole_weights
³
Erienwrhgrh/kngiiuzftt/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1rienwrhgrh/kngiiuzftt/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2rienwrhgrh/kngiiuzftt/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights
µ
Frienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2rienwrhgrh/kngiiuzftt/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42rienwrhgrh/kngiiuzftt/output_gate_peephole_weights
µ
Frienwrhgrh/kngiiuzftt/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2rienwrhgrh/kngiiuzftt/output_gate_peephole_weights*
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
RMSprop/kjxhlaztnm/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/kjxhlaztnm/kernel/rms

1RMSprop/kjxhlaztnm/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/kjxhlaztnm/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/kjxhlaztnm/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/kjxhlaztnm/bias/rms

/RMSprop/kjxhlaztnm/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/kjxhlaztnm/bias/rms*
_output_shapes
:*
dtype0

RMSprop/uilnjhxhrx/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/uilnjhxhrx/kernel/rms

1RMSprop/uilnjhxhrx/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/uilnjhxhrx/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/uilnjhxhrx/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/uilnjhxhrx/bias/rms

/RMSprop/uilnjhxhrx/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/uilnjhxhrx/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/lctpanrywj/dhxpxqfhna/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/lctpanrywj/dhxpxqfhna/kernel/rms
¦
<RMSprop/lctpanrywj/dhxpxqfhna/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/lctpanrywj/dhxpxqfhna/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rms
º
FRMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/lctpanrywj/dhxpxqfhna/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/lctpanrywj/dhxpxqfhna/bias/rms

:RMSprop/lctpanrywj/dhxpxqfhna/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/lctpanrywj/dhxpxqfhna/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rms
Ë
QRMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rms
Í
RRMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rms
Í
RRMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/rienwrhgrh/kngiiuzftt/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/rienwrhgrh/kngiiuzftt/kernel/rms
¦
<RMSprop/rienwrhgrh/kngiiuzftt/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/rienwrhgrh/kngiiuzftt/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rms
º
FRMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/rienwrhgrh/kngiiuzftt/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/rienwrhgrh/kngiiuzftt/bias/rms

:RMSprop/rienwrhgrh/kngiiuzftt/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/rienwrhgrh/kngiiuzftt/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rms
Ë
QRMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rms
Í
RRMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rms
Í
RRMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0

NoOpNoOp
ôC
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¯C
value¥CB¢C BC
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
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
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
­
trainable_variables
	variables
9layer_regularization_losses
	regularization_losses
:non_trainable_variables
;metrics
<layer_metrics

=layers
 
][
VARIABLE_VALUEkjxhlaztnm/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEkjxhlaztnm/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
>layer_regularization_losses
	variables
regularization_losses
?non_trainable_variables
@metrics
Alayer_metrics

Blayers
 
 
 
­
trainable_variables
Clayer_regularization_losses
	variables
regularization_losses
Dnon_trainable_variables
Emetrics
Flayer_metrics

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
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
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
 
¹
trainable_variables
	variables
Mlayer_regularization_losses

Nlayers
regularization_losses
Onon_trainable_variables
Pmetrics
Qlayer_metrics

Rstates
ó
S
state_size

3kernel
4recurrent_kernel
5bias
6input_gate_peephole_weights
 7forget_gate_peephole_weights
 8output_gate_peephole_weights
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
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
 
¹
trainable_variables
	variables
Xlayer_regularization_losses

Ylayers
 regularization_losses
Znon_trainable_variables
[metrics
\layer_metrics

]states
][
VARIABLE_VALUEuilnjhxhrx/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEuilnjhxhrx/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
­
$trainable_variables
^layer_regularization_losses
%	variables
&regularization_losses
_non_trainable_variables
`metrics
alayer_metrics

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
b`
VARIABLE_VALUElctpanrywj/dhxpxqfhna/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&lctpanrywj/dhxpxqfhna/recurrent_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUElctpanrywj/dhxpxqfhna/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1lctpanrywj/dhxpxqfhna/input_gate_peephole_weights0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2lctpanrywj/dhxpxqfhna/output_gate_peephole_weights0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUErienwrhgrh/kngiiuzftt/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&rienwrhgrh/kngiiuzftt/recurrent_kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUErienwrhgrh/kngiiuzftt/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1rienwrhgrh/kngiiuzftt/input_gate_peephole_weights1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2rienwrhgrh/kngiiuzftt/output_gate_peephole_weights1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
 

c0
 
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
 
­
Itrainable_variables
dlayer_regularization_losses
J	variables
Kregularization_losses
enon_trainable_variables
fmetrics
glayer_metrics

hlayers
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
*
30
41
52
63
74
85
 
­
Ttrainable_variables
ilayer_regularization_losses
U	variables
Vregularization_losses
jnon_trainable_variables
kmetrics
llayer_metrics

mlayers
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

VARIABLE_VALUERMSprop/kjxhlaztnm/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/kjxhlaztnm/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/uilnjhxhrx/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/uilnjhxhrx/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/lctpanrywj/dhxpxqfhna/kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/lctpanrywj/dhxpxqfhna/bias/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE=RMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE>RMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE>RMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rmsNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/rienwrhgrh/kngiiuzftt/kernel/rmsNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rmsNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/rienwrhgrh/kngiiuzftt/bias/rmsOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE=RMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rmsOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¤¡
VARIABLE_VALUE>RMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rmsOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¤¡
VARIABLE_VALUE>RMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rmsOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_dtmnvweekcPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_dtmnvweekckjxhlaztnm/kernelkjxhlaztnm/biaslctpanrywj/dhxpxqfhna/kernel&lctpanrywj/dhxpxqfhna/recurrent_kernellctpanrywj/dhxpxqfhna/bias1lctpanrywj/dhxpxqfhna/input_gate_peephole_weights2lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights2lctpanrywj/dhxpxqfhna/output_gate_peephole_weightsrienwrhgrh/kngiiuzftt/kernel&rienwrhgrh/kngiiuzftt/recurrent_kernelrienwrhgrh/kngiiuzftt/bias1rienwrhgrh/kngiiuzftt/input_gate_peephole_weights2rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights2rienwrhgrh/kngiiuzftt/output_gate_peephole_weightsuilnjhxhrx/kerneluilnjhxhrx/bias*
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
%__inference_signature_wrapper_1103328
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%kjxhlaztnm/kernel/Read/ReadVariableOp#kjxhlaztnm/bias/Read/ReadVariableOp%uilnjhxhrx/kernel/Read/ReadVariableOp#uilnjhxhrx/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0lctpanrywj/dhxpxqfhna/kernel/Read/ReadVariableOp:lctpanrywj/dhxpxqfhna/recurrent_kernel/Read/ReadVariableOp.lctpanrywj/dhxpxqfhna/bias/Read/ReadVariableOpElctpanrywj/dhxpxqfhna/input_gate_peephole_weights/Read/ReadVariableOpFlctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/Read/ReadVariableOpFlctpanrywj/dhxpxqfhna/output_gate_peephole_weights/Read/ReadVariableOp0rienwrhgrh/kngiiuzftt/kernel/Read/ReadVariableOp:rienwrhgrh/kngiiuzftt/recurrent_kernel/Read/ReadVariableOp.rienwrhgrh/kngiiuzftt/bias/Read/ReadVariableOpErienwrhgrh/kngiiuzftt/input_gate_peephole_weights/Read/ReadVariableOpFrienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/Read/ReadVariableOpFrienwrhgrh/kngiiuzftt/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/kjxhlaztnm/kernel/rms/Read/ReadVariableOp/RMSprop/kjxhlaztnm/bias/rms/Read/ReadVariableOp1RMSprop/uilnjhxhrx/kernel/rms/Read/ReadVariableOp/RMSprop/uilnjhxhrx/bias/rms/Read/ReadVariableOp<RMSprop/lctpanrywj/dhxpxqfhna/kernel/rms/Read/ReadVariableOpFRMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/lctpanrywj/dhxpxqfhna/bias/rms/Read/ReadVariableOpQRMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/rienwrhgrh/kngiiuzftt/kernel/rms/Read/ReadVariableOpFRMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/rienwrhgrh/kngiiuzftt/bias/rms/Read/ReadVariableOpQRMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
 __inference__traced_save_1106277
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamekjxhlaztnm/kernelkjxhlaztnm/biasuilnjhxhrx/kerneluilnjhxhrx/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rholctpanrywj/dhxpxqfhna/kernel&lctpanrywj/dhxpxqfhna/recurrent_kernellctpanrywj/dhxpxqfhna/bias1lctpanrywj/dhxpxqfhna/input_gate_peephole_weights2lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights2lctpanrywj/dhxpxqfhna/output_gate_peephole_weightsrienwrhgrh/kngiiuzftt/kernel&rienwrhgrh/kngiiuzftt/recurrent_kernelrienwrhgrh/kngiiuzftt/bias1rienwrhgrh/kngiiuzftt/input_gate_peephole_weights2rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights2rienwrhgrh/kngiiuzftt/output_gate_peephole_weightstotalcountRMSprop/kjxhlaztnm/kernel/rmsRMSprop/kjxhlaztnm/bias/rmsRMSprop/uilnjhxhrx/kernel/rmsRMSprop/uilnjhxhrx/bias/rms(RMSprop/lctpanrywj/dhxpxqfhna/kernel/rms2RMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rms&RMSprop/lctpanrywj/dhxpxqfhna/bias/rms=RMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rms>RMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rms>RMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rms(RMSprop/rienwrhgrh/kngiiuzftt/kernel/rms2RMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rms&RMSprop/rienwrhgrh/kngiiuzftt/bias/rms=RMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rms>RMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rms>RMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rms*3
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
#__inference__traced_restore_1106404¢ä-
Ó	
ø
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_1105860

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
¹'
µ
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1106091

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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1101421

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
Û

,__inference_lctpanrywj_layer_call_fn_1105045

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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11023362
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


,__inference_sequential_layer_call_fn_1103201

dtmnvweekc
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
dtmnvweekcunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_11031292
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
dtmnvweekc


í
while_cond_1101440
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1101440___redundant_placeholder05
1while_while_cond_1101440___redundant_placeholder15
1while_while_cond_1101440___redundant_placeholder25
1while_while_cond_1101440___redundant_placeholder35
1while_while_cond_1101440___redundant_placeholder45
1while_while_cond_1101440___redundant_placeholder55
1while_while_cond_1101440___redundant_placeholder6
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
,__inference_lctpanrywj_layer_call_fn_1105011
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11007632
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
Ó	
ø
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_1102553

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
G__inference_sequential_layer_call_and_return_conditional_losses_1103242

dtmnvweekc(
kjxhlaztnm_1103204: 
kjxhlaztnm_1103206:%
lctpanrywj_1103210:	%
lctpanrywj_1103212:	 !
lctpanrywj_1103214:	 
lctpanrywj_1103216:  
lctpanrywj_1103218:  
lctpanrywj_1103220: %
rienwrhgrh_1103223:	 %
rienwrhgrh_1103225:	 !
rienwrhgrh_1103227:	 
rienwrhgrh_1103229:  
rienwrhgrh_1103231:  
rienwrhgrh_1103233: $
uilnjhxhrx_1103236:  
uilnjhxhrx_1103238:
identity¢"kjxhlaztnm/StatefulPartitionedCall¢"lctpanrywj/StatefulPartitionedCall¢"rienwrhgrh/StatefulPartitionedCall¢"uilnjhxhrx/StatefulPartitionedCall°
"kjxhlaztnm/StatefulPartitionedCallStatefulPartitionedCall
dtmnvweekckjxhlaztnm_1103204kjxhlaztnm_1103206*
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
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_11021362$
"kjxhlaztnm/StatefulPartitionedCall
tzzrzfazij/PartitionedCallPartitionedCall+kjxhlaztnm/StatefulPartitionedCall:output:0*
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
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_11021552
tzzrzfazij/PartitionedCall
"lctpanrywj/StatefulPartitionedCallStatefulPartitionedCall#tzzrzfazij/PartitionedCall:output:0lctpanrywj_1103210lctpanrywj_1103212lctpanrywj_1103214lctpanrywj_1103216lctpanrywj_1103218lctpanrywj_1103220*
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11023362$
"lctpanrywj/StatefulPartitionedCall¡
"rienwrhgrh/StatefulPartitionedCallStatefulPartitionedCall+lctpanrywj/StatefulPartitionedCall:output:0rienwrhgrh_1103223rienwrhgrh_1103225rienwrhgrh_1103227rienwrhgrh_1103229rienwrhgrh_1103231rienwrhgrh_1103233*
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11025292$
"rienwrhgrh/StatefulPartitionedCallÉ
"uilnjhxhrx/StatefulPartitionedCallStatefulPartitionedCall+rienwrhgrh/StatefulPartitionedCall:output:0uilnjhxhrx_1103236uilnjhxhrx_1103238*
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
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_11025532$
"uilnjhxhrx/StatefulPartitionedCall
IdentityIdentity+uilnjhxhrx/StatefulPartitionedCall:output:0#^kjxhlaztnm/StatefulPartitionedCall#^lctpanrywj/StatefulPartitionedCall#^rienwrhgrh/StatefulPartitionedCall#^uilnjhxhrx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kjxhlaztnm/StatefulPartitionedCall"kjxhlaztnm/StatefulPartitionedCall2H
"lctpanrywj/StatefulPartitionedCall"lctpanrywj/StatefulPartitionedCall2H
"rienwrhgrh/StatefulPartitionedCall"rienwrhgrh/StatefulPartitionedCall2H
"uilnjhxhrx/StatefulPartitionedCall"uilnjhxhrx/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
dtmnvweekc


í
while_cond_1102702
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1102702___redundant_placeholder05
1while_while_cond_1102702___redundant_placeholder15
1while_while_cond_1102702___redundant_placeholder25
1while_while_cond_1102702___redundant_placeholder35
1while_while_cond_1102702___redundant_placeholder45
1while_while_cond_1102702___redundant_placeholder55
1while_while_cond_1102702___redundant_placeholder6
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
ì

%__inference_signature_wrapper_1103328

dtmnvweekc
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
dtmnvweekcunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_11005762
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
dtmnvweekc
p
Ê
rienwrhgrh_while_body_11040292
.rienwrhgrh_while_rienwrhgrh_while_loop_counter8
4rienwrhgrh_while_rienwrhgrh_while_maximum_iterations 
rienwrhgrh_while_placeholder"
rienwrhgrh_while_placeholder_1"
rienwrhgrh_while_placeholder_2"
rienwrhgrh_while_placeholder_31
-rienwrhgrh_while_rienwrhgrh_strided_slice_1_0m
irienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0O
<rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0:	 Q
>rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 L
=rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0:	C
5rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0: E
7rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0: E
7rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0: 
rienwrhgrh_while_identity
rienwrhgrh_while_identity_1
rienwrhgrh_while_identity_2
rienwrhgrh_while_identity_3
rienwrhgrh_while_identity_4
rienwrhgrh_while_identity_5/
+rienwrhgrh_while_rienwrhgrh_strided_slice_1k
grienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensorM
:rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource:	 O
<rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource:	 J
;rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource:	A
3rienwrhgrh_while_kngiiuzftt_readvariableop_resource: C
5rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource: C
5rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource: ¢2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp¢1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp¢3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp¢*rienwrhgrh/while/kngiiuzftt/ReadVariableOp¢,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1¢,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2Ù
Brienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Brienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shape
4rienwrhgrh/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemirienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0rienwrhgrh_while_placeholderKrienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4rienwrhgrh/while/TensorArrayV2Read/TensorListGetItemä
1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp<rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOpý
"rienwrhgrh/while/kngiiuzftt/MatMulMatMul;rienwrhgrh/while/TensorArrayV2Read/TensorListGetItem:item:09rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"rienwrhgrh/while/kngiiuzftt/MatMulê
3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp>rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOpæ
$rienwrhgrh/while/kngiiuzftt/MatMul_1MatMulrienwrhgrh_while_placeholder_2;rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rienwrhgrh/while/kngiiuzftt/MatMul_1Ü
rienwrhgrh/while/kngiiuzftt/addAddV2,rienwrhgrh/while/kngiiuzftt/MatMul:product:0.rienwrhgrh/while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
rienwrhgrh/while/kngiiuzftt/addã
2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp=rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOpé
#rienwrhgrh/while/kngiiuzftt/BiasAddBiasAdd#rienwrhgrh/while/kngiiuzftt/add:z:0:rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rienwrhgrh/while/kngiiuzftt/BiasAdd
+rienwrhgrh/while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+rienwrhgrh/while/kngiiuzftt/split/split_dim¯
!rienwrhgrh/while/kngiiuzftt/splitSplit4rienwrhgrh/while/kngiiuzftt/split/split_dim:output:0,rienwrhgrh/while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!rienwrhgrh/while/kngiiuzftt/splitÊ
*rienwrhgrh/while/kngiiuzftt/ReadVariableOpReadVariableOp5rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*rienwrhgrh/while/kngiiuzftt/ReadVariableOpÏ
rienwrhgrh/while/kngiiuzftt/mulMul2rienwrhgrh/while/kngiiuzftt/ReadVariableOp:value:0rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rienwrhgrh/while/kngiiuzftt/mulÒ
!rienwrhgrh/while/kngiiuzftt/add_1AddV2*rienwrhgrh/while/kngiiuzftt/split:output:0#rienwrhgrh/while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_1®
#rienwrhgrh/while/kngiiuzftt/SigmoidSigmoid%rienwrhgrh/while/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#rienwrhgrh/while/kngiiuzftt/SigmoidÐ
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1ReadVariableOp7rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1Õ
!rienwrhgrh/while/kngiiuzftt/mul_1Mul4rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1:value:0rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_1Ô
!rienwrhgrh/while/kngiiuzftt/add_2AddV2*rienwrhgrh/while/kngiiuzftt/split:output:1%rienwrhgrh/while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_2²
%rienwrhgrh/while/kngiiuzftt/Sigmoid_1Sigmoid%rienwrhgrh/while/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%rienwrhgrh/while/kngiiuzftt/Sigmoid_1Ê
!rienwrhgrh/while/kngiiuzftt/mul_2Mul)rienwrhgrh/while/kngiiuzftt/Sigmoid_1:y:0rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_2ª
 rienwrhgrh/while/kngiiuzftt/TanhTanh*rienwrhgrh/while/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rienwrhgrh/while/kngiiuzftt/TanhÎ
!rienwrhgrh/while/kngiiuzftt/mul_3Mul'rienwrhgrh/while/kngiiuzftt/Sigmoid:y:0$rienwrhgrh/while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_3Ï
!rienwrhgrh/while/kngiiuzftt/add_3AddV2%rienwrhgrh/while/kngiiuzftt/mul_2:z:0%rienwrhgrh/while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_3Ð
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2ReadVariableOp7rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2Ü
!rienwrhgrh/while/kngiiuzftt/mul_4Mul4rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2:value:0%rienwrhgrh/while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_4Ô
!rienwrhgrh/while/kngiiuzftt/add_4AddV2*rienwrhgrh/while/kngiiuzftt/split:output:3%rienwrhgrh/while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_4²
%rienwrhgrh/while/kngiiuzftt/Sigmoid_2Sigmoid%rienwrhgrh/while/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%rienwrhgrh/while/kngiiuzftt/Sigmoid_2©
"rienwrhgrh/while/kngiiuzftt/Tanh_1Tanh%rienwrhgrh/while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rienwrhgrh/while/kngiiuzftt/Tanh_1Ò
!rienwrhgrh/while/kngiiuzftt/mul_5Mul)rienwrhgrh/while/kngiiuzftt/Sigmoid_2:y:0&rienwrhgrh/while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_5
5rienwrhgrh/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrienwrhgrh_while_placeholder_1rienwrhgrh_while_placeholder%rienwrhgrh/while/kngiiuzftt/mul_5:z:0*
_output_shapes
: *
element_dtype027
5rienwrhgrh/while/TensorArrayV2Write/TensorListSetItemr
rienwrhgrh/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rienwrhgrh/while/add/y
rienwrhgrh/while/addAddV2rienwrhgrh_while_placeholderrienwrhgrh/while/add/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/while/addv
rienwrhgrh/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rienwrhgrh/while/add_1/y­
rienwrhgrh/while/add_1AddV2.rienwrhgrh_while_rienwrhgrh_while_loop_counter!rienwrhgrh/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/while/add_1©
rienwrhgrh/while/IdentityIdentityrienwrhgrh/while/add_1:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/IdentityÇ
rienwrhgrh/while/Identity_1Identity4rienwrhgrh_while_rienwrhgrh_while_maximum_iterations3^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/Identity_1«
rienwrhgrh/while/Identity_2Identityrienwrhgrh/while/add:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/Identity_2Ø
rienwrhgrh/while/Identity_3IdentityErienwrhgrh/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/Identity_3É
rienwrhgrh/while/Identity_4Identity%rienwrhgrh/while/kngiiuzftt/mul_5:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/while/Identity_4É
rienwrhgrh/while/Identity_5Identity%rienwrhgrh/while/kngiiuzftt/add_3:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/while/Identity_5"?
rienwrhgrh_while_identity"rienwrhgrh/while/Identity:output:0"C
rienwrhgrh_while_identity_1$rienwrhgrh/while/Identity_1:output:0"C
rienwrhgrh_while_identity_2$rienwrhgrh/while/Identity_2:output:0"C
rienwrhgrh_while_identity_3$rienwrhgrh/while/Identity_3:output:0"C
rienwrhgrh_while_identity_4$rienwrhgrh/while/Identity_4:output:0"C
rienwrhgrh_while_identity_5$rienwrhgrh/while/Identity_5:output:0"|
;rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource=rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0"~
<rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource>rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0"z
:rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource<rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0"p
5rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource7rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0"p
5rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource7rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0"l
3rienwrhgrh_while_kngiiuzftt_readvariableop_resource5rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0"\
+rienwrhgrh_while_rienwrhgrh_strided_slice_1-rienwrhgrh_while_rienwrhgrh_strided_slice_1_0"Ô
grienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensoririenwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2f
1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp2j
3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp2X
*rienwrhgrh/while/kngiiuzftt/ReadVariableOp*rienwrhgrh/while/kngiiuzftt/ReadVariableOp2\
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_12\
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2: 
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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1100850

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

À
,__inference_dhxpxqfhna_layer_call_fn_1105980

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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_11006632
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
¡h

G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104994

inputs<
)dhxpxqfhna_matmul_readvariableop_resource:	>
+dhxpxqfhna_matmul_1_readvariableop_resource:	 9
*dhxpxqfhna_biasadd_readvariableop_resource:	0
"dhxpxqfhna_readvariableop_resource: 2
$dhxpxqfhna_readvariableop_1_resource: 2
$dhxpxqfhna_readvariableop_2_resource: 
identity¢!dhxpxqfhna/BiasAdd/ReadVariableOp¢ dhxpxqfhna/MatMul/ReadVariableOp¢"dhxpxqfhna/MatMul_1/ReadVariableOp¢dhxpxqfhna/ReadVariableOp¢dhxpxqfhna/ReadVariableOp_1¢dhxpxqfhna/ReadVariableOp_2¢whileD
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
 dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp)dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dhxpxqfhna/MatMul/ReadVariableOp§
dhxpxqfhna/MatMulMatMulstrided_slice_2:output:0(dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMulµ
"dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp+dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dhxpxqfhna/MatMul_1/ReadVariableOp£
dhxpxqfhna/MatMul_1MatMulzeros:output:0*dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMul_1
dhxpxqfhna/addAddV2dhxpxqfhna/MatMul:product:0dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/add®
!dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp*dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dhxpxqfhna/BiasAdd/ReadVariableOp¥
dhxpxqfhna/BiasAddBiasAdddhxpxqfhna/add:z:0)dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/BiasAddz
dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dhxpxqfhna/split/split_dimë
dhxpxqfhna/splitSplit#dhxpxqfhna/split/split_dim:output:0dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dhxpxqfhna/split
dhxpxqfhna/ReadVariableOpReadVariableOp"dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp
dhxpxqfhna/mulMul!dhxpxqfhna/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul
dhxpxqfhna/add_1AddV2dhxpxqfhna/split:output:0dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_1{
dhxpxqfhna/SigmoidSigmoiddhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid
dhxpxqfhna/ReadVariableOp_1ReadVariableOp$dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_1
dhxpxqfhna/mul_1Mul#dhxpxqfhna/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_1
dhxpxqfhna/add_2AddV2dhxpxqfhna/split:output:1dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_2
dhxpxqfhna/Sigmoid_1Sigmoiddhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_1
dhxpxqfhna/mul_2Muldhxpxqfhna/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_2w
dhxpxqfhna/TanhTanhdhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh
dhxpxqfhna/mul_3Muldhxpxqfhna/Sigmoid:y:0dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_3
dhxpxqfhna/add_3AddV2dhxpxqfhna/mul_2:z:0dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_3
dhxpxqfhna/ReadVariableOp_2ReadVariableOp$dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_2
dhxpxqfhna/mul_4Mul#dhxpxqfhna/ReadVariableOp_2:value:0dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_4
dhxpxqfhna/add_4AddV2dhxpxqfhna/split:output:3dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_4
dhxpxqfhna/Sigmoid_2Sigmoiddhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_2v
dhxpxqfhna/Tanh_1Tanhdhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh_1
dhxpxqfhna/mul_5Muldhxpxqfhna/Sigmoid_2:y:0dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dhxpxqfhna_matmul_readvariableop_resource+dhxpxqfhna_matmul_1_readvariableop_resource*dhxpxqfhna_biasadd_readvariableop_resource"dhxpxqfhna_readvariableop_resource$dhxpxqfhna_readvariableop_1_resource$dhxpxqfhna_readvariableop_2_resource*
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
while_body_1104893*
condR
while_cond_1104892*Q
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
IdentityIdentitytranspose_1:y:0"^dhxpxqfhna/BiasAdd/ReadVariableOp!^dhxpxqfhna/MatMul/ReadVariableOp#^dhxpxqfhna/MatMul_1/ReadVariableOp^dhxpxqfhna/ReadVariableOp^dhxpxqfhna/ReadVariableOp_1^dhxpxqfhna/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dhxpxqfhna/BiasAdd/ReadVariableOp!dhxpxqfhna/BiasAdd/ReadVariableOp2D
 dhxpxqfhna/MatMul/ReadVariableOp dhxpxqfhna/MatMul/ReadVariableOp2H
"dhxpxqfhna/MatMul_1/ReadVariableOp"dhxpxqfhna/MatMul_1/ReadVariableOp26
dhxpxqfhna/ReadVariableOpdhxpxqfhna/ReadVariableOp2:
dhxpxqfhna/ReadVariableOp_1dhxpxqfhna/ReadVariableOp_12:
dhxpxqfhna/ReadVariableOp_2dhxpxqfhna/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àY

while_body_1105141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kngiiuzftt_matmul_readvariableop_resource_0:	 F
3while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 A
2while_kngiiuzftt_biasadd_readvariableop_resource_0:	8
*while_kngiiuzftt_readvariableop_resource_0: :
,while_kngiiuzftt_readvariableop_1_resource_0: :
,while_kngiiuzftt_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kngiiuzftt_matmul_readvariableop_resource:	 D
1while_kngiiuzftt_matmul_1_readvariableop_resource:	 ?
0while_kngiiuzftt_biasadd_readvariableop_resource:	6
(while_kngiiuzftt_readvariableop_resource: 8
*while_kngiiuzftt_readvariableop_1_resource: 8
*while_kngiiuzftt_readvariableop_2_resource: ¢'while/kngiiuzftt/BiasAdd/ReadVariableOp¢&while/kngiiuzftt/MatMul/ReadVariableOp¢(while/kngiiuzftt/MatMul_1/ReadVariableOp¢while/kngiiuzftt/ReadVariableOp¢!while/kngiiuzftt/ReadVariableOp_1¢!while/kngiiuzftt/ReadVariableOp_2Ã
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
&while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp1while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/kngiiuzftt/MatMul/ReadVariableOpÑ
while/kngiiuzftt/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMulÉ
(while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp3while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kngiiuzftt/MatMul_1/ReadVariableOpº
while/kngiiuzftt/MatMul_1MatMulwhile_placeholder_20while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMul_1°
while/kngiiuzftt/addAddV2!while/kngiiuzftt/MatMul:product:0#while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/addÂ
'while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp2while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kngiiuzftt/BiasAdd/ReadVariableOp½
while/kngiiuzftt/BiasAddBiasAddwhile/kngiiuzftt/add:z:0/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/BiasAdd
 while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kngiiuzftt/split/split_dim
while/kngiiuzftt/splitSplit)while/kngiiuzftt/split/split_dim:output:0!while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kngiiuzftt/split©
while/kngiiuzftt/ReadVariableOpReadVariableOp*while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kngiiuzftt/ReadVariableOp£
while/kngiiuzftt/mulMul'while/kngiiuzftt/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul¦
while/kngiiuzftt/add_1AddV2while/kngiiuzftt/split:output:0while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_1
while/kngiiuzftt/SigmoidSigmoidwhile/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid¯
!while/kngiiuzftt/ReadVariableOp_1ReadVariableOp,while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_1©
while/kngiiuzftt/mul_1Mul)while/kngiiuzftt/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_1¨
while/kngiiuzftt/add_2AddV2while/kngiiuzftt/split:output:1while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_2
while/kngiiuzftt/Sigmoid_1Sigmoidwhile/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_1
while/kngiiuzftt/mul_2Mulwhile/kngiiuzftt/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_2
while/kngiiuzftt/TanhTanhwhile/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh¢
while/kngiiuzftt/mul_3Mulwhile/kngiiuzftt/Sigmoid:y:0while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_3£
while/kngiiuzftt/add_3AddV2while/kngiiuzftt/mul_2:z:0while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_3¯
!while/kngiiuzftt/ReadVariableOp_2ReadVariableOp,while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_2°
while/kngiiuzftt/mul_4Mul)while/kngiiuzftt/ReadVariableOp_2:value:0while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_4¨
while/kngiiuzftt/add_4AddV2while/kngiiuzftt/split:output:3while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_4
while/kngiiuzftt/Sigmoid_2Sigmoidwhile/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_2
while/kngiiuzftt/Tanh_1Tanhwhile/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh_1¦
while/kngiiuzftt/mul_5Mulwhile/kngiiuzftt/Sigmoid_2:y:0while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kngiiuzftt/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kngiiuzftt/mul_5:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kngiiuzftt/add_3:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
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
0while_kngiiuzftt_biasadd_readvariableop_resource2while_kngiiuzftt_biasadd_readvariableop_resource_0"h
1while_kngiiuzftt_matmul_1_readvariableop_resource3while_kngiiuzftt_matmul_1_readvariableop_resource_0"d
/while_kngiiuzftt_matmul_readvariableop_resource1while_kngiiuzftt_matmul_readvariableop_resource_0"Z
*while_kngiiuzftt_readvariableop_1_resource,while_kngiiuzftt_readvariableop_1_resource_0"Z
*while_kngiiuzftt_readvariableop_2_resource,while_kngiiuzftt_readvariableop_2_resource_0"V
(while_kngiiuzftt_readvariableop_resource*while_kngiiuzftt_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kngiiuzftt/BiasAdd/ReadVariableOp'while/kngiiuzftt/BiasAdd/ReadVariableOp2P
&while/kngiiuzftt/MatMul/ReadVariableOp&while/kngiiuzftt/MatMul/ReadVariableOp2T
(while/kngiiuzftt/MatMul_1/ReadVariableOp(while/kngiiuzftt/MatMul_1/ReadVariableOp2B
while/kngiiuzftt/ReadVariableOpwhile/kngiiuzftt/ReadVariableOp2F
!while/kngiiuzftt/ReadVariableOp_1!while/kngiiuzftt/ReadVariableOp_12F
!while/kngiiuzftt/ReadVariableOp_2!while/kngiiuzftt/ReadVariableOp_2: 
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
,__inference_rienwrhgrh_layer_call_fn_1105816
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11017842
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
¦h

G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1102529

inputs<
)kngiiuzftt_matmul_readvariableop_resource:	 >
+kngiiuzftt_matmul_1_readvariableop_resource:	 9
*kngiiuzftt_biasadd_readvariableop_resource:	0
"kngiiuzftt_readvariableop_resource: 2
$kngiiuzftt_readvariableop_1_resource: 2
$kngiiuzftt_readvariableop_2_resource: 
identity¢!kngiiuzftt/BiasAdd/ReadVariableOp¢ kngiiuzftt/MatMul/ReadVariableOp¢"kngiiuzftt/MatMul_1/ReadVariableOp¢kngiiuzftt/ReadVariableOp¢kngiiuzftt/ReadVariableOp_1¢kngiiuzftt/ReadVariableOp_2¢whileD
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
 kngiiuzftt/MatMul/ReadVariableOpReadVariableOp)kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 kngiiuzftt/MatMul/ReadVariableOp§
kngiiuzftt/MatMulMatMulstrided_slice_2:output:0(kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMulµ
"kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp+kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kngiiuzftt/MatMul_1/ReadVariableOp£
kngiiuzftt/MatMul_1MatMulzeros:output:0*kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMul_1
kngiiuzftt/addAddV2kngiiuzftt/MatMul:product:0kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/add®
!kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp*kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kngiiuzftt/BiasAdd/ReadVariableOp¥
kngiiuzftt/BiasAddBiasAddkngiiuzftt/add:z:0)kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/BiasAddz
kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kngiiuzftt/split/split_dimë
kngiiuzftt/splitSplit#kngiiuzftt/split/split_dim:output:0kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kngiiuzftt/split
kngiiuzftt/ReadVariableOpReadVariableOp"kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp
kngiiuzftt/mulMul!kngiiuzftt/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul
kngiiuzftt/add_1AddV2kngiiuzftt/split:output:0kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_1{
kngiiuzftt/SigmoidSigmoidkngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid
kngiiuzftt/ReadVariableOp_1ReadVariableOp$kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_1
kngiiuzftt/mul_1Mul#kngiiuzftt/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_1
kngiiuzftt/add_2AddV2kngiiuzftt/split:output:1kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_2
kngiiuzftt/Sigmoid_1Sigmoidkngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_1
kngiiuzftt/mul_2Mulkngiiuzftt/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_2w
kngiiuzftt/TanhTanhkngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh
kngiiuzftt/mul_3Mulkngiiuzftt/Sigmoid:y:0kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_3
kngiiuzftt/add_3AddV2kngiiuzftt/mul_2:z:0kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_3
kngiiuzftt/ReadVariableOp_2ReadVariableOp$kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_2
kngiiuzftt/mul_4Mul#kngiiuzftt/ReadVariableOp_2:value:0kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_4
kngiiuzftt/add_4AddV2kngiiuzftt/split:output:3kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_4
kngiiuzftt/Sigmoid_2Sigmoidkngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_2v
kngiiuzftt/Tanh_1Tanhkngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh_1
kngiiuzftt/mul_5Mulkngiiuzftt/Sigmoid_2:y:0kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kngiiuzftt_matmul_readvariableop_resource+kngiiuzftt_matmul_1_readvariableop_resource*kngiiuzftt_biasadd_readvariableop_resource"kngiiuzftt_readvariableop_resource$kngiiuzftt_readvariableop_1_resource$kngiiuzftt_readvariableop_2_resource*
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
while_body_1102428*
condR
while_cond_1102427*Q
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
IdentityIdentitystrided_slice_3:output:0"^kngiiuzftt/BiasAdd/ReadVariableOp!^kngiiuzftt/MatMul/ReadVariableOp#^kngiiuzftt/MatMul_1/ReadVariableOp^kngiiuzftt/ReadVariableOp^kngiiuzftt/ReadVariableOp_1^kngiiuzftt/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!kngiiuzftt/BiasAdd/ReadVariableOp!kngiiuzftt/BiasAdd/ReadVariableOp2D
 kngiiuzftt/MatMul/ReadVariableOp kngiiuzftt/MatMul/ReadVariableOp2H
"kngiiuzftt/MatMul_1/ReadVariableOp"kngiiuzftt/MatMul_1/ReadVariableOp26
kngiiuzftt/ReadVariableOpkngiiuzftt/ReadVariableOp2:
kngiiuzftt/ReadVariableOp_1kngiiuzftt/ReadVariableOp_12:
kngiiuzftt/ReadVariableOp_2kngiiuzftt/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¡h

G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1103018

inputs<
)dhxpxqfhna_matmul_readvariableop_resource:	>
+dhxpxqfhna_matmul_1_readvariableop_resource:	 9
*dhxpxqfhna_biasadd_readvariableop_resource:	0
"dhxpxqfhna_readvariableop_resource: 2
$dhxpxqfhna_readvariableop_1_resource: 2
$dhxpxqfhna_readvariableop_2_resource: 
identity¢!dhxpxqfhna/BiasAdd/ReadVariableOp¢ dhxpxqfhna/MatMul/ReadVariableOp¢"dhxpxqfhna/MatMul_1/ReadVariableOp¢dhxpxqfhna/ReadVariableOp¢dhxpxqfhna/ReadVariableOp_1¢dhxpxqfhna/ReadVariableOp_2¢whileD
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
 dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp)dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dhxpxqfhna/MatMul/ReadVariableOp§
dhxpxqfhna/MatMulMatMulstrided_slice_2:output:0(dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMulµ
"dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp+dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dhxpxqfhna/MatMul_1/ReadVariableOp£
dhxpxqfhna/MatMul_1MatMulzeros:output:0*dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMul_1
dhxpxqfhna/addAddV2dhxpxqfhna/MatMul:product:0dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/add®
!dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp*dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dhxpxqfhna/BiasAdd/ReadVariableOp¥
dhxpxqfhna/BiasAddBiasAdddhxpxqfhna/add:z:0)dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/BiasAddz
dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dhxpxqfhna/split/split_dimë
dhxpxqfhna/splitSplit#dhxpxqfhna/split/split_dim:output:0dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dhxpxqfhna/split
dhxpxqfhna/ReadVariableOpReadVariableOp"dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp
dhxpxqfhna/mulMul!dhxpxqfhna/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul
dhxpxqfhna/add_1AddV2dhxpxqfhna/split:output:0dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_1{
dhxpxqfhna/SigmoidSigmoiddhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid
dhxpxqfhna/ReadVariableOp_1ReadVariableOp$dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_1
dhxpxqfhna/mul_1Mul#dhxpxqfhna/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_1
dhxpxqfhna/add_2AddV2dhxpxqfhna/split:output:1dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_2
dhxpxqfhna/Sigmoid_1Sigmoiddhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_1
dhxpxqfhna/mul_2Muldhxpxqfhna/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_2w
dhxpxqfhna/TanhTanhdhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh
dhxpxqfhna/mul_3Muldhxpxqfhna/Sigmoid:y:0dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_3
dhxpxqfhna/add_3AddV2dhxpxqfhna/mul_2:z:0dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_3
dhxpxqfhna/ReadVariableOp_2ReadVariableOp$dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_2
dhxpxqfhna/mul_4Mul#dhxpxqfhna/ReadVariableOp_2:value:0dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_4
dhxpxqfhna/add_4AddV2dhxpxqfhna/split:output:3dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_4
dhxpxqfhna/Sigmoid_2Sigmoiddhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_2v
dhxpxqfhna/Tanh_1Tanhdhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh_1
dhxpxqfhna/mul_5Muldhxpxqfhna/Sigmoid_2:y:0dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dhxpxqfhna_matmul_readvariableop_resource+dhxpxqfhna_matmul_1_readvariableop_resource*dhxpxqfhna_biasadd_readvariableop_resource"dhxpxqfhna_readvariableop_resource$dhxpxqfhna_readvariableop_1_resource$dhxpxqfhna_readvariableop_2_resource*
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
while_body_1102917*
condR
while_cond_1102916*Q
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
IdentityIdentitytranspose_1:y:0"^dhxpxqfhna/BiasAdd/ReadVariableOp!^dhxpxqfhna/MatMul/ReadVariableOp#^dhxpxqfhna/MatMul_1/ReadVariableOp^dhxpxqfhna/ReadVariableOp^dhxpxqfhna/ReadVariableOp_1^dhxpxqfhna/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dhxpxqfhna/BiasAdd/ReadVariableOp!dhxpxqfhna/BiasAdd/ReadVariableOp2D
 dhxpxqfhna/MatMul/ReadVariableOp dhxpxqfhna/MatMul/ReadVariableOp2H
"dhxpxqfhna/MatMul_1/ReadVariableOp"dhxpxqfhna/MatMul_1/ReadVariableOp26
dhxpxqfhna/ReadVariableOpdhxpxqfhna/ReadVariableOp2:
dhxpxqfhna/ReadVariableOp_1dhxpxqfhna/ReadVariableOp_12:
dhxpxqfhna/ReadVariableOp_2dhxpxqfhna/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦h

G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105602

inputs<
)kngiiuzftt_matmul_readvariableop_resource:	 >
+kngiiuzftt_matmul_1_readvariableop_resource:	 9
*kngiiuzftt_biasadd_readvariableop_resource:	0
"kngiiuzftt_readvariableop_resource: 2
$kngiiuzftt_readvariableop_1_resource: 2
$kngiiuzftt_readvariableop_2_resource: 
identity¢!kngiiuzftt/BiasAdd/ReadVariableOp¢ kngiiuzftt/MatMul/ReadVariableOp¢"kngiiuzftt/MatMul_1/ReadVariableOp¢kngiiuzftt/ReadVariableOp¢kngiiuzftt/ReadVariableOp_1¢kngiiuzftt/ReadVariableOp_2¢whileD
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
 kngiiuzftt/MatMul/ReadVariableOpReadVariableOp)kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 kngiiuzftt/MatMul/ReadVariableOp§
kngiiuzftt/MatMulMatMulstrided_slice_2:output:0(kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMulµ
"kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp+kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kngiiuzftt/MatMul_1/ReadVariableOp£
kngiiuzftt/MatMul_1MatMulzeros:output:0*kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMul_1
kngiiuzftt/addAddV2kngiiuzftt/MatMul:product:0kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/add®
!kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp*kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kngiiuzftt/BiasAdd/ReadVariableOp¥
kngiiuzftt/BiasAddBiasAddkngiiuzftt/add:z:0)kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/BiasAddz
kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kngiiuzftt/split/split_dimë
kngiiuzftt/splitSplit#kngiiuzftt/split/split_dim:output:0kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kngiiuzftt/split
kngiiuzftt/ReadVariableOpReadVariableOp"kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp
kngiiuzftt/mulMul!kngiiuzftt/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul
kngiiuzftt/add_1AddV2kngiiuzftt/split:output:0kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_1{
kngiiuzftt/SigmoidSigmoidkngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid
kngiiuzftt/ReadVariableOp_1ReadVariableOp$kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_1
kngiiuzftt/mul_1Mul#kngiiuzftt/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_1
kngiiuzftt/add_2AddV2kngiiuzftt/split:output:1kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_2
kngiiuzftt/Sigmoid_1Sigmoidkngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_1
kngiiuzftt/mul_2Mulkngiiuzftt/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_2w
kngiiuzftt/TanhTanhkngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh
kngiiuzftt/mul_3Mulkngiiuzftt/Sigmoid:y:0kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_3
kngiiuzftt/add_3AddV2kngiiuzftt/mul_2:z:0kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_3
kngiiuzftt/ReadVariableOp_2ReadVariableOp$kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_2
kngiiuzftt/mul_4Mul#kngiiuzftt/ReadVariableOp_2:value:0kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_4
kngiiuzftt/add_4AddV2kngiiuzftt/split:output:3kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_4
kngiiuzftt/Sigmoid_2Sigmoidkngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_2v
kngiiuzftt/Tanh_1Tanhkngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh_1
kngiiuzftt/mul_5Mulkngiiuzftt/Sigmoid_2:y:0kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kngiiuzftt_matmul_readvariableop_resource+kngiiuzftt_matmul_1_readvariableop_resource*kngiiuzftt_biasadd_readvariableop_resource"kngiiuzftt_readvariableop_resource$kngiiuzftt_readvariableop_1_resource$kngiiuzftt_readvariableop_2_resource*
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
while_body_1105501*
condR
while_cond_1105500*Q
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
IdentityIdentitystrided_slice_3:output:0"^kngiiuzftt/BiasAdd/ReadVariableOp!^kngiiuzftt/MatMul/ReadVariableOp#^kngiiuzftt/MatMul_1/ReadVariableOp^kngiiuzftt/ReadVariableOp^kngiiuzftt/ReadVariableOp_1^kngiiuzftt/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!kngiiuzftt/BiasAdd/ReadVariableOp!kngiiuzftt/BiasAdd/ReadVariableOp2D
 kngiiuzftt/MatMul/ReadVariableOp kngiiuzftt/MatMul/ReadVariableOp2H
"kngiiuzftt/MatMul_1/ReadVariableOp"kngiiuzftt/MatMul_1/ReadVariableOp26
kngiiuzftt/ReadVariableOpkngiiuzftt/ReadVariableOp2:
kngiiuzftt/ReadVariableOp_1kngiiuzftt/ReadVariableOp_12:
kngiiuzftt/ReadVariableOp_2kngiiuzftt/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
àY

while_body_1105681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kngiiuzftt_matmul_readvariableop_resource_0:	 F
3while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 A
2while_kngiiuzftt_biasadd_readvariableop_resource_0:	8
*while_kngiiuzftt_readvariableop_resource_0: :
,while_kngiiuzftt_readvariableop_1_resource_0: :
,while_kngiiuzftt_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kngiiuzftt_matmul_readvariableop_resource:	 D
1while_kngiiuzftt_matmul_1_readvariableop_resource:	 ?
0while_kngiiuzftt_biasadd_readvariableop_resource:	6
(while_kngiiuzftt_readvariableop_resource: 8
*while_kngiiuzftt_readvariableop_1_resource: 8
*while_kngiiuzftt_readvariableop_2_resource: ¢'while/kngiiuzftt/BiasAdd/ReadVariableOp¢&while/kngiiuzftt/MatMul/ReadVariableOp¢(while/kngiiuzftt/MatMul_1/ReadVariableOp¢while/kngiiuzftt/ReadVariableOp¢!while/kngiiuzftt/ReadVariableOp_1¢!while/kngiiuzftt/ReadVariableOp_2Ã
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
&while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp1while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/kngiiuzftt/MatMul/ReadVariableOpÑ
while/kngiiuzftt/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMulÉ
(while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp3while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kngiiuzftt/MatMul_1/ReadVariableOpº
while/kngiiuzftt/MatMul_1MatMulwhile_placeholder_20while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMul_1°
while/kngiiuzftt/addAddV2!while/kngiiuzftt/MatMul:product:0#while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/addÂ
'while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp2while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kngiiuzftt/BiasAdd/ReadVariableOp½
while/kngiiuzftt/BiasAddBiasAddwhile/kngiiuzftt/add:z:0/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/BiasAdd
 while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kngiiuzftt/split/split_dim
while/kngiiuzftt/splitSplit)while/kngiiuzftt/split/split_dim:output:0!while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kngiiuzftt/split©
while/kngiiuzftt/ReadVariableOpReadVariableOp*while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kngiiuzftt/ReadVariableOp£
while/kngiiuzftt/mulMul'while/kngiiuzftt/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul¦
while/kngiiuzftt/add_1AddV2while/kngiiuzftt/split:output:0while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_1
while/kngiiuzftt/SigmoidSigmoidwhile/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid¯
!while/kngiiuzftt/ReadVariableOp_1ReadVariableOp,while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_1©
while/kngiiuzftt/mul_1Mul)while/kngiiuzftt/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_1¨
while/kngiiuzftt/add_2AddV2while/kngiiuzftt/split:output:1while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_2
while/kngiiuzftt/Sigmoid_1Sigmoidwhile/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_1
while/kngiiuzftt/mul_2Mulwhile/kngiiuzftt/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_2
while/kngiiuzftt/TanhTanhwhile/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh¢
while/kngiiuzftt/mul_3Mulwhile/kngiiuzftt/Sigmoid:y:0while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_3£
while/kngiiuzftt/add_3AddV2while/kngiiuzftt/mul_2:z:0while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_3¯
!while/kngiiuzftt/ReadVariableOp_2ReadVariableOp,while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_2°
while/kngiiuzftt/mul_4Mul)while/kngiiuzftt/ReadVariableOp_2:value:0while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_4¨
while/kngiiuzftt/add_4AddV2while/kngiiuzftt/split:output:3while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_4
while/kngiiuzftt/Sigmoid_2Sigmoidwhile/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_2
while/kngiiuzftt/Tanh_1Tanhwhile/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh_1¦
while/kngiiuzftt/mul_5Mulwhile/kngiiuzftt/Sigmoid_2:y:0while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kngiiuzftt/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kngiiuzftt/mul_5:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kngiiuzftt/add_3:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
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
0while_kngiiuzftt_biasadd_readvariableop_resource2while_kngiiuzftt_biasadd_readvariableop_resource_0"h
1while_kngiiuzftt_matmul_1_readvariableop_resource3while_kngiiuzftt_matmul_1_readvariableop_resource_0"d
/while_kngiiuzftt_matmul_readvariableop_resource1while_kngiiuzftt_matmul_readvariableop_resource_0"Z
*while_kngiiuzftt_readvariableop_1_resource,while_kngiiuzftt_readvariableop_1_resource_0"Z
*while_kngiiuzftt_readvariableop_2_resource,while_kngiiuzftt_readvariableop_2_resource_0"V
(while_kngiiuzftt_readvariableop_resource*while_kngiiuzftt_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kngiiuzftt/BiasAdd/ReadVariableOp'while/kngiiuzftt/BiasAdd/ReadVariableOp2P
&while/kngiiuzftt/MatMul/ReadVariableOp&while/kngiiuzftt/MatMul/ReadVariableOp2T
(while/kngiiuzftt/MatMul_1/ReadVariableOp(while/kngiiuzftt/MatMul_1/ReadVariableOp2B
while/kngiiuzftt/ReadVariableOpwhile/kngiiuzftt/ReadVariableOp2F
!while/kngiiuzftt/ReadVariableOp_1!while/kngiiuzftt/ReadVariableOp_12F
!while/kngiiuzftt/ReadVariableOp_2!while/kngiiuzftt/ReadVariableOp_2: 
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1101784

inputs%
kngiiuzftt_1101685:	 %
kngiiuzftt_1101687:	 !
kngiiuzftt_1101689:	 
kngiiuzftt_1101691:  
kngiiuzftt_1101693:  
kngiiuzftt_1101695: 
identity¢"kngiiuzftt/StatefulPartitionedCall¢whileD
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
"kngiiuzftt/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0kngiiuzftt_1101685kngiiuzftt_1101687kngiiuzftt_1101689kngiiuzftt_1101691kngiiuzftt_1101693kngiiuzftt_1101695*
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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_11016082$
"kngiiuzftt/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kngiiuzftt_1101685kngiiuzftt_1101687kngiiuzftt_1101689kngiiuzftt_1101691kngiiuzftt_1101693kngiiuzftt_1101695*
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
while_body_1101704*
condR
while_cond_1101703*Q
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
IdentityIdentitystrided_slice_3:output:0#^kngiiuzftt/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"kngiiuzftt/StatefulPartitionedCall"kngiiuzftt/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_1103283

dtmnvweekc(
kjxhlaztnm_1103245: 
kjxhlaztnm_1103247:%
lctpanrywj_1103251:	%
lctpanrywj_1103253:	 !
lctpanrywj_1103255:	 
lctpanrywj_1103257:  
lctpanrywj_1103259:  
lctpanrywj_1103261: %
rienwrhgrh_1103264:	 %
rienwrhgrh_1103266:	 !
rienwrhgrh_1103268:	 
rienwrhgrh_1103270:  
rienwrhgrh_1103272:  
rienwrhgrh_1103274: $
uilnjhxhrx_1103277:  
uilnjhxhrx_1103279:
identity¢"kjxhlaztnm/StatefulPartitionedCall¢"lctpanrywj/StatefulPartitionedCall¢"rienwrhgrh/StatefulPartitionedCall¢"uilnjhxhrx/StatefulPartitionedCall°
"kjxhlaztnm/StatefulPartitionedCallStatefulPartitionedCall
dtmnvweekckjxhlaztnm_1103245kjxhlaztnm_1103247*
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
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_11021362$
"kjxhlaztnm/StatefulPartitionedCall
tzzrzfazij/PartitionedCallPartitionedCall+kjxhlaztnm/StatefulPartitionedCall:output:0*
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
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_11021552
tzzrzfazij/PartitionedCall
"lctpanrywj/StatefulPartitionedCallStatefulPartitionedCall#tzzrzfazij/PartitionedCall:output:0lctpanrywj_1103251lctpanrywj_1103253lctpanrywj_1103255lctpanrywj_1103257lctpanrywj_1103259lctpanrywj_1103261*
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11030182$
"lctpanrywj/StatefulPartitionedCall¡
"rienwrhgrh/StatefulPartitionedCallStatefulPartitionedCall+lctpanrywj/StatefulPartitionedCall:output:0rienwrhgrh_1103264rienwrhgrh_1103266rienwrhgrh_1103268rienwrhgrh_1103270rienwrhgrh_1103272rienwrhgrh_1103274*
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11028042$
"rienwrhgrh/StatefulPartitionedCallÉ
"uilnjhxhrx/StatefulPartitionedCallStatefulPartitionedCall+rienwrhgrh/StatefulPartitionedCall:output:0uilnjhxhrx_1103277uilnjhxhrx_1103279*
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
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_11025532$
"uilnjhxhrx/StatefulPartitionedCall
IdentityIdentity+uilnjhxhrx/StatefulPartitionedCall:output:0#^kjxhlaztnm/StatefulPartitionedCall#^lctpanrywj/StatefulPartitionedCall#^rienwrhgrh/StatefulPartitionedCall#^uilnjhxhrx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kjxhlaztnm/StatefulPartitionedCall"kjxhlaztnm/StatefulPartitionedCall2H
"lctpanrywj/StatefulPartitionedCall"lctpanrywj/StatefulPartitionedCall2H
"rienwrhgrh/StatefulPartitionedCall"rienwrhgrh/StatefulPartitionedCall2H
"uilnjhxhrx/StatefulPartitionedCall"uilnjhxhrx/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
dtmnvweekc
Üh

G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105422
inputs_0<
)kngiiuzftt_matmul_readvariableop_resource:	 >
+kngiiuzftt_matmul_1_readvariableop_resource:	 9
*kngiiuzftt_biasadd_readvariableop_resource:	0
"kngiiuzftt_readvariableop_resource: 2
$kngiiuzftt_readvariableop_1_resource: 2
$kngiiuzftt_readvariableop_2_resource: 
identity¢!kngiiuzftt/BiasAdd/ReadVariableOp¢ kngiiuzftt/MatMul/ReadVariableOp¢"kngiiuzftt/MatMul_1/ReadVariableOp¢kngiiuzftt/ReadVariableOp¢kngiiuzftt/ReadVariableOp_1¢kngiiuzftt/ReadVariableOp_2¢whileF
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
 kngiiuzftt/MatMul/ReadVariableOpReadVariableOp)kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 kngiiuzftt/MatMul/ReadVariableOp§
kngiiuzftt/MatMulMatMulstrided_slice_2:output:0(kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMulµ
"kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp+kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kngiiuzftt/MatMul_1/ReadVariableOp£
kngiiuzftt/MatMul_1MatMulzeros:output:0*kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMul_1
kngiiuzftt/addAddV2kngiiuzftt/MatMul:product:0kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/add®
!kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp*kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kngiiuzftt/BiasAdd/ReadVariableOp¥
kngiiuzftt/BiasAddBiasAddkngiiuzftt/add:z:0)kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/BiasAddz
kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kngiiuzftt/split/split_dimë
kngiiuzftt/splitSplit#kngiiuzftt/split/split_dim:output:0kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kngiiuzftt/split
kngiiuzftt/ReadVariableOpReadVariableOp"kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp
kngiiuzftt/mulMul!kngiiuzftt/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul
kngiiuzftt/add_1AddV2kngiiuzftt/split:output:0kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_1{
kngiiuzftt/SigmoidSigmoidkngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid
kngiiuzftt/ReadVariableOp_1ReadVariableOp$kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_1
kngiiuzftt/mul_1Mul#kngiiuzftt/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_1
kngiiuzftt/add_2AddV2kngiiuzftt/split:output:1kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_2
kngiiuzftt/Sigmoid_1Sigmoidkngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_1
kngiiuzftt/mul_2Mulkngiiuzftt/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_2w
kngiiuzftt/TanhTanhkngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh
kngiiuzftt/mul_3Mulkngiiuzftt/Sigmoid:y:0kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_3
kngiiuzftt/add_3AddV2kngiiuzftt/mul_2:z:0kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_3
kngiiuzftt/ReadVariableOp_2ReadVariableOp$kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_2
kngiiuzftt/mul_4Mul#kngiiuzftt/ReadVariableOp_2:value:0kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_4
kngiiuzftt/add_4AddV2kngiiuzftt/split:output:3kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_4
kngiiuzftt/Sigmoid_2Sigmoidkngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_2v
kngiiuzftt/Tanh_1Tanhkngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh_1
kngiiuzftt/mul_5Mulkngiiuzftt/Sigmoid_2:y:0kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kngiiuzftt_matmul_readvariableop_resource+kngiiuzftt_matmul_1_readvariableop_resource*kngiiuzftt_biasadd_readvariableop_resource"kngiiuzftt_readvariableop_resource$kngiiuzftt_readvariableop_1_resource$kngiiuzftt_readvariableop_2_resource*
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
while_body_1105321*
condR
while_cond_1105320*Q
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
IdentityIdentitystrided_slice_3:output:0"^kngiiuzftt/BiasAdd/ReadVariableOp!^kngiiuzftt/MatMul/ReadVariableOp#^kngiiuzftt/MatMul_1/ReadVariableOp^kngiiuzftt/ReadVariableOp^kngiiuzftt/ReadVariableOp_1^kngiiuzftt/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!kngiiuzftt/BiasAdd/ReadVariableOp!kngiiuzftt/BiasAdd/ReadVariableOp2D
 kngiiuzftt/MatMul/ReadVariableOp kngiiuzftt/MatMul/ReadVariableOp2H
"kngiiuzftt/MatMul_1/ReadVariableOp"kngiiuzftt/MatMul_1/ReadVariableOp26
kngiiuzftt/ReadVariableOpkngiiuzftt/ReadVariableOp2:
kngiiuzftt/ReadVariableOp_1kngiiuzftt/ReadVariableOp_12:
kngiiuzftt/ReadVariableOp_2kngiiuzftt/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Ý
H
,__inference_tzzrzfazij_layer_call_fn_1104274

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
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_11021552
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
while_cond_1100945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1100945___redundant_placeholder05
1while_while_cond_1100945___redundant_placeholder15
1while_while_cond_1100945___redundant_placeholder25
1while_while_cond_1100945___redundant_placeholder35
1while_while_cond_1100945___redundant_placeholder45
1while_while_cond_1100945___redundant_placeholder55
1while_while_cond_1100945___redundant_placeholder6
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
¦h

G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105782

inputs<
)kngiiuzftt_matmul_readvariableop_resource:	 >
+kngiiuzftt_matmul_1_readvariableop_resource:	 9
*kngiiuzftt_biasadd_readvariableop_resource:	0
"kngiiuzftt_readvariableop_resource: 2
$kngiiuzftt_readvariableop_1_resource: 2
$kngiiuzftt_readvariableop_2_resource: 
identity¢!kngiiuzftt/BiasAdd/ReadVariableOp¢ kngiiuzftt/MatMul/ReadVariableOp¢"kngiiuzftt/MatMul_1/ReadVariableOp¢kngiiuzftt/ReadVariableOp¢kngiiuzftt/ReadVariableOp_1¢kngiiuzftt/ReadVariableOp_2¢whileD
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
 kngiiuzftt/MatMul/ReadVariableOpReadVariableOp)kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 kngiiuzftt/MatMul/ReadVariableOp§
kngiiuzftt/MatMulMatMulstrided_slice_2:output:0(kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMulµ
"kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp+kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kngiiuzftt/MatMul_1/ReadVariableOp£
kngiiuzftt/MatMul_1MatMulzeros:output:0*kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMul_1
kngiiuzftt/addAddV2kngiiuzftt/MatMul:product:0kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/add®
!kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp*kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kngiiuzftt/BiasAdd/ReadVariableOp¥
kngiiuzftt/BiasAddBiasAddkngiiuzftt/add:z:0)kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/BiasAddz
kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kngiiuzftt/split/split_dimë
kngiiuzftt/splitSplit#kngiiuzftt/split/split_dim:output:0kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kngiiuzftt/split
kngiiuzftt/ReadVariableOpReadVariableOp"kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp
kngiiuzftt/mulMul!kngiiuzftt/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul
kngiiuzftt/add_1AddV2kngiiuzftt/split:output:0kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_1{
kngiiuzftt/SigmoidSigmoidkngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid
kngiiuzftt/ReadVariableOp_1ReadVariableOp$kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_1
kngiiuzftt/mul_1Mul#kngiiuzftt/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_1
kngiiuzftt/add_2AddV2kngiiuzftt/split:output:1kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_2
kngiiuzftt/Sigmoid_1Sigmoidkngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_1
kngiiuzftt/mul_2Mulkngiiuzftt/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_2w
kngiiuzftt/TanhTanhkngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh
kngiiuzftt/mul_3Mulkngiiuzftt/Sigmoid:y:0kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_3
kngiiuzftt/add_3AddV2kngiiuzftt/mul_2:z:0kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_3
kngiiuzftt/ReadVariableOp_2ReadVariableOp$kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_2
kngiiuzftt/mul_4Mul#kngiiuzftt/ReadVariableOp_2:value:0kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_4
kngiiuzftt/add_4AddV2kngiiuzftt/split:output:3kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_4
kngiiuzftt/Sigmoid_2Sigmoidkngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_2v
kngiiuzftt/Tanh_1Tanhkngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh_1
kngiiuzftt/mul_5Mulkngiiuzftt/Sigmoid_2:y:0kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kngiiuzftt_matmul_readvariableop_resource+kngiiuzftt_matmul_1_readvariableop_resource*kngiiuzftt_biasadd_readvariableop_resource"kngiiuzftt_readvariableop_resource$kngiiuzftt_readvariableop_1_resource$kngiiuzftt_readvariableop_2_resource*
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
while_body_1105681*
condR
while_cond_1105680*Q
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
IdentityIdentitystrided_slice_3:output:0"^kngiiuzftt/BiasAdd/ReadVariableOp!^kngiiuzftt/MatMul/ReadVariableOp#^kngiiuzftt/MatMul_1/ReadVariableOp^kngiiuzftt/ReadVariableOp^kngiiuzftt/ReadVariableOp_1^kngiiuzftt/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!kngiiuzftt/BiasAdd/ReadVariableOp!kngiiuzftt/BiasAdd/ReadVariableOp2D
 kngiiuzftt/MatMul/ReadVariableOp kngiiuzftt/MatMul/ReadVariableOp2H
"kngiiuzftt/MatMul_1/ReadVariableOp"kngiiuzftt/MatMul_1/ReadVariableOp26
kngiiuzftt/ReadVariableOpkngiiuzftt/ReadVariableOp2:
kngiiuzftt/ReadVariableOp_1kngiiuzftt/ReadVariableOp_12:
kngiiuzftt/ReadVariableOp_2kngiiuzftt/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
àh

G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104634
inputs_0<
)dhxpxqfhna_matmul_readvariableop_resource:	>
+dhxpxqfhna_matmul_1_readvariableop_resource:	 9
*dhxpxqfhna_biasadd_readvariableop_resource:	0
"dhxpxqfhna_readvariableop_resource: 2
$dhxpxqfhna_readvariableop_1_resource: 2
$dhxpxqfhna_readvariableop_2_resource: 
identity¢!dhxpxqfhna/BiasAdd/ReadVariableOp¢ dhxpxqfhna/MatMul/ReadVariableOp¢"dhxpxqfhna/MatMul_1/ReadVariableOp¢dhxpxqfhna/ReadVariableOp¢dhxpxqfhna/ReadVariableOp_1¢dhxpxqfhna/ReadVariableOp_2¢whileF
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
 dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp)dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dhxpxqfhna/MatMul/ReadVariableOp§
dhxpxqfhna/MatMulMatMulstrided_slice_2:output:0(dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMulµ
"dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp+dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dhxpxqfhna/MatMul_1/ReadVariableOp£
dhxpxqfhna/MatMul_1MatMulzeros:output:0*dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMul_1
dhxpxqfhna/addAddV2dhxpxqfhna/MatMul:product:0dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/add®
!dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp*dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dhxpxqfhna/BiasAdd/ReadVariableOp¥
dhxpxqfhna/BiasAddBiasAdddhxpxqfhna/add:z:0)dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/BiasAddz
dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dhxpxqfhna/split/split_dimë
dhxpxqfhna/splitSplit#dhxpxqfhna/split/split_dim:output:0dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dhxpxqfhna/split
dhxpxqfhna/ReadVariableOpReadVariableOp"dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp
dhxpxqfhna/mulMul!dhxpxqfhna/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul
dhxpxqfhna/add_1AddV2dhxpxqfhna/split:output:0dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_1{
dhxpxqfhna/SigmoidSigmoiddhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid
dhxpxqfhna/ReadVariableOp_1ReadVariableOp$dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_1
dhxpxqfhna/mul_1Mul#dhxpxqfhna/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_1
dhxpxqfhna/add_2AddV2dhxpxqfhna/split:output:1dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_2
dhxpxqfhna/Sigmoid_1Sigmoiddhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_1
dhxpxqfhna/mul_2Muldhxpxqfhna/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_2w
dhxpxqfhna/TanhTanhdhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh
dhxpxqfhna/mul_3Muldhxpxqfhna/Sigmoid:y:0dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_3
dhxpxqfhna/add_3AddV2dhxpxqfhna/mul_2:z:0dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_3
dhxpxqfhna/ReadVariableOp_2ReadVariableOp$dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_2
dhxpxqfhna/mul_4Mul#dhxpxqfhna/ReadVariableOp_2:value:0dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_4
dhxpxqfhna/add_4AddV2dhxpxqfhna/split:output:3dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_4
dhxpxqfhna/Sigmoid_2Sigmoiddhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_2v
dhxpxqfhna/Tanh_1Tanhdhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh_1
dhxpxqfhna/mul_5Muldhxpxqfhna/Sigmoid_2:y:0dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dhxpxqfhna_matmul_readvariableop_resource+dhxpxqfhna_matmul_1_readvariableop_resource*dhxpxqfhna_biasadd_readvariableop_resource"dhxpxqfhna_readvariableop_resource$dhxpxqfhna_readvariableop_1_resource$dhxpxqfhna_readvariableop_2_resource*
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
while_body_1104533*
condR
while_cond_1104532*Q
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
IdentityIdentitytranspose_1:y:0"^dhxpxqfhna/BiasAdd/ReadVariableOp!^dhxpxqfhna/MatMul/ReadVariableOp#^dhxpxqfhna/MatMul_1/ReadVariableOp^dhxpxqfhna/ReadVariableOp^dhxpxqfhna/ReadVariableOp_1^dhxpxqfhna/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dhxpxqfhna/BiasAdd/ReadVariableOp!dhxpxqfhna/BiasAdd/ReadVariableOp2D
 dhxpxqfhna/MatMul/ReadVariableOp dhxpxqfhna/MatMul/ReadVariableOp2H
"dhxpxqfhna/MatMul_1/ReadVariableOp"dhxpxqfhna/MatMul_1/ReadVariableOp26
dhxpxqfhna/ReadVariableOpdhxpxqfhna/ReadVariableOp2:
dhxpxqfhna/ReadVariableOp_1dhxpxqfhna/ReadVariableOp_12:
dhxpxqfhna/ReadVariableOp_2dhxpxqfhna/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
p
Ê
lctpanrywj_while_body_11034492
.lctpanrywj_while_lctpanrywj_while_loop_counter8
4lctpanrywj_while_lctpanrywj_while_maximum_iterations 
lctpanrywj_while_placeholder"
lctpanrywj_while_placeholder_1"
lctpanrywj_while_placeholder_2"
lctpanrywj_while_placeholder_31
-lctpanrywj_while_lctpanrywj_strided_slice_1_0m
ilctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0O
<lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0:	Q
>lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 L
=lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0:	C
5lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0: E
7lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0: E
7lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0: 
lctpanrywj_while_identity
lctpanrywj_while_identity_1
lctpanrywj_while_identity_2
lctpanrywj_while_identity_3
lctpanrywj_while_identity_4
lctpanrywj_while_identity_5/
+lctpanrywj_while_lctpanrywj_strided_slice_1k
glctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensorM
:lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource:	O
<lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource:	 J
;lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource:	A
3lctpanrywj_while_dhxpxqfhna_readvariableop_resource: C
5lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource: C
5lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource: ¢2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp¢1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp¢3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp¢*lctpanrywj/while/dhxpxqfhna/ReadVariableOp¢,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1¢,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2Ù
Blctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Blctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shape
4lctpanrywj/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemilctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0lctpanrywj_while_placeholderKlctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4lctpanrywj/while/TensorArrayV2Read/TensorListGetItemä
1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp<lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOpý
"lctpanrywj/while/dhxpxqfhna/MatMulMatMul;lctpanrywj/while/TensorArrayV2Read/TensorListGetItem:item:09lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lctpanrywj/while/dhxpxqfhna/MatMulê
3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp>lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOpæ
$lctpanrywj/while/dhxpxqfhna/MatMul_1MatMullctpanrywj_while_placeholder_2;lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lctpanrywj/while/dhxpxqfhna/MatMul_1Ü
lctpanrywj/while/dhxpxqfhna/addAddV2,lctpanrywj/while/dhxpxqfhna/MatMul:product:0.lctpanrywj/while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lctpanrywj/while/dhxpxqfhna/addã
2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp=lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOpé
#lctpanrywj/while/dhxpxqfhna/BiasAddBiasAdd#lctpanrywj/while/dhxpxqfhna/add:z:0:lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lctpanrywj/while/dhxpxqfhna/BiasAdd
+lctpanrywj/while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+lctpanrywj/while/dhxpxqfhna/split/split_dim¯
!lctpanrywj/while/dhxpxqfhna/splitSplit4lctpanrywj/while/dhxpxqfhna/split/split_dim:output:0,lctpanrywj/while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!lctpanrywj/while/dhxpxqfhna/splitÊ
*lctpanrywj/while/dhxpxqfhna/ReadVariableOpReadVariableOp5lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*lctpanrywj/while/dhxpxqfhna/ReadVariableOpÏ
lctpanrywj/while/dhxpxqfhna/mulMul2lctpanrywj/while/dhxpxqfhna/ReadVariableOp:value:0lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lctpanrywj/while/dhxpxqfhna/mulÒ
!lctpanrywj/while/dhxpxqfhna/add_1AddV2*lctpanrywj/while/dhxpxqfhna/split:output:0#lctpanrywj/while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_1®
#lctpanrywj/while/dhxpxqfhna/SigmoidSigmoid%lctpanrywj/while/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#lctpanrywj/while/dhxpxqfhna/SigmoidÐ
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp7lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1Õ
!lctpanrywj/while/dhxpxqfhna/mul_1Mul4lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1:value:0lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_1Ô
!lctpanrywj/while/dhxpxqfhna/add_2AddV2*lctpanrywj/while/dhxpxqfhna/split:output:1%lctpanrywj/while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_2²
%lctpanrywj/while/dhxpxqfhna/Sigmoid_1Sigmoid%lctpanrywj/while/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%lctpanrywj/while/dhxpxqfhna/Sigmoid_1Ê
!lctpanrywj/while/dhxpxqfhna/mul_2Mul)lctpanrywj/while/dhxpxqfhna/Sigmoid_1:y:0lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_2ª
 lctpanrywj/while/dhxpxqfhna/TanhTanh*lctpanrywj/while/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 lctpanrywj/while/dhxpxqfhna/TanhÎ
!lctpanrywj/while/dhxpxqfhna/mul_3Mul'lctpanrywj/while/dhxpxqfhna/Sigmoid:y:0$lctpanrywj/while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_3Ï
!lctpanrywj/while/dhxpxqfhna/add_3AddV2%lctpanrywj/while/dhxpxqfhna/mul_2:z:0%lctpanrywj/while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_3Ð
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp7lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2Ü
!lctpanrywj/while/dhxpxqfhna/mul_4Mul4lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2:value:0%lctpanrywj/while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_4Ô
!lctpanrywj/while/dhxpxqfhna/add_4AddV2*lctpanrywj/while/dhxpxqfhna/split:output:3%lctpanrywj/while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_4²
%lctpanrywj/while/dhxpxqfhna/Sigmoid_2Sigmoid%lctpanrywj/while/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%lctpanrywj/while/dhxpxqfhna/Sigmoid_2©
"lctpanrywj/while/dhxpxqfhna/Tanh_1Tanh%lctpanrywj/while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"lctpanrywj/while/dhxpxqfhna/Tanh_1Ò
!lctpanrywj/while/dhxpxqfhna/mul_5Mul)lctpanrywj/while/dhxpxqfhna/Sigmoid_2:y:0&lctpanrywj/while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_5
5lctpanrywj/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlctpanrywj_while_placeholder_1lctpanrywj_while_placeholder%lctpanrywj/while/dhxpxqfhna/mul_5:z:0*
_output_shapes
: *
element_dtype027
5lctpanrywj/while/TensorArrayV2Write/TensorListSetItemr
lctpanrywj/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lctpanrywj/while/add/y
lctpanrywj/while/addAddV2lctpanrywj_while_placeholderlctpanrywj/while/add/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/while/addv
lctpanrywj/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lctpanrywj/while/add_1/y­
lctpanrywj/while/add_1AddV2.lctpanrywj_while_lctpanrywj_while_loop_counter!lctpanrywj/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/while/add_1©
lctpanrywj/while/IdentityIdentitylctpanrywj/while/add_1:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/IdentityÇ
lctpanrywj/while/Identity_1Identity4lctpanrywj_while_lctpanrywj_while_maximum_iterations3^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/Identity_1«
lctpanrywj/while/Identity_2Identitylctpanrywj/while/add:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/Identity_2Ø
lctpanrywj/while/Identity_3IdentityElctpanrywj/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/Identity_3É
lctpanrywj/while/Identity_4Identity%lctpanrywj/while/dhxpxqfhna/mul_5:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/while/Identity_4É
lctpanrywj/while/Identity_5Identity%lctpanrywj/while/dhxpxqfhna/add_3:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/while/Identity_5"|
;lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource=lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0"~
<lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource>lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0"z
:lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource<lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0"p
5lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource7lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0"p
5lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource7lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0"l
3lctpanrywj_while_dhxpxqfhna_readvariableop_resource5lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0"?
lctpanrywj_while_identity"lctpanrywj/while/Identity:output:0"C
lctpanrywj_while_identity_1$lctpanrywj/while/Identity_1:output:0"C
lctpanrywj_while_identity_2$lctpanrywj/while/Identity_2:output:0"C
lctpanrywj_while_identity_3$lctpanrywj/while/Identity_3:output:0"C
lctpanrywj_while_identity_4$lctpanrywj/while/Identity_4:output:0"C
lctpanrywj_while_identity_5$lctpanrywj/while/Identity_5:output:0"\
+lctpanrywj_while_lctpanrywj_strided_slice_1-lctpanrywj_while_lctpanrywj_strided_slice_1_0"Ô
glctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensorilctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2f
1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp2j
3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp2X
*lctpanrywj/while/dhxpxqfhna/ReadVariableOp*lctpanrywj/while/dhxpxqfhna/ReadVariableOp2\
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_12\
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2: 
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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1104136

inputsL
6kjxhlaztnm_conv1d_expanddims_1_readvariableop_resource:K
=kjxhlaztnm_squeeze_batch_dims_biasadd_readvariableop_resource:G
4lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource:	I
6lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource:	 D
5lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource:	;
-lctpanrywj_dhxpxqfhna_readvariableop_resource: =
/lctpanrywj_dhxpxqfhna_readvariableop_1_resource: =
/lctpanrywj_dhxpxqfhna_readvariableop_2_resource: G
4rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource:	 I
6rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource:	 D
5rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource:	;
-rienwrhgrh_kngiiuzftt_readvariableop_resource: =
/rienwrhgrh_kngiiuzftt_readvariableop_1_resource: =
/rienwrhgrh_kngiiuzftt_readvariableop_2_resource: ;
)uilnjhxhrx_matmul_readvariableop_resource: 8
*uilnjhxhrx_biasadd_readvariableop_resource:
identity¢-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp¢4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp¢+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp¢-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp¢$lctpanrywj/dhxpxqfhna/ReadVariableOp¢&lctpanrywj/dhxpxqfhna/ReadVariableOp_1¢&lctpanrywj/dhxpxqfhna/ReadVariableOp_2¢lctpanrywj/while¢,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp¢+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp¢-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp¢$rienwrhgrh/kngiiuzftt/ReadVariableOp¢&rienwrhgrh/kngiiuzftt/ReadVariableOp_1¢&rienwrhgrh/kngiiuzftt/ReadVariableOp_2¢rienwrhgrh/while¢!uilnjhxhrx/BiasAdd/ReadVariableOp¢ uilnjhxhrx/MatMul/ReadVariableOp
 kjxhlaztnm/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 kjxhlaztnm/conv1d/ExpandDims/dim»
kjxhlaztnm/conv1d/ExpandDims
ExpandDimsinputs)kjxhlaztnm/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/ExpandDimsÙ
-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6kjxhlaztnm_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp
"kjxhlaztnm/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"kjxhlaztnm/conv1d/ExpandDims_1/dimã
kjxhlaztnm/conv1d/ExpandDims_1
ExpandDims5kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp:value:0+kjxhlaztnm/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
kjxhlaztnm/conv1d/ExpandDims_1
kjxhlaztnm/conv1d/ShapeShape%kjxhlaztnm/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
kjxhlaztnm/conv1d/Shape
%kjxhlaztnm/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%kjxhlaztnm/conv1d/strided_slice/stack¥
'kjxhlaztnm/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'kjxhlaztnm/conv1d/strided_slice/stack_1
'kjxhlaztnm/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'kjxhlaztnm/conv1d/strided_slice/stack_2Ì
kjxhlaztnm/conv1d/strided_sliceStridedSlice kjxhlaztnm/conv1d/Shape:output:0.kjxhlaztnm/conv1d/strided_slice/stack:output:00kjxhlaztnm/conv1d/strided_slice/stack_1:output:00kjxhlaztnm/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
kjxhlaztnm/conv1d/strided_slice
kjxhlaztnm/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
kjxhlaztnm/conv1d/Reshape/shapeÌ
kjxhlaztnm/conv1d/ReshapeReshape%kjxhlaztnm/conv1d/ExpandDims:output:0(kjxhlaztnm/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/Reshapeî
kjxhlaztnm/conv1d/Conv2DConv2D"kjxhlaztnm/conv1d/Reshape:output:0'kjxhlaztnm/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
kjxhlaztnm/conv1d/Conv2D
!kjxhlaztnm/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!kjxhlaztnm/conv1d/concat/values_1
kjxhlaztnm/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/concat/axisì
kjxhlaztnm/conv1d/concatConcatV2(kjxhlaztnm/conv1d/strided_slice:output:0*kjxhlaztnm/conv1d/concat/values_1:output:0&kjxhlaztnm/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
kjxhlaztnm/conv1d/concatÉ
kjxhlaztnm/conv1d/Reshape_1Reshape!kjxhlaztnm/conv1d/Conv2D:output:0!kjxhlaztnm/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/Reshape_1Á
kjxhlaztnm/conv1d/SqueezeSqueeze$kjxhlaztnm/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/Squeeze
#kjxhlaztnm/squeeze_batch_dims/ShapeShape"kjxhlaztnm/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#kjxhlaztnm/squeeze_batch_dims/Shape°
1kjxhlaztnm/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1kjxhlaztnm/squeeze_batch_dims/strided_slice/stack½
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1´
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2
+kjxhlaztnm/squeeze_batch_dims/strided_sliceStridedSlice,kjxhlaztnm/squeeze_batch_dims/Shape:output:0:kjxhlaztnm/squeeze_batch_dims/strided_slice/stack:output:0<kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1:output:0<kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+kjxhlaztnm/squeeze_batch_dims/strided_slice¯
+kjxhlaztnm/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+kjxhlaztnm/squeeze_batch_dims/Reshape/shapeé
%kjxhlaztnm/squeeze_batch_dims/ReshapeReshape"kjxhlaztnm/conv1d/Squeeze:output:04kjxhlaztnm/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%kjxhlaztnm/squeeze_batch_dims/Reshapeæ
4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=kjxhlaztnm_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%kjxhlaztnm/squeeze_batch_dims/BiasAddBiasAdd.kjxhlaztnm/squeeze_batch_dims/Reshape:output:0<kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%kjxhlaztnm/squeeze_batch_dims/BiasAdd¯
-kjxhlaztnm/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-kjxhlaztnm/squeeze_batch_dims/concat/values_1¡
)kjxhlaztnm/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)kjxhlaztnm/squeeze_batch_dims/concat/axis¨
$kjxhlaztnm/squeeze_batch_dims/concatConcatV24kjxhlaztnm/squeeze_batch_dims/strided_slice:output:06kjxhlaztnm/squeeze_batch_dims/concat/values_1:output:02kjxhlaztnm/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$kjxhlaztnm/squeeze_batch_dims/concatö
'kjxhlaztnm/squeeze_batch_dims/Reshape_1Reshape.kjxhlaztnm/squeeze_batch_dims/BiasAdd:output:0-kjxhlaztnm/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'kjxhlaztnm/squeeze_batch_dims/Reshape_1
tzzrzfazij/ShapeShape0kjxhlaztnm/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
tzzrzfazij/Shape
tzzrzfazij/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
tzzrzfazij/strided_slice/stack
 tzzrzfazij/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 tzzrzfazij/strided_slice/stack_1
 tzzrzfazij/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 tzzrzfazij/strided_slice/stack_2¤
tzzrzfazij/strided_sliceStridedSlicetzzrzfazij/Shape:output:0'tzzrzfazij/strided_slice/stack:output:0)tzzrzfazij/strided_slice/stack_1:output:0)tzzrzfazij/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tzzrzfazij/strided_slicez
tzzrzfazij/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tzzrzfazij/Reshape/shape/1z
tzzrzfazij/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tzzrzfazij/Reshape/shape/2×
tzzrzfazij/Reshape/shapePack!tzzrzfazij/strided_slice:output:0#tzzrzfazij/Reshape/shape/1:output:0#tzzrzfazij/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
tzzrzfazij/Reshape/shape¾
tzzrzfazij/ReshapeReshape0kjxhlaztnm/squeeze_batch_dims/Reshape_1:output:0!tzzrzfazij/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tzzrzfazij/Reshapeo
lctpanrywj/ShapeShapetzzrzfazij/Reshape:output:0*
T0*
_output_shapes
:2
lctpanrywj/Shape
lctpanrywj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lctpanrywj/strided_slice/stack
 lctpanrywj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lctpanrywj/strided_slice/stack_1
 lctpanrywj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lctpanrywj/strided_slice/stack_2¤
lctpanrywj/strided_sliceStridedSlicelctpanrywj/Shape:output:0'lctpanrywj/strided_slice/stack:output:0)lctpanrywj/strided_slice/stack_1:output:0)lctpanrywj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lctpanrywj/strided_slicer
lctpanrywj/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros/mul/y
lctpanrywj/zeros/mulMul!lctpanrywj/strided_slice:output:0lctpanrywj/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros/mulu
lctpanrywj/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lctpanrywj/zeros/Less/y
lctpanrywj/zeros/LessLesslctpanrywj/zeros/mul:z:0 lctpanrywj/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros/Lessx
lctpanrywj/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros/packed/1¯
lctpanrywj/zeros/packedPack!lctpanrywj/strided_slice:output:0"lctpanrywj/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lctpanrywj/zeros/packedu
lctpanrywj/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lctpanrywj/zeros/Const¡
lctpanrywj/zerosFill lctpanrywj/zeros/packed:output:0lctpanrywj/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/zerosv
lctpanrywj/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros_1/mul/y
lctpanrywj/zeros_1/mulMul!lctpanrywj/strided_slice:output:0!lctpanrywj/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros_1/muly
lctpanrywj/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lctpanrywj/zeros_1/Less/y
lctpanrywj/zeros_1/LessLesslctpanrywj/zeros_1/mul:z:0"lctpanrywj/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros_1/Less|
lctpanrywj/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros_1/packed/1µ
lctpanrywj/zeros_1/packedPack!lctpanrywj/strided_slice:output:0$lctpanrywj/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lctpanrywj/zeros_1/packedy
lctpanrywj/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lctpanrywj/zeros_1/Const©
lctpanrywj/zeros_1Fill"lctpanrywj/zeros_1/packed:output:0!lctpanrywj/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/zeros_1
lctpanrywj/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lctpanrywj/transpose/perm°
lctpanrywj/transpose	Transposetzzrzfazij/Reshape:output:0"lctpanrywj/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/transposep
lctpanrywj/Shape_1Shapelctpanrywj/transpose:y:0*
T0*
_output_shapes
:2
lctpanrywj/Shape_1
 lctpanrywj/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 lctpanrywj/strided_slice_1/stack
"lctpanrywj/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_1/stack_1
"lctpanrywj/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_1/stack_2°
lctpanrywj/strided_slice_1StridedSlicelctpanrywj/Shape_1:output:0)lctpanrywj/strided_slice_1/stack:output:0+lctpanrywj/strided_slice_1/stack_1:output:0+lctpanrywj/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lctpanrywj/strided_slice_1
&lctpanrywj/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&lctpanrywj/TensorArrayV2/element_shapeÞ
lctpanrywj/TensorArrayV2TensorListReserve/lctpanrywj/TensorArrayV2/element_shape:output:0#lctpanrywj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lctpanrywj/TensorArrayV2Õ
@lctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@lctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2lctpanrywj/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlctpanrywj/transpose:y:0Ilctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2lctpanrywj/TensorArrayUnstack/TensorListFromTensor
 lctpanrywj/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 lctpanrywj/strided_slice_2/stack
"lctpanrywj/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_2/stack_1
"lctpanrywj/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_2/stack_2¾
lctpanrywj/strided_slice_2StridedSlicelctpanrywj/transpose:y:0)lctpanrywj/strided_slice_2/stack:output:0+lctpanrywj/strided_slice_2/stack_1:output:0+lctpanrywj/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lctpanrywj/strided_slice_2Ð
+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp4lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOpÓ
lctpanrywj/dhxpxqfhna/MatMulMatMul#lctpanrywj/strided_slice_2:output:03lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/dhxpxqfhna/MatMulÖ
-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp6lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOpÏ
lctpanrywj/dhxpxqfhna/MatMul_1MatMullctpanrywj/zeros:output:05lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lctpanrywj/dhxpxqfhna/MatMul_1Ä
lctpanrywj/dhxpxqfhna/addAddV2&lctpanrywj/dhxpxqfhna/MatMul:product:0(lctpanrywj/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/dhxpxqfhna/addÏ
,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp5lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOpÑ
lctpanrywj/dhxpxqfhna/BiasAddBiasAddlctpanrywj/dhxpxqfhna/add:z:04lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/dhxpxqfhna/BiasAdd
%lctpanrywj/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%lctpanrywj/dhxpxqfhna/split/split_dim
lctpanrywj/dhxpxqfhna/splitSplit.lctpanrywj/dhxpxqfhna/split/split_dim:output:0&lctpanrywj/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lctpanrywj/dhxpxqfhna/split¶
$lctpanrywj/dhxpxqfhna/ReadVariableOpReadVariableOp-lctpanrywj_dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02&
$lctpanrywj/dhxpxqfhna/ReadVariableOpº
lctpanrywj/dhxpxqfhna/mulMul,lctpanrywj/dhxpxqfhna/ReadVariableOp:value:0lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mulº
lctpanrywj/dhxpxqfhna/add_1AddV2$lctpanrywj/dhxpxqfhna/split:output:0lctpanrywj/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_1
lctpanrywj/dhxpxqfhna/SigmoidSigmoidlctpanrywj/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/Sigmoid¼
&lctpanrywj/dhxpxqfhna/ReadVariableOp_1ReadVariableOp/lctpanrywj_dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&lctpanrywj/dhxpxqfhna/ReadVariableOp_1À
lctpanrywj/dhxpxqfhna/mul_1Mul.lctpanrywj/dhxpxqfhna/ReadVariableOp_1:value:0lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_1¼
lctpanrywj/dhxpxqfhna/add_2AddV2$lctpanrywj/dhxpxqfhna/split:output:1lctpanrywj/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_2 
lctpanrywj/dhxpxqfhna/Sigmoid_1Sigmoidlctpanrywj/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lctpanrywj/dhxpxqfhna/Sigmoid_1µ
lctpanrywj/dhxpxqfhna/mul_2Mul#lctpanrywj/dhxpxqfhna/Sigmoid_1:y:0lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_2
lctpanrywj/dhxpxqfhna/TanhTanh$lctpanrywj/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/Tanh¶
lctpanrywj/dhxpxqfhna/mul_3Mul!lctpanrywj/dhxpxqfhna/Sigmoid:y:0lctpanrywj/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_3·
lctpanrywj/dhxpxqfhna/add_3AddV2lctpanrywj/dhxpxqfhna/mul_2:z:0lctpanrywj/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_3¼
&lctpanrywj/dhxpxqfhna/ReadVariableOp_2ReadVariableOp/lctpanrywj_dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&lctpanrywj/dhxpxqfhna/ReadVariableOp_2Ä
lctpanrywj/dhxpxqfhna/mul_4Mul.lctpanrywj/dhxpxqfhna/ReadVariableOp_2:value:0lctpanrywj/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_4¼
lctpanrywj/dhxpxqfhna/add_4AddV2$lctpanrywj/dhxpxqfhna/split:output:3lctpanrywj/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_4 
lctpanrywj/dhxpxqfhna/Sigmoid_2Sigmoidlctpanrywj/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lctpanrywj/dhxpxqfhna/Sigmoid_2
lctpanrywj/dhxpxqfhna/Tanh_1Tanhlctpanrywj/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/Tanh_1º
lctpanrywj/dhxpxqfhna/mul_5Mul#lctpanrywj/dhxpxqfhna/Sigmoid_2:y:0 lctpanrywj/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_5¥
(lctpanrywj/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(lctpanrywj/TensorArrayV2_1/element_shapeä
lctpanrywj/TensorArrayV2_1TensorListReserve1lctpanrywj/TensorArrayV2_1/element_shape:output:0#lctpanrywj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lctpanrywj/TensorArrayV2_1d
lctpanrywj/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/time
#lctpanrywj/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lctpanrywj/while/maximum_iterations
lctpanrywj/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/while/loop_counter²
lctpanrywj/whileWhile&lctpanrywj/while/loop_counter:output:0,lctpanrywj/while/maximum_iterations:output:0lctpanrywj/time:output:0#lctpanrywj/TensorArrayV2_1:handle:0lctpanrywj/zeros:output:0lctpanrywj/zeros_1:output:0#lctpanrywj/strided_slice_1:output:0Blctpanrywj/TensorArrayUnstack/TensorListFromTensor:output_handle:04lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource6lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource5lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource-lctpanrywj_dhxpxqfhna_readvariableop_resource/lctpanrywj_dhxpxqfhna_readvariableop_1_resource/lctpanrywj_dhxpxqfhna_readvariableop_2_resource*
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
lctpanrywj_while_body_1103853*)
cond!R
lctpanrywj_while_cond_1103852*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
lctpanrywj/whileË
;lctpanrywj/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;lctpanrywj/TensorArrayV2Stack/TensorListStack/element_shape
-lctpanrywj/TensorArrayV2Stack/TensorListStackTensorListStacklctpanrywj/while:output:3Dlctpanrywj/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-lctpanrywj/TensorArrayV2Stack/TensorListStack
 lctpanrywj/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 lctpanrywj/strided_slice_3/stack
"lctpanrywj/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"lctpanrywj/strided_slice_3/stack_1
"lctpanrywj/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_3/stack_2Ü
lctpanrywj/strided_slice_3StridedSlice6lctpanrywj/TensorArrayV2Stack/TensorListStack:tensor:0)lctpanrywj/strided_slice_3/stack:output:0+lctpanrywj/strided_slice_3/stack_1:output:0+lctpanrywj/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
lctpanrywj/strided_slice_3
lctpanrywj/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lctpanrywj/transpose_1/permÑ
lctpanrywj/transpose_1	Transpose6lctpanrywj/TensorArrayV2Stack/TensorListStack:tensor:0$lctpanrywj/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/transpose_1n
rienwrhgrh/ShapeShapelctpanrywj/transpose_1:y:0*
T0*
_output_shapes
:2
rienwrhgrh/Shape
rienwrhgrh/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
rienwrhgrh/strided_slice/stack
 rienwrhgrh/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 rienwrhgrh/strided_slice/stack_1
 rienwrhgrh/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 rienwrhgrh/strided_slice/stack_2¤
rienwrhgrh/strided_sliceStridedSlicerienwrhgrh/Shape:output:0'rienwrhgrh/strided_slice/stack:output:0)rienwrhgrh/strided_slice/stack_1:output:0)rienwrhgrh/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rienwrhgrh/strided_slicer
rienwrhgrh/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros/mul/y
rienwrhgrh/zeros/mulMul!rienwrhgrh/strided_slice:output:0rienwrhgrh/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros/mulu
rienwrhgrh/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rienwrhgrh/zeros/Less/y
rienwrhgrh/zeros/LessLessrienwrhgrh/zeros/mul:z:0 rienwrhgrh/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros/Lessx
rienwrhgrh/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros/packed/1¯
rienwrhgrh/zeros/packedPack!rienwrhgrh/strided_slice:output:0"rienwrhgrh/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rienwrhgrh/zeros/packedu
rienwrhgrh/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rienwrhgrh/zeros/Const¡
rienwrhgrh/zerosFill rienwrhgrh/zeros/packed:output:0rienwrhgrh/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/zerosv
rienwrhgrh/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros_1/mul/y
rienwrhgrh/zeros_1/mulMul!rienwrhgrh/strided_slice:output:0!rienwrhgrh/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros_1/muly
rienwrhgrh/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rienwrhgrh/zeros_1/Less/y
rienwrhgrh/zeros_1/LessLessrienwrhgrh/zeros_1/mul:z:0"rienwrhgrh/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros_1/Less|
rienwrhgrh/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros_1/packed/1µ
rienwrhgrh/zeros_1/packedPack!rienwrhgrh/strided_slice:output:0$rienwrhgrh/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rienwrhgrh/zeros_1/packedy
rienwrhgrh/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rienwrhgrh/zeros_1/Const©
rienwrhgrh/zeros_1Fill"rienwrhgrh/zeros_1/packed:output:0!rienwrhgrh/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/zeros_1
rienwrhgrh/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rienwrhgrh/transpose/perm¯
rienwrhgrh/transpose	Transposelctpanrywj/transpose_1:y:0"rienwrhgrh/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/transposep
rienwrhgrh/Shape_1Shaperienwrhgrh/transpose:y:0*
T0*
_output_shapes
:2
rienwrhgrh/Shape_1
 rienwrhgrh/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rienwrhgrh/strided_slice_1/stack
"rienwrhgrh/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_1/stack_1
"rienwrhgrh/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_1/stack_2°
rienwrhgrh/strided_slice_1StridedSlicerienwrhgrh/Shape_1:output:0)rienwrhgrh/strided_slice_1/stack:output:0+rienwrhgrh/strided_slice_1/stack_1:output:0+rienwrhgrh/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rienwrhgrh/strided_slice_1
&rienwrhgrh/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&rienwrhgrh/TensorArrayV2/element_shapeÞ
rienwrhgrh/TensorArrayV2TensorListReserve/rienwrhgrh/TensorArrayV2/element_shape:output:0#rienwrhgrh/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rienwrhgrh/TensorArrayV2Õ
@rienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@rienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2rienwrhgrh/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrienwrhgrh/transpose:y:0Irienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2rienwrhgrh/TensorArrayUnstack/TensorListFromTensor
 rienwrhgrh/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rienwrhgrh/strided_slice_2/stack
"rienwrhgrh/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_2/stack_1
"rienwrhgrh/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_2/stack_2¾
rienwrhgrh/strided_slice_2StridedSlicerienwrhgrh/transpose:y:0)rienwrhgrh/strided_slice_2/stack:output:0+rienwrhgrh/strided_slice_2/stack_1:output:0+rienwrhgrh/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rienwrhgrh/strided_slice_2Ð
+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp4rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOpÓ
rienwrhgrh/kngiiuzftt/MatMulMatMul#rienwrhgrh/strided_slice_2:output:03rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rienwrhgrh/kngiiuzftt/MatMulÖ
-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp6rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOpÏ
rienwrhgrh/kngiiuzftt/MatMul_1MatMulrienwrhgrh/zeros:output:05rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rienwrhgrh/kngiiuzftt/MatMul_1Ä
rienwrhgrh/kngiiuzftt/addAddV2&rienwrhgrh/kngiiuzftt/MatMul:product:0(rienwrhgrh/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rienwrhgrh/kngiiuzftt/addÏ
,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp5rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOpÑ
rienwrhgrh/kngiiuzftt/BiasAddBiasAddrienwrhgrh/kngiiuzftt/add:z:04rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rienwrhgrh/kngiiuzftt/BiasAdd
%rienwrhgrh/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%rienwrhgrh/kngiiuzftt/split/split_dim
rienwrhgrh/kngiiuzftt/splitSplit.rienwrhgrh/kngiiuzftt/split/split_dim:output:0&rienwrhgrh/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rienwrhgrh/kngiiuzftt/split¶
$rienwrhgrh/kngiiuzftt/ReadVariableOpReadVariableOp-rienwrhgrh_kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02&
$rienwrhgrh/kngiiuzftt/ReadVariableOpº
rienwrhgrh/kngiiuzftt/mulMul,rienwrhgrh/kngiiuzftt/ReadVariableOp:value:0rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mulº
rienwrhgrh/kngiiuzftt/add_1AddV2$rienwrhgrh/kngiiuzftt/split:output:0rienwrhgrh/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_1
rienwrhgrh/kngiiuzftt/SigmoidSigmoidrienwrhgrh/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/Sigmoid¼
&rienwrhgrh/kngiiuzftt/ReadVariableOp_1ReadVariableOp/rienwrhgrh_kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&rienwrhgrh/kngiiuzftt/ReadVariableOp_1À
rienwrhgrh/kngiiuzftt/mul_1Mul.rienwrhgrh/kngiiuzftt/ReadVariableOp_1:value:0rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_1¼
rienwrhgrh/kngiiuzftt/add_2AddV2$rienwrhgrh/kngiiuzftt/split:output:1rienwrhgrh/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_2 
rienwrhgrh/kngiiuzftt/Sigmoid_1Sigmoidrienwrhgrh/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rienwrhgrh/kngiiuzftt/Sigmoid_1µ
rienwrhgrh/kngiiuzftt/mul_2Mul#rienwrhgrh/kngiiuzftt/Sigmoid_1:y:0rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_2
rienwrhgrh/kngiiuzftt/TanhTanh$rienwrhgrh/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/Tanh¶
rienwrhgrh/kngiiuzftt/mul_3Mul!rienwrhgrh/kngiiuzftt/Sigmoid:y:0rienwrhgrh/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_3·
rienwrhgrh/kngiiuzftt/add_3AddV2rienwrhgrh/kngiiuzftt/mul_2:z:0rienwrhgrh/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_3¼
&rienwrhgrh/kngiiuzftt/ReadVariableOp_2ReadVariableOp/rienwrhgrh_kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&rienwrhgrh/kngiiuzftt/ReadVariableOp_2Ä
rienwrhgrh/kngiiuzftt/mul_4Mul.rienwrhgrh/kngiiuzftt/ReadVariableOp_2:value:0rienwrhgrh/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_4¼
rienwrhgrh/kngiiuzftt/add_4AddV2$rienwrhgrh/kngiiuzftt/split:output:3rienwrhgrh/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_4 
rienwrhgrh/kngiiuzftt/Sigmoid_2Sigmoidrienwrhgrh/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rienwrhgrh/kngiiuzftt/Sigmoid_2
rienwrhgrh/kngiiuzftt/Tanh_1Tanhrienwrhgrh/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/Tanh_1º
rienwrhgrh/kngiiuzftt/mul_5Mul#rienwrhgrh/kngiiuzftt/Sigmoid_2:y:0 rienwrhgrh/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_5¥
(rienwrhgrh/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(rienwrhgrh/TensorArrayV2_1/element_shapeä
rienwrhgrh/TensorArrayV2_1TensorListReserve1rienwrhgrh/TensorArrayV2_1/element_shape:output:0#rienwrhgrh/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rienwrhgrh/TensorArrayV2_1d
rienwrhgrh/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/time
#rienwrhgrh/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#rienwrhgrh/while/maximum_iterations
rienwrhgrh/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/while/loop_counter²
rienwrhgrh/whileWhile&rienwrhgrh/while/loop_counter:output:0,rienwrhgrh/while/maximum_iterations:output:0rienwrhgrh/time:output:0#rienwrhgrh/TensorArrayV2_1:handle:0rienwrhgrh/zeros:output:0rienwrhgrh/zeros_1:output:0#rienwrhgrh/strided_slice_1:output:0Brienwrhgrh/TensorArrayUnstack/TensorListFromTensor:output_handle:04rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource6rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource5rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource-rienwrhgrh_kngiiuzftt_readvariableop_resource/rienwrhgrh_kngiiuzftt_readvariableop_1_resource/rienwrhgrh_kngiiuzftt_readvariableop_2_resource*
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
rienwrhgrh_while_body_1104029*)
cond!R
rienwrhgrh_while_cond_1104028*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
rienwrhgrh/whileË
;rienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;rienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shape
-rienwrhgrh/TensorArrayV2Stack/TensorListStackTensorListStackrienwrhgrh/while:output:3Drienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-rienwrhgrh/TensorArrayV2Stack/TensorListStack
 rienwrhgrh/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 rienwrhgrh/strided_slice_3/stack
"rienwrhgrh/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"rienwrhgrh/strided_slice_3/stack_1
"rienwrhgrh/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_3/stack_2Ü
rienwrhgrh/strided_slice_3StridedSlice6rienwrhgrh/TensorArrayV2Stack/TensorListStack:tensor:0)rienwrhgrh/strided_slice_3/stack:output:0+rienwrhgrh/strided_slice_3/stack_1:output:0+rienwrhgrh/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rienwrhgrh/strided_slice_3
rienwrhgrh/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rienwrhgrh/transpose_1/permÑ
rienwrhgrh/transpose_1	Transpose6rienwrhgrh/TensorArrayV2Stack/TensorListStack:tensor:0$rienwrhgrh/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/transpose_1®
 uilnjhxhrx/MatMul/ReadVariableOpReadVariableOp)uilnjhxhrx_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 uilnjhxhrx/MatMul/ReadVariableOp±
uilnjhxhrx/MatMulMatMul#rienwrhgrh/strided_slice_3:output:0(uilnjhxhrx/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uilnjhxhrx/MatMul­
!uilnjhxhrx/BiasAdd/ReadVariableOpReadVariableOp*uilnjhxhrx_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!uilnjhxhrx/BiasAdd/ReadVariableOp­
uilnjhxhrx/BiasAddBiasAdduilnjhxhrx/MatMul:product:0)uilnjhxhrx/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uilnjhxhrx/BiasAddÏ
IdentityIdentityuilnjhxhrx/BiasAdd:output:0.^kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp5^kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp-^lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp,^lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp.^lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp%^lctpanrywj/dhxpxqfhna/ReadVariableOp'^lctpanrywj/dhxpxqfhna/ReadVariableOp_1'^lctpanrywj/dhxpxqfhna/ReadVariableOp_2^lctpanrywj/while-^rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp,^rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp.^rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp%^rienwrhgrh/kngiiuzftt/ReadVariableOp'^rienwrhgrh/kngiiuzftt/ReadVariableOp_1'^rienwrhgrh/kngiiuzftt/ReadVariableOp_2^rienwrhgrh/while"^uilnjhxhrx/BiasAdd/ReadVariableOp!^uilnjhxhrx/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp2l
4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp2Z
+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp2^
-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp2L
$lctpanrywj/dhxpxqfhna/ReadVariableOp$lctpanrywj/dhxpxqfhna/ReadVariableOp2P
&lctpanrywj/dhxpxqfhna/ReadVariableOp_1&lctpanrywj/dhxpxqfhna/ReadVariableOp_12P
&lctpanrywj/dhxpxqfhna/ReadVariableOp_2&lctpanrywj/dhxpxqfhna/ReadVariableOp_22$
lctpanrywj/whilelctpanrywj/while2\
,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp2Z
+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp2^
-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp2L
$rienwrhgrh/kngiiuzftt/ReadVariableOp$rienwrhgrh/kngiiuzftt/ReadVariableOp2P
&rienwrhgrh/kngiiuzftt/ReadVariableOp_1&rienwrhgrh/kngiiuzftt/ReadVariableOp_12P
&rienwrhgrh/kngiiuzftt/ReadVariableOp_2&rienwrhgrh/kngiiuzftt/ReadVariableOp_22$
rienwrhgrh/whilerienwrhgrh/while2F
!uilnjhxhrx/BiasAdd/ReadVariableOp!uilnjhxhrx/BiasAdd/ReadVariableOp2D
 uilnjhxhrx/MatMul/ReadVariableOp uilnjhxhrx/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±'
³
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1100663

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
³F
ê
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1101026

inputs%
dhxpxqfhna_1100927:	%
dhxpxqfhna_1100929:	 !
dhxpxqfhna_1100931:	 
dhxpxqfhna_1100933:  
dhxpxqfhna_1100935:  
dhxpxqfhna_1100937: 
identity¢"dhxpxqfhna/StatefulPartitionedCall¢whileD
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
"dhxpxqfhna/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0dhxpxqfhna_1100927dhxpxqfhna_1100929dhxpxqfhna_1100931dhxpxqfhna_1100933dhxpxqfhna_1100935dhxpxqfhna_1100937*
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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_11008502$
"dhxpxqfhna/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0dhxpxqfhna_1100927dhxpxqfhna_1100929dhxpxqfhna_1100931dhxpxqfhna_1100933dhxpxqfhna_1100935dhxpxqfhna_1100937*
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
while_body_1100946*
condR
while_cond_1100945*Q
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
IdentityIdentitytranspose_1:y:0#^dhxpxqfhna/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"dhxpxqfhna/StatefulPartitionedCall"dhxpxqfhna/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
,__inference_dhxpxqfhna_layer_call_fn_1106003

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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_11008502
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


í
while_cond_1105140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1105140___redundant_placeholder05
1while_while_cond_1105140___redundant_placeholder15
1while_while_cond_1105140___redundant_placeholder25
1while_while_cond_1105140___redundant_placeholder35
1while_while_cond_1105140___redundant_placeholder45
1while_while_cond_1105140___redundant_placeholder55
1while_while_cond_1105140___redundant_placeholder6
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
p
Ê
rienwrhgrh_while_body_11036252
.rienwrhgrh_while_rienwrhgrh_while_loop_counter8
4rienwrhgrh_while_rienwrhgrh_while_maximum_iterations 
rienwrhgrh_while_placeholder"
rienwrhgrh_while_placeholder_1"
rienwrhgrh_while_placeholder_2"
rienwrhgrh_while_placeholder_31
-rienwrhgrh_while_rienwrhgrh_strided_slice_1_0m
irienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0O
<rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0:	 Q
>rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 L
=rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0:	C
5rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0: E
7rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0: E
7rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0: 
rienwrhgrh_while_identity
rienwrhgrh_while_identity_1
rienwrhgrh_while_identity_2
rienwrhgrh_while_identity_3
rienwrhgrh_while_identity_4
rienwrhgrh_while_identity_5/
+rienwrhgrh_while_rienwrhgrh_strided_slice_1k
grienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensorM
:rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource:	 O
<rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource:	 J
;rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource:	A
3rienwrhgrh_while_kngiiuzftt_readvariableop_resource: C
5rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource: C
5rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource: ¢2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp¢1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp¢3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp¢*rienwrhgrh/while/kngiiuzftt/ReadVariableOp¢,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1¢,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2Ù
Brienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Brienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shape
4rienwrhgrh/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemirienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0rienwrhgrh_while_placeholderKrienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4rienwrhgrh/while/TensorArrayV2Read/TensorListGetItemä
1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp<rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOpý
"rienwrhgrh/while/kngiiuzftt/MatMulMatMul;rienwrhgrh/while/TensorArrayV2Read/TensorListGetItem:item:09rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"rienwrhgrh/while/kngiiuzftt/MatMulê
3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp>rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOpæ
$rienwrhgrh/while/kngiiuzftt/MatMul_1MatMulrienwrhgrh_while_placeholder_2;rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rienwrhgrh/while/kngiiuzftt/MatMul_1Ü
rienwrhgrh/while/kngiiuzftt/addAddV2,rienwrhgrh/while/kngiiuzftt/MatMul:product:0.rienwrhgrh/while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
rienwrhgrh/while/kngiiuzftt/addã
2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp=rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOpé
#rienwrhgrh/while/kngiiuzftt/BiasAddBiasAdd#rienwrhgrh/while/kngiiuzftt/add:z:0:rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rienwrhgrh/while/kngiiuzftt/BiasAdd
+rienwrhgrh/while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+rienwrhgrh/while/kngiiuzftt/split/split_dim¯
!rienwrhgrh/while/kngiiuzftt/splitSplit4rienwrhgrh/while/kngiiuzftt/split/split_dim:output:0,rienwrhgrh/while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!rienwrhgrh/while/kngiiuzftt/splitÊ
*rienwrhgrh/while/kngiiuzftt/ReadVariableOpReadVariableOp5rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*rienwrhgrh/while/kngiiuzftt/ReadVariableOpÏ
rienwrhgrh/while/kngiiuzftt/mulMul2rienwrhgrh/while/kngiiuzftt/ReadVariableOp:value:0rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rienwrhgrh/while/kngiiuzftt/mulÒ
!rienwrhgrh/while/kngiiuzftt/add_1AddV2*rienwrhgrh/while/kngiiuzftt/split:output:0#rienwrhgrh/while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_1®
#rienwrhgrh/while/kngiiuzftt/SigmoidSigmoid%rienwrhgrh/while/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#rienwrhgrh/while/kngiiuzftt/SigmoidÐ
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1ReadVariableOp7rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1Õ
!rienwrhgrh/while/kngiiuzftt/mul_1Mul4rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1:value:0rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_1Ô
!rienwrhgrh/while/kngiiuzftt/add_2AddV2*rienwrhgrh/while/kngiiuzftt/split:output:1%rienwrhgrh/while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_2²
%rienwrhgrh/while/kngiiuzftt/Sigmoid_1Sigmoid%rienwrhgrh/while/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%rienwrhgrh/while/kngiiuzftt/Sigmoid_1Ê
!rienwrhgrh/while/kngiiuzftt/mul_2Mul)rienwrhgrh/while/kngiiuzftt/Sigmoid_1:y:0rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_2ª
 rienwrhgrh/while/kngiiuzftt/TanhTanh*rienwrhgrh/while/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 rienwrhgrh/while/kngiiuzftt/TanhÎ
!rienwrhgrh/while/kngiiuzftt/mul_3Mul'rienwrhgrh/while/kngiiuzftt/Sigmoid:y:0$rienwrhgrh/while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_3Ï
!rienwrhgrh/while/kngiiuzftt/add_3AddV2%rienwrhgrh/while/kngiiuzftt/mul_2:z:0%rienwrhgrh/while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_3Ð
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2ReadVariableOp7rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2Ü
!rienwrhgrh/while/kngiiuzftt/mul_4Mul4rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2:value:0%rienwrhgrh/while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_4Ô
!rienwrhgrh/while/kngiiuzftt/add_4AddV2*rienwrhgrh/while/kngiiuzftt/split:output:3%rienwrhgrh/while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/add_4²
%rienwrhgrh/while/kngiiuzftt/Sigmoid_2Sigmoid%rienwrhgrh/while/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%rienwrhgrh/while/kngiiuzftt/Sigmoid_2©
"rienwrhgrh/while/kngiiuzftt/Tanh_1Tanh%rienwrhgrh/while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"rienwrhgrh/while/kngiiuzftt/Tanh_1Ò
!rienwrhgrh/while/kngiiuzftt/mul_5Mul)rienwrhgrh/while/kngiiuzftt/Sigmoid_2:y:0&rienwrhgrh/while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!rienwrhgrh/while/kngiiuzftt/mul_5
5rienwrhgrh/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrienwrhgrh_while_placeholder_1rienwrhgrh_while_placeholder%rienwrhgrh/while/kngiiuzftt/mul_5:z:0*
_output_shapes
: *
element_dtype027
5rienwrhgrh/while/TensorArrayV2Write/TensorListSetItemr
rienwrhgrh/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rienwrhgrh/while/add/y
rienwrhgrh/while/addAddV2rienwrhgrh_while_placeholderrienwrhgrh/while/add/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/while/addv
rienwrhgrh/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rienwrhgrh/while/add_1/y­
rienwrhgrh/while/add_1AddV2.rienwrhgrh_while_rienwrhgrh_while_loop_counter!rienwrhgrh/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/while/add_1©
rienwrhgrh/while/IdentityIdentityrienwrhgrh/while/add_1:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/IdentityÇ
rienwrhgrh/while/Identity_1Identity4rienwrhgrh_while_rienwrhgrh_while_maximum_iterations3^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/Identity_1«
rienwrhgrh/while/Identity_2Identityrienwrhgrh/while/add:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/Identity_2Ø
rienwrhgrh/while/Identity_3IdentityErienwrhgrh/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
rienwrhgrh/while/Identity_3É
rienwrhgrh/while/Identity_4Identity%rienwrhgrh/while/kngiiuzftt/mul_5:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/while/Identity_4É
rienwrhgrh/while/Identity_5Identity%rienwrhgrh/while/kngiiuzftt/add_3:z:03^rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2^rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp4^rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp+^rienwrhgrh/while/kngiiuzftt/ReadVariableOp-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1-^rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/while/Identity_5"?
rienwrhgrh_while_identity"rienwrhgrh/while/Identity:output:0"C
rienwrhgrh_while_identity_1$rienwrhgrh/while/Identity_1:output:0"C
rienwrhgrh_while_identity_2$rienwrhgrh/while/Identity_2:output:0"C
rienwrhgrh_while_identity_3$rienwrhgrh/while/Identity_3:output:0"C
rienwrhgrh_while_identity_4$rienwrhgrh/while/Identity_4:output:0"C
rienwrhgrh_while_identity_5$rienwrhgrh/while/Identity_5:output:0"|
;rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource=rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0"~
<rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource>rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0"z
:rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource<rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0"p
5rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource7rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0"p
5rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource7rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0"l
3rienwrhgrh_while_kngiiuzftt_readvariableop_resource5rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0"\
+rienwrhgrh_while_rienwrhgrh_strided_slice_1-rienwrhgrh_while_rienwrhgrh_strided_slice_1_0"Ô
grienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensoririenwrhgrh_while_tensorarrayv2read_tensorlistgetitem_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2f
1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp1rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp2j
3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp3rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp2X
*rienwrhgrh/while/kngiiuzftt/ReadVariableOp*rienwrhgrh/while/kngiiuzftt/ReadVariableOp2\
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_12\
,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2,rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2: 
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
G__inference_sequential_layer_call_and_return_conditional_losses_1102560

inputs(
kjxhlaztnm_1102137: 
kjxhlaztnm_1102139:%
lctpanrywj_1102337:	%
lctpanrywj_1102339:	 !
lctpanrywj_1102341:	 
lctpanrywj_1102343:  
lctpanrywj_1102345:  
lctpanrywj_1102347: %
rienwrhgrh_1102530:	 %
rienwrhgrh_1102532:	 !
rienwrhgrh_1102534:	 
rienwrhgrh_1102536:  
rienwrhgrh_1102538:  
rienwrhgrh_1102540: $
uilnjhxhrx_1102554:  
uilnjhxhrx_1102556:
identity¢"kjxhlaztnm/StatefulPartitionedCall¢"lctpanrywj/StatefulPartitionedCall¢"rienwrhgrh/StatefulPartitionedCall¢"uilnjhxhrx/StatefulPartitionedCall¬
"kjxhlaztnm/StatefulPartitionedCallStatefulPartitionedCallinputskjxhlaztnm_1102137kjxhlaztnm_1102139*
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
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_11021362$
"kjxhlaztnm/StatefulPartitionedCall
tzzrzfazij/PartitionedCallPartitionedCall+kjxhlaztnm/StatefulPartitionedCall:output:0*
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
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_11021552
tzzrzfazij/PartitionedCall
"lctpanrywj/StatefulPartitionedCallStatefulPartitionedCall#tzzrzfazij/PartitionedCall:output:0lctpanrywj_1102337lctpanrywj_1102339lctpanrywj_1102341lctpanrywj_1102343lctpanrywj_1102345lctpanrywj_1102347*
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11023362$
"lctpanrywj/StatefulPartitionedCall¡
"rienwrhgrh/StatefulPartitionedCallStatefulPartitionedCall+lctpanrywj/StatefulPartitionedCall:output:0rienwrhgrh_1102530rienwrhgrh_1102532rienwrhgrh_1102534rienwrhgrh_1102536rienwrhgrh_1102538rienwrhgrh_1102540*
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11025292$
"rienwrhgrh/StatefulPartitionedCallÉ
"uilnjhxhrx/StatefulPartitionedCallStatefulPartitionedCall+rienwrhgrh/StatefulPartitionedCall:output:0uilnjhxhrx_1102554uilnjhxhrx_1102556*
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
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_11025532$
"uilnjhxhrx/StatefulPartitionedCall
IdentityIdentity+uilnjhxhrx/StatefulPartitionedCall:output:0#^kjxhlaztnm/StatefulPartitionedCall#^lctpanrywj/StatefulPartitionedCall#^rienwrhgrh/StatefulPartitionedCall#^uilnjhxhrx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kjxhlaztnm/StatefulPartitionedCall"kjxhlaztnm/StatefulPartitionedCall2H
"lctpanrywj/StatefulPartitionedCall"lctpanrywj/StatefulPartitionedCall2H
"rienwrhgrh/StatefulPartitionedCall"rienwrhgrh/StatefulPartitionedCall2H
"uilnjhxhrx/StatefulPartitionedCall"uilnjhxhrx/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1101703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1101703___redundant_placeholder05
1while_while_cond_1101703___redundant_placeholder15
1while_while_cond_1101703___redundant_placeholder25
1while_while_cond_1101703___redundant_placeholder35
1while_while_cond_1101703___redundant_placeholder45
1while_while_cond_1101703___redundant_placeholder55
1while_while_cond_1101703___redundant_placeholder6
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
p
Ê
lctpanrywj_while_body_11038532
.lctpanrywj_while_lctpanrywj_while_loop_counter8
4lctpanrywj_while_lctpanrywj_while_maximum_iterations 
lctpanrywj_while_placeholder"
lctpanrywj_while_placeholder_1"
lctpanrywj_while_placeholder_2"
lctpanrywj_while_placeholder_31
-lctpanrywj_while_lctpanrywj_strided_slice_1_0m
ilctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0O
<lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0:	Q
>lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 L
=lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0:	C
5lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0: E
7lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0: E
7lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0: 
lctpanrywj_while_identity
lctpanrywj_while_identity_1
lctpanrywj_while_identity_2
lctpanrywj_while_identity_3
lctpanrywj_while_identity_4
lctpanrywj_while_identity_5/
+lctpanrywj_while_lctpanrywj_strided_slice_1k
glctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensorM
:lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource:	O
<lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource:	 J
;lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource:	A
3lctpanrywj_while_dhxpxqfhna_readvariableop_resource: C
5lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource: C
5lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource: ¢2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp¢1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp¢3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp¢*lctpanrywj/while/dhxpxqfhna/ReadVariableOp¢,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1¢,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2Ù
Blctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Blctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shape
4lctpanrywj/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemilctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0lctpanrywj_while_placeholderKlctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4lctpanrywj/while/TensorArrayV2Read/TensorListGetItemä
1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp<lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOpý
"lctpanrywj/while/dhxpxqfhna/MatMulMatMul;lctpanrywj/while/TensorArrayV2Read/TensorListGetItem:item:09lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lctpanrywj/while/dhxpxqfhna/MatMulê
3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp>lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOpæ
$lctpanrywj/while/dhxpxqfhna/MatMul_1MatMullctpanrywj_while_placeholder_2;lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lctpanrywj/while/dhxpxqfhna/MatMul_1Ü
lctpanrywj/while/dhxpxqfhna/addAddV2,lctpanrywj/while/dhxpxqfhna/MatMul:product:0.lctpanrywj/while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lctpanrywj/while/dhxpxqfhna/addã
2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp=lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOpé
#lctpanrywj/while/dhxpxqfhna/BiasAddBiasAdd#lctpanrywj/while/dhxpxqfhna/add:z:0:lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lctpanrywj/while/dhxpxqfhna/BiasAdd
+lctpanrywj/while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+lctpanrywj/while/dhxpxqfhna/split/split_dim¯
!lctpanrywj/while/dhxpxqfhna/splitSplit4lctpanrywj/while/dhxpxqfhna/split/split_dim:output:0,lctpanrywj/while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!lctpanrywj/while/dhxpxqfhna/splitÊ
*lctpanrywj/while/dhxpxqfhna/ReadVariableOpReadVariableOp5lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*lctpanrywj/while/dhxpxqfhna/ReadVariableOpÏ
lctpanrywj/while/dhxpxqfhna/mulMul2lctpanrywj/while/dhxpxqfhna/ReadVariableOp:value:0lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lctpanrywj/while/dhxpxqfhna/mulÒ
!lctpanrywj/while/dhxpxqfhna/add_1AddV2*lctpanrywj/while/dhxpxqfhna/split:output:0#lctpanrywj/while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_1®
#lctpanrywj/while/dhxpxqfhna/SigmoidSigmoid%lctpanrywj/while/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#lctpanrywj/while/dhxpxqfhna/SigmoidÐ
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp7lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1Õ
!lctpanrywj/while/dhxpxqfhna/mul_1Mul4lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1:value:0lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_1Ô
!lctpanrywj/while/dhxpxqfhna/add_2AddV2*lctpanrywj/while/dhxpxqfhna/split:output:1%lctpanrywj/while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_2²
%lctpanrywj/while/dhxpxqfhna/Sigmoid_1Sigmoid%lctpanrywj/while/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%lctpanrywj/while/dhxpxqfhna/Sigmoid_1Ê
!lctpanrywj/while/dhxpxqfhna/mul_2Mul)lctpanrywj/while/dhxpxqfhna/Sigmoid_1:y:0lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_2ª
 lctpanrywj/while/dhxpxqfhna/TanhTanh*lctpanrywj/while/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 lctpanrywj/while/dhxpxqfhna/TanhÎ
!lctpanrywj/while/dhxpxqfhna/mul_3Mul'lctpanrywj/while/dhxpxqfhna/Sigmoid:y:0$lctpanrywj/while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_3Ï
!lctpanrywj/while/dhxpxqfhna/add_3AddV2%lctpanrywj/while/dhxpxqfhna/mul_2:z:0%lctpanrywj/while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_3Ð
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp7lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2Ü
!lctpanrywj/while/dhxpxqfhna/mul_4Mul4lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2:value:0%lctpanrywj/while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_4Ô
!lctpanrywj/while/dhxpxqfhna/add_4AddV2*lctpanrywj/while/dhxpxqfhna/split:output:3%lctpanrywj/while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/add_4²
%lctpanrywj/while/dhxpxqfhna/Sigmoid_2Sigmoid%lctpanrywj/while/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%lctpanrywj/while/dhxpxqfhna/Sigmoid_2©
"lctpanrywj/while/dhxpxqfhna/Tanh_1Tanh%lctpanrywj/while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"lctpanrywj/while/dhxpxqfhna/Tanh_1Ò
!lctpanrywj/while/dhxpxqfhna/mul_5Mul)lctpanrywj/while/dhxpxqfhna/Sigmoid_2:y:0&lctpanrywj/while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lctpanrywj/while/dhxpxqfhna/mul_5
5lctpanrywj/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlctpanrywj_while_placeholder_1lctpanrywj_while_placeholder%lctpanrywj/while/dhxpxqfhna/mul_5:z:0*
_output_shapes
: *
element_dtype027
5lctpanrywj/while/TensorArrayV2Write/TensorListSetItemr
lctpanrywj/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lctpanrywj/while/add/y
lctpanrywj/while/addAddV2lctpanrywj_while_placeholderlctpanrywj/while/add/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/while/addv
lctpanrywj/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lctpanrywj/while/add_1/y­
lctpanrywj/while/add_1AddV2.lctpanrywj_while_lctpanrywj_while_loop_counter!lctpanrywj/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/while/add_1©
lctpanrywj/while/IdentityIdentitylctpanrywj/while/add_1:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/IdentityÇ
lctpanrywj/while/Identity_1Identity4lctpanrywj_while_lctpanrywj_while_maximum_iterations3^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/Identity_1«
lctpanrywj/while/Identity_2Identitylctpanrywj/while/add:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/Identity_2Ø
lctpanrywj/while/Identity_3IdentityElctpanrywj/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
lctpanrywj/while/Identity_3É
lctpanrywj/while/Identity_4Identity%lctpanrywj/while/dhxpxqfhna/mul_5:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/while/Identity_4É
lctpanrywj/while/Identity_5Identity%lctpanrywj/while/dhxpxqfhna/add_3:z:03^lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2^lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp4^lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp+^lctpanrywj/while/dhxpxqfhna/ReadVariableOp-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1-^lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/while/Identity_5"|
;lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource=lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0"~
<lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource>lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0"z
:lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource<lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0"p
5lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource7lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0"p
5lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource7lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0"l
3lctpanrywj_while_dhxpxqfhna_readvariableop_resource5lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0"?
lctpanrywj_while_identity"lctpanrywj/while/Identity:output:0"C
lctpanrywj_while_identity_1$lctpanrywj/while/Identity_1:output:0"C
lctpanrywj_while_identity_2$lctpanrywj/while/Identity_2:output:0"C
lctpanrywj_while_identity_3$lctpanrywj/while/Identity_3:output:0"C
lctpanrywj_while_identity_4$lctpanrywj/while/Identity_4:output:0"C
lctpanrywj_while_identity_5$lctpanrywj/while/Identity_5:output:0"\
+lctpanrywj_while_lctpanrywj_strided_slice_1-lctpanrywj_while_lctpanrywj_strided_slice_1_0"Ô
glctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensorilctpanrywj_while_tensorarrayv2read_tensorlistgetitem_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2f
1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp1lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp2j
3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp3lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp2X
*lctpanrywj/while/dhxpxqfhna/ReadVariableOp*lctpanrywj/while/dhxpxqfhna/ReadVariableOp2\
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_12\
,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2,lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2: 
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
¤

,__inference_uilnjhxhrx_layer_call_fn_1105869

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
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_11025532
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
³F
ê
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1100763

inputs%
dhxpxqfhna_1100664:	%
dhxpxqfhna_1100666:	 !
dhxpxqfhna_1100668:	 
dhxpxqfhna_1100670:  
dhxpxqfhna_1100672:  
dhxpxqfhna_1100674: 
identity¢"dhxpxqfhna/StatefulPartitionedCall¢whileD
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
"dhxpxqfhna/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0dhxpxqfhna_1100664dhxpxqfhna_1100666dhxpxqfhna_1100668dhxpxqfhna_1100670dhxpxqfhna_1100672dhxpxqfhna_1100674*
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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_11006632$
"dhxpxqfhna/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0dhxpxqfhna_1100664dhxpxqfhna_1100666dhxpxqfhna_1100668dhxpxqfhna_1100670dhxpxqfhna_1100672dhxpxqfhna_1100674*
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
while_body_1100683*
condR
while_cond_1100682*Q
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
IdentityIdentitytranspose_1:y:0#^dhxpxqfhna/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"dhxpxqfhna/StatefulPartitionedCall"dhxpxqfhna/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
×
"__inference__wrapped_model_1100576

dtmnvweekcW
Asequential_kjxhlaztnm_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_kjxhlaztnm_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource:	T
Asequential_lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource:	 O
@sequential_lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource:	F
8sequential_lctpanrywj_dhxpxqfhna_readvariableop_resource: H
:sequential_lctpanrywj_dhxpxqfhna_readvariableop_1_resource: H
:sequential_lctpanrywj_dhxpxqfhna_readvariableop_2_resource: R
?sequential_rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource:	 T
Asequential_rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource:	 O
@sequential_rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource:	F
8sequential_rienwrhgrh_kngiiuzftt_readvariableop_resource: H
:sequential_rienwrhgrh_kngiiuzftt_readvariableop_1_resource: H
:sequential_rienwrhgrh_kngiiuzftt_readvariableop_2_resource: F
4sequential_uilnjhxhrx_matmul_readvariableop_resource: C
5sequential_uilnjhxhrx_biasadd_readvariableop_resource:
identity¢8sequential/kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7sequential/lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp¢6sequential/lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp¢8sequential/lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp¢/sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp¢1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_1¢1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_2¢sequential/lctpanrywj/while¢7sequential/rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp¢6sequential/rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp¢8sequential/rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp¢/sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp¢1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_1¢1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_2¢sequential/rienwrhgrh/while¢,sequential/uilnjhxhrx/BiasAdd/ReadVariableOp¢+sequential/uilnjhxhrx/MatMul/ReadVariableOp¥
+sequential/kjxhlaztnm/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/kjxhlaztnm/conv1d/ExpandDims/dimà
'sequential/kjxhlaztnm/conv1d/ExpandDims
ExpandDims
dtmnvweekc4sequential/kjxhlaztnm/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/kjxhlaztnm/conv1d/ExpandDimsú
8sequential/kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_kjxhlaztnm_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/kjxhlaztnm/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/kjxhlaztnm/conv1d/ExpandDims_1/dim
)sequential/kjxhlaztnm/conv1d/ExpandDims_1
ExpandDims@sequential/kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/kjxhlaztnm/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/kjxhlaztnm/conv1d/ExpandDims_1¨
"sequential/kjxhlaztnm/conv1d/ShapeShape0sequential/kjxhlaztnm/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/kjxhlaztnm/conv1d/Shape®
0sequential/kjxhlaztnm/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/kjxhlaztnm/conv1d/strided_slice/stack»
2sequential/kjxhlaztnm/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/kjxhlaztnm/conv1d/strided_slice/stack_1²
2sequential/kjxhlaztnm/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/kjxhlaztnm/conv1d/strided_slice/stack_2
*sequential/kjxhlaztnm/conv1d/strided_sliceStridedSlice+sequential/kjxhlaztnm/conv1d/Shape:output:09sequential/kjxhlaztnm/conv1d/strided_slice/stack:output:0;sequential/kjxhlaztnm/conv1d/strided_slice/stack_1:output:0;sequential/kjxhlaztnm/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/kjxhlaztnm/conv1d/strided_slice±
*sequential/kjxhlaztnm/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/kjxhlaztnm/conv1d/Reshape/shapeø
$sequential/kjxhlaztnm/conv1d/ReshapeReshape0sequential/kjxhlaztnm/conv1d/ExpandDims:output:03sequential/kjxhlaztnm/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/kjxhlaztnm/conv1d/Reshape
#sequential/kjxhlaztnm/conv1d/Conv2DConv2D-sequential/kjxhlaztnm/conv1d/Reshape:output:02sequential/kjxhlaztnm/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/kjxhlaztnm/conv1d/Conv2D±
,sequential/kjxhlaztnm/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/kjxhlaztnm/conv1d/concat/values_1
(sequential/kjxhlaztnm/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/kjxhlaztnm/conv1d/concat/axis£
#sequential/kjxhlaztnm/conv1d/concatConcatV23sequential/kjxhlaztnm/conv1d/strided_slice:output:05sequential/kjxhlaztnm/conv1d/concat/values_1:output:01sequential/kjxhlaztnm/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/kjxhlaztnm/conv1d/concatõ
&sequential/kjxhlaztnm/conv1d/Reshape_1Reshape,sequential/kjxhlaztnm/conv1d/Conv2D:output:0,sequential/kjxhlaztnm/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/kjxhlaztnm/conv1d/Reshape_1â
$sequential/kjxhlaztnm/conv1d/SqueezeSqueeze/sequential/kjxhlaztnm/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/kjxhlaztnm/conv1d/Squeeze½
.sequential/kjxhlaztnm/squeeze_batch_dims/ShapeShape-sequential/kjxhlaztnm/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/kjxhlaztnm/squeeze_batch_dims/ShapeÆ
<sequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stackÓ
>sequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/kjxhlaztnm/squeeze_batch_dims/strided_sliceStridedSlice7sequential/kjxhlaztnm/squeeze_batch_dims/Shape:output:0Esequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/kjxhlaztnm/squeeze_batch_dims/strided_sliceÅ
6sequential/kjxhlaztnm/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/kjxhlaztnm/squeeze_batch_dims/Reshape/shape
0sequential/kjxhlaztnm/squeeze_batch_dims/ReshapeReshape-sequential/kjxhlaztnm/conv1d/Squeeze:output:0?sequential/kjxhlaztnm/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/kjxhlaztnm/squeeze_batch_dims/Reshape
?sequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_kjxhlaztnm_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/kjxhlaztnm/squeeze_batch_dims/BiasAddBiasAdd9sequential/kjxhlaztnm/squeeze_batch_dims/Reshape:output:0Gsequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/kjxhlaztnm/squeeze_batch_dims/BiasAddÅ
8sequential/kjxhlaztnm/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/kjxhlaztnm/squeeze_batch_dims/concat/values_1·
4sequential/kjxhlaztnm/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/kjxhlaztnm/squeeze_batch_dims/concat/axisß
/sequential/kjxhlaztnm/squeeze_batch_dims/concatConcatV2?sequential/kjxhlaztnm/squeeze_batch_dims/strided_slice:output:0Asequential/kjxhlaztnm/squeeze_batch_dims/concat/values_1:output:0=sequential/kjxhlaztnm/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/kjxhlaztnm/squeeze_batch_dims/concat¢
2sequential/kjxhlaztnm/squeeze_batch_dims/Reshape_1Reshape9sequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd:output:08sequential/kjxhlaztnm/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/kjxhlaztnm/squeeze_batch_dims/Reshape_1¥
sequential/tzzrzfazij/ShapeShape;sequential/kjxhlaztnm/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/tzzrzfazij/Shape 
)sequential/tzzrzfazij/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/tzzrzfazij/strided_slice/stack¤
+sequential/tzzrzfazij/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/tzzrzfazij/strided_slice/stack_1¤
+sequential/tzzrzfazij/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/tzzrzfazij/strided_slice/stack_2æ
#sequential/tzzrzfazij/strided_sliceStridedSlice$sequential/tzzrzfazij/Shape:output:02sequential/tzzrzfazij/strided_slice/stack:output:04sequential/tzzrzfazij/strided_slice/stack_1:output:04sequential/tzzrzfazij/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/tzzrzfazij/strided_slice
%sequential/tzzrzfazij/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/tzzrzfazij/Reshape/shape/1
%sequential/tzzrzfazij/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/tzzrzfazij/Reshape/shape/2
#sequential/tzzrzfazij/Reshape/shapePack,sequential/tzzrzfazij/strided_slice:output:0.sequential/tzzrzfazij/Reshape/shape/1:output:0.sequential/tzzrzfazij/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/tzzrzfazij/Reshape/shapeê
sequential/tzzrzfazij/ReshapeReshape;sequential/kjxhlaztnm/squeeze_batch_dims/Reshape_1:output:0,sequential/tzzrzfazij/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/tzzrzfazij/Reshape
sequential/lctpanrywj/ShapeShape&sequential/tzzrzfazij/Reshape:output:0*
T0*
_output_shapes
:2
sequential/lctpanrywj/Shape 
)sequential/lctpanrywj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/lctpanrywj/strided_slice/stack¤
+sequential/lctpanrywj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/lctpanrywj/strided_slice/stack_1¤
+sequential/lctpanrywj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/lctpanrywj/strided_slice/stack_2æ
#sequential/lctpanrywj/strided_sliceStridedSlice$sequential/lctpanrywj/Shape:output:02sequential/lctpanrywj/strided_slice/stack:output:04sequential/lctpanrywj/strided_slice/stack_1:output:04sequential/lctpanrywj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/lctpanrywj/strided_slice
!sequential/lctpanrywj/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/lctpanrywj/zeros/mul/yÄ
sequential/lctpanrywj/zeros/mulMul,sequential/lctpanrywj/strided_slice:output:0*sequential/lctpanrywj/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/lctpanrywj/zeros/mul
"sequential/lctpanrywj/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/lctpanrywj/zeros/Less/y¿
 sequential/lctpanrywj/zeros/LessLess#sequential/lctpanrywj/zeros/mul:z:0+sequential/lctpanrywj/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/lctpanrywj/zeros/Less
$sequential/lctpanrywj/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/lctpanrywj/zeros/packed/1Û
"sequential/lctpanrywj/zeros/packedPack,sequential/lctpanrywj/strided_slice:output:0-sequential/lctpanrywj/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/lctpanrywj/zeros/packed
!sequential/lctpanrywj/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/lctpanrywj/zeros/ConstÍ
sequential/lctpanrywj/zerosFill+sequential/lctpanrywj/zeros/packed:output:0*sequential/lctpanrywj/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/lctpanrywj/zeros
#sequential/lctpanrywj/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/lctpanrywj/zeros_1/mul/yÊ
!sequential/lctpanrywj/zeros_1/mulMul,sequential/lctpanrywj/strided_slice:output:0,sequential/lctpanrywj/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/lctpanrywj/zeros_1/mul
$sequential/lctpanrywj/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/lctpanrywj/zeros_1/Less/yÇ
"sequential/lctpanrywj/zeros_1/LessLess%sequential/lctpanrywj/zeros_1/mul:z:0-sequential/lctpanrywj/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/lctpanrywj/zeros_1/Less
&sequential/lctpanrywj/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/lctpanrywj/zeros_1/packed/1á
$sequential/lctpanrywj/zeros_1/packedPack,sequential/lctpanrywj/strided_slice:output:0/sequential/lctpanrywj/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/lctpanrywj/zeros_1/packed
#sequential/lctpanrywj/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/lctpanrywj/zeros_1/ConstÕ
sequential/lctpanrywj/zeros_1Fill-sequential/lctpanrywj/zeros_1/packed:output:0,sequential/lctpanrywj/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/lctpanrywj/zeros_1¡
$sequential/lctpanrywj/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/lctpanrywj/transpose/permÜ
sequential/lctpanrywj/transpose	Transpose&sequential/tzzrzfazij/Reshape:output:0-sequential/lctpanrywj/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/lctpanrywj/transpose
sequential/lctpanrywj/Shape_1Shape#sequential/lctpanrywj/transpose:y:0*
T0*
_output_shapes
:2
sequential/lctpanrywj/Shape_1¤
+sequential/lctpanrywj/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/lctpanrywj/strided_slice_1/stack¨
-sequential/lctpanrywj/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/lctpanrywj/strided_slice_1/stack_1¨
-sequential/lctpanrywj/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/lctpanrywj/strided_slice_1/stack_2ò
%sequential/lctpanrywj/strided_slice_1StridedSlice&sequential/lctpanrywj/Shape_1:output:04sequential/lctpanrywj/strided_slice_1/stack:output:06sequential/lctpanrywj/strided_slice_1/stack_1:output:06sequential/lctpanrywj/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/lctpanrywj/strided_slice_1±
1sequential/lctpanrywj/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/lctpanrywj/TensorArrayV2/element_shape
#sequential/lctpanrywj/TensorArrayV2TensorListReserve:sequential/lctpanrywj/TensorArrayV2/element_shape:output:0.sequential/lctpanrywj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/lctpanrywj/TensorArrayV2ë
Ksequential/lctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/lctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/lctpanrywj/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/lctpanrywj/transpose:y:0Tsequential/lctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/lctpanrywj/TensorArrayUnstack/TensorListFromTensor¤
+sequential/lctpanrywj/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/lctpanrywj/strided_slice_2/stack¨
-sequential/lctpanrywj/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/lctpanrywj/strided_slice_2/stack_1¨
-sequential/lctpanrywj/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/lctpanrywj/strided_slice_2/stack_2
%sequential/lctpanrywj/strided_slice_2StridedSlice#sequential/lctpanrywj/transpose:y:04sequential/lctpanrywj/strided_slice_2/stack:output:06sequential/lctpanrywj/strided_slice_2/stack_1:output:06sequential/lctpanrywj/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/lctpanrywj/strided_slice_2ñ
6sequential/lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp?sequential_lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOpÿ
'sequential/lctpanrywj/dhxpxqfhna/MatMulMatMul.sequential/lctpanrywj/strided_slice_2:output:0>sequential/lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/lctpanrywj/dhxpxqfhna/MatMul÷
8sequential/lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOpAsequential_lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOpû
)sequential/lctpanrywj/dhxpxqfhna/MatMul_1MatMul$sequential/lctpanrywj/zeros:output:0@sequential/lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/lctpanrywj/dhxpxqfhna/MatMul_1ð
$sequential/lctpanrywj/dhxpxqfhna/addAddV21sequential/lctpanrywj/dhxpxqfhna/MatMul:product:03sequential/lctpanrywj/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/lctpanrywj/dhxpxqfhna/addð
7sequential/lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp@sequential_lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOpý
(sequential/lctpanrywj/dhxpxqfhna/BiasAddBiasAdd(sequential/lctpanrywj/dhxpxqfhna/add:z:0?sequential/lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/lctpanrywj/dhxpxqfhna/BiasAdd¦
0sequential/lctpanrywj/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/lctpanrywj/dhxpxqfhna/split/split_dimÃ
&sequential/lctpanrywj/dhxpxqfhna/splitSplit9sequential/lctpanrywj/dhxpxqfhna/split/split_dim:output:01sequential/lctpanrywj/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/lctpanrywj/dhxpxqfhna/split×
/sequential/lctpanrywj/dhxpxqfhna/ReadVariableOpReadVariableOp8sequential_lctpanrywj_dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/lctpanrywj/dhxpxqfhna/ReadVariableOpæ
$sequential/lctpanrywj/dhxpxqfhna/mulMul7sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp:value:0&sequential/lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/lctpanrywj/dhxpxqfhna/mulæ
&sequential/lctpanrywj/dhxpxqfhna/add_1AddV2/sequential/lctpanrywj/dhxpxqfhna/split:output:0(sequential/lctpanrywj/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/add_1½
(sequential/lctpanrywj/dhxpxqfhna/SigmoidSigmoid*sequential/lctpanrywj/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/lctpanrywj/dhxpxqfhna/SigmoidÝ
1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_1ReadVariableOp:sequential_lctpanrywj_dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_1ì
&sequential/lctpanrywj/dhxpxqfhna/mul_1Mul9sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_1:value:0&sequential/lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/mul_1è
&sequential/lctpanrywj/dhxpxqfhna/add_2AddV2/sequential/lctpanrywj/dhxpxqfhna/split:output:1*sequential/lctpanrywj/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/add_2Á
*sequential/lctpanrywj/dhxpxqfhna/Sigmoid_1Sigmoid*sequential/lctpanrywj/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/lctpanrywj/dhxpxqfhna/Sigmoid_1á
&sequential/lctpanrywj/dhxpxqfhna/mul_2Mul.sequential/lctpanrywj/dhxpxqfhna/Sigmoid_1:y:0&sequential/lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/mul_2¹
%sequential/lctpanrywj/dhxpxqfhna/TanhTanh/sequential/lctpanrywj/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/lctpanrywj/dhxpxqfhna/Tanhâ
&sequential/lctpanrywj/dhxpxqfhna/mul_3Mul,sequential/lctpanrywj/dhxpxqfhna/Sigmoid:y:0)sequential/lctpanrywj/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/mul_3ã
&sequential/lctpanrywj/dhxpxqfhna/add_3AddV2*sequential/lctpanrywj/dhxpxqfhna/mul_2:z:0*sequential/lctpanrywj/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/add_3Ý
1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_2ReadVariableOp:sequential_lctpanrywj_dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_2ð
&sequential/lctpanrywj/dhxpxqfhna/mul_4Mul9sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_2:value:0*sequential/lctpanrywj/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/mul_4è
&sequential/lctpanrywj/dhxpxqfhna/add_4AddV2/sequential/lctpanrywj/dhxpxqfhna/split:output:3*sequential/lctpanrywj/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/add_4Á
*sequential/lctpanrywj/dhxpxqfhna/Sigmoid_2Sigmoid*sequential/lctpanrywj/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/lctpanrywj/dhxpxqfhna/Sigmoid_2¸
'sequential/lctpanrywj/dhxpxqfhna/Tanh_1Tanh*sequential/lctpanrywj/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/lctpanrywj/dhxpxqfhna/Tanh_1æ
&sequential/lctpanrywj/dhxpxqfhna/mul_5Mul.sequential/lctpanrywj/dhxpxqfhna/Sigmoid_2:y:0+sequential/lctpanrywj/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/dhxpxqfhna/mul_5»
3sequential/lctpanrywj/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/lctpanrywj/TensorArrayV2_1/element_shape
%sequential/lctpanrywj/TensorArrayV2_1TensorListReserve<sequential/lctpanrywj/TensorArrayV2_1/element_shape:output:0.sequential/lctpanrywj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/lctpanrywj/TensorArrayV2_1z
sequential/lctpanrywj/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lctpanrywj/time«
.sequential/lctpanrywj/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/lctpanrywj/while/maximum_iterations
(sequential/lctpanrywj/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/lctpanrywj/while/loop_counterø	
sequential/lctpanrywj/whileWhile1sequential/lctpanrywj/while/loop_counter:output:07sequential/lctpanrywj/while/maximum_iterations:output:0#sequential/lctpanrywj/time:output:0.sequential/lctpanrywj/TensorArrayV2_1:handle:0$sequential/lctpanrywj/zeros:output:0&sequential/lctpanrywj/zeros_1:output:0.sequential/lctpanrywj/strided_slice_1:output:0Msequential/lctpanrywj/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_lctpanrywj_dhxpxqfhna_matmul_readvariableop_resourceAsequential_lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource@sequential_lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource8sequential_lctpanrywj_dhxpxqfhna_readvariableop_resource:sequential_lctpanrywj_dhxpxqfhna_readvariableop_1_resource:sequential_lctpanrywj_dhxpxqfhna_readvariableop_2_resource*
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
(sequential_lctpanrywj_while_body_1100293*4
cond,R*
(sequential_lctpanrywj_while_cond_1100292*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/lctpanrywj/whileá
Fsequential/lctpanrywj/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/lctpanrywj/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/lctpanrywj/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/lctpanrywj/while:output:3Osequential/lctpanrywj/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/lctpanrywj/TensorArrayV2Stack/TensorListStack­
+sequential/lctpanrywj/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/lctpanrywj/strided_slice_3/stack¨
-sequential/lctpanrywj/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/lctpanrywj/strided_slice_3/stack_1¨
-sequential/lctpanrywj/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/lctpanrywj/strided_slice_3/stack_2
%sequential/lctpanrywj/strided_slice_3StridedSliceAsequential/lctpanrywj/TensorArrayV2Stack/TensorListStack:tensor:04sequential/lctpanrywj/strided_slice_3/stack:output:06sequential/lctpanrywj/strided_slice_3/stack_1:output:06sequential/lctpanrywj/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/lctpanrywj/strided_slice_3¥
&sequential/lctpanrywj/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/lctpanrywj/transpose_1/permý
!sequential/lctpanrywj/transpose_1	TransposeAsequential/lctpanrywj/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/lctpanrywj/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/lctpanrywj/transpose_1
sequential/rienwrhgrh/ShapeShape%sequential/lctpanrywj/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/rienwrhgrh/Shape 
)sequential/rienwrhgrh/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/rienwrhgrh/strided_slice/stack¤
+sequential/rienwrhgrh/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/rienwrhgrh/strided_slice/stack_1¤
+sequential/rienwrhgrh/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/rienwrhgrh/strided_slice/stack_2æ
#sequential/rienwrhgrh/strided_sliceStridedSlice$sequential/rienwrhgrh/Shape:output:02sequential/rienwrhgrh/strided_slice/stack:output:04sequential/rienwrhgrh/strided_slice/stack_1:output:04sequential/rienwrhgrh/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/rienwrhgrh/strided_slice
!sequential/rienwrhgrh/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/rienwrhgrh/zeros/mul/yÄ
sequential/rienwrhgrh/zeros/mulMul,sequential/rienwrhgrh/strided_slice:output:0*sequential/rienwrhgrh/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/rienwrhgrh/zeros/mul
"sequential/rienwrhgrh/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/rienwrhgrh/zeros/Less/y¿
 sequential/rienwrhgrh/zeros/LessLess#sequential/rienwrhgrh/zeros/mul:z:0+sequential/rienwrhgrh/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/rienwrhgrh/zeros/Less
$sequential/rienwrhgrh/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/rienwrhgrh/zeros/packed/1Û
"sequential/rienwrhgrh/zeros/packedPack,sequential/rienwrhgrh/strided_slice:output:0-sequential/rienwrhgrh/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/rienwrhgrh/zeros/packed
!sequential/rienwrhgrh/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/rienwrhgrh/zeros/ConstÍ
sequential/rienwrhgrh/zerosFill+sequential/rienwrhgrh/zeros/packed:output:0*sequential/rienwrhgrh/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rienwrhgrh/zeros
#sequential/rienwrhgrh/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/rienwrhgrh/zeros_1/mul/yÊ
!sequential/rienwrhgrh/zeros_1/mulMul,sequential/rienwrhgrh/strided_slice:output:0,sequential/rienwrhgrh/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/rienwrhgrh/zeros_1/mul
$sequential/rienwrhgrh/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/rienwrhgrh/zeros_1/Less/yÇ
"sequential/rienwrhgrh/zeros_1/LessLess%sequential/rienwrhgrh/zeros_1/mul:z:0-sequential/rienwrhgrh/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/rienwrhgrh/zeros_1/Less
&sequential/rienwrhgrh/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/rienwrhgrh/zeros_1/packed/1á
$sequential/rienwrhgrh/zeros_1/packedPack,sequential/rienwrhgrh/strided_slice:output:0/sequential/rienwrhgrh/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/rienwrhgrh/zeros_1/packed
#sequential/rienwrhgrh/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/rienwrhgrh/zeros_1/ConstÕ
sequential/rienwrhgrh/zeros_1Fill-sequential/rienwrhgrh/zeros_1/packed:output:0,sequential/rienwrhgrh/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/rienwrhgrh/zeros_1¡
$sequential/rienwrhgrh/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/rienwrhgrh/transpose/permÛ
sequential/rienwrhgrh/transpose	Transpose%sequential/lctpanrywj/transpose_1:y:0-sequential/rienwrhgrh/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/rienwrhgrh/transpose
sequential/rienwrhgrh/Shape_1Shape#sequential/rienwrhgrh/transpose:y:0*
T0*
_output_shapes
:2
sequential/rienwrhgrh/Shape_1¤
+sequential/rienwrhgrh/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/rienwrhgrh/strided_slice_1/stack¨
-sequential/rienwrhgrh/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/rienwrhgrh/strided_slice_1/stack_1¨
-sequential/rienwrhgrh/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/rienwrhgrh/strided_slice_1/stack_2ò
%sequential/rienwrhgrh/strided_slice_1StridedSlice&sequential/rienwrhgrh/Shape_1:output:04sequential/rienwrhgrh/strided_slice_1/stack:output:06sequential/rienwrhgrh/strided_slice_1/stack_1:output:06sequential/rienwrhgrh/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/rienwrhgrh/strided_slice_1±
1sequential/rienwrhgrh/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/rienwrhgrh/TensorArrayV2/element_shape
#sequential/rienwrhgrh/TensorArrayV2TensorListReserve:sequential/rienwrhgrh/TensorArrayV2/element_shape:output:0.sequential/rienwrhgrh/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/rienwrhgrh/TensorArrayV2ë
Ksequential/rienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/rienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/rienwrhgrh/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/rienwrhgrh/transpose:y:0Tsequential/rienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/rienwrhgrh/TensorArrayUnstack/TensorListFromTensor¤
+sequential/rienwrhgrh/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/rienwrhgrh/strided_slice_2/stack¨
-sequential/rienwrhgrh/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/rienwrhgrh/strided_slice_2/stack_1¨
-sequential/rienwrhgrh/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/rienwrhgrh/strided_slice_2/stack_2
%sequential/rienwrhgrh/strided_slice_2StridedSlice#sequential/rienwrhgrh/transpose:y:04sequential/rienwrhgrh/strided_slice_2/stack:output:06sequential/rienwrhgrh/strided_slice_2/stack_1:output:06sequential/rienwrhgrh/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/rienwrhgrh/strided_slice_2ñ
6sequential/rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp?sequential_rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOpÿ
'sequential/rienwrhgrh/kngiiuzftt/MatMulMatMul.sequential/rienwrhgrh/strided_slice_2:output:0>sequential/rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/rienwrhgrh/kngiiuzftt/MatMul÷
8sequential/rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOpAsequential_rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOpû
)sequential/rienwrhgrh/kngiiuzftt/MatMul_1MatMul$sequential/rienwrhgrh/zeros:output:0@sequential/rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/rienwrhgrh/kngiiuzftt/MatMul_1ð
$sequential/rienwrhgrh/kngiiuzftt/addAddV21sequential/rienwrhgrh/kngiiuzftt/MatMul:product:03sequential/rienwrhgrh/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/rienwrhgrh/kngiiuzftt/addð
7sequential/rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp@sequential_rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOpý
(sequential/rienwrhgrh/kngiiuzftt/BiasAddBiasAdd(sequential/rienwrhgrh/kngiiuzftt/add:z:0?sequential/rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/rienwrhgrh/kngiiuzftt/BiasAdd¦
0sequential/rienwrhgrh/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/rienwrhgrh/kngiiuzftt/split/split_dimÃ
&sequential/rienwrhgrh/kngiiuzftt/splitSplit9sequential/rienwrhgrh/kngiiuzftt/split/split_dim:output:01sequential/rienwrhgrh/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/rienwrhgrh/kngiiuzftt/split×
/sequential/rienwrhgrh/kngiiuzftt/ReadVariableOpReadVariableOp8sequential_rienwrhgrh_kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/rienwrhgrh/kngiiuzftt/ReadVariableOpæ
$sequential/rienwrhgrh/kngiiuzftt/mulMul7sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp:value:0&sequential/rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/rienwrhgrh/kngiiuzftt/mulæ
&sequential/rienwrhgrh/kngiiuzftt/add_1AddV2/sequential/rienwrhgrh/kngiiuzftt/split:output:0(sequential/rienwrhgrh/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/add_1½
(sequential/rienwrhgrh/kngiiuzftt/SigmoidSigmoid*sequential/rienwrhgrh/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/rienwrhgrh/kngiiuzftt/SigmoidÝ
1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_1ReadVariableOp:sequential_rienwrhgrh_kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_1ì
&sequential/rienwrhgrh/kngiiuzftt/mul_1Mul9sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_1:value:0&sequential/rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/mul_1è
&sequential/rienwrhgrh/kngiiuzftt/add_2AddV2/sequential/rienwrhgrh/kngiiuzftt/split:output:1*sequential/rienwrhgrh/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/add_2Á
*sequential/rienwrhgrh/kngiiuzftt/Sigmoid_1Sigmoid*sequential/rienwrhgrh/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/rienwrhgrh/kngiiuzftt/Sigmoid_1á
&sequential/rienwrhgrh/kngiiuzftt/mul_2Mul.sequential/rienwrhgrh/kngiiuzftt/Sigmoid_1:y:0&sequential/rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/mul_2¹
%sequential/rienwrhgrh/kngiiuzftt/TanhTanh/sequential/rienwrhgrh/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/rienwrhgrh/kngiiuzftt/Tanhâ
&sequential/rienwrhgrh/kngiiuzftt/mul_3Mul,sequential/rienwrhgrh/kngiiuzftt/Sigmoid:y:0)sequential/rienwrhgrh/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/mul_3ã
&sequential/rienwrhgrh/kngiiuzftt/add_3AddV2*sequential/rienwrhgrh/kngiiuzftt/mul_2:z:0*sequential/rienwrhgrh/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/add_3Ý
1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_2ReadVariableOp:sequential_rienwrhgrh_kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_2ð
&sequential/rienwrhgrh/kngiiuzftt/mul_4Mul9sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_2:value:0*sequential/rienwrhgrh/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/mul_4è
&sequential/rienwrhgrh/kngiiuzftt/add_4AddV2/sequential/rienwrhgrh/kngiiuzftt/split:output:3*sequential/rienwrhgrh/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/add_4Á
*sequential/rienwrhgrh/kngiiuzftt/Sigmoid_2Sigmoid*sequential/rienwrhgrh/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/rienwrhgrh/kngiiuzftt/Sigmoid_2¸
'sequential/rienwrhgrh/kngiiuzftt/Tanh_1Tanh*sequential/rienwrhgrh/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/rienwrhgrh/kngiiuzftt/Tanh_1æ
&sequential/rienwrhgrh/kngiiuzftt/mul_5Mul.sequential/rienwrhgrh/kngiiuzftt/Sigmoid_2:y:0+sequential/rienwrhgrh/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/kngiiuzftt/mul_5»
3sequential/rienwrhgrh/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/rienwrhgrh/TensorArrayV2_1/element_shape
%sequential/rienwrhgrh/TensorArrayV2_1TensorListReserve<sequential/rienwrhgrh/TensorArrayV2_1/element_shape:output:0.sequential/rienwrhgrh/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/rienwrhgrh/TensorArrayV2_1z
sequential/rienwrhgrh/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/rienwrhgrh/time«
.sequential/rienwrhgrh/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/rienwrhgrh/while/maximum_iterations
(sequential/rienwrhgrh/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/rienwrhgrh/while/loop_counterø	
sequential/rienwrhgrh/whileWhile1sequential/rienwrhgrh/while/loop_counter:output:07sequential/rienwrhgrh/while/maximum_iterations:output:0#sequential/rienwrhgrh/time:output:0.sequential/rienwrhgrh/TensorArrayV2_1:handle:0$sequential/rienwrhgrh/zeros:output:0&sequential/rienwrhgrh/zeros_1:output:0.sequential/rienwrhgrh/strided_slice_1:output:0Msequential/rienwrhgrh/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_rienwrhgrh_kngiiuzftt_matmul_readvariableop_resourceAsequential_rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource@sequential_rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource8sequential_rienwrhgrh_kngiiuzftt_readvariableop_resource:sequential_rienwrhgrh_kngiiuzftt_readvariableop_1_resource:sequential_rienwrhgrh_kngiiuzftt_readvariableop_2_resource*
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
(sequential_rienwrhgrh_while_body_1100469*4
cond,R*
(sequential_rienwrhgrh_while_cond_1100468*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/rienwrhgrh/whileá
Fsequential/rienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/rienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/rienwrhgrh/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/rienwrhgrh/while:output:3Osequential/rienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/rienwrhgrh/TensorArrayV2Stack/TensorListStack­
+sequential/rienwrhgrh/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/rienwrhgrh/strided_slice_3/stack¨
-sequential/rienwrhgrh/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/rienwrhgrh/strided_slice_3/stack_1¨
-sequential/rienwrhgrh/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/rienwrhgrh/strided_slice_3/stack_2
%sequential/rienwrhgrh/strided_slice_3StridedSliceAsequential/rienwrhgrh/TensorArrayV2Stack/TensorListStack:tensor:04sequential/rienwrhgrh/strided_slice_3/stack:output:06sequential/rienwrhgrh/strided_slice_3/stack_1:output:06sequential/rienwrhgrh/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/rienwrhgrh/strided_slice_3¥
&sequential/rienwrhgrh/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/rienwrhgrh/transpose_1/permý
!sequential/rienwrhgrh/transpose_1	TransposeAsequential/rienwrhgrh/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/rienwrhgrh/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/rienwrhgrh/transpose_1Ï
+sequential/uilnjhxhrx/MatMul/ReadVariableOpReadVariableOp4sequential_uilnjhxhrx_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/uilnjhxhrx/MatMul/ReadVariableOpÝ
sequential/uilnjhxhrx/MatMulMatMul.sequential/rienwrhgrh/strided_slice_3:output:03sequential/uilnjhxhrx/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/uilnjhxhrx/MatMulÎ
,sequential/uilnjhxhrx/BiasAdd/ReadVariableOpReadVariableOp5sequential_uilnjhxhrx_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/uilnjhxhrx/BiasAdd/ReadVariableOpÙ
sequential/uilnjhxhrx/BiasAddBiasAdd&sequential/uilnjhxhrx/MatMul:product:04sequential/uilnjhxhrx/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/uilnjhxhrx/BiasAdd 
IdentityIdentity&sequential/uilnjhxhrx/BiasAdd:output:09^sequential/kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp@^sequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp8^sequential/lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp7^sequential/lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp9^sequential/lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp0^sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp2^sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_12^sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_2^sequential/lctpanrywj/while8^sequential/rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp7^sequential/rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp9^sequential/rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp0^sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp2^sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_12^sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_2^sequential/rienwrhgrh/while-^sequential/uilnjhxhrx/BiasAdd/ReadVariableOp,^sequential/uilnjhxhrx/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2t
8sequential/kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp8sequential/kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7sequential/lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp7sequential/lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp2p
6sequential/lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp6sequential/lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp2t
8sequential/lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp8sequential/lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp2b
/sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp/sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp2f
1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_11sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_12f
1sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_21sequential/lctpanrywj/dhxpxqfhna/ReadVariableOp_22:
sequential/lctpanrywj/whilesequential/lctpanrywj/while2r
7sequential/rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp7sequential/rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp2p
6sequential/rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp6sequential/rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp2t
8sequential/rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp8sequential/rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp2b
/sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp/sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp2f
1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_11sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_12f
1sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_21sequential/rienwrhgrh/kngiiuzftt/ReadVariableOp_22:
sequential/rienwrhgrh/whilesequential/rienwrhgrh/while2\
,sequential/uilnjhxhrx/BiasAdd/ReadVariableOp,sequential/uilnjhxhrx/BiasAdd/ReadVariableOp2Z
+sequential/uilnjhxhrx/MatMul/ReadVariableOp+sequential/uilnjhxhrx/MatMul/ReadVariableOp:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
dtmnvweekc
¹'
µ
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1106047

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
while_body_1100946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_dhxpxqfhna_1100970_0:	-
while_dhxpxqfhna_1100972_0:	 )
while_dhxpxqfhna_1100974_0:	(
while_dhxpxqfhna_1100976_0: (
while_dhxpxqfhna_1100978_0: (
while_dhxpxqfhna_1100980_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_dhxpxqfhna_1100970:	+
while_dhxpxqfhna_1100972:	 '
while_dhxpxqfhna_1100974:	&
while_dhxpxqfhna_1100976: &
while_dhxpxqfhna_1100978: &
while_dhxpxqfhna_1100980: ¢(while/dhxpxqfhna/StatefulPartitionedCallÃ
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
(while/dhxpxqfhna/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_dhxpxqfhna_1100970_0while_dhxpxqfhna_1100972_0while_dhxpxqfhna_1100974_0while_dhxpxqfhna_1100976_0while_dhxpxqfhna_1100978_0while_dhxpxqfhna_1100980_0*
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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_11008502*
(while/dhxpxqfhna/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/dhxpxqfhna/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/dhxpxqfhna/StatefulPartitionedCall:output:1)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/dhxpxqfhna/StatefulPartitionedCall:output:2)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_dhxpxqfhna_1100970while_dhxpxqfhna_1100970_0"6
while_dhxpxqfhna_1100972while_dhxpxqfhna_1100972_0"6
while_dhxpxqfhna_1100974while_dhxpxqfhna_1100974_0"6
while_dhxpxqfhna_1100976while_dhxpxqfhna_1100976_0"6
while_dhxpxqfhna_1100978while_dhxpxqfhna_1100978_0"6
while_dhxpxqfhna_1100980while_dhxpxqfhna_1100980_0")
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
(while/dhxpxqfhna/StatefulPartitionedCall(while/dhxpxqfhna/StatefulPartitionedCall: 
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
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_1102136

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
¹'
µ
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1105957

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
àh

G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104454
inputs_0<
)dhxpxqfhna_matmul_readvariableop_resource:	>
+dhxpxqfhna_matmul_1_readvariableop_resource:	 9
*dhxpxqfhna_biasadd_readvariableop_resource:	0
"dhxpxqfhna_readvariableop_resource: 2
$dhxpxqfhna_readvariableop_1_resource: 2
$dhxpxqfhna_readvariableop_2_resource: 
identity¢!dhxpxqfhna/BiasAdd/ReadVariableOp¢ dhxpxqfhna/MatMul/ReadVariableOp¢"dhxpxqfhna/MatMul_1/ReadVariableOp¢dhxpxqfhna/ReadVariableOp¢dhxpxqfhna/ReadVariableOp_1¢dhxpxqfhna/ReadVariableOp_2¢whileF
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
 dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp)dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dhxpxqfhna/MatMul/ReadVariableOp§
dhxpxqfhna/MatMulMatMulstrided_slice_2:output:0(dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMulµ
"dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp+dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dhxpxqfhna/MatMul_1/ReadVariableOp£
dhxpxqfhna/MatMul_1MatMulzeros:output:0*dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMul_1
dhxpxqfhna/addAddV2dhxpxqfhna/MatMul:product:0dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/add®
!dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp*dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dhxpxqfhna/BiasAdd/ReadVariableOp¥
dhxpxqfhna/BiasAddBiasAdddhxpxqfhna/add:z:0)dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/BiasAddz
dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dhxpxqfhna/split/split_dimë
dhxpxqfhna/splitSplit#dhxpxqfhna/split/split_dim:output:0dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dhxpxqfhna/split
dhxpxqfhna/ReadVariableOpReadVariableOp"dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp
dhxpxqfhna/mulMul!dhxpxqfhna/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul
dhxpxqfhna/add_1AddV2dhxpxqfhna/split:output:0dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_1{
dhxpxqfhna/SigmoidSigmoiddhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid
dhxpxqfhna/ReadVariableOp_1ReadVariableOp$dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_1
dhxpxqfhna/mul_1Mul#dhxpxqfhna/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_1
dhxpxqfhna/add_2AddV2dhxpxqfhna/split:output:1dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_2
dhxpxqfhna/Sigmoid_1Sigmoiddhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_1
dhxpxqfhna/mul_2Muldhxpxqfhna/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_2w
dhxpxqfhna/TanhTanhdhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh
dhxpxqfhna/mul_3Muldhxpxqfhna/Sigmoid:y:0dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_3
dhxpxqfhna/add_3AddV2dhxpxqfhna/mul_2:z:0dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_3
dhxpxqfhna/ReadVariableOp_2ReadVariableOp$dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_2
dhxpxqfhna/mul_4Mul#dhxpxqfhna/ReadVariableOp_2:value:0dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_4
dhxpxqfhna/add_4AddV2dhxpxqfhna/split:output:3dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_4
dhxpxqfhna/Sigmoid_2Sigmoiddhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_2v
dhxpxqfhna/Tanh_1Tanhdhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh_1
dhxpxqfhna/mul_5Muldhxpxqfhna/Sigmoid_2:y:0dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dhxpxqfhna_matmul_readvariableop_resource+dhxpxqfhna_matmul_1_readvariableop_resource*dhxpxqfhna_biasadd_readvariableop_resource"dhxpxqfhna_readvariableop_resource$dhxpxqfhna_readvariableop_1_resource$dhxpxqfhna_readvariableop_2_resource*
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
while_body_1104353*
condR
while_cond_1104352*Q
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
IdentityIdentitytranspose_1:y:0"^dhxpxqfhna/BiasAdd/ReadVariableOp!^dhxpxqfhna/MatMul/ReadVariableOp#^dhxpxqfhna/MatMul_1/ReadVariableOp^dhxpxqfhna/ReadVariableOp^dhxpxqfhna/ReadVariableOp_1^dhxpxqfhna/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dhxpxqfhna/BiasAdd/ReadVariableOp!dhxpxqfhna/BiasAdd/ReadVariableOp2D
 dhxpxqfhna/MatMul/ReadVariableOp dhxpxqfhna/MatMul/ReadVariableOp2H
"dhxpxqfhna/MatMul_1/ReadVariableOp"dhxpxqfhna/MatMul_1/ReadVariableOp26
dhxpxqfhna/ReadVariableOpdhxpxqfhna/ReadVariableOp2:
dhxpxqfhna/ReadVariableOp_1dhxpxqfhna/ReadVariableOp_12:
dhxpxqfhna/ReadVariableOp_2dhxpxqfhna/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
È

,__inference_kjxhlaztnm_layer_call_fn_1104256

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
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_11021362
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
,__inference_sequential_layer_call_fn_1104173

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
G__inference_sequential_layer_call_and_return_conditional_losses_11025602
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
¯F
ê
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1101521

inputs%
kngiiuzftt_1101422:	 %
kngiiuzftt_1101424:	 !
kngiiuzftt_1101426:	 
kngiiuzftt_1101428:  
kngiiuzftt_1101430:  
kngiiuzftt_1101432: 
identity¢"kngiiuzftt/StatefulPartitionedCall¢whileD
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
"kngiiuzftt/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0kngiiuzftt_1101422kngiiuzftt_1101424kngiiuzftt_1101426kngiiuzftt_1101428kngiiuzftt_1101430kngiiuzftt_1101432*
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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_11014212$
"kngiiuzftt/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kngiiuzftt_1101422kngiiuzftt_1101424kngiiuzftt_1101426kngiiuzftt_1101428kngiiuzftt_1101430kngiiuzftt_1101432*
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
while_body_1101441*
condR
while_cond_1101440*Q
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
IdentityIdentitystrided_slice_3:output:0#^kngiiuzftt/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"kngiiuzftt/StatefulPartitionedCall"kngiiuzftt/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


,__inference_sequential_layer_call_fn_1104210

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
G__inference_sequential_layer_call_and_return_conditional_losses_11031292
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

c
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_1104269

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


lctpanrywj_while_cond_11034482
.lctpanrywj_while_lctpanrywj_while_loop_counter8
4lctpanrywj_while_lctpanrywj_while_maximum_iterations 
lctpanrywj_while_placeholder"
lctpanrywj_while_placeholder_1"
lctpanrywj_while_placeholder_2"
lctpanrywj_while_placeholder_34
0lctpanrywj_while_less_lctpanrywj_strided_slice_1K
Glctpanrywj_while_lctpanrywj_while_cond_1103448___redundant_placeholder0K
Glctpanrywj_while_lctpanrywj_while_cond_1103448___redundant_placeholder1K
Glctpanrywj_while_lctpanrywj_while_cond_1103448___redundant_placeholder2K
Glctpanrywj_while_lctpanrywj_while_cond_1103448___redundant_placeholder3K
Glctpanrywj_while_lctpanrywj_while_cond_1103448___redundant_placeholder4K
Glctpanrywj_while_lctpanrywj_while_cond_1103448___redundant_placeholder5K
Glctpanrywj_while_lctpanrywj_while_cond_1103448___redundant_placeholder6
lctpanrywj_while_identity
§
lctpanrywj/while/LessLesslctpanrywj_while_placeholder0lctpanrywj_while_less_lctpanrywj_strided_slice_1*
T0*
_output_shapes
: 2
lctpanrywj/while/Less~
lctpanrywj/while/IdentityIdentitylctpanrywj/while/Less:z:0*
T0
*
_output_shapes
: 2
lctpanrywj/while/Identity"?
lctpanrywj_while_identity"lctpanrywj/while/Identity:output:0*(
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
while_body_1102703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kngiiuzftt_matmul_readvariableop_resource_0:	 F
3while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 A
2while_kngiiuzftt_biasadd_readvariableop_resource_0:	8
*while_kngiiuzftt_readvariableop_resource_0: :
,while_kngiiuzftt_readvariableop_1_resource_0: :
,while_kngiiuzftt_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kngiiuzftt_matmul_readvariableop_resource:	 D
1while_kngiiuzftt_matmul_1_readvariableop_resource:	 ?
0while_kngiiuzftt_biasadd_readvariableop_resource:	6
(while_kngiiuzftt_readvariableop_resource: 8
*while_kngiiuzftt_readvariableop_1_resource: 8
*while_kngiiuzftt_readvariableop_2_resource: ¢'while/kngiiuzftt/BiasAdd/ReadVariableOp¢&while/kngiiuzftt/MatMul/ReadVariableOp¢(while/kngiiuzftt/MatMul_1/ReadVariableOp¢while/kngiiuzftt/ReadVariableOp¢!while/kngiiuzftt/ReadVariableOp_1¢!while/kngiiuzftt/ReadVariableOp_2Ã
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
&while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp1while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/kngiiuzftt/MatMul/ReadVariableOpÑ
while/kngiiuzftt/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMulÉ
(while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp3while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kngiiuzftt/MatMul_1/ReadVariableOpº
while/kngiiuzftt/MatMul_1MatMulwhile_placeholder_20while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMul_1°
while/kngiiuzftt/addAddV2!while/kngiiuzftt/MatMul:product:0#while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/addÂ
'while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp2while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kngiiuzftt/BiasAdd/ReadVariableOp½
while/kngiiuzftt/BiasAddBiasAddwhile/kngiiuzftt/add:z:0/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/BiasAdd
 while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kngiiuzftt/split/split_dim
while/kngiiuzftt/splitSplit)while/kngiiuzftt/split/split_dim:output:0!while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kngiiuzftt/split©
while/kngiiuzftt/ReadVariableOpReadVariableOp*while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kngiiuzftt/ReadVariableOp£
while/kngiiuzftt/mulMul'while/kngiiuzftt/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul¦
while/kngiiuzftt/add_1AddV2while/kngiiuzftt/split:output:0while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_1
while/kngiiuzftt/SigmoidSigmoidwhile/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid¯
!while/kngiiuzftt/ReadVariableOp_1ReadVariableOp,while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_1©
while/kngiiuzftt/mul_1Mul)while/kngiiuzftt/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_1¨
while/kngiiuzftt/add_2AddV2while/kngiiuzftt/split:output:1while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_2
while/kngiiuzftt/Sigmoid_1Sigmoidwhile/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_1
while/kngiiuzftt/mul_2Mulwhile/kngiiuzftt/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_2
while/kngiiuzftt/TanhTanhwhile/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh¢
while/kngiiuzftt/mul_3Mulwhile/kngiiuzftt/Sigmoid:y:0while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_3£
while/kngiiuzftt/add_3AddV2while/kngiiuzftt/mul_2:z:0while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_3¯
!while/kngiiuzftt/ReadVariableOp_2ReadVariableOp,while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_2°
while/kngiiuzftt/mul_4Mul)while/kngiiuzftt/ReadVariableOp_2:value:0while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_4¨
while/kngiiuzftt/add_4AddV2while/kngiiuzftt/split:output:3while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_4
while/kngiiuzftt/Sigmoid_2Sigmoidwhile/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_2
while/kngiiuzftt/Tanh_1Tanhwhile/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh_1¦
while/kngiiuzftt/mul_5Mulwhile/kngiiuzftt/Sigmoid_2:y:0while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kngiiuzftt/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kngiiuzftt/mul_5:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kngiiuzftt/add_3:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
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
0while_kngiiuzftt_biasadd_readvariableop_resource2while_kngiiuzftt_biasadd_readvariableop_resource_0"h
1while_kngiiuzftt_matmul_1_readvariableop_resource3while_kngiiuzftt_matmul_1_readvariableop_resource_0"d
/while_kngiiuzftt_matmul_readvariableop_resource1while_kngiiuzftt_matmul_readvariableop_resource_0"Z
*while_kngiiuzftt_readvariableop_1_resource,while_kngiiuzftt_readvariableop_1_resource_0"Z
*while_kngiiuzftt_readvariableop_2_resource,while_kngiiuzftt_readvariableop_2_resource_0"V
(while_kngiiuzftt_readvariableop_resource*while_kngiiuzftt_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kngiiuzftt/BiasAdd/ReadVariableOp'while/kngiiuzftt/BiasAdd/ReadVariableOp2P
&while/kngiiuzftt/MatMul/ReadVariableOp&while/kngiiuzftt/MatMul/ReadVariableOp2T
(while/kngiiuzftt/MatMul_1/ReadVariableOp(while/kngiiuzftt/MatMul_1/ReadVariableOp2B
while/kngiiuzftt/ReadVariableOpwhile/kngiiuzftt/ReadVariableOp2F
!while/kngiiuzftt/ReadVariableOp_1!while/kngiiuzftt/ReadVariableOp_12F
!while/kngiiuzftt/ReadVariableOp_2!while/kngiiuzftt/ReadVariableOp_2: 
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
ú[
Ë
 __inference__traced_save_1106277
file_prefix0
,savev2_kjxhlaztnm_kernel_read_readvariableop.
*savev2_kjxhlaztnm_bias_read_readvariableop0
,savev2_uilnjhxhrx_kernel_read_readvariableop.
*savev2_uilnjhxhrx_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_lctpanrywj_dhxpxqfhna_kernel_read_readvariableopE
Asavev2_lctpanrywj_dhxpxqfhna_recurrent_kernel_read_readvariableop9
5savev2_lctpanrywj_dhxpxqfhna_bias_read_readvariableopP
Lsavev2_lctpanrywj_dhxpxqfhna_input_gate_peephole_weights_read_readvariableopQ
Msavev2_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_lctpanrywj_dhxpxqfhna_output_gate_peephole_weights_read_readvariableop;
7savev2_rienwrhgrh_kngiiuzftt_kernel_read_readvariableopE
Asavev2_rienwrhgrh_kngiiuzftt_recurrent_kernel_read_readvariableop9
5savev2_rienwrhgrh_kngiiuzftt_bias_read_readvariableopP
Lsavev2_rienwrhgrh_kngiiuzftt_input_gate_peephole_weights_read_readvariableopQ
Msavev2_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_rienwrhgrh_kngiiuzftt_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_kjxhlaztnm_kernel_rms_read_readvariableop:
6savev2_rmsprop_kjxhlaztnm_bias_rms_read_readvariableop<
8savev2_rmsprop_uilnjhxhrx_kernel_rms_read_readvariableop:
6savev2_rmsprop_uilnjhxhrx_bias_rms_read_readvariableopG
Csavev2_rmsprop_lctpanrywj_dhxpxqfhna_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_lctpanrywj_dhxpxqfhna_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_lctpanrywj_dhxpxqfhna_bias_rms_read_readvariableop\
Xsavev2_rmsprop_lctpanrywj_dhxpxqfhna_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_lctpanrywj_dhxpxqfhna_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_rienwrhgrh_kngiiuzftt_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_rienwrhgrh_kngiiuzftt_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_rienwrhgrh_kngiiuzftt_bias_rms_read_readvariableop\
Xsavev2_rmsprop_rienwrhgrh_kngiiuzftt_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_rienwrhgrh_kngiiuzftt_output_gate_peephole_weights_rms_read_readvariableop
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
ShardedFilenameÁ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Ó
valueÉBÆ(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesØ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices£
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_kjxhlaztnm_kernel_read_readvariableop*savev2_kjxhlaztnm_bias_read_readvariableop,savev2_uilnjhxhrx_kernel_read_readvariableop*savev2_uilnjhxhrx_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_lctpanrywj_dhxpxqfhna_kernel_read_readvariableopAsavev2_lctpanrywj_dhxpxqfhna_recurrent_kernel_read_readvariableop5savev2_lctpanrywj_dhxpxqfhna_bias_read_readvariableopLsavev2_lctpanrywj_dhxpxqfhna_input_gate_peephole_weights_read_readvariableopMsavev2_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weights_read_readvariableopMsavev2_lctpanrywj_dhxpxqfhna_output_gate_peephole_weights_read_readvariableop7savev2_rienwrhgrh_kngiiuzftt_kernel_read_readvariableopAsavev2_rienwrhgrh_kngiiuzftt_recurrent_kernel_read_readvariableop5savev2_rienwrhgrh_kngiiuzftt_bias_read_readvariableopLsavev2_rienwrhgrh_kngiiuzftt_input_gate_peephole_weights_read_readvariableopMsavev2_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weights_read_readvariableopMsavev2_rienwrhgrh_kngiiuzftt_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_kjxhlaztnm_kernel_rms_read_readvariableop6savev2_rmsprop_kjxhlaztnm_bias_rms_read_readvariableop8savev2_rmsprop_uilnjhxhrx_kernel_rms_read_readvariableop6savev2_rmsprop_uilnjhxhrx_bias_rms_read_readvariableopCsavev2_rmsprop_lctpanrywj_dhxpxqfhna_kernel_rms_read_readvariableopMsavev2_rmsprop_lctpanrywj_dhxpxqfhna_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_lctpanrywj_dhxpxqfhna_bias_rms_read_readvariableopXsavev2_rmsprop_lctpanrywj_dhxpxqfhna_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_lctpanrywj_dhxpxqfhna_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_rienwrhgrh_kngiiuzftt_kernel_rms_read_readvariableopMsavev2_rmsprop_rienwrhgrh_kngiiuzftt_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_rienwrhgrh_kngiiuzftt_bias_rms_read_readvariableopXsavev2_rmsprop_rienwrhgrh_kngiiuzftt_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_rienwrhgrh_kngiiuzftt_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
àY

while_body_1105501
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kngiiuzftt_matmul_readvariableop_resource_0:	 F
3while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 A
2while_kngiiuzftt_biasadd_readvariableop_resource_0:	8
*while_kngiiuzftt_readvariableop_resource_0: :
,while_kngiiuzftt_readvariableop_1_resource_0: :
,while_kngiiuzftt_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kngiiuzftt_matmul_readvariableop_resource:	 D
1while_kngiiuzftt_matmul_1_readvariableop_resource:	 ?
0while_kngiiuzftt_biasadd_readvariableop_resource:	6
(while_kngiiuzftt_readvariableop_resource: 8
*while_kngiiuzftt_readvariableop_1_resource: 8
*while_kngiiuzftt_readvariableop_2_resource: ¢'while/kngiiuzftt/BiasAdd/ReadVariableOp¢&while/kngiiuzftt/MatMul/ReadVariableOp¢(while/kngiiuzftt/MatMul_1/ReadVariableOp¢while/kngiiuzftt/ReadVariableOp¢!while/kngiiuzftt/ReadVariableOp_1¢!while/kngiiuzftt/ReadVariableOp_2Ã
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
&while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp1while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/kngiiuzftt/MatMul/ReadVariableOpÑ
while/kngiiuzftt/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMulÉ
(while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp3while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kngiiuzftt/MatMul_1/ReadVariableOpº
while/kngiiuzftt/MatMul_1MatMulwhile_placeholder_20while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMul_1°
while/kngiiuzftt/addAddV2!while/kngiiuzftt/MatMul:product:0#while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/addÂ
'while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp2while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kngiiuzftt/BiasAdd/ReadVariableOp½
while/kngiiuzftt/BiasAddBiasAddwhile/kngiiuzftt/add:z:0/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/BiasAdd
 while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kngiiuzftt/split/split_dim
while/kngiiuzftt/splitSplit)while/kngiiuzftt/split/split_dim:output:0!while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kngiiuzftt/split©
while/kngiiuzftt/ReadVariableOpReadVariableOp*while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kngiiuzftt/ReadVariableOp£
while/kngiiuzftt/mulMul'while/kngiiuzftt/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul¦
while/kngiiuzftt/add_1AddV2while/kngiiuzftt/split:output:0while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_1
while/kngiiuzftt/SigmoidSigmoidwhile/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid¯
!while/kngiiuzftt/ReadVariableOp_1ReadVariableOp,while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_1©
while/kngiiuzftt/mul_1Mul)while/kngiiuzftt/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_1¨
while/kngiiuzftt/add_2AddV2while/kngiiuzftt/split:output:1while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_2
while/kngiiuzftt/Sigmoid_1Sigmoidwhile/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_1
while/kngiiuzftt/mul_2Mulwhile/kngiiuzftt/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_2
while/kngiiuzftt/TanhTanhwhile/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh¢
while/kngiiuzftt/mul_3Mulwhile/kngiiuzftt/Sigmoid:y:0while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_3£
while/kngiiuzftt/add_3AddV2while/kngiiuzftt/mul_2:z:0while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_3¯
!while/kngiiuzftt/ReadVariableOp_2ReadVariableOp,while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_2°
while/kngiiuzftt/mul_4Mul)while/kngiiuzftt/ReadVariableOp_2:value:0while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_4¨
while/kngiiuzftt/add_4AddV2while/kngiiuzftt/split:output:3while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_4
while/kngiiuzftt/Sigmoid_2Sigmoidwhile/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_2
while/kngiiuzftt/Tanh_1Tanhwhile/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh_1¦
while/kngiiuzftt/mul_5Mulwhile/kngiiuzftt/Sigmoid_2:y:0while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kngiiuzftt/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kngiiuzftt/mul_5:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kngiiuzftt/add_3:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
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
0while_kngiiuzftt_biasadd_readvariableop_resource2while_kngiiuzftt_biasadd_readvariableop_resource_0"h
1while_kngiiuzftt_matmul_1_readvariableop_resource3while_kngiiuzftt_matmul_1_readvariableop_resource_0"d
/while_kngiiuzftt_matmul_readvariableop_resource1while_kngiiuzftt_matmul_readvariableop_resource_0"Z
*while_kngiiuzftt_readvariableop_1_resource,while_kngiiuzftt_readvariableop_1_resource_0"Z
*while_kngiiuzftt_readvariableop_2_resource,while_kngiiuzftt_readvariableop_2_resource_0"V
(while_kngiiuzftt_readvariableop_resource*while_kngiiuzftt_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kngiiuzftt/BiasAdd/ReadVariableOp'while/kngiiuzftt/BiasAdd/ReadVariableOp2P
&while/kngiiuzftt/MatMul/ReadVariableOp&while/kngiiuzftt/MatMul/ReadVariableOp2T
(while/kngiiuzftt/MatMul_1/ReadVariableOp(while/kngiiuzftt/MatMul_1/ReadVariableOp2B
while/kngiiuzftt/ReadVariableOpwhile/kngiiuzftt/ReadVariableOp2F
!while/kngiiuzftt/ReadVariableOp_1!while/kngiiuzftt/ReadVariableOp_12F
!while/kngiiuzftt/ReadVariableOp_2!while/kngiiuzftt/ReadVariableOp_2: 
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
Û

,__inference_lctpanrywj_layer_call_fn_1105062

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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11030182
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


rienwrhgrh_while_cond_11036242
.rienwrhgrh_while_rienwrhgrh_while_loop_counter8
4rienwrhgrh_while_rienwrhgrh_while_maximum_iterations 
rienwrhgrh_while_placeholder"
rienwrhgrh_while_placeholder_1"
rienwrhgrh_while_placeholder_2"
rienwrhgrh_while_placeholder_34
0rienwrhgrh_while_less_rienwrhgrh_strided_slice_1K
Grienwrhgrh_while_rienwrhgrh_while_cond_1103624___redundant_placeholder0K
Grienwrhgrh_while_rienwrhgrh_while_cond_1103624___redundant_placeholder1K
Grienwrhgrh_while_rienwrhgrh_while_cond_1103624___redundant_placeholder2K
Grienwrhgrh_while_rienwrhgrh_while_cond_1103624___redundant_placeholder3K
Grienwrhgrh_while_rienwrhgrh_while_cond_1103624___redundant_placeholder4K
Grienwrhgrh_while_rienwrhgrh_while_cond_1103624___redundant_placeholder5K
Grienwrhgrh_while_rienwrhgrh_while_cond_1103624___redundant_placeholder6
rienwrhgrh_while_identity
§
rienwrhgrh/while/LessLessrienwrhgrh_while_placeholder0rienwrhgrh_while_less_rienwrhgrh_strided_slice_1*
T0*
_output_shapes
: 2
rienwrhgrh/while/Less~
rienwrhgrh/while/IdentityIdentityrienwrhgrh/while/Less:z:0*
T0
*
_output_shapes
: 2
rienwrhgrh/while/Identity"?
rienwrhgrh_while_identity"rienwrhgrh/while/Identity:output:0*(
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

,__inference_sequential_layer_call_fn_1102595

dtmnvweekc
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
dtmnvweekcunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_11025602
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
dtmnvweekc
ç)
Ò
while_body_1101704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_kngiiuzftt_1101728_0:	 -
while_kngiiuzftt_1101730_0:	 )
while_kngiiuzftt_1101732_0:	(
while_kngiiuzftt_1101734_0: (
while_kngiiuzftt_1101736_0: (
while_kngiiuzftt_1101738_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_kngiiuzftt_1101728:	 +
while_kngiiuzftt_1101730:	 '
while_kngiiuzftt_1101732:	&
while_kngiiuzftt_1101734: &
while_kngiiuzftt_1101736: &
while_kngiiuzftt_1101738: ¢(while/kngiiuzftt/StatefulPartitionedCallÃ
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
(while/kngiiuzftt/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_kngiiuzftt_1101728_0while_kngiiuzftt_1101730_0while_kngiiuzftt_1101732_0while_kngiiuzftt_1101734_0while_kngiiuzftt_1101736_0while_kngiiuzftt_1101738_0*
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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_11016082*
(while/kngiiuzftt/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/kngiiuzftt/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/kngiiuzftt/StatefulPartitionedCall:output:1)^while/kngiiuzftt/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/kngiiuzftt/StatefulPartitionedCall:output:2)^while/kngiiuzftt/StatefulPartitionedCall*
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
while_kngiiuzftt_1101728while_kngiiuzftt_1101728_0"6
while_kngiiuzftt_1101730while_kngiiuzftt_1101730_0"6
while_kngiiuzftt_1101732while_kngiiuzftt_1101732_0"6
while_kngiiuzftt_1101734while_kngiiuzftt_1101734_0"6
while_kngiiuzftt_1101736while_kngiiuzftt_1101736_0"6
while_kngiiuzftt_1101738while_kngiiuzftt_1101738_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/kngiiuzftt/StatefulPartitionedCall(while/kngiiuzftt/StatefulPartitionedCall: 
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
while_cond_1104712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1104712___redundant_placeholder05
1while_while_cond_1104712___redundant_placeholder15
1while_while_cond_1104712___redundant_placeholder25
1while_while_cond_1104712___redundant_placeholder35
1while_while_cond_1104712___redundant_placeholder45
1while_while_cond_1104712___redundant_placeholder55
1while_while_cond_1104712___redundant_placeholder6
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
,__inference_kngiiuzftt_layer_call_fn_1106114

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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_11014212
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
while_body_1100683
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_dhxpxqfhna_1100707_0:	-
while_dhxpxqfhna_1100709_0:	 )
while_dhxpxqfhna_1100711_0:	(
while_dhxpxqfhna_1100713_0: (
while_dhxpxqfhna_1100715_0: (
while_dhxpxqfhna_1100717_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_dhxpxqfhna_1100707:	+
while_dhxpxqfhna_1100709:	 '
while_dhxpxqfhna_1100711:	&
while_dhxpxqfhna_1100713: &
while_dhxpxqfhna_1100715: &
while_dhxpxqfhna_1100717: ¢(while/dhxpxqfhna/StatefulPartitionedCallÃ
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
(while/dhxpxqfhna/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_dhxpxqfhna_1100707_0while_dhxpxqfhna_1100709_0while_dhxpxqfhna_1100711_0while_dhxpxqfhna_1100713_0while_dhxpxqfhna_1100715_0while_dhxpxqfhna_1100717_0*
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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_11006632*
(while/dhxpxqfhna/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/dhxpxqfhna/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/dhxpxqfhna/StatefulPartitionedCall:output:1)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/dhxpxqfhna/StatefulPartitionedCall:output:2)^while/dhxpxqfhna/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_dhxpxqfhna_1100707while_dhxpxqfhna_1100707_0"6
while_dhxpxqfhna_1100709while_dhxpxqfhna_1100709_0"6
while_dhxpxqfhna_1100711while_dhxpxqfhna_1100711_0"6
while_dhxpxqfhna_1100713while_dhxpxqfhna_1100713_0"6
while_dhxpxqfhna_1100715while_dhxpxqfhna_1100715_0"6
while_dhxpxqfhna_1100717while_dhxpxqfhna_1100717_0")
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
(while/dhxpxqfhna/StatefulPartitionedCall(while/dhxpxqfhna/StatefulPartitionedCall: 
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

lctpanrywj_while_cond_11038522
.lctpanrywj_while_lctpanrywj_while_loop_counter8
4lctpanrywj_while_lctpanrywj_while_maximum_iterations 
lctpanrywj_while_placeholder"
lctpanrywj_while_placeholder_1"
lctpanrywj_while_placeholder_2"
lctpanrywj_while_placeholder_34
0lctpanrywj_while_less_lctpanrywj_strided_slice_1K
Glctpanrywj_while_lctpanrywj_while_cond_1103852___redundant_placeholder0K
Glctpanrywj_while_lctpanrywj_while_cond_1103852___redundant_placeholder1K
Glctpanrywj_while_lctpanrywj_while_cond_1103852___redundant_placeholder2K
Glctpanrywj_while_lctpanrywj_while_cond_1103852___redundant_placeholder3K
Glctpanrywj_while_lctpanrywj_while_cond_1103852___redundant_placeholder4K
Glctpanrywj_while_lctpanrywj_while_cond_1103852___redundant_placeholder5K
Glctpanrywj_while_lctpanrywj_while_cond_1103852___redundant_placeholder6
lctpanrywj_while_identity
§
lctpanrywj/while/LessLesslctpanrywj_while_placeholder0lctpanrywj_while_less_lctpanrywj_strided_slice_1*
T0*
_output_shapes
: 2
lctpanrywj/while/Less~
lctpanrywj/while/IdentityIdentitylctpanrywj/while/Less:z:0*
T0
*
_output_shapes
: 2
lctpanrywj/while/Identity"?
lctpanrywj_while_identity"lctpanrywj/while/Identity:output:0*(
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
while_cond_1104352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1104352___redundant_placeholder05
1while_while_cond_1104352___redundant_placeholder15
1while_while_cond_1104352___redundant_placeholder25
1while_while_cond_1104352___redundant_placeholder35
1while_while_cond_1104352___redundant_placeholder45
1while_while_cond_1104352___redundant_placeholder55
1while_while_cond_1104352___redundant_placeholder6
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
while_cond_1104532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1104532___redundant_placeholder05
1while_while_cond_1104532___redundant_placeholder15
1while_while_cond_1104532___redundant_placeholder25
1while_while_cond_1104532___redundant_placeholder35
1while_while_cond_1104532___redundant_placeholder45
1while_while_cond_1104532___redundant_placeholder55
1while_while_cond_1104532___redundant_placeholder6
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
Ü

(sequential_rienwrhgrh_while_body_1100469H
Dsequential_rienwrhgrh_while_sequential_rienwrhgrh_while_loop_counterN
Jsequential_rienwrhgrh_while_sequential_rienwrhgrh_while_maximum_iterations+
'sequential_rienwrhgrh_while_placeholder-
)sequential_rienwrhgrh_while_placeholder_1-
)sequential_rienwrhgrh_while_placeholder_2-
)sequential_rienwrhgrh_while_placeholder_3G
Csequential_rienwrhgrh_while_sequential_rienwrhgrh_strided_slice_1_0
sequential_rienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_sequential_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0:	 \
Isequential_rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 W
Hsequential_rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0:	N
@sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0: P
Bsequential_rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0: P
Bsequential_rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0: (
$sequential_rienwrhgrh_while_identity*
&sequential_rienwrhgrh_while_identity_1*
&sequential_rienwrhgrh_while_identity_2*
&sequential_rienwrhgrh_while_identity_3*
&sequential_rienwrhgrh_while_identity_4*
&sequential_rienwrhgrh_while_identity_5E
Asequential_rienwrhgrh_while_sequential_rienwrhgrh_strided_slice_1
}sequential_rienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_sequential_rienwrhgrh_tensorarrayunstack_tensorlistfromtensorX
Esequential_rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource:	 Z
Gsequential_rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource:	 U
Fsequential_rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource:	L
>sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_resource: N
@sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource: N
@sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource: ¢=sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp¢<sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp¢>sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp¢5sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp¢7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1¢7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2ï
Msequential/rienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/rienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/rienwrhgrh/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_rienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_sequential_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0'sequential_rienwrhgrh_while_placeholderVsequential/rienwrhgrh/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/rienwrhgrh/while/TensorArrayV2Read/TensorListGetItem
<sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOpGsequential_rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp©
-sequential/rienwrhgrh/while/kngiiuzftt/MatMulMatMulFsequential/rienwrhgrh/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/rienwrhgrh/while/kngiiuzftt/MatMul
>sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOpIsequential_rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp
/sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1MatMul)sequential_rienwrhgrh_while_placeholder_2Fsequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1
*sequential/rienwrhgrh/while/kngiiuzftt/addAddV27sequential/rienwrhgrh/while/kngiiuzftt/MatMul:product:09sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/rienwrhgrh/while/kngiiuzftt/add
=sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOpHsequential_rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp
.sequential/rienwrhgrh/while/kngiiuzftt/BiasAddBiasAdd.sequential/rienwrhgrh/while/kngiiuzftt/add:z:0Esequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd²
6sequential/rienwrhgrh/while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/rienwrhgrh/while/kngiiuzftt/split/split_dimÛ
,sequential/rienwrhgrh/while/kngiiuzftt/splitSplit?sequential/rienwrhgrh/while/kngiiuzftt/split/split_dim:output:07sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/rienwrhgrh/while/kngiiuzftt/splitë
5sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOpReadVariableOp@sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOpû
*sequential/rienwrhgrh/while/kngiiuzftt/mulMul=sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp:value:0)sequential_rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/rienwrhgrh/while/kngiiuzftt/mulþ
,sequential/rienwrhgrh/while/kngiiuzftt/add_1AddV25sequential/rienwrhgrh/while/kngiiuzftt/split:output:0.sequential/rienwrhgrh/while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/add_1Ï
.sequential/rienwrhgrh/while/kngiiuzftt/SigmoidSigmoid0sequential/rienwrhgrh/while/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/rienwrhgrh/while/kngiiuzftt/Sigmoidñ
7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1ReadVariableOpBsequential_rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1
,sequential/rienwrhgrh/while/kngiiuzftt/mul_1Mul?sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_1:value:0)sequential_rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/mul_1
,sequential/rienwrhgrh/while/kngiiuzftt/add_2AddV25sequential/rienwrhgrh/while/kngiiuzftt/split:output:10sequential/rienwrhgrh/while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/add_2Ó
0sequential/rienwrhgrh/while/kngiiuzftt/Sigmoid_1Sigmoid0sequential/rienwrhgrh/while/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/rienwrhgrh/while/kngiiuzftt/Sigmoid_1ö
,sequential/rienwrhgrh/while/kngiiuzftt/mul_2Mul4sequential/rienwrhgrh/while/kngiiuzftt/Sigmoid_1:y:0)sequential_rienwrhgrh_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/mul_2Ë
+sequential/rienwrhgrh/while/kngiiuzftt/TanhTanh5sequential/rienwrhgrh/while/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/rienwrhgrh/while/kngiiuzftt/Tanhú
,sequential/rienwrhgrh/while/kngiiuzftt/mul_3Mul2sequential/rienwrhgrh/while/kngiiuzftt/Sigmoid:y:0/sequential/rienwrhgrh/while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/mul_3û
,sequential/rienwrhgrh/while/kngiiuzftt/add_3AddV20sequential/rienwrhgrh/while/kngiiuzftt/mul_2:z:00sequential/rienwrhgrh/while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/add_3ñ
7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2ReadVariableOpBsequential_rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2
,sequential/rienwrhgrh/while/kngiiuzftt/mul_4Mul?sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2:value:00sequential/rienwrhgrh/while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/mul_4
,sequential/rienwrhgrh/while/kngiiuzftt/add_4AddV25sequential/rienwrhgrh/while/kngiiuzftt/split:output:30sequential/rienwrhgrh/while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/add_4Ó
0sequential/rienwrhgrh/while/kngiiuzftt/Sigmoid_2Sigmoid0sequential/rienwrhgrh/while/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/rienwrhgrh/while/kngiiuzftt/Sigmoid_2Ê
-sequential/rienwrhgrh/while/kngiiuzftt/Tanh_1Tanh0sequential/rienwrhgrh/while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/rienwrhgrh/while/kngiiuzftt/Tanh_1þ
,sequential/rienwrhgrh/while/kngiiuzftt/mul_5Mul4sequential/rienwrhgrh/while/kngiiuzftt/Sigmoid_2:y:01sequential/rienwrhgrh/while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/rienwrhgrh/while/kngiiuzftt/mul_5Ì
@sequential/rienwrhgrh/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_rienwrhgrh_while_placeholder_1'sequential_rienwrhgrh_while_placeholder0sequential/rienwrhgrh/while/kngiiuzftt/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/rienwrhgrh/while/TensorArrayV2Write/TensorListSetItem
!sequential/rienwrhgrh/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/rienwrhgrh/while/add/yÁ
sequential/rienwrhgrh/while/addAddV2'sequential_rienwrhgrh_while_placeholder*sequential/rienwrhgrh/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/rienwrhgrh/while/add
#sequential/rienwrhgrh/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/rienwrhgrh/while/add_1/yä
!sequential/rienwrhgrh/while/add_1AddV2Dsequential_rienwrhgrh_while_sequential_rienwrhgrh_while_loop_counter,sequential/rienwrhgrh/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/rienwrhgrh/while/add_1
$sequential/rienwrhgrh/while/IdentityIdentity%sequential/rienwrhgrh/while/add_1:z:0>^sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp=^sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp?^sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp6^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp8^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_18^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/rienwrhgrh/while/Identityµ
&sequential/rienwrhgrh/while/Identity_1IdentityJsequential_rienwrhgrh_while_sequential_rienwrhgrh_while_maximum_iterations>^sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp=^sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp?^sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp6^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp8^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_18^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/rienwrhgrh/while/Identity_1
&sequential/rienwrhgrh/while/Identity_2Identity#sequential/rienwrhgrh/while/add:z:0>^sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp=^sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp?^sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp6^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp8^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_18^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/rienwrhgrh/while/Identity_2»
&sequential/rienwrhgrh/while/Identity_3IdentityPsequential/rienwrhgrh/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp=^sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp?^sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp6^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp8^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_18^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/rienwrhgrh/while/Identity_3¬
&sequential/rienwrhgrh/while/Identity_4Identity0sequential/rienwrhgrh/while/kngiiuzftt/mul_5:z:0>^sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp=^sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp?^sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp6^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp8^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_18^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/while/Identity_4¬
&sequential/rienwrhgrh/while/Identity_5Identity0sequential/rienwrhgrh/while/kngiiuzftt/add_3:z:0>^sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp=^sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp?^sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp6^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp8^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_18^sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/rienwrhgrh/while/Identity_5"U
$sequential_rienwrhgrh_while_identity-sequential/rienwrhgrh/while/Identity:output:0"Y
&sequential_rienwrhgrh_while_identity_1/sequential/rienwrhgrh/while/Identity_1:output:0"Y
&sequential_rienwrhgrh_while_identity_2/sequential/rienwrhgrh/while/Identity_2:output:0"Y
&sequential_rienwrhgrh_while_identity_3/sequential/rienwrhgrh/while/Identity_3:output:0"Y
&sequential_rienwrhgrh_while_identity_4/sequential/rienwrhgrh/while/Identity_4:output:0"Y
&sequential_rienwrhgrh_while_identity_5/sequential/rienwrhgrh/while/Identity_5:output:0"
Fsequential_rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resourceHsequential_rienwrhgrh_while_kngiiuzftt_biasadd_readvariableop_resource_0"
Gsequential_rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resourceIsequential_rienwrhgrh_while_kngiiuzftt_matmul_1_readvariableop_resource_0"
Esequential_rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resourceGsequential_rienwrhgrh_while_kngiiuzftt_matmul_readvariableop_resource_0"
@sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_1_resourceBsequential_rienwrhgrh_while_kngiiuzftt_readvariableop_1_resource_0"
@sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_2_resourceBsequential_rienwrhgrh_while_kngiiuzftt_readvariableop_2_resource_0"
>sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_resource@sequential_rienwrhgrh_while_kngiiuzftt_readvariableop_resource_0"
Asequential_rienwrhgrh_while_sequential_rienwrhgrh_strided_slice_1Csequential_rienwrhgrh_while_sequential_rienwrhgrh_strided_slice_1_0"
}sequential_rienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_sequential_rienwrhgrh_tensorarrayunstack_tensorlistfromtensorsequential_rienwrhgrh_while_tensorarrayv2read_tensorlistgetitem_sequential_rienwrhgrh_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp=sequential/rienwrhgrh/while/kngiiuzftt/BiasAdd/ReadVariableOp2|
<sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp<sequential/rienwrhgrh/while/kngiiuzftt/MatMul/ReadVariableOp2
>sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp>sequential/rienwrhgrh/while/kngiiuzftt/MatMul_1/ReadVariableOp2n
5sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp5sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp2r
7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_17sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_12r
7sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_27sequential/rienwrhgrh/while/kngiiuzftt/ReadVariableOp_2: 
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
while_cond_1105320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1105320___redundant_placeholder05
1while_while_cond_1105320___redundant_placeholder15
1while_while_cond_1105320___redundant_placeholder25
1while_while_cond_1105320___redundant_placeholder35
1while_while_cond_1105320___redundant_placeholder45
1while_while_cond_1105320___redundant_placeholder55
1while_while_cond_1105320___redundant_placeholder6
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
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_1102155

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

À
,__inference_kngiiuzftt_layer_call_fn_1106137

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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_11016082
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
Üh

G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105242
inputs_0<
)kngiiuzftt_matmul_readvariableop_resource:	 >
+kngiiuzftt_matmul_1_readvariableop_resource:	 9
*kngiiuzftt_biasadd_readvariableop_resource:	0
"kngiiuzftt_readvariableop_resource: 2
$kngiiuzftt_readvariableop_1_resource: 2
$kngiiuzftt_readvariableop_2_resource: 
identity¢!kngiiuzftt/BiasAdd/ReadVariableOp¢ kngiiuzftt/MatMul/ReadVariableOp¢"kngiiuzftt/MatMul_1/ReadVariableOp¢kngiiuzftt/ReadVariableOp¢kngiiuzftt/ReadVariableOp_1¢kngiiuzftt/ReadVariableOp_2¢whileF
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
 kngiiuzftt/MatMul/ReadVariableOpReadVariableOp)kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 kngiiuzftt/MatMul/ReadVariableOp§
kngiiuzftt/MatMulMatMulstrided_slice_2:output:0(kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMulµ
"kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp+kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kngiiuzftt/MatMul_1/ReadVariableOp£
kngiiuzftt/MatMul_1MatMulzeros:output:0*kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMul_1
kngiiuzftt/addAddV2kngiiuzftt/MatMul:product:0kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/add®
!kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp*kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kngiiuzftt/BiasAdd/ReadVariableOp¥
kngiiuzftt/BiasAddBiasAddkngiiuzftt/add:z:0)kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/BiasAddz
kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kngiiuzftt/split/split_dimë
kngiiuzftt/splitSplit#kngiiuzftt/split/split_dim:output:0kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kngiiuzftt/split
kngiiuzftt/ReadVariableOpReadVariableOp"kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp
kngiiuzftt/mulMul!kngiiuzftt/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul
kngiiuzftt/add_1AddV2kngiiuzftt/split:output:0kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_1{
kngiiuzftt/SigmoidSigmoidkngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid
kngiiuzftt/ReadVariableOp_1ReadVariableOp$kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_1
kngiiuzftt/mul_1Mul#kngiiuzftt/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_1
kngiiuzftt/add_2AddV2kngiiuzftt/split:output:1kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_2
kngiiuzftt/Sigmoid_1Sigmoidkngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_1
kngiiuzftt/mul_2Mulkngiiuzftt/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_2w
kngiiuzftt/TanhTanhkngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh
kngiiuzftt/mul_3Mulkngiiuzftt/Sigmoid:y:0kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_3
kngiiuzftt/add_3AddV2kngiiuzftt/mul_2:z:0kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_3
kngiiuzftt/ReadVariableOp_2ReadVariableOp$kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_2
kngiiuzftt/mul_4Mul#kngiiuzftt/ReadVariableOp_2:value:0kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_4
kngiiuzftt/add_4AddV2kngiiuzftt/split:output:3kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_4
kngiiuzftt/Sigmoid_2Sigmoidkngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_2v
kngiiuzftt/Tanh_1Tanhkngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh_1
kngiiuzftt/mul_5Mulkngiiuzftt/Sigmoid_2:y:0kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kngiiuzftt_matmul_readvariableop_resource+kngiiuzftt_matmul_1_readvariableop_resource*kngiiuzftt_biasadd_readvariableop_resource"kngiiuzftt_readvariableop_resource$kngiiuzftt_readvariableop_1_resource$kngiiuzftt_readvariableop_2_resource*
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
while_body_1105141*
condR
while_cond_1105140*Q
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
IdentityIdentitystrided_slice_3:output:0"^kngiiuzftt/BiasAdd/ReadVariableOp!^kngiiuzftt/MatMul/ReadVariableOp#^kngiiuzftt/MatMul_1/ReadVariableOp^kngiiuzftt/ReadVariableOp^kngiiuzftt/ReadVariableOp_1^kngiiuzftt/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!kngiiuzftt/BiasAdd/ReadVariableOp!kngiiuzftt/BiasAdd/ReadVariableOp2D
 kngiiuzftt/MatMul/ReadVariableOp kngiiuzftt/MatMul/ReadVariableOp2H
"kngiiuzftt/MatMul_1/ReadVariableOp"kngiiuzftt/MatMul_1/ReadVariableOp26
kngiiuzftt/ReadVariableOpkngiiuzftt/ReadVariableOp2:
kngiiuzftt/ReadVariableOp_1kngiiuzftt/ReadVariableOp_12:
kngiiuzftt/ReadVariableOp_2kngiiuzftt/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
¦h

G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1102804

inputs<
)kngiiuzftt_matmul_readvariableop_resource:	 >
+kngiiuzftt_matmul_1_readvariableop_resource:	 9
*kngiiuzftt_biasadd_readvariableop_resource:	0
"kngiiuzftt_readvariableop_resource: 2
$kngiiuzftt_readvariableop_1_resource: 2
$kngiiuzftt_readvariableop_2_resource: 
identity¢!kngiiuzftt/BiasAdd/ReadVariableOp¢ kngiiuzftt/MatMul/ReadVariableOp¢"kngiiuzftt/MatMul_1/ReadVariableOp¢kngiiuzftt/ReadVariableOp¢kngiiuzftt/ReadVariableOp_1¢kngiiuzftt/ReadVariableOp_2¢whileD
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
 kngiiuzftt/MatMul/ReadVariableOpReadVariableOp)kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 kngiiuzftt/MatMul/ReadVariableOp§
kngiiuzftt/MatMulMatMulstrided_slice_2:output:0(kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMulµ
"kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp+kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kngiiuzftt/MatMul_1/ReadVariableOp£
kngiiuzftt/MatMul_1MatMulzeros:output:0*kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/MatMul_1
kngiiuzftt/addAddV2kngiiuzftt/MatMul:product:0kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/add®
!kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp*kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kngiiuzftt/BiasAdd/ReadVariableOp¥
kngiiuzftt/BiasAddBiasAddkngiiuzftt/add:z:0)kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kngiiuzftt/BiasAddz
kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kngiiuzftt/split/split_dimë
kngiiuzftt/splitSplit#kngiiuzftt/split/split_dim:output:0kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kngiiuzftt/split
kngiiuzftt/ReadVariableOpReadVariableOp"kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp
kngiiuzftt/mulMul!kngiiuzftt/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul
kngiiuzftt/add_1AddV2kngiiuzftt/split:output:0kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_1{
kngiiuzftt/SigmoidSigmoidkngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid
kngiiuzftt/ReadVariableOp_1ReadVariableOp$kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_1
kngiiuzftt/mul_1Mul#kngiiuzftt/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_1
kngiiuzftt/add_2AddV2kngiiuzftt/split:output:1kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_2
kngiiuzftt/Sigmoid_1Sigmoidkngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_1
kngiiuzftt/mul_2Mulkngiiuzftt/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_2w
kngiiuzftt/TanhTanhkngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh
kngiiuzftt/mul_3Mulkngiiuzftt/Sigmoid:y:0kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_3
kngiiuzftt/add_3AddV2kngiiuzftt/mul_2:z:0kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_3
kngiiuzftt/ReadVariableOp_2ReadVariableOp$kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02
kngiiuzftt/ReadVariableOp_2
kngiiuzftt/mul_4Mul#kngiiuzftt/ReadVariableOp_2:value:0kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_4
kngiiuzftt/add_4AddV2kngiiuzftt/split:output:3kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/add_4
kngiiuzftt/Sigmoid_2Sigmoidkngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Sigmoid_2v
kngiiuzftt/Tanh_1Tanhkngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/Tanh_1
kngiiuzftt/mul_5Mulkngiiuzftt/Sigmoid_2:y:0kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kngiiuzftt/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kngiiuzftt_matmul_readvariableop_resource+kngiiuzftt_matmul_1_readvariableop_resource*kngiiuzftt_biasadd_readvariableop_resource"kngiiuzftt_readvariableop_resource$kngiiuzftt_readvariableop_1_resource$kngiiuzftt_readvariableop_2_resource*
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
while_body_1102703*
condR
while_cond_1102702*Q
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
IdentityIdentitystrided_slice_3:output:0"^kngiiuzftt/BiasAdd/ReadVariableOp!^kngiiuzftt/MatMul/ReadVariableOp#^kngiiuzftt/MatMul_1/ReadVariableOp^kngiiuzftt/ReadVariableOp^kngiiuzftt/ReadVariableOp_1^kngiiuzftt/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!kngiiuzftt/BiasAdd/ReadVariableOp!kngiiuzftt/BiasAdd/ReadVariableOp2D
 kngiiuzftt/MatMul/ReadVariableOp kngiiuzftt/MatMul/ReadVariableOp2H
"kngiiuzftt/MatMul_1/ReadVariableOp"kngiiuzftt/MatMul_1/ReadVariableOp26
kngiiuzftt/ReadVariableOpkngiiuzftt/ReadVariableOp2:
kngiiuzftt/ReadVariableOp_1kngiiuzftt/ReadVariableOp_12:
kngiiuzftt/ReadVariableOp_2kngiiuzftt/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
©	
(sequential_rienwrhgrh_while_cond_1100468H
Dsequential_rienwrhgrh_while_sequential_rienwrhgrh_while_loop_counterN
Jsequential_rienwrhgrh_while_sequential_rienwrhgrh_while_maximum_iterations+
'sequential_rienwrhgrh_while_placeholder-
)sequential_rienwrhgrh_while_placeholder_1-
)sequential_rienwrhgrh_while_placeholder_2-
)sequential_rienwrhgrh_while_placeholder_3J
Fsequential_rienwrhgrh_while_less_sequential_rienwrhgrh_strided_slice_1a
]sequential_rienwrhgrh_while_sequential_rienwrhgrh_while_cond_1100468___redundant_placeholder0a
]sequential_rienwrhgrh_while_sequential_rienwrhgrh_while_cond_1100468___redundant_placeholder1a
]sequential_rienwrhgrh_while_sequential_rienwrhgrh_while_cond_1100468___redundant_placeholder2a
]sequential_rienwrhgrh_while_sequential_rienwrhgrh_while_cond_1100468___redundant_placeholder3a
]sequential_rienwrhgrh_while_sequential_rienwrhgrh_while_cond_1100468___redundant_placeholder4a
]sequential_rienwrhgrh_while_sequential_rienwrhgrh_while_cond_1100468___redundant_placeholder5a
]sequential_rienwrhgrh_while_sequential_rienwrhgrh_while_cond_1100468___redundant_placeholder6(
$sequential_rienwrhgrh_while_identity
Þ
 sequential/rienwrhgrh/while/LessLess'sequential_rienwrhgrh_while_placeholderFsequential_rienwrhgrh_while_less_sequential_rienwrhgrh_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/rienwrhgrh/while/Less
$sequential/rienwrhgrh/while/IdentityIdentity$sequential/rienwrhgrh/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/rienwrhgrh/while/Identity"U
$sequential_rienwrhgrh_while_identity-sequential/rienwrhgrh/while/Identity:output:0*(
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
while_cond_1100682
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1100682___redundant_placeholder05
1while_while_cond_1100682___redundant_placeholder15
1while_while_cond_1100682___redundant_placeholder25
1while_while_cond_1100682___redundant_placeholder35
1while_while_cond_1100682___redundant_placeholder45
1while_while_cond_1100682___redundant_placeholder55
1while_while_cond_1100682___redundant_placeholder6
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
while_body_1104533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dhxpxqfhna_matmul_readvariableop_resource_0:	F
3while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 A
2while_dhxpxqfhna_biasadd_readvariableop_resource_0:	8
*while_dhxpxqfhna_readvariableop_resource_0: :
,while_dhxpxqfhna_readvariableop_1_resource_0: :
,while_dhxpxqfhna_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dhxpxqfhna_matmul_readvariableop_resource:	D
1while_dhxpxqfhna_matmul_1_readvariableop_resource:	 ?
0while_dhxpxqfhna_biasadd_readvariableop_resource:	6
(while_dhxpxqfhna_readvariableop_resource: 8
*while_dhxpxqfhna_readvariableop_1_resource: 8
*while_dhxpxqfhna_readvariableop_2_resource: ¢'while/dhxpxqfhna/BiasAdd/ReadVariableOp¢&while/dhxpxqfhna/MatMul/ReadVariableOp¢(while/dhxpxqfhna/MatMul_1/ReadVariableOp¢while/dhxpxqfhna/ReadVariableOp¢!while/dhxpxqfhna/ReadVariableOp_1¢!while/dhxpxqfhna/ReadVariableOp_2Ã
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
&while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp1while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dhxpxqfhna/MatMul/ReadVariableOpÑ
while/dhxpxqfhna/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMulÉ
(while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp3while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dhxpxqfhna/MatMul_1/ReadVariableOpº
while/dhxpxqfhna/MatMul_1MatMulwhile_placeholder_20while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMul_1°
while/dhxpxqfhna/addAddV2!while/dhxpxqfhna/MatMul:product:0#while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/addÂ
'while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp2while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dhxpxqfhna/BiasAdd/ReadVariableOp½
while/dhxpxqfhna/BiasAddBiasAddwhile/dhxpxqfhna/add:z:0/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/BiasAdd
 while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dhxpxqfhna/split/split_dim
while/dhxpxqfhna/splitSplit)while/dhxpxqfhna/split/split_dim:output:0!while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dhxpxqfhna/split©
while/dhxpxqfhna/ReadVariableOpReadVariableOp*while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dhxpxqfhna/ReadVariableOp£
while/dhxpxqfhna/mulMul'while/dhxpxqfhna/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul¦
while/dhxpxqfhna/add_1AddV2while/dhxpxqfhna/split:output:0while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_1
while/dhxpxqfhna/SigmoidSigmoidwhile/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid¯
!while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp,while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_1©
while/dhxpxqfhna/mul_1Mul)while/dhxpxqfhna/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_1¨
while/dhxpxqfhna/add_2AddV2while/dhxpxqfhna/split:output:1while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_2
while/dhxpxqfhna/Sigmoid_1Sigmoidwhile/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_1
while/dhxpxqfhna/mul_2Mulwhile/dhxpxqfhna/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_2
while/dhxpxqfhna/TanhTanhwhile/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh¢
while/dhxpxqfhna/mul_3Mulwhile/dhxpxqfhna/Sigmoid:y:0while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_3£
while/dhxpxqfhna/add_3AddV2while/dhxpxqfhna/mul_2:z:0while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_3¯
!while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp,while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_2°
while/dhxpxqfhna/mul_4Mul)while/dhxpxqfhna/ReadVariableOp_2:value:0while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_4¨
while/dhxpxqfhna/add_4AddV2while/dhxpxqfhna/split:output:3while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_4
while/dhxpxqfhna/Sigmoid_2Sigmoidwhile/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_2
while/dhxpxqfhna/Tanh_1Tanhwhile/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh_1¦
while/dhxpxqfhna/mul_5Mulwhile/dhxpxqfhna/Sigmoid_2:y:0while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dhxpxqfhna/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dhxpxqfhna/mul_5:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dhxpxqfhna/add_3:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dhxpxqfhna_biasadd_readvariableop_resource2while_dhxpxqfhna_biasadd_readvariableop_resource_0"h
1while_dhxpxqfhna_matmul_1_readvariableop_resource3while_dhxpxqfhna_matmul_1_readvariableop_resource_0"d
/while_dhxpxqfhna_matmul_readvariableop_resource1while_dhxpxqfhna_matmul_readvariableop_resource_0"Z
*while_dhxpxqfhna_readvariableop_1_resource,while_dhxpxqfhna_readvariableop_1_resource_0"Z
*while_dhxpxqfhna_readvariableop_2_resource,while_dhxpxqfhna_readvariableop_2_resource_0"V
(while_dhxpxqfhna_readvariableop_resource*while_dhxpxqfhna_readvariableop_resource_0")
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
'while/dhxpxqfhna/BiasAdd/ReadVariableOp'while/dhxpxqfhna/BiasAdd/ReadVariableOp2P
&while/dhxpxqfhna/MatMul/ReadVariableOp&while/dhxpxqfhna/MatMul/ReadVariableOp2T
(while/dhxpxqfhna/MatMul_1/ReadVariableOp(while/dhxpxqfhna/MatMul_1/ReadVariableOp2B
while/dhxpxqfhna/ReadVariableOpwhile/dhxpxqfhna/ReadVariableOp2F
!while/dhxpxqfhna/ReadVariableOp_1!while/dhxpxqfhna/ReadVariableOp_12F
!while/dhxpxqfhna/ReadVariableOp_2!while/dhxpxqfhna/ReadVariableOp_2: 
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
while_body_1104353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dhxpxqfhna_matmul_readvariableop_resource_0:	F
3while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 A
2while_dhxpxqfhna_biasadd_readvariableop_resource_0:	8
*while_dhxpxqfhna_readvariableop_resource_0: :
,while_dhxpxqfhna_readvariableop_1_resource_0: :
,while_dhxpxqfhna_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dhxpxqfhna_matmul_readvariableop_resource:	D
1while_dhxpxqfhna_matmul_1_readvariableop_resource:	 ?
0while_dhxpxqfhna_biasadd_readvariableop_resource:	6
(while_dhxpxqfhna_readvariableop_resource: 8
*while_dhxpxqfhna_readvariableop_1_resource: 8
*while_dhxpxqfhna_readvariableop_2_resource: ¢'while/dhxpxqfhna/BiasAdd/ReadVariableOp¢&while/dhxpxqfhna/MatMul/ReadVariableOp¢(while/dhxpxqfhna/MatMul_1/ReadVariableOp¢while/dhxpxqfhna/ReadVariableOp¢!while/dhxpxqfhna/ReadVariableOp_1¢!while/dhxpxqfhna/ReadVariableOp_2Ã
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
&while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp1while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dhxpxqfhna/MatMul/ReadVariableOpÑ
while/dhxpxqfhna/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMulÉ
(while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp3while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dhxpxqfhna/MatMul_1/ReadVariableOpº
while/dhxpxqfhna/MatMul_1MatMulwhile_placeholder_20while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMul_1°
while/dhxpxqfhna/addAddV2!while/dhxpxqfhna/MatMul:product:0#while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/addÂ
'while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp2while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dhxpxqfhna/BiasAdd/ReadVariableOp½
while/dhxpxqfhna/BiasAddBiasAddwhile/dhxpxqfhna/add:z:0/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/BiasAdd
 while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dhxpxqfhna/split/split_dim
while/dhxpxqfhna/splitSplit)while/dhxpxqfhna/split/split_dim:output:0!while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dhxpxqfhna/split©
while/dhxpxqfhna/ReadVariableOpReadVariableOp*while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dhxpxqfhna/ReadVariableOp£
while/dhxpxqfhna/mulMul'while/dhxpxqfhna/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul¦
while/dhxpxqfhna/add_1AddV2while/dhxpxqfhna/split:output:0while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_1
while/dhxpxqfhna/SigmoidSigmoidwhile/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid¯
!while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp,while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_1©
while/dhxpxqfhna/mul_1Mul)while/dhxpxqfhna/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_1¨
while/dhxpxqfhna/add_2AddV2while/dhxpxqfhna/split:output:1while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_2
while/dhxpxqfhna/Sigmoid_1Sigmoidwhile/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_1
while/dhxpxqfhna/mul_2Mulwhile/dhxpxqfhna/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_2
while/dhxpxqfhna/TanhTanhwhile/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh¢
while/dhxpxqfhna/mul_3Mulwhile/dhxpxqfhna/Sigmoid:y:0while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_3£
while/dhxpxqfhna/add_3AddV2while/dhxpxqfhna/mul_2:z:0while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_3¯
!while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp,while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_2°
while/dhxpxqfhna/mul_4Mul)while/dhxpxqfhna/ReadVariableOp_2:value:0while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_4¨
while/dhxpxqfhna/add_4AddV2while/dhxpxqfhna/split:output:3while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_4
while/dhxpxqfhna/Sigmoid_2Sigmoidwhile/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_2
while/dhxpxqfhna/Tanh_1Tanhwhile/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh_1¦
while/dhxpxqfhna/mul_5Mulwhile/dhxpxqfhna/Sigmoid_2:y:0while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dhxpxqfhna/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dhxpxqfhna/mul_5:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dhxpxqfhna/add_3:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dhxpxqfhna_biasadd_readvariableop_resource2while_dhxpxqfhna_biasadd_readvariableop_resource_0"h
1while_dhxpxqfhna_matmul_1_readvariableop_resource3while_dhxpxqfhna_matmul_1_readvariableop_resource_0"d
/while_dhxpxqfhna_matmul_readvariableop_resource1while_dhxpxqfhna_matmul_readvariableop_resource_0"Z
*while_dhxpxqfhna_readvariableop_1_resource,while_dhxpxqfhna_readvariableop_1_resource_0"Z
*while_dhxpxqfhna_readvariableop_2_resource,while_dhxpxqfhna_readvariableop_2_resource_0"V
(while_dhxpxqfhna_readvariableop_resource*while_dhxpxqfhna_readvariableop_resource_0")
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
'while/dhxpxqfhna/BiasAdd/ReadVariableOp'while/dhxpxqfhna/BiasAdd/ReadVariableOp2P
&while/dhxpxqfhna/MatMul/ReadVariableOp&while/dhxpxqfhna/MatMul/ReadVariableOp2T
(while/dhxpxqfhna/MatMul_1/ReadVariableOp(while/dhxpxqfhna/MatMul_1/ReadVariableOp2B
while/dhxpxqfhna/ReadVariableOpwhile/dhxpxqfhna/ReadVariableOp2F
!while/dhxpxqfhna/ReadVariableOp_1!while/dhxpxqfhna/ReadVariableOp_12F
!while/dhxpxqfhna/ReadVariableOp_2!while/dhxpxqfhna/ReadVariableOp_2: 
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
¡h

G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1102336

inputs<
)dhxpxqfhna_matmul_readvariableop_resource:	>
+dhxpxqfhna_matmul_1_readvariableop_resource:	 9
*dhxpxqfhna_biasadd_readvariableop_resource:	0
"dhxpxqfhna_readvariableop_resource: 2
$dhxpxqfhna_readvariableop_1_resource: 2
$dhxpxqfhna_readvariableop_2_resource: 
identity¢!dhxpxqfhna/BiasAdd/ReadVariableOp¢ dhxpxqfhna/MatMul/ReadVariableOp¢"dhxpxqfhna/MatMul_1/ReadVariableOp¢dhxpxqfhna/ReadVariableOp¢dhxpxqfhna/ReadVariableOp_1¢dhxpxqfhna/ReadVariableOp_2¢whileD
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
 dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp)dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dhxpxqfhna/MatMul/ReadVariableOp§
dhxpxqfhna/MatMulMatMulstrided_slice_2:output:0(dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMulµ
"dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp+dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dhxpxqfhna/MatMul_1/ReadVariableOp£
dhxpxqfhna/MatMul_1MatMulzeros:output:0*dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMul_1
dhxpxqfhna/addAddV2dhxpxqfhna/MatMul:product:0dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/add®
!dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp*dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dhxpxqfhna/BiasAdd/ReadVariableOp¥
dhxpxqfhna/BiasAddBiasAdddhxpxqfhna/add:z:0)dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/BiasAddz
dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dhxpxqfhna/split/split_dimë
dhxpxqfhna/splitSplit#dhxpxqfhna/split/split_dim:output:0dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dhxpxqfhna/split
dhxpxqfhna/ReadVariableOpReadVariableOp"dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp
dhxpxqfhna/mulMul!dhxpxqfhna/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul
dhxpxqfhna/add_1AddV2dhxpxqfhna/split:output:0dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_1{
dhxpxqfhna/SigmoidSigmoiddhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid
dhxpxqfhna/ReadVariableOp_1ReadVariableOp$dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_1
dhxpxqfhna/mul_1Mul#dhxpxqfhna/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_1
dhxpxqfhna/add_2AddV2dhxpxqfhna/split:output:1dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_2
dhxpxqfhna/Sigmoid_1Sigmoiddhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_1
dhxpxqfhna/mul_2Muldhxpxqfhna/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_2w
dhxpxqfhna/TanhTanhdhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh
dhxpxqfhna/mul_3Muldhxpxqfhna/Sigmoid:y:0dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_3
dhxpxqfhna/add_3AddV2dhxpxqfhna/mul_2:z:0dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_3
dhxpxqfhna/ReadVariableOp_2ReadVariableOp$dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_2
dhxpxqfhna/mul_4Mul#dhxpxqfhna/ReadVariableOp_2:value:0dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_4
dhxpxqfhna/add_4AddV2dhxpxqfhna/split:output:3dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_4
dhxpxqfhna/Sigmoid_2Sigmoiddhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_2v
dhxpxqfhna/Tanh_1Tanhdhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh_1
dhxpxqfhna/mul_5Muldhxpxqfhna/Sigmoid_2:y:0dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dhxpxqfhna_matmul_readvariableop_resource+dhxpxqfhna_matmul_1_readvariableop_resource*dhxpxqfhna_biasadd_readvariableop_resource"dhxpxqfhna_readvariableop_resource$dhxpxqfhna_readvariableop_1_resource$dhxpxqfhna_readvariableop_2_resource*
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
while_body_1102235*
condR
while_cond_1102234*Q
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
IdentityIdentitytranspose_1:y:0"^dhxpxqfhna/BiasAdd/ReadVariableOp!^dhxpxqfhna/MatMul/ReadVariableOp#^dhxpxqfhna/MatMul_1/ReadVariableOp^dhxpxqfhna/ReadVariableOp^dhxpxqfhna/ReadVariableOp_1^dhxpxqfhna/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dhxpxqfhna/BiasAdd/ReadVariableOp!dhxpxqfhna/BiasAdd/ReadVariableOp2D
 dhxpxqfhna/MatMul/ReadVariableOp dhxpxqfhna/MatMul/ReadVariableOp2H
"dhxpxqfhna/MatMul_1/ReadVariableOp"dhxpxqfhna/MatMul_1/ReadVariableOp26
dhxpxqfhna/ReadVariableOpdhxpxqfhna/ReadVariableOp2:
dhxpxqfhna/ReadVariableOp_1dhxpxqfhna/ReadVariableOp_12:
dhxpxqfhna/ReadVariableOp_2dhxpxqfhna/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
¼
G__inference_sequential_layer_call_and_return_conditional_losses_1103129

inputs(
kjxhlaztnm_1103091: 
kjxhlaztnm_1103093:%
lctpanrywj_1103097:	%
lctpanrywj_1103099:	 !
lctpanrywj_1103101:	 
lctpanrywj_1103103:  
lctpanrywj_1103105:  
lctpanrywj_1103107: %
rienwrhgrh_1103110:	 %
rienwrhgrh_1103112:	 !
rienwrhgrh_1103114:	 
rienwrhgrh_1103116:  
rienwrhgrh_1103118:  
rienwrhgrh_1103120: $
uilnjhxhrx_1103123:  
uilnjhxhrx_1103125:
identity¢"kjxhlaztnm/StatefulPartitionedCall¢"lctpanrywj/StatefulPartitionedCall¢"rienwrhgrh/StatefulPartitionedCall¢"uilnjhxhrx/StatefulPartitionedCall¬
"kjxhlaztnm/StatefulPartitionedCallStatefulPartitionedCallinputskjxhlaztnm_1103091kjxhlaztnm_1103093*
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
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_11021362$
"kjxhlaztnm/StatefulPartitionedCall
tzzrzfazij/PartitionedCallPartitionedCall+kjxhlaztnm/StatefulPartitionedCall:output:0*
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
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_11021552
tzzrzfazij/PartitionedCall
"lctpanrywj/StatefulPartitionedCallStatefulPartitionedCall#tzzrzfazij/PartitionedCall:output:0lctpanrywj_1103097lctpanrywj_1103099lctpanrywj_1103101lctpanrywj_1103103lctpanrywj_1103105lctpanrywj_1103107*
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11030182$
"lctpanrywj/StatefulPartitionedCall¡
"rienwrhgrh/StatefulPartitionedCallStatefulPartitionedCall+lctpanrywj/StatefulPartitionedCall:output:0rienwrhgrh_1103110rienwrhgrh_1103112rienwrhgrh_1103114rienwrhgrh_1103116rienwrhgrh_1103118rienwrhgrh_1103120*
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11028042$
"rienwrhgrh/StatefulPartitionedCallÉ
"uilnjhxhrx/StatefulPartitionedCallStatefulPartitionedCall+rienwrhgrh/StatefulPartitionedCall:output:0uilnjhxhrx_1103123uilnjhxhrx_1103125*
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
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_11025532$
"uilnjhxhrx/StatefulPartitionedCall
IdentityIdentity+uilnjhxhrx/StatefulPartitionedCall:output:0#^kjxhlaztnm/StatefulPartitionedCall#^lctpanrywj/StatefulPartitionedCall#^rienwrhgrh/StatefulPartitionedCall#^uilnjhxhrx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kjxhlaztnm/StatefulPartitionedCall"kjxhlaztnm/StatefulPartitionedCall2H
"lctpanrywj/StatefulPartitionedCall"lctpanrywj/StatefulPartitionedCall2H
"rienwrhgrh/StatefulPartitionedCall"rienwrhgrh/StatefulPartitionedCall2H
"uilnjhxhrx/StatefulPartitionedCall"uilnjhxhrx/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹'
µ
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1105913

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


í
while_cond_1102916
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1102916___redundant_placeholder05
1while_while_cond_1102916___redundant_placeholder15
1while_while_cond_1102916___redundant_placeholder25
1while_while_cond_1102916___redundant_placeholder35
1while_while_cond_1102916___redundant_placeholder45
1while_while_cond_1102916___redundant_placeholder55
1while_while_cond_1102916___redundant_placeholder6
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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1101608

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
while_cond_1102234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1102234___redundant_placeholder05
1while_while_cond_1102234___redundant_placeholder15
1while_while_cond_1102234___redundant_placeholder25
1while_while_cond_1102234___redundant_placeholder35
1while_while_cond_1102234___redundant_placeholder45
1while_while_cond_1102234___redundant_placeholder55
1while_while_cond_1102234___redundant_placeholder6
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
while_body_1102235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dhxpxqfhna_matmul_readvariableop_resource_0:	F
3while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 A
2while_dhxpxqfhna_biasadd_readvariableop_resource_0:	8
*while_dhxpxqfhna_readvariableop_resource_0: :
,while_dhxpxqfhna_readvariableop_1_resource_0: :
,while_dhxpxqfhna_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dhxpxqfhna_matmul_readvariableop_resource:	D
1while_dhxpxqfhna_matmul_1_readvariableop_resource:	 ?
0while_dhxpxqfhna_biasadd_readvariableop_resource:	6
(while_dhxpxqfhna_readvariableop_resource: 8
*while_dhxpxqfhna_readvariableop_1_resource: 8
*while_dhxpxqfhna_readvariableop_2_resource: ¢'while/dhxpxqfhna/BiasAdd/ReadVariableOp¢&while/dhxpxqfhna/MatMul/ReadVariableOp¢(while/dhxpxqfhna/MatMul_1/ReadVariableOp¢while/dhxpxqfhna/ReadVariableOp¢!while/dhxpxqfhna/ReadVariableOp_1¢!while/dhxpxqfhna/ReadVariableOp_2Ã
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
&while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp1while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dhxpxqfhna/MatMul/ReadVariableOpÑ
while/dhxpxqfhna/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMulÉ
(while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp3while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dhxpxqfhna/MatMul_1/ReadVariableOpº
while/dhxpxqfhna/MatMul_1MatMulwhile_placeholder_20while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMul_1°
while/dhxpxqfhna/addAddV2!while/dhxpxqfhna/MatMul:product:0#while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/addÂ
'while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp2while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dhxpxqfhna/BiasAdd/ReadVariableOp½
while/dhxpxqfhna/BiasAddBiasAddwhile/dhxpxqfhna/add:z:0/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/BiasAdd
 while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dhxpxqfhna/split/split_dim
while/dhxpxqfhna/splitSplit)while/dhxpxqfhna/split/split_dim:output:0!while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dhxpxqfhna/split©
while/dhxpxqfhna/ReadVariableOpReadVariableOp*while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dhxpxqfhna/ReadVariableOp£
while/dhxpxqfhna/mulMul'while/dhxpxqfhna/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul¦
while/dhxpxqfhna/add_1AddV2while/dhxpxqfhna/split:output:0while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_1
while/dhxpxqfhna/SigmoidSigmoidwhile/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid¯
!while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp,while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_1©
while/dhxpxqfhna/mul_1Mul)while/dhxpxqfhna/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_1¨
while/dhxpxqfhna/add_2AddV2while/dhxpxqfhna/split:output:1while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_2
while/dhxpxqfhna/Sigmoid_1Sigmoidwhile/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_1
while/dhxpxqfhna/mul_2Mulwhile/dhxpxqfhna/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_2
while/dhxpxqfhna/TanhTanhwhile/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh¢
while/dhxpxqfhna/mul_3Mulwhile/dhxpxqfhna/Sigmoid:y:0while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_3£
while/dhxpxqfhna/add_3AddV2while/dhxpxqfhna/mul_2:z:0while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_3¯
!while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp,while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_2°
while/dhxpxqfhna/mul_4Mul)while/dhxpxqfhna/ReadVariableOp_2:value:0while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_4¨
while/dhxpxqfhna/add_4AddV2while/dhxpxqfhna/split:output:3while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_4
while/dhxpxqfhna/Sigmoid_2Sigmoidwhile/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_2
while/dhxpxqfhna/Tanh_1Tanhwhile/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh_1¦
while/dhxpxqfhna/mul_5Mulwhile/dhxpxqfhna/Sigmoid_2:y:0while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dhxpxqfhna/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dhxpxqfhna/mul_5:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dhxpxqfhna/add_3:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dhxpxqfhna_biasadd_readvariableop_resource2while_dhxpxqfhna_biasadd_readvariableop_resource_0"h
1while_dhxpxqfhna_matmul_1_readvariableop_resource3while_dhxpxqfhna_matmul_1_readvariableop_resource_0"d
/while_dhxpxqfhna_matmul_readvariableop_resource1while_dhxpxqfhna_matmul_readvariableop_resource_0"Z
*while_dhxpxqfhna_readvariableop_1_resource,while_dhxpxqfhna_readvariableop_1_resource_0"Z
*while_dhxpxqfhna_readvariableop_2_resource,while_dhxpxqfhna_readvariableop_2_resource_0"V
(while_dhxpxqfhna_readvariableop_resource*while_dhxpxqfhna_readvariableop_resource_0")
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
'while/dhxpxqfhna/BiasAdd/ReadVariableOp'while/dhxpxqfhna/BiasAdd/ReadVariableOp2P
&while/dhxpxqfhna/MatMul/ReadVariableOp&while/dhxpxqfhna/MatMul/ReadVariableOp2T
(while/dhxpxqfhna/MatMul_1/ReadVariableOp(while/dhxpxqfhna/MatMul_1/ReadVariableOp2B
while/dhxpxqfhna/ReadVariableOpwhile/dhxpxqfhna/ReadVariableOp2F
!while/dhxpxqfhna/ReadVariableOp_1!while/dhxpxqfhna/ReadVariableOp_12F
!while/dhxpxqfhna/ReadVariableOp_2!while/dhxpxqfhna/ReadVariableOp_2: 
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
while_body_1104893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dhxpxqfhna_matmul_readvariableop_resource_0:	F
3while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 A
2while_dhxpxqfhna_biasadd_readvariableop_resource_0:	8
*while_dhxpxqfhna_readvariableop_resource_0: :
,while_dhxpxqfhna_readvariableop_1_resource_0: :
,while_dhxpxqfhna_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dhxpxqfhna_matmul_readvariableop_resource:	D
1while_dhxpxqfhna_matmul_1_readvariableop_resource:	 ?
0while_dhxpxqfhna_biasadd_readvariableop_resource:	6
(while_dhxpxqfhna_readvariableop_resource: 8
*while_dhxpxqfhna_readvariableop_1_resource: 8
*while_dhxpxqfhna_readvariableop_2_resource: ¢'while/dhxpxqfhna/BiasAdd/ReadVariableOp¢&while/dhxpxqfhna/MatMul/ReadVariableOp¢(while/dhxpxqfhna/MatMul_1/ReadVariableOp¢while/dhxpxqfhna/ReadVariableOp¢!while/dhxpxqfhna/ReadVariableOp_1¢!while/dhxpxqfhna/ReadVariableOp_2Ã
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
&while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp1while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dhxpxqfhna/MatMul/ReadVariableOpÑ
while/dhxpxqfhna/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMulÉ
(while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp3while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dhxpxqfhna/MatMul_1/ReadVariableOpº
while/dhxpxqfhna/MatMul_1MatMulwhile_placeholder_20while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMul_1°
while/dhxpxqfhna/addAddV2!while/dhxpxqfhna/MatMul:product:0#while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/addÂ
'while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp2while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dhxpxqfhna/BiasAdd/ReadVariableOp½
while/dhxpxqfhna/BiasAddBiasAddwhile/dhxpxqfhna/add:z:0/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/BiasAdd
 while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dhxpxqfhna/split/split_dim
while/dhxpxqfhna/splitSplit)while/dhxpxqfhna/split/split_dim:output:0!while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dhxpxqfhna/split©
while/dhxpxqfhna/ReadVariableOpReadVariableOp*while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dhxpxqfhna/ReadVariableOp£
while/dhxpxqfhna/mulMul'while/dhxpxqfhna/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul¦
while/dhxpxqfhna/add_1AddV2while/dhxpxqfhna/split:output:0while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_1
while/dhxpxqfhna/SigmoidSigmoidwhile/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid¯
!while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp,while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_1©
while/dhxpxqfhna/mul_1Mul)while/dhxpxqfhna/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_1¨
while/dhxpxqfhna/add_2AddV2while/dhxpxqfhna/split:output:1while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_2
while/dhxpxqfhna/Sigmoid_1Sigmoidwhile/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_1
while/dhxpxqfhna/mul_2Mulwhile/dhxpxqfhna/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_2
while/dhxpxqfhna/TanhTanhwhile/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh¢
while/dhxpxqfhna/mul_3Mulwhile/dhxpxqfhna/Sigmoid:y:0while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_3£
while/dhxpxqfhna/add_3AddV2while/dhxpxqfhna/mul_2:z:0while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_3¯
!while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp,while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_2°
while/dhxpxqfhna/mul_4Mul)while/dhxpxqfhna/ReadVariableOp_2:value:0while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_4¨
while/dhxpxqfhna/add_4AddV2while/dhxpxqfhna/split:output:3while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_4
while/dhxpxqfhna/Sigmoid_2Sigmoidwhile/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_2
while/dhxpxqfhna/Tanh_1Tanhwhile/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh_1¦
while/dhxpxqfhna/mul_5Mulwhile/dhxpxqfhna/Sigmoid_2:y:0while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dhxpxqfhna/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dhxpxqfhna/mul_5:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dhxpxqfhna/add_3:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dhxpxqfhna_biasadd_readvariableop_resource2while_dhxpxqfhna_biasadd_readvariableop_resource_0"h
1while_dhxpxqfhna_matmul_1_readvariableop_resource3while_dhxpxqfhna_matmul_1_readvariableop_resource_0"d
/while_dhxpxqfhna_matmul_readvariableop_resource1while_dhxpxqfhna_matmul_readvariableop_resource_0"Z
*while_dhxpxqfhna_readvariableop_1_resource,while_dhxpxqfhna_readvariableop_1_resource_0"Z
*while_dhxpxqfhna_readvariableop_2_resource,while_dhxpxqfhna_readvariableop_2_resource_0"V
(while_dhxpxqfhna_readvariableop_resource*while_dhxpxqfhna_readvariableop_resource_0")
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
'while/dhxpxqfhna/BiasAdd/ReadVariableOp'while/dhxpxqfhna/BiasAdd/ReadVariableOp2P
&while/dhxpxqfhna/MatMul/ReadVariableOp&while/dhxpxqfhna/MatMul/ReadVariableOp2T
(while/dhxpxqfhna/MatMul_1/ReadVariableOp(while/dhxpxqfhna/MatMul_1/ReadVariableOp2B
while/dhxpxqfhna/ReadVariableOpwhile/dhxpxqfhna/ReadVariableOp2F
!while/dhxpxqfhna/ReadVariableOp_1!while/dhxpxqfhna/ReadVariableOp_12F
!while/dhxpxqfhna/ReadVariableOp_2!while/dhxpxqfhna/ReadVariableOp_2: 
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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1103732

inputsL
6kjxhlaztnm_conv1d_expanddims_1_readvariableop_resource:K
=kjxhlaztnm_squeeze_batch_dims_biasadd_readvariableop_resource:G
4lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource:	I
6lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource:	 D
5lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource:	;
-lctpanrywj_dhxpxqfhna_readvariableop_resource: =
/lctpanrywj_dhxpxqfhna_readvariableop_1_resource: =
/lctpanrywj_dhxpxqfhna_readvariableop_2_resource: G
4rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource:	 I
6rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource:	 D
5rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource:	;
-rienwrhgrh_kngiiuzftt_readvariableop_resource: =
/rienwrhgrh_kngiiuzftt_readvariableop_1_resource: =
/rienwrhgrh_kngiiuzftt_readvariableop_2_resource: ;
)uilnjhxhrx_matmul_readvariableop_resource: 8
*uilnjhxhrx_biasadd_readvariableop_resource:
identity¢-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp¢4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp¢+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp¢-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp¢$lctpanrywj/dhxpxqfhna/ReadVariableOp¢&lctpanrywj/dhxpxqfhna/ReadVariableOp_1¢&lctpanrywj/dhxpxqfhna/ReadVariableOp_2¢lctpanrywj/while¢,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp¢+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp¢-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp¢$rienwrhgrh/kngiiuzftt/ReadVariableOp¢&rienwrhgrh/kngiiuzftt/ReadVariableOp_1¢&rienwrhgrh/kngiiuzftt/ReadVariableOp_2¢rienwrhgrh/while¢!uilnjhxhrx/BiasAdd/ReadVariableOp¢ uilnjhxhrx/MatMul/ReadVariableOp
 kjxhlaztnm/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 kjxhlaztnm/conv1d/ExpandDims/dim»
kjxhlaztnm/conv1d/ExpandDims
ExpandDimsinputs)kjxhlaztnm/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/ExpandDimsÙ
-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6kjxhlaztnm_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp
"kjxhlaztnm/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"kjxhlaztnm/conv1d/ExpandDims_1/dimã
kjxhlaztnm/conv1d/ExpandDims_1
ExpandDims5kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp:value:0+kjxhlaztnm/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
kjxhlaztnm/conv1d/ExpandDims_1
kjxhlaztnm/conv1d/ShapeShape%kjxhlaztnm/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
kjxhlaztnm/conv1d/Shape
%kjxhlaztnm/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%kjxhlaztnm/conv1d/strided_slice/stack¥
'kjxhlaztnm/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'kjxhlaztnm/conv1d/strided_slice/stack_1
'kjxhlaztnm/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'kjxhlaztnm/conv1d/strided_slice/stack_2Ì
kjxhlaztnm/conv1d/strided_sliceStridedSlice kjxhlaztnm/conv1d/Shape:output:0.kjxhlaztnm/conv1d/strided_slice/stack:output:00kjxhlaztnm/conv1d/strided_slice/stack_1:output:00kjxhlaztnm/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
kjxhlaztnm/conv1d/strided_slice
kjxhlaztnm/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
kjxhlaztnm/conv1d/Reshape/shapeÌ
kjxhlaztnm/conv1d/ReshapeReshape%kjxhlaztnm/conv1d/ExpandDims:output:0(kjxhlaztnm/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/Reshapeî
kjxhlaztnm/conv1d/Conv2DConv2D"kjxhlaztnm/conv1d/Reshape:output:0'kjxhlaztnm/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
kjxhlaztnm/conv1d/Conv2D
!kjxhlaztnm/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!kjxhlaztnm/conv1d/concat/values_1
kjxhlaztnm/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/concat/axisì
kjxhlaztnm/conv1d/concatConcatV2(kjxhlaztnm/conv1d/strided_slice:output:0*kjxhlaztnm/conv1d/concat/values_1:output:0&kjxhlaztnm/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
kjxhlaztnm/conv1d/concatÉ
kjxhlaztnm/conv1d/Reshape_1Reshape!kjxhlaztnm/conv1d/Conv2D:output:0!kjxhlaztnm/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/Reshape_1Á
kjxhlaztnm/conv1d/SqueezeSqueeze$kjxhlaztnm/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
kjxhlaztnm/conv1d/Squeeze
#kjxhlaztnm/squeeze_batch_dims/ShapeShape"kjxhlaztnm/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#kjxhlaztnm/squeeze_batch_dims/Shape°
1kjxhlaztnm/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1kjxhlaztnm/squeeze_batch_dims/strided_slice/stack½
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1´
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2
+kjxhlaztnm/squeeze_batch_dims/strided_sliceStridedSlice,kjxhlaztnm/squeeze_batch_dims/Shape:output:0:kjxhlaztnm/squeeze_batch_dims/strided_slice/stack:output:0<kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_1:output:0<kjxhlaztnm/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+kjxhlaztnm/squeeze_batch_dims/strided_slice¯
+kjxhlaztnm/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+kjxhlaztnm/squeeze_batch_dims/Reshape/shapeé
%kjxhlaztnm/squeeze_batch_dims/ReshapeReshape"kjxhlaztnm/conv1d/Squeeze:output:04kjxhlaztnm/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%kjxhlaztnm/squeeze_batch_dims/Reshapeæ
4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=kjxhlaztnm_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%kjxhlaztnm/squeeze_batch_dims/BiasAddBiasAdd.kjxhlaztnm/squeeze_batch_dims/Reshape:output:0<kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%kjxhlaztnm/squeeze_batch_dims/BiasAdd¯
-kjxhlaztnm/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-kjxhlaztnm/squeeze_batch_dims/concat/values_1¡
)kjxhlaztnm/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)kjxhlaztnm/squeeze_batch_dims/concat/axis¨
$kjxhlaztnm/squeeze_batch_dims/concatConcatV24kjxhlaztnm/squeeze_batch_dims/strided_slice:output:06kjxhlaztnm/squeeze_batch_dims/concat/values_1:output:02kjxhlaztnm/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$kjxhlaztnm/squeeze_batch_dims/concatö
'kjxhlaztnm/squeeze_batch_dims/Reshape_1Reshape.kjxhlaztnm/squeeze_batch_dims/BiasAdd:output:0-kjxhlaztnm/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'kjxhlaztnm/squeeze_batch_dims/Reshape_1
tzzrzfazij/ShapeShape0kjxhlaztnm/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
tzzrzfazij/Shape
tzzrzfazij/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
tzzrzfazij/strided_slice/stack
 tzzrzfazij/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 tzzrzfazij/strided_slice/stack_1
 tzzrzfazij/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 tzzrzfazij/strided_slice/stack_2¤
tzzrzfazij/strided_sliceStridedSlicetzzrzfazij/Shape:output:0'tzzrzfazij/strided_slice/stack:output:0)tzzrzfazij/strided_slice/stack_1:output:0)tzzrzfazij/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
tzzrzfazij/strided_slicez
tzzrzfazij/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tzzrzfazij/Reshape/shape/1z
tzzrzfazij/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tzzrzfazij/Reshape/shape/2×
tzzrzfazij/Reshape/shapePack!tzzrzfazij/strided_slice:output:0#tzzrzfazij/Reshape/shape/1:output:0#tzzrzfazij/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
tzzrzfazij/Reshape/shape¾
tzzrzfazij/ReshapeReshape0kjxhlaztnm/squeeze_batch_dims/Reshape_1:output:0!tzzrzfazij/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tzzrzfazij/Reshapeo
lctpanrywj/ShapeShapetzzrzfazij/Reshape:output:0*
T0*
_output_shapes
:2
lctpanrywj/Shape
lctpanrywj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
lctpanrywj/strided_slice/stack
 lctpanrywj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 lctpanrywj/strided_slice/stack_1
 lctpanrywj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 lctpanrywj/strided_slice/stack_2¤
lctpanrywj/strided_sliceStridedSlicelctpanrywj/Shape:output:0'lctpanrywj/strided_slice/stack:output:0)lctpanrywj/strided_slice/stack_1:output:0)lctpanrywj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lctpanrywj/strided_slicer
lctpanrywj/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros/mul/y
lctpanrywj/zeros/mulMul!lctpanrywj/strided_slice:output:0lctpanrywj/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros/mulu
lctpanrywj/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lctpanrywj/zeros/Less/y
lctpanrywj/zeros/LessLesslctpanrywj/zeros/mul:z:0 lctpanrywj/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros/Lessx
lctpanrywj/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros/packed/1¯
lctpanrywj/zeros/packedPack!lctpanrywj/strided_slice:output:0"lctpanrywj/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lctpanrywj/zeros/packedu
lctpanrywj/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lctpanrywj/zeros/Const¡
lctpanrywj/zerosFill lctpanrywj/zeros/packed:output:0lctpanrywj/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/zerosv
lctpanrywj/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros_1/mul/y
lctpanrywj/zeros_1/mulMul!lctpanrywj/strided_slice:output:0!lctpanrywj/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros_1/muly
lctpanrywj/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lctpanrywj/zeros_1/Less/y
lctpanrywj/zeros_1/LessLesslctpanrywj/zeros_1/mul:z:0"lctpanrywj/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lctpanrywj/zeros_1/Less|
lctpanrywj/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/zeros_1/packed/1µ
lctpanrywj/zeros_1/packedPack!lctpanrywj/strided_slice:output:0$lctpanrywj/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lctpanrywj/zeros_1/packedy
lctpanrywj/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lctpanrywj/zeros_1/Const©
lctpanrywj/zeros_1Fill"lctpanrywj/zeros_1/packed:output:0!lctpanrywj/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/zeros_1
lctpanrywj/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lctpanrywj/transpose/perm°
lctpanrywj/transpose	Transposetzzrzfazij/Reshape:output:0"lctpanrywj/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/transposep
lctpanrywj/Shape_1Shapelctpanrywj/transpose:y:0*
T0*
_output_shapes
:2
lctpanrywj/Shape_1
 lctpanrywj/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 lctpanrywj/strided_slice_1/stack
"lctpanrywj/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_1/stack_1
"lctpanrywj/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_1/stack_2°
lctpanrywj/strided_slice_1StridedSlicelctpanrywj/Shape_1:output:0)lctpanrywj/strided_slice_1/stack:output:0+lctpanrywj/strided_slice_1/stack_1:output:0+lctpanrywj/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lctpanrywj/strided_slice_1
&lctpanrywj/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&lctpanrywj/TensorArrayV2/element_shapeÞ
lctpanrywj/TensorArrayV2TensorListReserve/lctpanrywj/TensorArrayV2/element_shape:output:0#lctpanrywj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lctpanrywj/TensorArrayV2Õ
@lctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@lctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2lctpanrywj/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlctpanrywj/transpose:y:0Ilctpanrywj/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2lctpanrywj/TensorArrayUnstack/TensorListFromTensor
 lctpanrywj/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 lctpanrywj/strided_slice_2/stack
"lctpanrywj/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_2/stack_1
"lctpanrywj/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_2/stack_2¾
lctpanrywj/strided_slice_2StridedSlicelctpanrywj/transpose:y:0)lctpanrywj/strided_slice_2/stack:output:0+lctpanrywj/strided_slice_2/stack_1:output:0+lctpanrywj/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lctpanrywj/strided_slice_2Ð
+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp4lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOpÓ
lctpanrywj/dhxpxqfhna/MatMulMatMul#lctpanrywj/strided_slice_2:output:03lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/dhxpxqfhna/MatMulÖ
-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp6lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOpÏ
lctpanrywj/dhxpxqfhna/MatMul_1MatMullctpanrywj/zeros:output:05lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lctpanrywj/dhxpxqfhna/MatMul_1Ä
lctpanrywj/dhxpxqfhna/addAddV2&lctpanrywj/dhxpxqfhna/MatMul:product:0(lctpanrywj/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/dhxpxqfhna/addÏ
,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp5lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOpÑ
lctpanrywj/dhxpxqfhna/BiasAddBiasAddlctpanrywj/dhxpxqfhna/add:z:04lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lctpanrywj/dhxpxqfhna/BiasAdd
%lctpanrywj/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%lctpanrywj/dhxpxqfhna/split/split_dim
lctpanrywj/dhxpxqfhna/splitSplit.lctpanrywj/dhxpxqfhna/split/split_dim:output:0&lctpanrywj/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lctpanrywj/dhxpxqfhna/split¶
$lctpanrywj/dhxpxqfhna/ReadVariableOpReadVariableOp-lctpanrywj_dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02&
$lctpanrywj/dhxpxqfhna/ReadVariableOpº
lctpanrywj/dhxpxqfhna/mulMul,lctpanrywj/dhxpxqfhna/ReadVariableOp:value:0lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mulº
lctpanrywj/dhxpxqfhna/add_1AddV2$lctpanrywj/dhxpxqfhna/split:output:0lctpanrywj/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_1
lctpanrywj/dhxpxqfhna/SigmoidSigmoidlctpanrywj/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/Sigmoid¼
&lctpanrywj/dhxpxqfhna/ReadVariableOp_1ReadVariableOp/lctpanrywj_dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&lctpanrywj/dhxpxqfhna/ReadVariableOp_1À
lctpanrywj/dhxpxqfhna/mul_1Mul.lctpanrywj/dhxpxqfhna/ReadVariableOp_1:value:0lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_1¼
lctpanrywj/dhxpxqfhna/add_2AddV2$lctpanrywj/dhxpxqfhna/split:output:1lctpanrywj/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_2 
lctpanrywj/dhxpxqfhna/Sigmoid_1Sigmoidlctpanrywj/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lctpanrywj/dhxpxqfhna/Sigmoid_1µ
lctpanrywj/dhxpxqfhna/mul_2Mul#lctpanrywj/dhxpxqfhna/Sigmoid_1:y:0lctpanrywj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_2
lctpanrywj/dhxpxqfhna/TanhTanh$lctpanrywj/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/Tanh¶
lctpanrywj/dhxpxqfhna/mul_3Mul!lctpanrywj/dhxpxqfhna/Sigmoid:y:0lctpanrywj/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_3·
lctpanrywj/dhxpxqfhna/add_3AddV2lctpanrywj/dhxpxqfhna/mul_2:z:0lctpanrywj/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_3¼
&lctpanrywj/dhxpxqfhna/ReadVariableOp_2ReadVariableOp/lctpanrywj_dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&lctpanrywj/dhxpxqfhna/ReadVariableOp_2Ä
lctpanrywj/dhxpxqfhna/mul_4Mul.lctpanrywj/dhxpxqfhna/ReadVariableOp_2:value:0lctpanrywj/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_4¼
lctpanrywj/dhxpxqfhna/add_4AddV2$lctpanrywj/dhxpxqfhna/split:output:3lctpanrywj/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/add_4 
lctpanrywj/dhxpxqfhna/Sigmoid_2Sigmoidlctpanrywj/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
lctpanrywj/dhxpxqfhna/Sigmoid_2
lctpanrywj/dhxpxqfhna/Tanh_1Tanhlctpanrywj/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/Tanh_1º
lctpanrywj/dhxpxqfhna/mul_5Mul#lctpanrywj/dhxpxqfhna/Sigmoid_2:y:0 lctpanrywj/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/dhxpxqfhna/mul_5¥
(lctpanrywj/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(lctpanrywj/TensorArrayV2_1/element_shapeä
lctpanrywj/TensorArrayV2_1TensorListReserve1lctpanrywj/TensorArrayV2_1/element_shape:output:0#lctpanrywj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lctpanrywj/TensorArrayV2_1d
lctpanrywj/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/time
#lctpanrywj/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lctpanrywj/while/maximum_iterations
lctpanrywj/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lctpanrywj/while/loop_counter²
lctpanrywj/whileWhile&lctpanrywj/while/loop_counter:output:0,lctpanrywj/while/maximum_iterations:output:0lctpanrywj/time:output:0#lctpanrywj/TensorArrayV2_1:handle:0lctpanrywj/zeros:output:0lctpanrywj/zeros_1:output:0#lctpanrywj/strided_slice_1:output:0Blctpanrywj/TensorArrayUnstack/TensorListFromTensor:output_handle:04lctpanrywj_dhxpxqfhna_matmul_readvariableop_resource6lctpanrywj_dhxpxqfhna_matmul_1_readvariableop_resource5lctpanrywj_dhxpxqfhna_biasadd_readvariableop_resource-lctpanrywj_dhxpxqfhna_readvariableop_resource/lctpanrywj_dhxpxqfhna_readvariableop_1_resource/lctpanrywj_dhxpxqfhna_readvariableop_2_resource*
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
lctpanrywj_while_body_1103449*)
cond!R
lctpanrywj_while_cond_1103448*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
lctpanrywj/whileË
;lctpanrywj/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;lctpanrywj/TensorArrayV2Stack/TensorListStack/element_shape
-lctpanrywj/TensorArrayV2Stack/TensorListStackTensorListStacklctpanrywj/while:output:3Dlctpanrywj/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-lctpanrywj/TensorArrayV2Stack/TensorListStack
 lctpanrywj/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 lctpanrywj/strided_slice_3/stack
"lctpanrywj/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"lctpanrywj/strided_slice_3/stack_1
"lctpanrywj/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"lctpanrywj/strided_slice_3/stack_2Ü
lctpanrywj/strided_slice_3StridedSlice6lctpanrywj/TensorArrayV2Stack/TensorListStack:tensor:0)lctpanrywj/strided_slice_3/stack:output:0+lctpanrywj/strided_slice_3/stack_1:output:0+lctpanrywj/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
lctpanrywj/strided_slice_3
lctpanrywj/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lctpanrywj/transpose_1/permÑ
lctpanrywj/transpose_1	Transpose6lctpanrywj/TensorArrayV2Stack/TensorListStack:tensor:0$lctpanrywj/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lctpanrywj/transpose_1n
rienwrhgrh/ShapeShapelctpanrywj/transpose_1:y:0*
T0*
_output_shapes
:2
rienwrhgrh/Shape
rienwrhgrh/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
rienwrhgrh/strided_slice/stack
 rienwrhgrh/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 rienwrhgrh/strided_slice/stack_1
 rienwrhgrh/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 rienwrhgrh/strided_slice/stack_2¤
rienwrhgrh/strided_sliceStridedSlicerienwrhgrh/Shape:output:0'rienwrhgrh/strided_slice/stack:output:0)rienwrhgrh/strided_slice/stack_1:output:0)rienwrhgrh/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rienwrhgrh/strided_slicer
rienwrhgrh/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros/mul/y
rienwrhgrh/zeros/mulMul!rienwrhgrh/strided_slice:output:0rienwrhgrh/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros/mulu
rienwrhgrh/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rienwrhgrh/zeros/Less/y
rienwrhgrh/zeros/LessLessrienwrhgrh/zeros/mul:z:0 rienwrhgrh/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros/Lessx
rienwrhgrh/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros/packed/1¯
rienwrhgrh/zeros/packedPack!rienwrhgrh/strided_slice:output:0"rienwrhgrh/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rienwrhgrh/zeros/packedu
rienwrhgrh/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rienwrhgrh/zeros/Const¡
rienwrhgrh/zerosFill rienwrhgrh/zeros/packed:output:0rienwrhgrh/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/zerosv
rienwrhgrh/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros_1/mul/y
rienwrhgrh/zeros_1/mulMul!rienwrhgrh/strided_slice:output:0!rienwrhgrh/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros_1/muly
rienwrhgrh/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rienwrhgrh/zeros_1/Less/y
rienwrhgrh/zeros_1/LessLessrienwrhgrh/zeros_1/mul:z:0"rienwrhgrh/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rienwrhgrh/zeros_1/Less|
rienwrhgrh/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/zeros_1/packed/1µ
rienwrhgrh/zeros_1/packedPack!rienwrhgrh/strided_slice:output:0$rienwrhgrh/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rienwrhgrh/zeros_1/packedy
rienwrhgrh/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rienwrhgrh/zeros_1/Const©
rienwrhgrh/zeros_1Fill"rienwrhgrh/zeros_1/packed:output:0!rienwrhgrh/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/zeros_1
rienwrhgrh/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rienwrhgrh/transpose/perm¯
rienwrhgrh/transpose	Transposelctpanrywj/transpose_1:y:0"rienwrhgrh/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/transposep
rienwrhgrh/Shape_1Shaperienwrhgrh/transpose:y:0*
T0*
_output_shapes
:2
rienwrhgrh/Shape_1
 rienwrhgrh/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rienwrhgrh/strided_slice_1/stack
"rienwrhgrh/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_1/stack_1
"rienwrhgrh/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_1/stack_2°
rienwrhgrh/strided_slice_1StridedSlicerienwrhgrh/Shape_1:output:0)rienwrhgrh/strided_slice_1/stack:output:0+rienwrhgrh/strided_slice_1/stack_1:output:0+rienwrhgrh/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rienwrhgrh/strided_slice_1
&rienwrhgrh/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&rienwrhgrh/TensorArrayV2/element_shapeÞ
rienwrhgrh/TensorArrayV2TensorListReserve/rienwrhgrh/TensorArrayV2/element_shape:output:0#rienwrhgrh/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rienwrhgrh/TensorArrayV2Õ
@rienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@rienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2rienwrhgrh/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrienwrhgrh/transpose:y:0Irienwrhgrh/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2rienwrhgrh/TensorArrayUnstack/TensorListFromTensor
 rienwrhgrh/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 rienwrhgrh/strided_slice_2/stack
"rienwrhgrh/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_2/stack_1
"rienwrhgrh/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_2/stack_2¾
rienwrhgrh/strided_slice_2StridedSlicerienwrhgrh/transpose:y:0)rienwrhgrh/strided_slice_2/stack:output:0+rienwrhgrh/strided_slice_2/stack_1:output:0+rienwrhgrh/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rienwrhgrh/strided_slice_2Ð
+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp4rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOpÓ
rienwrhgrh/kngiiuzftt/MatMulMatMul#rienwrhgrh/strided_slice_2:output:03rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rienwrhgrh/kngiiuzftt/MatMulÖ
-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp6rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOpÏ
rienwrhgrh/kngiiuzftt/MatMul_1MatMulrienwrhgrh/zeros:output:05rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
rienwrhgrh/kngiiuzftt/MatMul_1Ä
rienwrhgrh/kngiiuzftt/addAddV2&rienwrhgrh/kngiiuzftt/MatMul:product:0(rienwrhgrh/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rienwrhgrh/kngiiuzftt/addÏ
,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp5rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOpÑ
rienwrhgrh/kngiiuzftt/BiasAddBiasAddrienwrhgrh/kngiiuzftt/add:z:04rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rienwrhgrh/kngiiuzftt/BiasAdd
%rienwrhgrh/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%rienwrhgrh/kngiiuzftt/split/split_dim
rienwrhgrh/kngiiuzftt/splitSplit.rienwrhgrh/kngiiuzftt/split/split_dim:output:0&rienwrhgrh/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
rienwrhgrh/kngiiuzftt/split¶
$rienwrhgrh/kngiiuzftt/ReadVariableOpReadVariableOp-rienwrhgrh_kngiiuzftt_readvariableop_resource*
_output_shapes
: *
dtype02&
$rienwrhgrh/kngiiuzftt/ReadVariableOpº
rienwrhgrh/kngiiuzftt/mulMul,rienwrhgrh/kngiiuzftt/ReadVariableOp:value:0rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mulº
rienwrhgrh/kngiiuzftt/add_1AddV2$rienwrhgrh/kngiiuzftt/split:output:0rienwrhgrh/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_1
rienwrhgrh/kngiiuzftt/SigmoidSigmoidrienwrhgrh/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/Sigmoid¼
&rienwrhgrh/kngiiuzftt/ReadVariableOp_1ReadVariableOp/rienwrhgrh_kngiiuzftt_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&rienwrhgrh/kngiiuzftt/ReadVariableOp_1À
rienwrhgrh/kngiiuzftt/mul_1Mul.rienwrhgrh/kngiiuzftt/ReadVariableOp_1:value:0rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_1¼
rienwrhgrh/kngiiuzftt/add_2AddV2$rienwrhgrh/kngiiuzftt/split:output:1rienwrhgrh/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_2 
rienwrhgrh/kngiiuzftt/Sigmoid_1Sigmoidrienwrhgrh/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rienwrhgrh/kngiiuzftt/Sigmoid_1µ
rienwrhgrh/kngiiuzftt/mul_2Mul#rienwrhgrh/kngiiuzftt/Sigmoid_1:y:0rienwrhgrh/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_2
rienwrhgrh/kngiiuzftt/TanhTanh$rienwrhgrh/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/Tanh¶
rienwrhgrh/kngiiuzftt/mul_3Mul!rienwrhgrh/kngiiuzftt/Sigmoid:y:0rienwrhgrh/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_3·
rienwrhgrh/kngiiuzftt/add_3AddV2rienwrhgrh/kngiiuzftt/mul_2:z:0rienwrhgrh/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_3¼
&rienwrhgrh/kngiiuzftt/ReadVariableOp_2ReadVariableOp/rienwrhgrh_kngiiuzftt_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&rienwrhgrh/kngiiuzftt/ReadVariableOp_2Ä
rienwrhgrh/kngiiuzftt/mul_4Mul.rienwrhgrh/kngiiuzftt/ReadVariableOp_2:value:0rienwrhgrh/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_4¼
rienwrhgrh/kngiiuzftt/add_4AddV2$rienwrhgrh/kngiiuzftt/split:output:3rienwrhgrh/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/add_4 
rienwrhgrh/kngiiuzftt/Sigmoid_2Sigmoidrienwrhgrh/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
rienwrhgrh/kngiiuzftt/Sigmoid_2
rienwrhgrh/kngiiuzftt/Tanh_1Tanhrienwrhgrh/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/Tanh_1º
rienwrhgrh/kngiiuzftt/mul_5Mul#rienwrhgrh/kngiiuzftt/Sigmoid_2:y:0 rienwrhgrh/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/kngiiuzftt/mul_5¥
(rienwrhgrh/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(rienwrhgrh/TensorArrayV2_1/element_shapeä
rienwrhgrh/TensorArrayV2_1TensorListReserve1rienwrhgrh/TensorArrayV2_1/element_shape:output:0#rienwrhgrh/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rienwrhgrh/TensorArrayV2_1d
rienwrhgrh/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/time
#rienwrhgrh/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#rienwrhgrh/while/maximum_iterations
rienwrhgrh/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rienwrhgrh/while/loop_counter²
rienwrhgrh/whileWhile&rienwrhgrh/while/loop_counter:output:0,rienwrhgrh/while/maximum_iterations:output:0rienwrhgrh/time:output:0#rienwrhgrh/TensorArrayV2_1:handle:0rienwrhgrh/zeros:output:0rienwrhgrh/zeros_1:output:0#rienwrhgrh/strided_slice_1:output:0Brienwrhgrh/TensorArrayUnstack/TensorListFromTensor:output_handle:04rienwrhgrh_kngiiuzftt_matmul_readvariableop_resource6rienwrhgrh_kngiiuzftt_matmul_1_readvariableop_resource5rienwrhgrh_kngiiuzftt_biasadd_readvariableop_resource-rienwrhgrh_kngiiuzftt_readvariableop_resource/rienwrhgrh_kngiiuzftt_readvariableop_1_resource/rienwrhgrh_kngiiuzftt_readvariableop_2_resource*
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
rienwrhgrh_while_body_1103625*)
cond!R
rienwrhgrh_while_cond_1103624*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
rienwrhgrh/whileË
;rienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;rienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shape
-rienwrhgrh/TensorArrayV2Stack/TensorListStackTensorListStackrienwrhgrh/while:output:3Drienwrhgrh/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-rienwrhgrh/TensorArrayV2Stack/TensorListStack
 rienwrhgrh/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 rienwrhgrh/strided_slice_3/stack
"rienwrhgrh/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"rienwrhgrh/strided_slice_3/stack_1
"rienwrhgrh/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"rienwrhgrh/strided_slice_3/stack_2Ü
rienwrhgrh/strided_slice_3StridedSlice6rienwrhgrh/TensorArrayV2Stack/TensorListStack:tensor:0)rienwrhgrh/strided_slice_3/stack:output:0+rienwrhgrh/strided_slice_3/stack_1:output:0+rienwrhgrh/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
rienwrhgrh/strided_slice_3
rienwrhgrh/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rienwrhgrh/transpose_1/permÑ
rienwrhgrh/transpose_1	Transpose6rienwrhgrh/TensorArrayV2Stack/TensorListStack:tensor:0$rienwrhgrh/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
rienwrhgrh/transpose_1®
 uilnjhxhrx/MatMul/ReadVariableOpReadVariableOp)uilnjhxhrx_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 uilnjhxhrx/MatMul/ReadVariableOp±
uilnjhxhrx/MatMulMatMul#rienwrhgrh/strided_slice_3:output:0(uilnjhxhrx/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uilnjhxhrx/MatMul­
!uilnjhxhrx/BiasAdd/ReadVariableOpReadVariableOp*uilnjhxhrx_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!uilnjhxhrx/BiasAdd/ReadVariableOp­
uilnjhxhrx/BiasAddBiasAdduilnjhxhrx/MatMul:product:0)uilnjhxhrx/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uilnjhxhrx/BiasAddÏ
IdentityIdentityuilnjhxhrx/BiasAdd:output:0.^kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp5^kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp-^lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp,^lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp.^lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp%^lctpanrywj/dhxpxqfhna/ReadVariableOp'^lctpanrywj/dhxpxqfhna/ReadVariableOp_1'^lctpanrywj/dhxpxqfhna/ReadVariableOp_2^lctpanrywj/while-^rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp,^rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp.^rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp%^rienwrhgrh/kngiiuzftt/ReadVariableOp'^rienwrhgrh/kngiiuzftt/ReadVariableOp_1'^rienwrhgrh/kngiiuzftt/ReadVariableOp_2^rienwrhgrh/while"^uilnjhxhrx/BiasAdd/ReadVariableOp!^uilnjhxhrx/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp-kjxhlaztnm/conv1d/ExpandDims_1/ReadVariableOp2l
4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp4kjxhlaztnm/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp,lctpanrywj/dhxpxqfhna/BiasAdd/ReadVariableOp2Z
+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp+lctpanrywj/dhxpxqfhna/MatMul/ReadVariableOp2^
-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp-lctpanrywj/dhxpxqfhna/MatMul_1/ReadVariableOp2L
$lctpanrywj/dhxpxqfhna/ReadVariableOp$lctpanrywj/dhxpxqfhna/ReadVariableOp2P
&lctpanrywj/dhxpxqfhna/ReadVariableOp_1&lctpanrywj/dhxpxqfhna/ReadVariableOp_12P
&lctpanrywj/dhxpxqfhna/ReadVariableOp_2&lctpanrywj/dhxpxqfhna/ReadVariableOp_22$
lctpanrywj/whilelctpanrywj/while2\
,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp,rienwrhgrh/kngiiuzftt/BiasAdd/ReadVariableOp2Z
+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp+rienwrhgrh/kngiiuzftt/MatMul/ReadVariableOp2^
-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp-rienwrhgrh/kngiiuzftt/MatMul_1/ReadVariableOp2L
$rienwrhgrh/kngiiuzftt/ReadVariableOp$rienwrhgrh/kngiiuzftt/ReadVariableOp2P
&rienwrhgrh/kngiiuzftt/ReadVariableOp_1&rienwrhgrh/kngiiuzftt/ReadVariableOp_12P
&rienwrhgrh/kngiiuzftt/ReadVariableOp_2&rienwrhgrh/kngiiuzftt/ReadVariableOp_22$
rienwrhgrh/whilerienwrhgrh/while2F
!uilnjhxhrx/BiasAdd/ReadVariableOp!uilnjhxhrx/BiasAdd/ReadVariableOp2D
 uilnjhxhrx/MatMul/ReadVariableOp uilnjhxhrx/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àY

while_body_1102917
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dhxpxqfhna_matmul_readvariableop_resource_0:	F
3while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 A
2while_dhxpxqfhna_biasadd_readvariableop_resource_0:	8
*while_dhxpxqfhna_readvariableop_resource_0: :
,while_dhxpxqfhna_readvariableop_1_resource_0: :
,while_dhxpxqfhna_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dhxpxqfhna_matmul_readvariableop_resource:	D
1while_dhxpxqfhna_matmul_1_readvariableop_resource:	 ?
0while_dhxpxqfhna_biasadd_readvariableop_resource:	6
(while_dhxpxqfhna_readvariableop_resource: 8
*while_dhxpxqfhna_readvariableop_1_resource: 8
*while_dhxpxqfhna_readvariableop_2_resource: ¢'while/dhxpxqfhna/BiasAdd/ReadVariableOp¢&while/dhxpxqfhna/MatMul/ReadVariableOp¢(while/dhxpxqfhna/MatMul_1/ReadVariableOp¢while/dhxpxqfhna/ReadVariableOp¢!while/dhxpxqfhna/ReadVariableOp_1¢!while/dhxpxqfhna/ReadVariableOp_2Ã
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
&while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp1while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dhxpxqfhna/MatMul/ReadVariableOpÑ
while/dhxpxqfhna/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMulÉ
(while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp3while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dhxpxqfhna/MatMul_1/ReadVariableOpº
while/dhxpxqfhna/MatMul_1MatMulwhile_placeholder_20while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMul_1°
while/dhxpxqfhna/addAddV2!while/dhxpxqfhna/MatMul:product:0#while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/addÂ
'while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp2while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dhxpxqfhna/BiasAdd/ReadVariableOp½
while/dhxpxqfhna/BiasAddBiasAddwhile/dhxpxqfhna/add:z:0/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/BiasAdd
 while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dhxpxqfhna/split/split_dim
while/dhxpxqfhna/splitSplit)while/dhxpxqfhna/split/split_dim:output:0!while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dhxpxqfhna/split©
while/dhxpxqfhna/ReadVariableOpReadVariableOp*while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dhxpxqfhna/ReadVariableOp£
while/dhxpxqfhna/mulMul'while/dhxpxqfhna/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul¦
while/dhxpxqfhna/add_1AddV2while/dhxpxqfhna/split:output:0while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_1
while/dhxpxqfhna/SigmoidSigmoidwhile/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid¯
!while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp,while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_1©
while/dhxpxqfhna/mul_1Mul)while/dhxpxqfhna/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_1¨
while/dhxpxqfhna/add_2AddV2while/dhxpxqfhna/split:output:1while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_2
while/dhxpxqfhna/Sigmoid_1Sigmoidwhile/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_1
while/dhxpxqfhna/mul_2Mulwhile/dhxpxqfhna/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_2
while/dhxpxqfhna/TanhTanhwhile/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh¢
while/dhxpxqfhna/mul_3Mulwhile/dhxpxqfhna/Sigmoid:y:0while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_3£
while/dhxpxqfhna/add_3AddV2while/dhxpxqfhna/mul_2:z:0while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_3¯
!while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp,while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_2°
while/dhxpxqfhna/mul_4Mul)while/dhxpxqfhna/ReadVariableOp_2:value:0while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_4¨
while/dhxpxqfhna/add_4AddV2while/dhxpxqfhna/split:output:3while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_4
while/dhxpxqfhna/Sigmoid_2Sigmoidwhile/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_2
while/dhxpxqfhna/Tanh_1Tanhwhile/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh_1¦
while/dhxpxqfhna/mul_5Mulwhile/dhxpxqfhna/Sigmoid_2:y:0while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dhxpxqfhna/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dhxpxqfhna/mul_5:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dhxpxqfhna/add_3:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dhxpxqfhna_biasadd_readvariableop_resource2while_dhxpxqfhna_biasadd_readvariableop_resource_0"h
1while_dhxpxqfhna_matmul_1_readvariableop_resource3while_dhxpxqfhna_matmul_1_readvariableop_resource_0"d
/while_dhxpxqfhna_matmul_readvariableop_resource1while_dhxpxqfhna_matmul_readvariableop_resource_0"Z
*while_dhxpxqfhna_readvariableop_1_resource,while_dhxpxqfhna_readvariableop_1_resource_0"Z
*while_dhxpxqfhna_readvariableop_2_resource,while_dhxpxqfhna_readvariableop_2_resource_0"V
(while_dhxpxqfhna_readvariableop_resource*while_dhxpxqfhna_readvariableop_resource_0")
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
'while/dhxpxqfhna/BiasAdd/ReadVariableOp'while/dhxpxqfhna/BiasAdd/ReadVariableOp2P
&while/dhxpxqfhna/MatMul/ReadVariableOp&while/dhxpxqfhna/MatMul/ReadVariableOp2T
(while/dhxpxqfhna/MatMul_1/ReadVariableOp(while/dhxpxqfhna/MatMul_1/ReadVariableOp2B
while/dhxpxqfhna/ReadVariableOpwhile/dhxpxqfhna/ReadVariableOp2F
!while/dhxpxqfhna/ReadVariableOp_1!while/dhxpxqfhna/ReadVariableOp_12F
!while/dhxpxqfhna/ReadVariableOp_2!while/dhxpxqfhna/ReadVariableOp_2: 
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
while_body_1104713
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_dhxpxqfhna_matmul_readvariableop_resource_0:	F
3while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 A
2while_dhxpxqfhna_biasadd_readvariableop_resource_0:	8
*while_dhxpxqfhna_readvariableop_resource_0: :
,while_dhxpxqfhna_readvariableop_1_resource_0: :
,while_dhxpxqfhna_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_dhxpxqfhna_matmul_readvariableop_resource:	D
1while_dhxpxqfhna_matmul_1_readvariableop_resource:	 ?
0while_dhxpxqfhna_biasadd_readvariableop_resource:	6
(while_dhxpxqfhna_readvariableop_resource: 8
*while_dhxpxqfhna_readvariableop_1_resource: 8
*while_dhxpxqfhna_readvariableop_2_resource: ¢'while/dhxpxqfhna/BiasAdd/ReadVariableOp¢&while/dhxpxqfhna/MatMul/ReadVariableOp¢(while/dhxpxqfhna/MatMul_1/ReadVariableOp¢while/dhxpxqfhna/ReadVariableOp¢!while/dhxpxqfhna/ReadVariableOp_1¢!while/dhxpxqfhna/ReadVariableOp_2Ã
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
&while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp1while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/dhxpxqfhna/MatMul/ReadVariableOpÑ
while/dhxpxqfhna/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMulÉ
(while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp3while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/dhxpxqfhna/MatMul_1/ReadVariableOpº
while/dhxpxqfhna/MatMul_1MatMulwhile_placeholder_20while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/MatMul_1°
while/dhxpxqfhna/addAddV2!while/dhxpxqfhna/MatMul:product:0#while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/addÂ
'while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp2while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/dhxpxqfhna/BiasAdd/ReadVariableOp½
while/dhxpxqfhna/BiasAddBiasAddwhile/dhxpxqfhna/add:z:0/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/dhxpxqfhna/BiasAdd
 while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/dhxpxqfhna/split/split_dim
while/dhxpxqfhna/splitSplit)while/dhxpxqfhna/split/split_dim:output:0!while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/dhxpxqfhna/split©
while/dhxpxqfhna/ReadVariableOpReadVariableOp*while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/dhxpxqfhna/ReadVariableOp£
while/dhxpxqfhna/mulMul'while/dhxpxqfhna/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul¦
while/dhxpxqfhna/add_1AddV2while/dhxpxqfhna/split:output:0while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_1
while/dhxpxqfhna/SigmoidSigmoidwhile/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid¯
!while/dhxpxqfhna/ReadVariableOp_1ReadVariableOp,while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_1©
while/dhxpxqfhna/mul_1Mul)while/dhxpxqfhna/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_1¨
while/dhxpxqfhna/add_2AddV2while/dhxpxqfhna/split:output:1while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_2
while/dhxpxqfhna/Sigmoid_1Sigmoidwhile/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_1
while/dhxpxqfhna/mul_2Mulwhile/dhxpxqfhna/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_2
while/dhxpxqfhna/TanhTanhwhile/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh¢
while/dhxpxqfhna/mul_3Mulwhile/dhxpxqfhna/Sigmoid:y:0while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_3£
while/dhxpxqfhna/add_3AddV2while/dhxpxqfhna/mul_2:z:0while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_3¯
!while/dhxpxqfhna/ReadVariableOp_2ReadVariableOp,while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/dhxpxqfhna/ReadVariableOp_2°
while/dhxpxqfhna/mul_4Mul)while/dhxpxqfhna/ReadVariableOp_2:value:0while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_4¨
while/dhxpxqfhna/add_4AddV2while/dhxpxqfhna/split:output:3while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/add_4
while/dhxpxqfhna/Sigmoid_2Sigmoidwhile/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Sigmoid_2
while/dhxpxqfhna/Tanh_1Tanhwhile/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/Tanh_1¦
while/dhxpxqfhna/mul_5Mulwhile/dhxpxqfhna/Sigmoid_2:y:0while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/dhxpxqfhna/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/dhxpxqfhna/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/dhxpxqfhna/mul_5:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/dhxpxqfhna/add_3:z:0(^while/dhxpxqfhna/BiasAdd/ReadVariableOp'^while/dhxpxqfhna/MatMul/ReadVariableOp)^while/dhxpxqfhna/MatMul_1/ReadVariableOp ^while/dhxpxqfhna/ReadVariableOp"^while/dhxpxqfhna/ReadVariableOp_1"^while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_dhxpxqfhna_biasadd_readvariableop_resource2while_dhxpxqfhna_biasadd_readvariableop_resource_0"h
1while_dhxpxqfhna_matmul_1_readvariableop_resource3while_dhxpxqfhna_matmul_1_readvariableop_resource_0"d
/while_dhxpxqfhna_matmul_readvariableop_resource1while_dhxpxqfhna_matmul_readvariableop_resource_0"Z
*while_dhxpxqfhna_readvariableop_1_resource,while_dhxpxqfhna_readvariableop_1_resource_0"Z
*while_dhxpxqfhna_readvariableop_2_resource,while_dhxpxqfhna_readvariableop_2_resource_0"V
(while_dhxpxqfhna_readvariableop_resource*while_dhxpxqfhna_readvariableop_resource_0")
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
'while/dhxpxqfhna/BiasAdd/ReadVariableOp'while/dhxpxqfhna/BiasAdd/ReadVariableOp2P
&while/dhxpxqfhna/MatMul/ReadVariableOp&while/dhxpxqfhna/MatMul/ReadVariableOp2T
(while/dhxpxqfhna/MatMul_1/ReadVariableOp(while/dhxpxqfhna/MatMul_1/ReadVariableOp2B
while/dhxpxqfhna/ReadVariableOpwhile/dhxpxqfhna/ReadVariableOp2F
!while/dhxpxqfhna/ReadVariableOp_1!while/dhxpxqfhna/ReadVariableOp_12F
!while/dhxpxqfhna/ReadVariableOp_2!while/dhxpxqfhna/ReadVariableOp_2: 
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
	

,__inference_lctpanrywj_layer_call_fn_1105028
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_11010262
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
Ü

(sequential_lctpanrywj_while_body_1100293H
Dsequential_lctpanrywj_while_sequential_lctpanrywj_while_loop_counterN
Jsequential_lctpanrywj_while_sequential_lctpanrywj_while_maximum_iterations+
'sequential_lctpanrywj_while_placeholder-
)sequential_lctpanrywj_while_placeholder_1-
)sequential_lctpanrywj_while_placeholder_2-
)sequential_lctpanrywj_while_placeholder_3G
Csequential_lctpanrywj_while_sequential_lctpanrywj_strided_slice_1_0
sequential_lctpanrywj_while_tensorarrayv2read_tensorlistgetitem_sequential_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0:	\
Isequential_lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0:	 W
Hsequential_lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0:	N
@sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0: P
Bsequential_lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0: P
Bsequential_lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0: (
$sequential_lctpanrywj_while_identity*
&sequential_lctpanrywj_while_identity_1*
&sequential_lctpanrywj_while_identity_2*
&sequential_lctpanrywj_while_identity_3*
&sequential_lctpanrywj_while_identity_4*
&sequential_lctpanrywj_while_identity_5E
Asequential_lctpanrywj_while_sequential_lctpanrywj_strided_slice_1
}sequential_lctpanrywj_while_tensorarrayv2read_tensorlistgetitem_sequential_lctpanrywj_tensorarrayunstack_tensorlistfromtensorX
Esequential_lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource:	Z
Gsequential_lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource:	 U
Fsequential_lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource:	L
>sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_resource: N
@sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource: N
@sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource: ¢=sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp¢<sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp¢>sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp¢5sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp¢7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1¢7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2ï
Msequential/lctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/lctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/lctpanrywj/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_lctpanrywj_while_tensorarrayv2read_tensorlistgetitem_sequential_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0'sequential_lctpanrywj_while_placeholderVsequential/lctpanrywj/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/lctpanrywj/while/TensorArrayV2Read/TensorListGetItem
<sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOpReadVariableOpGsequential_lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp©
-sequential/lctpanrywj/while/dhxpxqfhna/MatMulMatMulFsequential/lctpanrywj/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/lctpanrywj/while/dhxpxqfhna/MatMul
>sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOpIsequential_lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp
/sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1MatMul)sequential_lctpanrywj_while_placeholder_2Fsequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1
*sequential/lctpanrywj/while/dhxpxqfhna/addAddV27sequential/lctpanrywj/while/dhxpxqfhna/MatMul:product:09sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/lctpanrywj/while/dhxpxqfhna/add
=sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOpHsequential_lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp
.sequential/lctpanrywj/while/dhxpxqfhna/BiasAddBiasAdd.sequential/lctpanrywj/while/dhxpxqfhna/add:z:0Esequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd²
6sequential/lctpanrywj/while/dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/lctpanrywj/while/dhxpxqfhna/split/split_dimÛ
,sequential/lctpanrywj/while/dhxpxqfhna/splitSplit?sequential/lctpanrywj/while/dhxpxqfhna/split/split_dim:output:07sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/lctpanrywj/while/dhxpxqfhna/splitë
5sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOpReadVariableOp@sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOpû
*sequential/lctpanrywj/while/dhxpxqfhna/mulMul=sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp:value:0)sequential_lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/lctpanrywj/while/dhxpxqfhna/mulþ
,sequential/lctpanrywj/while/dhxpxqfhna/add_1AddV25sequential/lctpanrywj/while/dhxpxqfhna/split:output:0.sequential/lctpanrywj/while/dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/add_1Ï
.sequential/lctpanrywj/while/dhxpxqfhna/SigmoidSigmoid0sequential/lctpanrywj/while/dhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/lctpanrywj/while/dhxpxqfhna/Sigmoidñ
7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1ReadVariableOpBsequential_lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1
,sequential/lctpanrywj/while/dhxpxqfhna/mul_1Mul?sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_1:value:0)sequential_lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/mul_1
,sequential/lctpanrywj/while/dhxpxqfhna/add_2AddV25sequential/lctpanrywj/while/dhxpxqfhna/split:output:10sequential/lctpanrywj/while/dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/add_2Ó
0sequential/lctpanrywj/while/dhxpxqfhna/Sigmoid_1Sigmoid0sequential/lctpanrywj/while/dhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/lctpanrywj/while/dhxpxqfhna/Sigmoid_1ö
,sequential/lctpanrywj/while/dhxpxqfhna/mul_2Mul4sequential/lctpanrywj/while/dhxpxqfhna/Sigmoid_1:y:0)sequential_lctpanrywj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/mul_2Ë
+sequential/lctpanrywj/while/dhxpxqfhna/TanhTanh5sequential/lctpanrywj/while/dhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/lctpanrywj/while/dhxpxqfhna/Tanhú
,sequential/lctpanrywj/while/dhxpxqfhna/mul_3Mul2sequential/lctpanrywj/while/dhxpxqfhna/Sigmoid:y:0/sequential/lctpanrywj/while/dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/mul_3û
,sequential/lctpanrywj/while/dhxpxqfhna/add_3AddV20sequential/lctpanrywj/while/dhxpxqfhna/mul_2:z:00sequential/lctpanrywj/while/dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/add_3ñ
7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2ReadVariableOpBsequential_lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2
,sequential/lctpanrywj/while/dhxpxqfhna/mul_4Mul?sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2:value:00sequential/lctpanrywj/while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/mul_4
,sequential/lctpanrywj/while/dhxpxqfhna/add_4AddV25sequential/lctpanrywj/while/dhxpxqfhna/split:output:30sequential/lctpanrywj/while/dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/add_4Ó
0sequential/lctpanrywj/while/dhxpxqfhna/Sigmoid_2Sigmoid0sequential/lctpanrywj/while/dhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/lctpanrywj/while/dhxpxqfhna/Sigmoid_2Ê
-sequential/lctpanrywj/while/dhxpxqfhna/Tanh_1Tanh0sequential/lctpanrywj/while/dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/lctpanrywj/while/dhxpxqfhna/Tanh_1þ
,sequential/lctpanrywj/while/dhxpxqfhna/mul_5Mul4sequential/lctpanrywj/while/dhxpxqfhna/Sigmoid_2:y:01sequential/lctpanrywj/while/dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/lctpanrywj/while/dhxpxqfhna/mul_5Ì
@sequential/lctpanrywj/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_lctpanrywj_while_placeholder_1'sequential_lctpanrywj_while_placeholder0sequential/lctpanrywj/while/dhxpxqfhna/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/lctpanrywj/while/TensorArrayV2Write/TensorListSetItem
!sequential/lctpanrywj/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/lctpanrywj/while/add/yÁ
sequential/lctpanrywj/while/addAddV2'sequential_lctpanrywj_while_placeholder*sequential/lctpanrywj/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/lctpanrywj/while/add
#sequential/lctpanrywj/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/lctpanrywj/while/add_1/yä
!sequential/lctpanrywj/while/add_1AddV2Dsequential_lctpanrywj_while_sequential_lctpanrywj_while_loop_counter,sequential/lctpanrywj/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/lctpanrywj/while/add_1
$sequential/lctpanrywj/while/IdentityIdentity%sequential/lctpanrywj/while/add_1:z:0>^sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp=^sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp?^sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp6^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp8^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_18^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/lctpanrywj/while/Identityµ
&sequential/lctpanrywj/while/Identity_1IdentityJsequential_lctpanrywj_while_sequential_lctpanrywj_while_maximum_iterations>^sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp=^sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp?^sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp6^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp8^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_18^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/lctpanrywj/while/Identity_1
&sequential/lctpanrywj/while/Identity_2Identity#sequential/lctpanrywj/while/add:z:0>^sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp=^sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp?^sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp6^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp8^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_18^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/lctpanrywj/while/Identity_2»
&sequential/lctpanrywj/while/Identity_3IdentityPsequential/lctpanrywj/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp=^sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp?^sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp6^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp8^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_18^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/lctpanrywj/while/Identity_3¬
&sequential/lctpanrywj/while/Identity_4Identity0sequential/lctpanrywj/while/dhxpxqfhna/mul_5:z:0>^sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp=^sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp?^sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp6^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp8^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_18^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/while/Identity_4¬
&sequential/lctpanrywj/while/Identity_5Identity0sequential/lctpanrywj/while/dhxpxqfhna/add_3:z:0>^sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp=^sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp?^sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp6^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp8^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_18^sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/lctpanrywj/while/Identity_5"
Fsequential_lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resourceHsequential_lctpanrywj_while_dhxpxqfhna_biasadd_readvariableop_resource_0"
Gsequential_lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resourceIsequential_lctpanrywj_while_dhxpxqfhna_matmul_1_readvariableop_resource_0"
Esequential_lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resourceGsequential_lctpanrywj_while_dhxpxqfhna_matmul_readvariableop_resource_0"
@sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_1_resourceBsequential_lctpanrywj_while_dhxpxqfhna_readvariableop_1_resource_0"
@sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_2_resourceBsequential_lctpanrywj_while_dhxpxqfhna_readvariableop_2_resource_0"
>sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_resource@sequential_lctpanrywj_while_dhxpxqfhna_readvariableop_resource_0"U
$sequential_lctpanrywj_while_identity-sequential/lctpanrywj/while/Identity:output:0"Y
&sequential_lctpanrywj_while_identity_1/sequential/lctpanrywj/while/Identity_1:output:0"Y
&sequential_lctpanrywj_while_identity_2/sequential/lctpanrywj/while/Identity_2:output:0"Y
&sequential_lctpanrywj_while_identity_3/sequential/lctpanrywj/while/Identity_3:output:0"Y
&sequential_lctpanrywj_while_identity_4/sequential/lctpanrywj/while/Identity_4:output:0"Y
&sequential_lctpanrywj_while_identity_5/sequential/lctpanrywj/while/Identity_5:output:0"
Asequential_lctpanrywj_while_sequential_lctpanrywj_strided_slice_1Csequential_lctpanrywj_while_sequential_lctpanrywj_strided_slice_1_0"
}sequential_lctpanrywj_while_tensorarrayv2read_tensorlistgetitem_sequential_lctpanrywj_tensorarrayunstack_tensorlistfromtensorsequential_lctpanrywj_while_tensorarrayv2read_tensorlistgetitem_sequential_lctpanrywj_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp=sequential/lctpanrywj/while/dhxpxqfhna/BiasAdd/ReadVariableOp2|
<sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp<sequential/lctpanrywj/while/dhxpxqfhna/MatMul/ReadVariableOp2
>sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp>sequential/lctpanrywj/while/dhxpxqfhna/MatMul_1/ReadVariableOp2n
5sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp5sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp2r
7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_17sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_12r
7sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_27sequential/lctpanrywj/while/dhxpxqfhna/ReadVariableOp_2: 
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
,__inference_rienwrhgrh_layer_call_fn_1105850

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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11028042
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
àY

while_body_1105321
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kngiiuzftt_matmul_readvariableop_resource_0:	 F
3while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 A
2while_kngiiuzftt_biasadd_readvariableop_resource_0:	8
*while_kngiiuzftt_readvariableop_resource_0: :
,while_kngiiuzftt_readvariableop_1_resource_0: :
,while_kngiiuzftt_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kngiiuzftt_matmul_readvariableop_resource:	 D
1while_kngiiuzftt_matmul_1_readvariableop_resource:	 ?
0while_kngiiuzftt_biasadd_readvariableop_resource:	6
(while_kngiiuzftt_readvariableop_resource: 8
*while_kngiiuzftt_readvariableop_1_resource: 8
*while_kngiiuzftt_readvariableop_2_resource: ¢'while/kngiiuzftt/BiasAdd/ReadVariableOp¢&while/kngiiuzftt/MatMul/ReadVariableOp¢(while/kngiiuzftt/MatMul_1/ReadVariableOp¢while/kngiiuzftt/ReadVariableOp¢!while/kngiiuzftt/ReadVariableOp_1¢!while/kngiiuzftt/ReadVariableOp_2Ã
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
&while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp1while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/kngiiuzftt/MatMul/ReadVariableOpÑ
while/kngiiuzftt/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMulÉ
(while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp3while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kngiiuzftt/MatMul_1/ReadVariableOpº
while/kngiiuzftt/MatMul_1MatMulwhile_placeholder_20while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMul_1°
while/kngiiuzftt/addAddV2!while/kngiiuzftt/MatMul:product:0#while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/addÂ
'while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp2while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kngiiuzftt/BiasAdd/ReadVariableOp½
while/kngiiuzftt/BiasAddBiasAddwhile/kngiiuzftt/add:z:0/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/BiasAdd
 while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kngiiuzftt/split/split_dim
while/kngiiuzftt/splitSplit)while/kngiiuzftt/split/split_dim:output:0!while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kngiiuzftt/split©
while/kngiiuzftt/ReadVariableOpReadVariableOp*while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kngiiuzftt/ReadVariableOp£
while/kngiiuzftt/mulMul'while/kngiiuzftt/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul¦
while/kngiiuzftt/add_1AddV2while/kngiiuzftt/split:output:0while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_1
while/kngiiuzftt/SigmoidSigmoidwhile/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid¯
!while/kngiiuzftt/ReadVariableOp_1ReadVariableOp,while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_1©
while/kngiiuzftt/mul_1Mul)while/kngiiuzftt/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_1¨
while/kngiiuzftt/add_2AddV2while/kngiiuzftt/split:output:1while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_2
while/kngiiuzftt/Sigmoid_1Sigmoidwhile/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_1
while/kngiiuzftt/mul_2Mulwhile/kngiiuzftt/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_2
while/kngiiuzftt/TanhTanhwhile/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh¢
while/kngiiuzftt/mul_3Mulwhile/kngiiuzftt/Sigmoid:y:0while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_3£
while/kngiiuzftt/add_3AddV2while/kngiiuzftt/mul_2:z:0while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_3¯
!while/kngiiuzftt/ReadVariableOp_2ReadVariableOp,while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_2°
while/kngiiuzftt/mul_4Mul)while/kngiiuzftt/ReadVariableOp_2:value:0while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_4¨
while/kngiiuzftt/add_4AddV2while/kngiiuzftt/split:output:3while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_4
while/kngiiuzftt/Sigmoid_2Sigmoidwhile/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_2
while/kngiiuzftt/Tanh_1Tanhwhile/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh_1¦
while/kngiiuzftt/mul_5Mulwhile/kngiiuzftt/Sigmoid_2:y:0while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kngiiuzftt/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kngiiuzftt/mul_5:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kngiiuzftt/add_3:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
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
0while_kngiiuzftt_biasadd_readvariableop_resource2while_kngiiuzftt_biasadd_readvariableop_resource_0"h
1while_kngiiuzftt_matmul_1_readvariableop_resource3while_kngiiuzftt_matmul_1_readvariableop_resource_0"d
/while_kngiiuzftt_matmul_readvariableop_resource1while_kngiiuzftt_matmul_readvariableop_resource_0"Z
*while_kngiiuzftt_readvariableop_1_resource,while_kngiiuzftt_readvariableop_1_resource_0"Z
*while_kngiiuzftt_readvariableop_2_resource,while_kngiiuzftt_readvariableop_2_resource_0"V
(while_kngiiuzftt_readvariableop_resource*while_kngiiuzftt_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kngiiuzftt/BiasAdd/ReadVariableOp'while/kngiiuzftt/BiasAdd/ReadVariableOp2P
&while/kngiiuzftt/MatMul/ReadVariableOp&while/kngiiuzftt/MatMul/ReadVariableOp2T
(while/kngiiuzftt/MatMul_1/ReadVariableOp(while/kngiiuzftt/MatMul_1/ReadVariableOp2B
while/kngiiuzftt/ReadVariableOpwhile/kngiiuzftt/ReadVariableOp2F
!while/kngiiuzftt/ReadVariableOp_1!while/kngiiuzftt/ReadVariableOp_12F
!while/kngiiuzftt/ReadVariableOp_2!while/kngiiuzftt/ReadVariableOp_2: 
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
while_cond_1105500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1105500___redundant_placeholder05
1while_while_cond_1105500___redundant_placeholder15
1while_while_cond_1105500___redundant_placeholder25
1while_while_cond_1105500___redundant_placeholder35
1while_while_cond_1105500___redundant_placeholder45
1while_while_cond_1105500___redundant_placeholder55
1while_while_cond_1105500___redundant_placeholder6
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
ç)
Ò
while_body_1101441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_kngiiuzftt_1101465_0:	 -
while_kngiiuzftt_1101467_0:	 )
while_kngiiuzftt_1101469_0:	(
while_kngiiuzftt_1101471_0: (
while_kngiiuzftt_1101473_0: (
while_kngiiuzftt_1101475_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_kngiiuzftt_1101465:	 +
while_kngiiuzftt_1101467:	 '
while_kngiiuzftt_1101469:	&
while_kngiiuzftt_1101471: &
while_kngiiuzftt_1101473: &
while_kngiiuzftt_1101475: ¢(while/kngiiuzftt/StatefulPartitionedCallÃ
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
(while/kngiiuzftt/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_kngiiuzftt_1101465_0while_kngiiuzftt_1101467_0while_kngiiuzftt_1101469_0while_kngiiuzftt_1101471_0while_kngiiuzftt_1101473_0while_kngiiuzftt_1101475_0*
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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_11014212*
(while/kngiiuzftt/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/kngiiuzftt/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/kngiiuzftt/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/kngiiuzftt/StatefulPartitionedCall:output:1)^while/kngiiuzftt/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/kngiiuzftt/StatefulPartitionedCall:output:2)^while/kngiiuzftt/StatefulPartitionedCall*
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
while_kngiiuzftt_1101465while_kngiiuzftt_1101465_0"6
while_kngiiuzftt_1101467while_kngiiuzftt_1101467_0"6
while_kngiiuzftt_1101469while_kngiiuzftt_1101469_0"6
while_kngiiuzftt_1101471while_kngiiuzftt_1101471_0"6
while_kngiiuzftt_1101473while_kngiiuzftt_1101473_0"6
while_kngiiuzftt_1101475while_kngiiuzftt_1101475_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/kngiiuzftt/StatefulPartitionedCall(while/kngiiuzftt/StatefulPartitionedCall: 
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
while_cond_1105680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1105680___redundant_placeholder05
1while_while_cond_1105680___redundant_placeholder15
1while_while_cond_1105680___redundant_placeholder25
1while_while_cond_1105680___redundant_placeholder35
1while_while_cond_1105680___redundant_placeholder45
1while_while_cond_1105680___redundant_placeholder55
1while_while_cond_1105680___redundant_placeholder6
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
while_cond_1102427
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1102427___redundant_placeholder05
1while_while_cond_1102427___redundant_placeholder15
1while_while_cond_1102427___redundant_placeholder25
1while_while_cond_1102427___redundant_placeholder35
1while_while_cond_1102427___redundant_placeholder45
1while_while_cond_1102427___redundant_placeholder55
1while_while_cond_1102427___redundant_placeholder6
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
while_cond_1104892
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1104892___redundant_placeholder05
1while_while_cond_1104892___redundant_placeholder15
1while_while_cond_1104892___redundant_placeholder25
1while_while_cond_1104892___redundant_placeholder35
1while_while_cond_1104892___redundant_placeholder45
1while_while_cond_1104892___redundant_placeholder55
1while_while_cond_1104892___redundant_placeholder6
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104814

inputs<
)dhxpxqfhna_matmul_readvariableop_resource:	>
+dhxpxqfhna_matmul_1_readvariableop_resource:	 9
*dhxpxqfhna_biasadd_readvariableop_resource:	0
"dhxpxqfhna_readvariableop_resource: 2
$dhxpxqfhna_readvariableop_1_resource: 2
$dhxpxqfhna_readvariableop_2_resource: 
identity¢!dhxpxqfhna/BiasAdd/ReadVariableOp¢ dhxpxqfhna/MatMul/ReadVariableOp¢"dhxpxqfhna/MatMul_1/ReadVariableOp¢dhxpxqfhna/ReadVariableOp¢dhxpxqfhna/ReadVariableOp_1¢dhxpxqfhna/ReadVariableOp_2¢whileD
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
 dhxpxqfhna/MatMul/ReadVariableOpReadVariableOp)dhxpxqfhna_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 dhxpxqfhna/MatMul/ReadVariableOp§
dhxpxqfhna/MatMulMatMulstrided_slice_2:output:0(dhxpxqfhna/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMulµ
"dhxpxqfhna/MatMul_1/ReadVariableOpReadVariableOp+dhxpxqfhna_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"dhxpxqfhna/MatMul_1/ReadVariableOp£
dhxpxqfhna/MatMul_1MatMulzeros:output:0*dhxpxqfhna/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/MatMul_1
dhxpxqfhna/addAddV2dhxpxqfhna/MatMul:product:0dhxpxqfhna/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/add®
!dhxpxqfhna/BiasAdd/ReadVariableOpReadVariableOp*dhxpxqfhna_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!dhxpxqfhna/BiasAdd/ReadVariableOp¥
dhxpxqfhna/BiasAddBiasAdddhxpxqfhna/add:z:0)dhxpxqfhna/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dhxpxqfhna/BiasAddz
dhxpxqfhna/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
dhxpxqfhna/split/split_dimë
dhxpxqfhna/splitSplit#dhxpxqfhna/split/split_dim:output:0dhxpxqfhna/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dhxpxqfhna/split
dhxpxqfhna/ReadVariableOpReadVariableOp"dhxpxqfhna_readvariableop_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp
dhxpxqfhna/mulMul!dhxpxqfhna/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul
dhxpxqfhna/add_1AddV2dhxpxqfhna/split:output:0dhxpxqfhna/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_1{
dhxpxqfhna/SigmoidSigmoiddhxpxqfhna/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid
dhxpxqfhna/ReadVariableOp_1ReadVariableOp$dhxpxqfhna_readvariableop_1_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_1
dhxpxqfhna/mul_1Mul#dhxpxqfhna/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_1
dhxpxqfhna/add_2AddV2dhxpxqfhna/split:output:1dhxpxqfhna/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_2
dhxpxqfhna/Sigmoid_1Sigmoiddhxpxqfhna/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_1
dhxpxqfhna/mul_2Muldhxpxqfhna/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_2w
dhxpxqfhna/TanhTanhdhxpxqfhna/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh
dhxpxqfhna/mul_3Muldhxpxqfhna/Sigmoid:y:0dhxpxqfhna/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_3
dhxpxqfhna/add_3AddV2dhxpxqfhna/mul_2:z:0dhxpxqfhna/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_3
dhxpxqfhna/ReadVariableOp_2ReadVariableOp$dhxpxqfhna_readvariableop_2_resource*
_output_shapes
: *
dtype02
dhxpxqfhna/ReadVariableOp_2
dhxpxqfhna/mul_4Mul#dhxpxqfhna/ReadVariableOp_2:value:0dhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_4
dhxpxqfhna/add_4AddV2dhxpxqfhna/split:output:3dhxpxqfhna/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/add_4
dhxpxqfhna/Sigmoid_2Sigmoiddhxpxqfhna/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Sigmoid_2v
dhxpxqfhna/Tanh_1Tanhdhxpxqfhna/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/Tanh_1
dhxpxqfhna/mul_5Muldhxpxqfhna/Sigmoid_2:y:0dhxpxqfhna/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dhxpxqfhna/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)dhxpxqfhna_matmul_readvariableop_resource+dhxpxqfhna_matmul_1_readvariableop_resource*dhxpxqfhna_biasadd_readvariableop_resource"dhxpxqfhna_readvariableop_resource$dhxpxqfhna_readvariableop_1_resource$dhxpxqfhna_readvariableop_2_resource*
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
while_body_1104713*
condR
while_cond_1104712*Q
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
IdentityIdentitytranspose_1:y:0"^dhxpxqfhna/BiasAdd/ReadVariableOp!^dhxpxqfhna/MatMul/ReadVariableOp#^dhxpxqfhna/MatMul_1/ReadVariableOp^dhxpxqfhna/ReadVariableOp^dhxpxqfhna/ReadVariableOp_1^dhxpxqfhna/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!dhxpxqfhna/BiasAdd/ReadVariableOp!dhxpxqfhna/BiasAdd/ReadVariableOp2D
 dhxpxqfhna/MatMul/ReadVariableOp dhxpxqfhna/MatMul/ReadVariableOp2H
"dhxpxqfhna/MatMul_1/ReadVariableOp"dhxpxqfhna/MatMul_1/ReadVariableOp26
dhxpxqfhna/ReadVariableOpdhxpxqfhna/ReadVariableOp2:
dhxpxqfhna/ReadVariableOp_1dhxpxqfhna/ReadVariableOp_12:
dhxpxqfhna/ReadVariableOp_2dhxpxqfhna/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

,__inference_rienwrhgrh_layer_call_fn_1105833

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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11025292
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
àY

while_body_1102428
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kngiiuzftt_matmul_readvariableop_resource_0:	 F
3while_kngiiuzftt_matmul_1_readvariableop_resource_0:	 A
2while_kngiiuzftt_biasadd_readvariableop_resource_0:	8
*while_kngiiuzftt_readvariableop_resource_0: :
,while_kngiiuzftt_readvariableop_1_resource_0: :
,while_kngiiuzftt_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kngiiuzftt_matmul_readvariableop_resource:	 D
1while_kngiiuzftt_matmul_1_readvariableop_resource:	 ?
0while_kngiiuzftt_biasadd_readvariableop_resource:	6
(while_kngiiuzftt_readvariableop_resource: 8
*while_kngiiuzftt_readvariableop_1_resource: 8
*while_kngiiuzftt_readvariableop_2_resource: ¢'while/kngiiuzftt/BiasAdd/ReadVariableOp¢&while/kngiiuzftt/MatMul/ReadVariableOp¢(while/kngiiuzftt/MatMul_1/ReadVariableOp¢while/kngiiuzftt/ReadVariableOp¢!while/kngiiuzftt/ReadVariableOp_1¢!while/kngiiuzftt/ReadVariableOp_2Ã
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
&while/kngiiuzftt/MatMul/ReadVariableOpReadVariableOp1while_kngiiuzftt_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/kngiiuzftt/MatMul/ReadVariableOpÑ
while/kngiiuzftt/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kngiiuzftt/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMulÉ
(while/kngiiuzftt/MatMul_1/ReadVariableOpReadVariableOp3while_kngiiuzftt_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kngiiuzftt/MatMul_1/ReadVariableOpº
while/kngiiuzftt/MatMul_1MatMulwhile_placeholder_20while/kngiiuzftt/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/MatMul_1°
while/kngiiuzftt/addAddV2!while/kngiiuzftt/MatMul:product:0#while/kngiiuzftt/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/addÂ
'while/kngiiuzftt/BiasAdd/ReadVariableOpReadVariableOp2while_kngiiuzftt_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kngiiuzftt/BiasAdd/ReadVariableOp½
while/kngiiuzftt/BiasAddBiasAddwhile/kngiiuzftt/add:z:0/while/kngiiuzftt/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kngiiuzftt/BiasAdd
 while/kngiiuzftt/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kngiiuzftt/split/split_dim
while/kngiiuzftt/splitSplit)while/kngiiuzftt/split/split_dim:output:0!while/kngiiuzftt/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kngiiuzftt/split©
while/kngiiuzftt/ReadVariableOpReadVariableOp*while_kngiiuzftt_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kngiiuzftt/ReadVariableOp£
while/kngiiuzftt/mulMul'while/kngiiuzftt/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul¦
while/kngiiuzftt/add_1AddV2while/kngiiuzftt/split:output:0while/kngiiuzftt/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_1
while/kngiiuzftt/SigmoidSigmoidwhile/kngiiuzftt/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid¯
!while/kngiiuzftt/ReadVariableOp_1ReadVariableOp,while_kngiiuzftt_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_1©
while/kngiiuzftt/mul_1Mul)while/kngiiuzftt/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_1¨
while/kngiiuzftt/add_2AddV2while/kngiiuzftt/split:output:1while/kngiiuzftt/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_2
while/kngiiuzftt/Sigmoid_1Sigmoidwhile/kngiiuzftt/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_1
while/kngiiuzftt/mul_2Mulwhile/kngiiuzftt/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_2
while/kngiiuzftt/TanhTanhwhile/kngiiuzftt/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh¢
while/kngiiuzftt/mul_3Mulwhile/kngiiuzftt/Sigmoid:y:0while/kngiiuzftt/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_3£
while/kngiiuzftt/add_3AddV2while/kngiiuzftt/mul_2:z:0while/kngiiuzftt/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_3¯
!while/kngiiuzftt/ReadVariableOp_2ReadVariableOp,while_kngiiuzftt_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kngiiuzftt/ReadVariableOp_2°
while/kngiiuzftt/mul_4Mul)while/kngiiuzftt/ReadVariableOp_2:value:0while/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_4¨
while/kngiiuzftt/add_4AddV2while/kngiiuzftt/split:output:3while/kngiiuzftt/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/add_4
while/kngiiuzftt/Sigmoid_2Sigmoidwhile/kngiiuzftt/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Sigmoid_2
while/kngiiuzftt/Tanh_1Tanhwhile/kngiiuzftt/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/Tanh_1¦
while/kngiiuzftt/mul_5Mulwhile/kngiiuzftt/Sigmoid_2:y:0while/kngiiuzftt/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kngiiuzftt/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kngiiuzftt/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kngiiuzftt/mul_5:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kngiiuzftt/add_3:z:0(^while/kngiiuzftt/BiasAdd/ReadVariableOp'^while/kngiiuzftt/MatMul/ReadVariableOp)^while/kngiiuzftt/MatMul_1/ReadVariableOp ^while/kngiiuzftt/ReadVariableOp"^while/kngiiuzftt/ReadVariableOp_1"^while/kngiiuzftt/ReadVariableOp_2*
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
0while_kngiiuzftt_biasadd_readvariableop_resource2while_kngiiuzftt_biasadd_readvariableop_resource_0"h
1while_kngiiuzftt_matmul_1_readvariableop_resource3while_kngiiuzftt_matmul_1_readvariableop_resource_0"d
/while_kngiiuzftt_matmul_readvariableop_resource1while_kngiiuzftt_matmul_readvariableop_resource_0"Z
*while_kngiiuzftt_readvariableop_1_resource,while_kngiiuzftt_readvariableop_1_resource_0"Z
*while_kngiiuzftt_readvariableop_2_resource,while_kngiiuzftt_readvariableop_2_resource_0"V
(while_kngiiuzftt_readvariableop_resource*while_kngiiuzftt_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kngiiuzftt/BiasAdd/ReadVariableOp'while/kngiiuzftt/BiasAdd/ReadVariableOp2P
&while/kngiiuzftt/MatMul/ReadVariableOp&while/kngiiuzftt/MatMul/ReadVariableOp2T
(while/kngiiuzftt/MatMul_1/ReadVariableOp(while/kngiiuzftt/MatMul_1/ReadVariableOp2B
while/kngiiuzftt/ReadVariableOpwhile/kngiiuzftt/ReadVariableOp2F
!while/kngiiuzftt/ReadVariableOp_1!while/kngiiuzftt/ReadVariableOp_12F
!while/kngiiuzftt/ReadVariableOp_2!while/kngiiuzftt/ReadVariableOp_2: 
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
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_1104247

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
¥
©	
(sequential_lctpanrywj_while_cond_1100292H
Dsequential_lctpanrywj_while_sequential_lctpanrywj_while_loop_counterN
Jsequential_lctpanrywj_while_sequential_lctpanrywj_while_maximum_iterations+
'sequential_lctpanrywj_while_placeholder-
)sequential_lctpanrywj_while_placeholder_1-
)sequential_lctpanrywj_while_placeholder_2-
)sequential_lctpanrywj_while_placeholder_3J
Fsequential_lctpanrywj_while_less_sequential_lctpanrywj_strided_slice_1a
]sequential_lctpanrywj_while_sequential_lctpanrywj_while_cond_1100292___redundant_placeholder0a
]sequential_lctpanrywj_while_sequential_lctpanrywj_while_cond_1100292___redundant_placeholder1a
]sequential_lctpanrywj_while_sequential_lctpanrywj_while_cond_1100292___redundant_placeholder2a
]sequential_lctpanrywj_while_sequential_lctpanrywj_while_cond_1100292___redundant_placeholder3a
]sequential_lctpanrywj_while_sequential_lctpanrywj_while_cond_1100292___redundant_placeholder4a
]sequential_lctpanrywj_while_sequential_lctpanrywj_while_cond_1100292___redundant_placeholder5a
]sequential_lctpanrywj_while_sequential_lctpanrywj_while_cond_1100292___redundant_placeholder6(
$sequential_lctpanrywj_while_identity
Þ
 sequential/lctpanrywj/while/LessLess'sequential_lctpanrywj_while_placeholderFsequential_lctpanrywj_while_less_sequential_lctpanrywj_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/lctpanrywj/while/Less
$sequential/lctpanrywj/while/IdentityIdentity$sequential/lctpanrywj/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/lctpanrywj/while/Identity"U
$sequential_lctpanrywj_while_identity-sequential/lctpanrywj/while/Identity:output:0*(
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
±

#__inference__traced_restore_1106404
file_prefix8
"assignvariableop_kjxhlaztnm_kernel:0
"assignvariableop_1_kjxhlaztnm_bias:6
$assignvariableop_2_uilnjhxhrx_kernel: 0
"assignvariableop_3_uilnjhxhrx_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_lctpanrywj_dhxpxqfhna_kernel:	M
:assignvariableop_10_lctpanrywj_dhxpxqfhna_recurrent_kernel:	 =
.assignvariableop_11_lctpanrywj_dhxpxqfhna_bias:	S
Eassignvariableop_12_lctpanrywj_dhxpxqfhna_input_gate_peephole_weights: T
Fassignvariableop_13_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weights: T
Fassignvariableop_14_lctpanrywj_dhxpxqfhna_output_gate_peephole_weights: C
0assignvariableop_15_rienwrhgrh_kngiiuzftt_kernel:	 M
:assignvariableop_16_rienwrhgrh_kngiiuzftt_recurrent_kernel:	 =
.assignvariableop_17_rienwrhgrh_kngiiuzftt_bias:	S
Eassignvariableop_18_rienwrhgrh_kngiiuzftt_input_gate_peephole_weights: T
Fassignvariableop_19_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weights: T
Fassignvariableop_20_rienwrhgrh_kngiiuzftt_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_kjxhlaztnm_kernel_rms:=
/assignvariableop_24_rmsprop_kjxhlaztnm_bias_rms:C
1assignvariableop_25_rmsprop_uilnjhxhrx_kernel_rms: =
/assignvariableop_26_rmsprop_uilnjhxhrx_bias_rms:O
<assignvariableop_27_rmsprop_lctpanrywj_dhxpxqfhna_kernel_rms:	Y
Fassignvariableop_28_rmsprop_lctpanrywj_dhxpxqfhna_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_lctpanrywj_dhxpxqfhna_bias_rms:	_
Qassignvariableop_30_rmsprop_lctpanrywj_dhxpxqfhna_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_lctpanrywj_dhxpxqfhna_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_rienwrhgrh_kngiiuzftt_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_rienwrhgrh_kngiiuzftt_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_rienwrhgrh_kngiiuzftt_bias_rms:	_
Qassignvariableop_36_rmsprop_rienwrhgrh_kngiiuzftt_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_rienwrhgrh_kngiiuzftt_output_gate_peephole_weights_rms: 
identity_40¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ç
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Ó
valueÉBÆ(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp"assignvariableop_kjxhlaztnm_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_kjxhlaztnm_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_uilnjhxhrx_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_uilnjhxhrx_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_lctpanrywj_dhxpxqfhna_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_lctpanrywj_dhxpxqfhna_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_lctpanrywj_dhxpxqfhna_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_lctpanrywj_dhxpxqfhna_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_lctpanrywj_dhxpxqfhna_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_rienwrhgrh_kngiiuzftt_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_rienwrhgrh_kngiiuzftt_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_rienwrhgrh_kngiiuzftt_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_rienwrhgrh_kngiiuzftt_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_rienwrhgrh_kngiiuzftt_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_kjxhlaztnm_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_kjxhlaztnm_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_uilnjhxhrx_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_uilnjhxhrx_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_lctpanrywj_dhxpxqfhna_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_lctpanrywj_dhxpxqfhna_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_lctpanrywj_dhxpxqfhna_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_lctpanrywj_dhxpxqfhna_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_lctpanrywj_dhxpxqfhna_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_lctpanrywj_dhxpxqfhna_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_rienwrhgrh_kngiiuzftt_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_rienwrhgrh_kngiiuzftt_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_rienwrhgrh_kngiiuzftt_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_rienwrhgrh_kngiiuzftt_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_rienwrhgrh_kngiiuzftt_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_rienwrhgrh_kngiiuzftt_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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


rienwrhgrh_while_cond_11040282
.rienwrhgrh_while_rienwrhgrh_while_loop_counter8
4rienwrhgrh_while_rienwrhgrh_while_maximum_iterations 
rienwrhgrh_while_placeholder"
rienwrhgrh_while_placeholder_1"
rienwrhgrh_while_placeholder_2"
rienwrhgrh_while_placeholder_34
0rienwrhgrh_while_less_rienwrhgrh_strided_slice_1K
Grienwrhgrh_while_rienwrhgrh_while_cond_1104028___redundant_placeholder0K
Grienwrhgrh_while_rienwrhgrh_while_cond_1104028___redundant_placeholder1K
Grienwrhgrh_while_rienwrhgrh_while_cond_1104028___redundant_placeholder2K
Grienwrhgrh_while_rienwrhgrh_while_cond_1104028___redundant_placeholder3K
Grienwrhgrh_while_rienwrhgrh_while_cond_1104028___redundant_placeholder4K
Grienwrhgrh_while_rienwrhgrh_while_cond_1104028___redundant_placeholder5K
Grienwrhgrh_while_rienwrhgrh_while_cond_1104028___redundant_placeholder6
rienwrhgrh_while_identity
§
rienwrhgrh/while/LessLessrienwrhgrh_while_placeholder0rienwrhgrh_while_less_rienwrhgrh_strided_slice_1*
T0*
_output_shapes
: 2
rienwrhgrh/while/Less~
rienwrhgrh/while/IdentityIdentityrienwrhgrh/while/Less:z:0*
T0
*
_output_shapes
: 2
rienwrhgrh/while/Identity"?
rienwrhgrh_while_identity"rienwrhgrh/while/Identity:output:0*(
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
ë

,__inference_rienwrhgrh_layer_call_fn_1105799
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_11015212
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
inputs/0"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
I

dtmnvweekc;
serving_default_dtmnvweekc:0ÿÿÿÿÿÿÿÿÿ>

uilnjhxhrx0
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
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"ÂA
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dtmnvweekc"}}, {"class_name": "Conv1D", "config": {"name": "kjxhlaztnm", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "tzzrzfazij", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "lctpanrywj", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dhxpxqfhna", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "rienwrhgrh", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kngiiuzftt", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "uilnjhxhrx", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "dtmnvweekc"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dtmnvweekc"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "kjxhlaztnm", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "tzzrzfazij", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "lctpanrywj", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dhxpxqfhna", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "rienwrhgrh", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kngiiuzftt", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "uilnjhxhrx", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
Ì

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"¥

_tf_keras_layer
{"name": "kjxhlaztnm", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "kjxhlaztnm", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"name": "tzzrzfazij", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "tzzrzfazij", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
­
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layerä{"name": "lctpanrywj", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "lctpanrywj", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dhxpxqfhna", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
³
cell

state_spec
trainable_variables
	variables
 regularization_losses
!	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layerê{"name": "rienwrhgrh", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "rienwrhgrh", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kngiiuzftt", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "uilnjhxhrx", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "uilnjhxhrx", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
Î
trainable_variables
	variables
9layer_regularization_losses
	regularization_losses
:non_trainable_variables
;metrics
<layer_metrics

=layers
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
':%2kjxhlaztnm/kernel
:2kjxhlaztnm/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
>layer_regularization_losses
	variables
regularization_losses
?non_trainable_variables
@metrics
Alayer_metrics

Blayers
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
trainable_variables
Clayer_regularization_losses
	variables
regularization_losses
Dnon_trainable_variables
Emetrics
Flayer_metrics

Glayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"name": "dhxpxqfhna", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "dhxpxqfhna", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
 "
trackable_list_wrapper
¼
trainable_variables
	variables
Mlayer_regularization_losses

Nlayers
regularization_losses
Onon_trainable_variables
Pmetrics
Qlayer_metrics

Rstates
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+&call_and_return_all_conditional_losses
__call__"Ú
_tf_keras_layerÀ{"name": "kngiiuzftt", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kngiiuzftt", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
 "
trackable_list_wrapper
¼
trainable_variables
	variables
Xlayer_regularization_losses

Ylayers
 regularization_losses
Znon_trainable_variables
[metrics
\layer_metrics

]states
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:! 2uilnjhxhrx/kernel
:2uilnjhxhrx/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
$trainable_variables
^layer_regularization_losses
%	variables
&regularization_losses
_non_trainable_variables
`metrics
alayer_metrics

blayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
/:-	2lctpanrywj/dhxpxqfhna/kernel
9:7	 2&lctpanrywj/dhxpxqfhna/recurrent_kernel
):'2lctpanrywj/dhxpxqfhna/bias
?:= 21lctpanrywj/dhxpxqfhna/input_gate_peephole_weights
@:> 22lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights
@:> 22lctpanrywj/dhxpxqfhna/output_gate_peephole_weights
/:-	 2rienwrhgrh/kngiiuzftt/kernel
9:7	 2&rienwrhgrh/kngiiuzftt/recurrent_kernel
):'2rienwrhgrh/kngiiuzftt/bias
?:= 21rienwrhgrh/kngiiuzftt/input_gate_peephole_weights
@:> 22rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights
@:> 22rienwrhgrh/kngiiuzftt/output_gate_peephole_weights
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
°
Itrainable_variables
dlayer_regularization_losses
J	variables
Kregularization_losses
enon_trainable_variables
fmetrics
glayer_metrics

hlayers
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
°
Ttrainable_variables
ilayer_regularization_losses
U	variables
Vregularization_losses
jnon_trainable_variables
kmetrics
llayer_metrics

mlayers
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
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
1:/2RMSprop/kjxhlaztnm/kernel/rms
':%2RMSprop/kjxhlaztnm/bias/rms
-:+ 2RMSprop/uilnjhxhrx/kernel/rms
':%2RMSprop/uilnjhxhrx/bias/rms
9:7	2(RMSprop/lctpanrywj/dhxpxqfhna/kernel/rms
C:A	 22RMSprop/lctpanrywj/dhxpxqfhna/recurrent_kernel/rms
3:12&RMSprop/lctpanrywj/dhxpxqfhna/bias/rms
I:G 2=RMSprop/lctpanrywj/dhxpxqfhna/input_gate_peephole_weights/rms
J:H 2>RMSprop/lctpanrywj/dhxpxqfhna/forget_gate_peephole_weights/rms
J:H 2>RMSprop/lctpanrywj/dhxpxqfhna/output_gate_peephole_weights/rms
9:7	 2(RMSprop/rienwrhgrh/kngiiuzftt/kernel/rms
C:A	 22RMSprop/rienwrhgrh/kngiiuzftt/recurrent_kernel/rms
3:12&RMSprop/rienwrhgrh/kngiiuzftt/bias/rms
I:G 2=RMSprop/rienwrhgrh/kngiiuzftt/input_gate_peephole_weights/rms
J:H 2>RMSprop/rienwrhgrh/kngiiuzftt/forget_gate_peephole_weights/rms
J:H 2>RMSprop/rienwrhgrh/kngiiuzftt/output_gate_peephole_weights/rms
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_1103732
G__inference_sequential_layer_call_and_return_conditional_losses_1104136
G__inference_sequential_layer_call_and_return_conditional_losses_1103242
G__inference_sequential_layer_call_and_return_conditional_losses_1103283À
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
,__inference_sequential_layer_call_fn_1102595
,__inference_sequential_layer_call_fn_1104173
,__inference_sequential_layer_call_fn_1104210
,__inference_sequential_layer_call_fn_1103201À
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
"__inference__wrapped_model_1100576Á
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

dtmnvweekcÿÿÿÿÿÿÿÿÿ
ñ2î
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_1104247¢
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
,__inference_kjxhlaztnm_layer_call_fn_1104256¢
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
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_1104269¢
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
,__inference_tzzrzfazij_layer_call_fn_1104274¢
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104454
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104634
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104814
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104994æ
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
,__inference_lctpanrywj_layer_call_fn_1105011
,__inference_lctpanrywj_layer_call_fn_1105028
,__inference_lctpanrywj_layer_call_fn_1105045
,__inference_lctpanrywj_layer_call_fn_1105062æ
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105242
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105422
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105602
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105782æ
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
,__inference_rienwrhgrh_layer_call_fn_1105799
,__inference_rienwrhgrh_layer_call_fn_1105816
,__inference_rienwrhgrh_layer_call_fn_1105833
,__inference_rienwrhgrh_layer_call_fn_1105850æ
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
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_1105860¢
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
,__inference_uilnjhxhrx_layer_call_fn_1105869¢
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
%__inference_signature_wrapper_1103328
dtmnvweekc"
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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1105913
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1105957¾
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
,__inference_dhxpxqfhna_layer_call_fn_1105980
,__inference_dhxpxqfhna_layer_call_fn_1106003¾
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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1106047
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1106091¾
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
,__inference_kngiiuzftt_layer_call_fn_1106114
,__inference_kngiiuzftt_layer_call_fn_1106137¾
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
"__inference__wrapped_model_1100576-./012345678"#;¢8
1¢.
,)

dtmnvweekcÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

uilnjhxhrx$!

uilnjhxhrxÿÿÿÿÿÿÿÿÿÌ
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1105913-./012¢}
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
G__inference_dhxpxqfhna_layer_call_and_return_conditional_losses_1105957-./012¢}
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
,__inference_dhxpxqfhna_layer_call_fn_1105980ð-./012¢}
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
,__inference_dhxpxqfhna_layer_call_fn_1106003ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ ·
G__inference_kjxhlaztnm_layer_call_and_return_conditional_losses_1104247l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_kjxhlaztnm_layer_call_fn_1104256_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÌ
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1106047345678¢}
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
G__inference_kngiiuzftt_layer_call_and_return_conditional_losses_1106091345678¢}
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
,__inference_kngiiuzftt_layer_call_fn_1106114ð345678¢}
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
,__inference_kngiiuzftt_layer_call_fn_1106137ð345678¢}
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104454-./012S¢P
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104634-./012S¢P
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104814x-./012C¢@
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
G__inference_lctpanrywj_layer_call_and_return_conditional_losses_1104994x-./012C¢@
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
,__inference_lctpanrywj_layer_call_fn_1105011-./012S¢P
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
,__inference_lctpanrywj_layer_call_fn_1105028-./012S¢P
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
,__inference_lctpanrywj_layer_call_fn_1105045k-./012C¢@
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
,__inference_lctpanrywj_layer_call_fn_1105062k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ Ð
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105242345678S¢P
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105422345678S¢P
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105602t345678C¢@
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
G__inference_rienwrhgrh_layer_call_and_return_conditional_losses_1105782t345678C¢@
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
,__inference_rienwrhgrh_layer_call_fn_1105799w345678S¢P
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
,__inference_rienwrhgrh_layer_call_fn_1105816w345678S¢P
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
,__inference_rienwrhgrh_layer_call_fn_1105833g345678C¢@
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
,__inference_rienwrhgrh_layer_call_fn_1105850g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ É
G__inference_sequential_layer_call_and_return_conditional_losses_1103242~-./012345678"#C¢@
9¢6
,)

dtmnvweekcÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_layer_call_and_return_conditional_losses_1103283~-./012345678"#C¢@
9¢6
,)

dtmnvweekcÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_1103732z-./012345678"#?¢<
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
G__inference_sequential_layer_call_and_return_conditional_losses_1104136z-./012345678"#?¢<
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
,__inference_sequential_layer_call_fn_1102595q-./012345678"#C¢@
9¢6
,)

dtmnvweekcÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_layer_call_fn_1103201q-./012345678"#C¢@
9¢6
,)

dtmnvweekcÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1104173m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1104210m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
%__inference_signature_wrapper_1103328-./012345678"#I¢F
¢ 
?ª<
:

dtmnvweekc,)

dtmnvweekcÿÿÿÿÿÿÿÿÿ"7ª4
2

uilnjhxhrx$!

uilnjhxhrxÿÿÿÿÿÿÿÿÿ¯
G__inference_tzzrzfazij_layer_call_and_return_conditional_losses_1104269d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_tzzrzfazij_layer_call_fn_1104274W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_uilnjhxhrx_layer_call_and_return_conditional_losses_1105860\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_uilnjhxhrx_layer_call_fn_1105869O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ