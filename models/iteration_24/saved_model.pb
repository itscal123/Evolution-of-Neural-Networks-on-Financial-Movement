ß2
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
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718´®/

ggbvrcpxtu/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameggbvrcpxtu/kernel
{
%ggbvrcpxtu/kernel/Read/ReadVariableOpReadVariableOpggbvrcpxtu/kernel*"
_output_shapes
:*
dtype0
v
ggbvrcpxtu/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameggbvrcpxtu/bias
o
#ggbvrcpxtu/bias/Read/ReadVariableOpReadVariableOpggbvrcpxtu/bias*
_output_shapes
:*
dtype0
~
supobtndkp/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namesupobtndkp/kernel
w
%supobtndkp/kernel/Read/ReadVariableOpReadVariableOpsupobtndkp/kernel*
_output_shapes

: *
dtype0
v
supobtndkp/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namesupobtndkp/bias
o
#supobtndkp/bias/Read/ReadVariableOpReadVariableOpsupobtndkp/bias*
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
gdmdltnblf/duhsngmesj/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namegdmdltnblf/duhsngmesj/kernel

0gdmdltnblf/duhsngmesj/kernel/Read/ReadVariableOpReadVariableOpgdmdltnblf/duhsngmesj/kernel*
_output_shapes
:	*
dtype0
©
&gdmdltnblf/duhsngmesj/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&gdmdltnblf/duhsngmesj/recurrent_kernel
¢
:gdmdltnblf/duhsngmesj/recurrent_kernel/Read/ReadVariableOpReadVariableOp&gdmdltnblf/duhsngmesj/recurrent_kernel*
_output_shapes
:	 *
dtype0

gdmdltnblf/duhsngmesj/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namegdmdltnblf/duhsngmesj/bias

.gdmdltnblf/duhsngmesj/bias/Read/ReadVariableOpReadVariableOpgdmdltnblf/duhsngmesj/bias*
_output_shapes	
:*
dtype0
º
1gdmdltnblf/duhsngmesj/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31gdmdltnblf/duhsngmesj/input_gate_peephole_weights
³
Egdmdltnblf/duhsngmesj/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1gdmdltnblf/duhsngmesj/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2gdmdltnblf/duhsngmesj/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42gdmdltnblf/duhsngmesj/forget_gate_peephole_weights
µ
Fgdmdltnblf/duhsngmesj/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2gdmdltnblf/duhsngmesj/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2gdmdltnblf/duhsngmesj/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42gdmdltnblf/duhsngmesj/output_gate_peephole_weights
µ
Fgdmdltnblf/duhsngmesj/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2gdmdltnblf/duhsngmesj/output_gate_peephole_weights*
_output_shapes
: *
dtype0

uazvpibasg/gddwjadkgr/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_nameuazvpibasg/gddwjadkgr/kernel

0uazvpibasg/gddwjadkgr/kernel/Read/ReadVariableOpReadVariableOpuazvpibasg/gddwjadkgr/kernel*
_output_shapes
:	 *
dtype0
©
&uazvpibasg/gddwjadkgr/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&uazvpibasg/gddwjadkgr/recurrent_kernel
¢
:uazvpibasg/gddwjadkgr/recurrent_kernel/Read/ReadVariableOpReadVariableOp&uazvpibasg/gddwjadkgr/recurrent_kernel*
_output_shapes
:	 *
dtype0

uazvpibasg/gddwjadkgr/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameuazvpibasg/gddwjadkgr/bias

.uazvpibasg/gddwjadkgr/bias/Read/ReadVariableOpReadVariableOpuazvpibasg/gddwjadkgr/bias*
_output_shapes	
:*
dtype0
º
1uazvpibasg/gddwjadkgr/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31uazvpibasg/gddwjadkgr/input_gate_peephole_weights
³
Euazvpibasg/gddwjadkgr/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1uazvpibasg/gddwjadkgr/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2uazvpibasg/gddwjadkgr/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42uazvpibasg/gddwjadkgr/forget_gate_peephole_weights
µ
Fuazvpibasg/gddwjadkgr/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2uazvpibasg/gddwjadkgr/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2uazvpibasg/gddwjadkgr/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42uazvpibasg/gddwjadkgr/output_gate_peephole_weights
µ
Fuazvpibasg/gddwjadkgr/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2uazvpibasg/gddwjadkgr/output_gate_peephole_weights*
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
RMSprop/ggbvrcpxtu/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/ggbvrcpxtu/kernel/rms

1RMSprop/ggbvrcpxtu/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/ggbvrcpxtu/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/ggbvrcpxtu/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/ggbvrcpxtu/bias/rms

/RMSprop/ggbvrcpxtu/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/ggbvrcpxtu/bias/rms*
_output_shapes
:*
dtype0

RMSprop/supobtndkp/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/supobtndkp/kernel/rms

1RMSprop/supobtndkp/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/supobtndkp/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/supobtndkp/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/supobtndkp/bias/rms

/RMSprop/supobtndkp/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/supobtndkp/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/gdmdltnblf/duhsngmesj/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/gdmdltnblf/duhsngmesj/kernel/rms
¦
<RMSprop/gdmdltnblf/duhsngmesj/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/gdmdltnblf/duhsngmesj/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rms
º
FRMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/gdmdltnblf/duhsngmesj/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/gdmdltnblf/duhsngmesj/bias/rms

:RMSprop/gdmdltnblf/duhsngmesj/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/gdmdltnblf/duhsngmesj/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rms
Ë
QRMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rms
Í
RRMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rms
Í
RRMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/uazvpibasg/gddwjadkgr/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/uazvpibasg/gddwjadkgr/kernel/rms
¦
<RMSprop/uazvpibasg/gddwjadkgr/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/uazvpibasg/gddwjadkgr/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rms
º
FRMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/uazvpibasg/gddwjadkgr/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/uazvpibasg/gddwjadkgr/bias/rms

:RMSprop/uazvpibasg/gddwjadkgr/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/uazvpibasg/gddwjadkgr/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rms
Ë
QRMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rms
Í
RRMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rms
Í
RRMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rms*
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
VARIABLE_VALUEggbvrcpxtu/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEggbvrcpxtu/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEsupobtndkp/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsupobtndkp/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEgdmdltnblf/duhsngmesj/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&gdmdltnblf/duhsngmesj/recurrent_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEgdmdltnblf/duhsngmesj/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1gdmdltnblf/duhsngmesj/input_gate_peephole_weights0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2gdmdltnblf/duhsngmesj/forget_gate_peephole_weights0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2gdmdltnblf/duhsngmesj/output_gate_peephole_weights0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEuazvpibasg/gddwjadkgr/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&uazvpibasg/gddwjadkgr/recurrent_kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEuazvpibasg/gddwjadkgr/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1uazvpibasg/gddwjadkgr/input_gate_peephole_weights1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2uazvpibasg/gddwjadkgr/forget_gate_peephole_weights1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2uazvpibasg/gddwjadkgr/output_gate_peephole_weights1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUERMSprop/ggbvrcpxtu/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/ggbvrcpxtu/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/supobtndkp/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/supobtndkp/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/gdmdltnblf/duhsngmesj/kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/gdmdltnblf/duhsngmesj/bias/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE=RMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE>RMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE>RMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rmsNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/uazvpibasg/gddwjadkgr/kernel/rmsNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rmsNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/uazvpibasg/gddwjadkgr/bias/rmsOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE=RMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rmsOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¤¡
VARIABLE_VALUE>RMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rmsOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¤¡
VARIABLE_VALUE>RMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rmsOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_unopekwlxvPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_unopekwlxvggbvrcpxtu/kernelggbvrcpxtu/biasgdmdltnblf/duhsngmesj/kernel&gdmdltnblf/duhsngmesj/recurrent_kernelgdmdltnblf/duhsngmesj/bias1gdmdltnblf/duhsngmesj/input_gate_peephole_weights2gdmdltnblf/duhsngmesj/forget_gate_peephole_weights2gdmdltnblf/duhsngmesj/output_gate_peephole_weightsuazvpibasg/gddwjadkgr/kernel&uazvpibasg/gddwjadkgr/recurrent_kerneluazvpibasg/gddwjadkgr/bias1uazvpibasg/gddwjadkgr/input_gate_peephole_weights2uazvpibasg/gddwjadkgr/forget_gate_peephole_weights2uazvpibasg/gddwjadkgr/output_gate_peephole_weightssupobtndkp/kernelsupobtndkp/bias*
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
GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_767389
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%ggbvrcpxtu/kernel/Read/ReadVariableOp#ggbvrcpxtu/bias/Read/ReadVariableOp%supobtndkp/kernel/Read/ReadVariableOp#supobtndkp/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0gdmdltnblf/duhsngmesj/kernel/Read/ReadVariableOp:gdmdltnblf/duhsngmesj/recurrent_kernel/Read/ReadVariableOp.gdmdltnblf/duhsngmesj/bias/Read/ReadVariableOpEgdmdltnblf/duhsngmesj/input_gate_peephole_weights/Read/ReadVariableOpFgdmdltnblf/duhsngmesj/forget_gate_peephole_weights/Read/ReadVariableOpFgdmdltnblf/duhsngmesj/output_gate_peephole_weights/Read/ReadVariableOp0uazvpibasg/gddwjadkgr/kernel/Read/ReadVariableOp:uazvpibasg/gddwjadkgr/recurrent_kernel/Read/ReadVariableOp.uazvpibasg/gddwjadkgr/bias/Read/ReadVariableOpEuazvpibasg/gddwjadkgr/input_gate_peephole_weights/Read/ReadVariableOpFuazvpibasg/gddwjadkgr/forget_gate_peephole_weights/Read/ReadVariableOpFuazvpibasg/gddwjadkgr/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/ggbvrcpxtu/kernel/rms/Read/ReadVariableOp/RMSprop/ggbvrcpxtu/bias/rms/Read/ReadVariableOp1RMSprop/supobtndkp/kernel/rms/Read/ReadVariableOp/RMSprop/supobtndkp/bias/rms/Read/ReadVariableOp<RMSprop/gdmdltnblf/duhsngmesj/kernel/rms/Read/ReadVariableOpFRMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/gdmdltnblf/duhsngmesj/bias/rms/Read/ReadVariableOpQRMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/uazvpibasg/gddwjadkgr/kernel/rms/Read/ReadVariableOpFRMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/uazvpibasg/gddwjadkgr/bias/rms/Read/ReadVariableOpQRMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_770338
å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameggbvrcpxtu/kernelggbvrcpxtu/biassupobtndkp/kernelsupobtndkp/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhogdmdltnblf/duhsngmesj/kernel&gdmdltnblf/duhsngmesj/recurrent_kernelgdmdltnblf/duhsngmesj/bias1gdmdltnblf/duhsngmesj/input_gate_peephole_weights2gdmdltnblf/duhsngmesj/forget_gate_peephole_weights2gdmdltnblf/duhsngmesj/output_gate_peephole_weightsuazvpibasg/gddwjadkgr/kernel&uazvpibasg/gddwjadkgr/recurrent_kerneluazvpibasg/gddwjadkgr/bias1uazvpibasg/gddwjadkgr/input_gate_peephole_weights2uazvpibasg/gddwjadkgr/forget_gate_peephole_weights2uazvpibasg/gddwjadkgr/output_gate_peephole_weightstotalcountRMSprop/ggbvrcpxtu/kernel/rmsRMSprop/ggbvrcpxtu/bias/rmsRMSprop/supobtndkp/kernel/rmsRMSprop/supobtndkp/bias/rms(RMSprop/gdmdltnblf/duhsngmesj/kernel/rms2RMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rms&RMSprop/gdmdltnblf/duhsngmesj/bias/rms=RMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rms>RMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rms>RMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rms(RMSprop/uazvpibasg/gddwjadkgr/kernel/rms2RMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rms&RMSprop/uazvpibasg/gddwjadkgr/bias/rms=RMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rms>RMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rms>RMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rms*3
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_770465ß-
°'
²
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_765482

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
£h

F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769663

inputs<
)gddwjadkgr_matmul_readvariableop_resource:	 >
+gddwjadkgr_matmul_1_readvariableop_resource:	 9
*gddwjadkgr_biasadd_readvariableop_resource:	0
"gddwjadkgr_readvariableop_resource: 2
$gddwjadkgr_readvariableop_1_resource: 2
$gddwjadkgr_readvariableop_2_resource: 
identity¢!gddwjadkgr/BiasAdd/ReadVariableOp¢ gddwjadkgr/MatMul/ReadVariableOp¢"gddwjadkgr/MatMul_1/ReadVariableOp¢gddwjadkgr/ReadVariableOp¢gddwjadkgr/ReadVariableOp_1¢gddwjadkgr/ReadVariableOp_2¢whileD
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
 gddwjadkgr/MatMul/ReadVariableOpReadVariableOp)gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 gddwjadkgr/MatMul/ReadVariableOp§
gddwjadkgr/MatMulMatMulstrided_slice_2:output:0(gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMulµ
"gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp+gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"gddwjadkgr/MatMul_1/ReadVariableOp£
gddwjadkgr/MatMul_1MatMulzeros:output:0*gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMul_1
gddwjadkgr/addAddV2gddwjadkgr/MatMul:product:0gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/add®
!gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp*gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!gddwjadkgr/BiasAdd/ReadVariableOp¥
gddwjadkgr/BiasAddBiasAddgddwjadkgr/add:z:0)gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/BiasAddz
gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gddwjadkgr/split/split_dimë
gddwjadkgr/splitSplit#gddwjadkgr/split/split_dim:output:0gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gddwjadkgr/split
gddwjadkgr/ReadVariableOpReadVariableOp"gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp
gddwjadkgr/mulMul!gddwjadkgr/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul
gddwjadkgr/add_1AddV2gddwjadkgr/split:output:0gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_1{
gddwjadkgr/SigmoidSigmoidgddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid
gddwjadkgr/ReadVariableOp_1ReadVariableOp$gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_1
gddwjadkgr/mul_1Mul#gddwjadkgr/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_1
gddwjadkgr/add_2AddV2gddwjadkgr/split:output:1gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_2
gddwjadkgr/Sigmoid_1Sigmoidgddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_1
gddwjadkgr/mul_2Mulgddwjadkgr/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_2w
gddwjadkgr/TanhTanhgddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh
gddwjadkgr/mul_3Mulgddwjadkgr/Sigmoid:y:0gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_3
gddwjadkgr/add_3AddV2gddwjadkgr/mul_2:z:0gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_3
gddwjadkgr/ReadVariableOp_2ReadVariableOp$gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_2
gddwjadkgr/mul_4Mul#gddwjadkgr/ReadVariableOp_2:value:0gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_4
gddwjadkgr/add_4AddV2gddwjadkgr/split:output:3gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_4
gddwjadkgr/Sigmoid_2Sigmoidgddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_2v
gddwjadkgr/Tanh_1Tanhgddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh_1
gddwjadkgr/mul_5Mulgddwjadkgr/Sigmoid_2:y:0gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gddwjadkgr_matmul_readvariableop_resource+gddwjadkgr_matmul_1_readvariableop_resource*gddwjadkgr_biasadd_readvariableop_resource"gddwjadkgr_readvariableop_resource$gddwjadkgr_readvariableop_1_resource$gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_769562*
condR
while_cond_769561*Q
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
IdentityIdentitystrided_slice_3:output:0"^gddwjadkgr/BiasAdd/ReadVariableOp!^gddwjadkgr/MatMul/ReadVariableOp#^gddwjadkgr/MatMul_1/ReadVariableOp^gddwjadkgr/ReadVariableOp^gddwjadkgr/ReadVariableOp_1^gddwjadkgr/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gddwjadkgr/BiasAdd/ReadVariableOp!gddwjadkgr/BiasAdd/ReadVariableOp2D
 gddwjadkgr/MatMul/ReadVariableOp gddwjadkgr/MatMul/ReadVariableOp2H
"gddwjadkgr/MatMul_1/ReadVariableOp"gddwjadkgr/MatMul_1/ReadVariableOp26
gddwjadkgr/ReadVariableOpgddwjadkgr/ReadVariableOp2:
gddwjadkgr/ReadVariableOp_1gddwjadkgr/ReadVariableOp_12:
gddwjadkgr/ReadVariableOp_2gddwjadkgr/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
p
É
gdmdltnblf_while_body_7679142
.gdmdltnblf_while_gdmdltnblf_while_loop_counter8
4gdmdltnblf_while_gdmdltnblf_while_maximum_iterations 
gdmdltnblf_while_placeholder"
gdmdltnblf_while_placeholder_1"
gdmdltnblf_while_placeholder_2"
gdmdltnblf_while_placeholder_31
-gdmdltnblf_while_gdmdltnblf_strided_slice_1_0m
igdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0O
<gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0:	Q
>gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0:	 L
=gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0:	C
5gdmdltnblf_while_duhsngmesj_readvariableop_resource_0: E
7gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0: E
7gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0: 
gdmdltnblf_while_identity
gdmdltnblf_while_identity_1
gdmdltnblf_while_identity_2
gdmdltnblf_while_identity_3
gdmdltnblf_while_identity_4
gdmdltnblf_while_identity_5/
+gdmdltnblf_while_gdmdltnblf_strided_slice_1k
ggdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensorM
:gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource:	O
<gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource:	 J
;gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource:	A
3gdmdltnblf_while_duhsngmesj_readvariableop_resource: C
5gdmdltnblf_while_duhsngmesj_readvariableop_1_resource: C
5gdmdltnblf_while_duhsngmesj_readvariableop_2_resource: ¢2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp¢1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp¢3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp¢*gdmdltnblf/while/duhsngmesj/ReadVariableOp¢,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1¢,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2Ù
Bgdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bgdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shape
4gdmdltnblf/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemigdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0gdmdltnblf_while_placeholderKgdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4gdmdltnblf/while/TensorArrayV2Read/TensorListGetItemä
1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp<gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOpý
"gdmdltnblf/while/duhsngmesj/MatMulMatMul;gdmdltnblf/while/TensorArrayV2Read/TensorListGetItem:item:09gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"gdmdltnblf/while/duhsngmesj/MatMulê
3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp>gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOpæ
$gdmdltnblf/while/duhsngmesj/MatMul_1MatMulgdmdltnblf_while_placeholder_2;gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$gdmdltnblf/while/duhsngmesj/MatMul_1Ü
gdmdltnblf/while/duhsngmesj/addAddV2,gdmdltnblf/while/duhsngmesj/MatMul:product:0.gdmdltnblf/while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
gdmdltnblf/while/duhsngmesj/addã
2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp=gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOpé
#gdmdltnblf/while/duhsngmesj/BiasAddBiasAdd#gdmdltnblf/while/duhsngmesj/add:z:0:gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#gdmdltnblf/while/duhsngmesj/BiasAdd
+gdmdltnblf/while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+gdmdltnblf/while/duhsngmesj/split/split_dim¯
!gdmdltnblf/while/duhsngmesj/splitSplit4gdmdltnblf/while/duhsngmesj/split/split_dim:output:0,gdmdltnblf/while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!gdmdltnblf/while/duhsngmesj/splitÊ
*gdmdltnblf/while/duhsngmesj/ReadVariableOpReadVariableOp5gdmdltnblf_while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*gdmdltnblf/while/duhsngmesj/ReadVariableOpÏ
gdmdltnblf/while/duhsngmesj/mulMul2gdmdltnblf/while/duhsngmesj/ReadVariableOp:value:0gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gdmdltnblf/while/duhsngmesj/mulÒ
!gdmdltnblf/while/duhsngmesj/add_1AddV2*gdmdltnblf/while/duhsngmesj/split:output:0#gdmdltnblf/while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_1®
#gdmdltnblf/while/duhsngmesj/SigmoidSigmoid%gdmdltnblf/while/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gdmdltnblf/while/duhsngmesj/SigmoidÐ
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1ReadVariableOp7gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1Õ
!gdmdltnblf/while/duhsngmesj/mul_1Mul4gdmdltnblf/while/duhsngmesj/ReadVariableOp_1:value:0gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_1Ô
!gdmdltnblf/while/duhsngmesj/add_2AddV2*gdmdltnblf/while/duhsngmesj/split:output:1%gdmdltnblf/while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_2²
%gdmdltnblf/while/duhsngmesj/Sigmoid_1Sigmoid%gdmdltnblf/while/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%gdmdltnblf/while/duhsngmesj/Sigmoid_1Ê
!gdmdltnblf/while/duhsngmesj/mul_2Mul)gdmdltnblf/while/duhsngmesj/Sigmoid_1:y:0gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_2ª
 gdmdltnblf/while/duhsngmesj/TanhTanh*gdmdltnblf/while/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 gdmdltnblf/while/duhsngmesj/TanhÎ
!gdmdltnblf/while/duhsngmesj/mul_3Mul'gdmdltnblf/while/duhsngmesj/Sigmoid:y:0$gdmdltnblf/while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_3Ï
!gdmdltnblf/while/duhsngmesj/add_3AddV2%gdmdltnblf/while/duhsngmesj/mul_2:z:0%gdmdltnblf/while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_3Ð
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2ReadVariableOp7gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2Ü
!gdmdltnblf/while/duhsngmesj/mul_4Mul4gdmdltnblf/while/duhsngmesj/ReadVariableOp_2:value:0%gdmdltnblf/while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_4Ô
!gdmdltnblf/while/duhsngmesj/add_4AddV2*gdmdltnblf/while/duhsngmesj/split:output:3%gdmdltnblf/while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_4²
%gdmdltnblf/while/duhsngmesj/Sigmoid_2Sigmoid%gdmdltnblf/while/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%gdmdltnblf/while/duhsngmesj/Sigmoid_2©
"gdmdltnblf/while/duhsngmesj/Tanh_1Tanh%gdmdltnblf/while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gdmdltnblf/while/duhsngmesj/Tanh_1Ò
!gdmdltnblf/while/duhsngmesj/mul_5Mul)gdmdltnblf/while/duhsngmesj/Sigmoid_2:y:0&gdmdltnblf/while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_5
5gdmdltnblf/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgdmdltnblf_while_placeholder_1gdmdltnblf_while_placeholder%gdmdltnblf/while/duhsngmesj/mul_5:z:0*
_output_shapes
: *
element_dtype027
5gdmdltnblf/while/TensorArrayV2Write/TensorListSetItemr
gdmdltnblf/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gdmdltnblf/while/add/y
gdmdltnblf/while/addAddV2gdmdltnblf_while_placeholdergdmdltnblf/while/add/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/while/addv
gdmdltnblf/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gdmdltnblf/while/add_1/y­
gdmdltnblf/while/add_1AddV2.gdmdltnblf_while_gdmdltnblf_while_loop_counter!gdmdltnblf/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/while/add_1©
gdmdltnblf/while/IdentityIdentitygdmdltnblf/while/add_1:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/IdentityÇ
gdmdltnblf/while/Identity_1Identity4gdmdltnblf_while_gdmdltnblf_while_maximum_iterations3^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/Identity_1«
gdmdltnblf/while/Identity_2Identitygdmdltnblf/while/add:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/Identity_2Ø
gdmdltnblf/while/Identity_3IdentityEgdmdltnblf/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/Identity_3É
gdmdltnblf/while/Identity_4Identity%gdmdltnblf/while/duhsngmesj/mul_5:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/while/Identity_4É
gdmdltnblf/while/Identity_5Identity%gdmdltnblf/while/duhsngmesj/add_3:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/while/Identity_5"|
;gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource=gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0"~
<gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource>gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0"z
:gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource<gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0"p
5gdmdltnblf_while_duhsngmesj_readvariableop_1_resource7gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0"p
5gdmdltnblf_while_duhsngmesj_readvariableop_2_resource7gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0"l
3gdmdltnblf_while_duhsngmesj_readvariableop_resource5gdmdltnblf_while_duhsngmesj_readvariableop_resource_0"\
+gdmdltnblf_while_gdmdltnblf_strided_slice_1-gdmdltnblf_while_gdmdltnblf_strided_slice_1_0"?
gdmdltnblf_while_identity"gdmdltnblf/while/Identity:output:0"C
gdmdltnblf_while_identity_1$gdmdltnblf/while/Identity_1:output:0"C
gdmdltnblf_while_identity_2$gdmdltnblf/while/Identity_2:output:0"C
gdmdltnblf_while_identity_3$gdmdltnblf/while/Identity_3:output:0"C
gdmdltnblf_while_identity_4$gdmdltnblf/while/Identity_4:output:0"C
gdmdltnblf_while_identity_5$gdmdltnblf/while/Identity_5:output:0"Ô
ggdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensorigdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2f
1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp2j
3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp2X
*gdmdltnblf/while/duhsngmesj/ReadVariableOp*gdmdltnblf/while/duhsngmesj/ReadVariableOp2\
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1,gdmdltnblf/while/duhsngmesj/ReadVariableOp_12\
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2: 
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
°'
²
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_764724

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
¸'
´
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_770108

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
ßY

while_body_768414
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_duhsngmesj_matmul_readvariableop_resource_0:	F
3while_duhsngmesj_matmul_1_readvariableop_resource_0:	 A
2while_duhsngmesj_biasadd_readvariableop_resource_0:	8
*while_duhsngmesj_readvariableop_resource_0: :
,while_duhsngmesj_readvariableop_1_resource_0: :
,while_duhsngmesj_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_duhsngmesj_matmul_readvariableop_resource:	D
1while_duhsngmesj_matmul_1_readvariableop_resource:	 ?
0while_duhsngmesj_biasadd_readvariableop_resource:	6
(while_duhsngmesj_readvariableop_resource: 8
*while_duhsngmesj_readvariableop_1_resource: 8
*while_duhsngmesj_readvariableop_2_resource: ¢'while/duhsngmesj/BiasAdd/ReadVariableOp¢&while/duhsngmesj/MatMul/ReadVariableOp¢(while/duhsngmesj/MatMul_1/ReadVariableOp¢while/duhsngmesj/ReadVariableOp¢!while/duhsngmesj/ReadVariableOp_1¢!while/duhsngmesj/ReadVariableOp_2Ã
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
&while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp1while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/duhsngmesj/MatMul/ReadVariableOpÑ
while/duhsngmesj/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMulÉ
(while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp3while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/duhsngmesj/MatMul_1/ReadVariableOpº
while/duhsngmesj/MatMul_1MatMulwhile_placeholder_20while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMul_1°
while/duhsngmesj/addAddV2!while/duhsngmesj/MatMul:product:0#while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/addÂ
'while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp2while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/duhsngmesj/BiasAdd/ReadVariableOp½
while/duhsngmesj/BiasAddBiasAddwhile/duhsngmesj/add:z:0/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/BiasAdd
 while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/duhsngmesj/split/split_dim
while/duhsngmesj/splitSplit)while/duhsngmesj/split/split_dim:output:0!while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/duhsngmesj/split©
while/duhsngmesj/ReadVariableOpReadVariableOp*while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/duhsngmesj/ReadVariableOp£
while/duhsngmesj/mulMul'while/duhsngmesj/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul¦
while/duhsngmesj/add_1AddV2while/duhsngmesj/split:output:0while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_1
while/duhsngmesj/SigmoidSigmoidwhile/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid¯
!while/duhsngmesj/ReadVariableOp_1ReadVariableOp,while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_1©
while/duhsngmesj/mul_1Mul)while/duhsngmesj/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_1¨
while/duhsngmesj/add_2AddV2while/duhsngmesj/split:output:1while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_2
while/duhsngmesj/Sigmoid_1Sigmoidwhile/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_1
while/duhsngmesj/mul_2Mulwhile/duhsngmesj/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_2
while/duhsngmesj/TanhTanhwhile/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh¢
while/duhsngmesj/mul_3Mulwhile/duhsngmesj/Sigmoid:y:0while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_3£
while/duhsngmesj/add_3AddV2while/duhsngmesj/mul_2:z:0while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_3¯
!while/duhsngmesj/ReadVariableOp_2ReadVariableOp,while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_2°
while/duhsngmesj/mul_4Mul)while/duhsngmesj/ReadVariableOp_2:value:0while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_4¨
while/duhsngmesj/add_4AddV2while/duhsngmesj/split:output:3while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_4
while/duhsngmesj/Sigmoid_2Sigmoidwhile/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_2
while/duhsngmesj/Tanh_1Tanhwhile/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh_1¦
while/duhsngmesj/mul_5Mulwhile/duhsngmesj/Sigmoid_2:y:0while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/duhsngmesj/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/duhsngmesj/mul_5:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/duhsngmesj/add_3:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_duhsngmesj_biasadd_readvariableop_resource2while_duhsngmesj_biasadd_readvariableop_resource_0"h
1while_duhsngmesj_matmul_1_readvariableop_resource3while_duhsngmesj_matmul_1_readvariableop_resource_0"d
/while_duhsngmesj_matmul_readvariableop_resource1while_duhsngmesj_matmul_readvariableop_resource_0"Z
*while_duhsngmesj_readvariableop_1_resource,while_duhsngmesj_readvariableop_1_resource_0"Z
*while_duhsngmesj_readvariableop_2_resource,while_duhsngmesj_readvariableop_2_resource_0"V
(while_duhsngmesj_readvariableop_resource*while_duhsngmesj_readvariableop_resource_0")
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
'while/duhsngmesj/BiasAdd/ReadVariableOp'while/duhsngmesj/BiasAdd/ReadVariableOp2P
&while/duhsngmesj/MatMul/ReadVariableOp&while/duhsngmesj/MatMul/ReadVariableOp2T
(while/duhsngmesj/MatMul_1/ReadVariableOp(while/duhsngmesj/MatMul_1/ReadVariableOp2B
while/duhsngmesj/ReadVariableOpwhile/duhsngmesj/ReadVariableOp2F
!while/duhsngmesj/ReadVariableOp_1!while/duhsngmesj/ReadVariableOp_12F
!while/duhsngmesj/ReadVariableOp_2!while/duhsngmesj/ReadVariableOp_2: 
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
¯
F__inference_sequential_layer_call_and_return_conditional_losses_767303

unopekwlxv'
ggbvrcpxtu_767265:
ggbvrcpxtu_767267:$
gdmdltnblf_767271:	$
gdmdltnblf_767273:	  
gdmdltnblf_767275:	
gdmdltnblf_767277: 
gdmdltnblf_767279: 
gdmdltnblf_767281: $
uazvpibasg_767284:	 $
uazvpibasg_767286:	  
uazvpibasg_767288:	
uazvpibasg_767290: 
uazvpibasg_767292: 
uazvpibasg_767294: #
supobtndkp_767297: 
supobtndkp_767299:
identity¢"gdmdltnblf/StatefulPartitionedCall¢"ggbvrcpxtu/StatefulPartitionedCall¢"supobtndkp/StatefulPartitionedCall¢"uazvpibasg/StatefulPartitionedCall­
"ggbvrcpxtu/StatefulPartitionedCallStatefulPartitionedCall
unopekwlxvggbvrcpxtu_767265ggbvrcpxtu_767267*
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
GPU2*0J 8 *O
fJRH
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_7661972$
"ggbvrcpxtu/StatefulPartitionedCall
vtqejmjhbd/PartitionedCallPartitionedCall+ggbvrcpxtu/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_7662162
vtqejmjhbd/PartitionedCall
"gdmdltnblf/StatefulPartitionedCallStatefulPartitionedCall#vtqejmjhbd/PartitionedCall:output:0gdmdltnblf_767271gdmdltnblf_767273gdmdltnblf_767275gdmdltnblf_767277gdmdltnblf_767279gdmdltnblf_767281*
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7663972$
"gdmdltnblf/StatefulPartitionedCall
"uazvpibasg/StatefulPartitionedCallStatefulPartitionedCall+gdmdltnblf/StatefulPartitionedCall:output:0uazvpibasg_767284uazvpibasg_767286uazvpibasg_767288uazvpibasg_767290uazvpibasg_767292uazvpibasg_767294*
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7665902$
"uazvpibasg/StatefulPartitionedCallÆ
"supobtndkp/StatefulPartitionedCallStatefulPartitionedCall+uazvpibasg/StatefulPartitionedCall:output:0supobtndkp_767297supobtndkp_767299*
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
GPU2*0J 8 *O
fJRH
F__inference_supobtndkp_layer_call_and_return_conditional_losses_7666142$
"supobtndkp/StatefulPartitionedCall
IdentityIdentity+supobtndkp/StatefulPartitionedCall:output:0#^gdmdltnblf/StatefulPartitionedCall#^ggbvrcpxtu/StatefulPartitionedCall#^supobtndkp/StatefulPartitionedCall#^uazvpibasg/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"gdmdltnblf/StatefulPartitionedCall"gdmdltnblf/StatefulPartitionedCall2H
"ggbvrcpxtu/StatefulPartitionedCall"ggbvrcpxtu/StatefulPartitionedCall2H
"supobtndkp/StatefulPartitionedCall"supobtndkp/StatefulPartitionedCall2H
"uazvpibasg/StatefulPartitionedCall"uazvpibasg/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
unopekwlxv

b
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_768330

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


+__inference_sequential_layer_call_fn_767262

unopekwlxv
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
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCall
unopekwlxvunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_7671902
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
unopekwlxv
p
É
uazvpibasg_while_body_7676862
.uazvpibasg_while_uazvpibasg_while_loop_counter8
4uazvpibasg_while_uazvpibasg_while_maximum_iterations 
uazvpibasg_while_placeholder"
uazvpibasg_while_placeholder_1"
uazvpibasg_while_placeholder_2"
uazvpibasg_while_placeholder_31
-uazvpibasg_while_uazvpibasg_strided_slice_1_0m
iuazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0O
<uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0:	 Q
>uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 L
=uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0:	C
5uazvpibasg_while_gddwjadkgr_readvariableop_resource_0: E
7uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0: E
7uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0: 
uazvpibasg_while_identity
uazvpibasg_while_identity_1
uazvpibasg_while_identity_2
uazvpibasg_while_identity_3
uazvpibasg_while_identity_4
uazvpibasg_while_identity_5/
+uazvpibasg_while_uazvpibasg_strided_slice_1k
guazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensorM
:uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource:	 O
<uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource:	 J
;uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource:	A
3uazvpibasg_while_gddwjadkgr_readvariableop_resource: C
5uazvpibasg_while_gddwjadkgr_readvariableop_1_resource: C
5uazvpibasg_while_gddwjadkgr_readvariableop_2_resource: ¢2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp¢1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp¢3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp¢*uazvpibasg/while/gddwjadkgr/ReadVariableOp¢,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1¢,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2Ù
Buazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Buazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shape
4uazvpibasg/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiuazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0uazvpibasg_while_placeholderKuazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4uazvpibasg/while/TensorArrayV2Read/TensorListGetItemä
1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp<uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOpý
"uazvpibasg/while/gddwjadkgr/MatMulMatMul;uazvpibasg/while/TensorArrayV2Read/TensorListGetItem:item:09uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"uazvpibasg/while/gddwjadkgr/MatMulê
3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp>uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOpæ
$uazvpibasg/while/gddwjadkgr/MatMul_1MatMuluazvpibasg_while_placeholder_2;uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$uazvpibasg/while/gddwjadkgr/MatMul_1Ü
uazvpibasg/while/gddwjadkgr/addAddV2,uazvpibasg/while/gddwjadkgr/MatMul:product:0.uazvpibasg/while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
uazvpibasg/while/gddwjadkgr/addã
2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp=uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOpé
#uazvpibasg/while/gddwjadkgr/BiasAddBiasAdd#uazvpibasg/while/gddwjadkgr/add:z:0:uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#uazvpibasg/while/gddwjadkgr/BiasAdd
+uazvpibasg/while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+uazvpibasg/while/gddwjadkgr/split/split_dim¯
!uazvpibasg/while/gddwjadkgr/splitSplit4uazvpibasg/while/gddwjadkgr/split/split_dim:output:0,uazvpibasg/while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!uazvpibasg/while/gddwjadkgr/splitÊ
*uazvpibasg/while/gddwjadkgr/ReadVariableOpReadVariableOp5uazvpibasg_while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*uazvpibasg/while/gddwjadkgr/ReadVariableOpÏ
uazvpibasg/while/gddwjadkgr/mulMul2uazvpibasg/while/gddwjadkgr/ReadVariableOp:value:0uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
uazvpibasg/while/gddwjadkgr/mulÒ
!uazvpibasg/while/gddwjadkgr/add_1AddV2*uazvpibasg/while/gddwjadkgr/split:output:0#uazvpibasg/while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_1®
#uazvpibasg/while/gddwjadkgr/SigmoidSigmoid%uazvpibasg/while/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#uazvpibasg/while/gddwjadkgr/SigmoidÐ
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1ReadVariableOp7uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1Õ
!uazvpibasg/while/gddwjadkgr/mul_1Mul4uazvpibasg/while/gddwjadkgr/ReadVariableOp_1:value:0uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_1Ô
!uazvpibasg/while/gddwjadkgr/add_2AddV2*uazvpibasg/while/gddwjadkgr/split:output:1%uazvpibasg/while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_2²
%uazvpibasg/while/gddwjadkgr/Sigmoid_1Sigmoid%uazvpibasg/while/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%uazvpibasg/while/gddwjadkgr/Sigmoid_1Ê
!uazvpibasg/while/gddwjadkgr/mul_2Mul)uazvpibasg/while/gddwjadkgr/Sigmoid_1:y:0uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_2ª
 uazvpibasg/while/gddwjadkgr/TanhTanh*uazvpibasg/while/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 uazvpibasg/while/gddwjadkgr/TanhÎ
!uazvpibasg/while/gddwjadkgr/mul_3Mul'uazvpibasg/while/gddwjadkgr/Sigmoid:y:0$uazvpibasg/while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_3Ï
!uazvpibasg/while/gddwjadkgr/add_3AddV2%uazvpibasg/while/gddwjadkgr/mul_2:z:0%uazvpibasg/while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_3Ð
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2ReadVariableOp7uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2Ü
!uazvpibasg/while/gddwjadkgr/mul_4Mul4uazvpibasg/while/gddwjadkgr/ReadVariableOp_2:value:0%uazvpibasg/while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_4Ô
!uazvpibasg/while/gddwjadkgr/add_4AddV2*uazvpibasg/while/gddwjadkgr/split:output:3%uazvpibasg/while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_4²
%uazvpibasg/while/gddwjadkgr/Sigmoid_2Sigmoid%uazvpibasg/while/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%uazvpibasg/while/gddwjadkgr/Sigmoid_2©
"uazvpibasg/while/gddwjadkgr/Tanh_1Tanh%uazvpibasg/while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"uazvpibasg/while/gddwjadkgr/Tanh_1Ò
!uazvpibasg/while/gddwjadkgr/mul_5Mul)uazvpibasg/while/gddwjadkgr/Sigmoid_2:y:0&uazvpibasg/while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_5
5uazvpibasg/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemuazvpibasg_while_placeholder_1uazvpibasg_while_placeholder%uazvpibasg/while/gddwjadkgr/mul_5:z:0*
_output_shapes
: *
element_dtype027
5uazvpibasg/while/TensorArrayV2Write/TensorListSetItemr
uazvpibasg/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
uazvpibasg/while/add/y
uazvpibasg/while/addAddV2uazvpibasg_while_placeholderuazvpibasg/while/add/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/while/addv
uazvpibasg/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
uazvpibasg/while/add_1/y­
uazvpibasg/while/add_1AddV2.uazvpibasg_while_uazvpibasg_while_loop_counter!uazvpibasg/while/add_1/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/while/add_1©
uazvpibasg/while/IdentityIdentityuazvpibasg/while/add_1:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/IdentityÇ
uazvpibasg/while/Identity_1Identity4uazvpibasg_while_uazvpibasg_while_maximum_iterations3^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/Identity_1«
uazvpibasg/while/Identity_2Identityuazvpibasg/while/add:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/Identity_2Ø
uazvpibasg/while/Identity_3IdentityEuazvpibasg/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/Identity_3É
uazvpibasg/while/Identity_4Identity%uazvpibasg/while/gddwjadkgr/mul_5:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/while/Identity_4É
uazvpibasg/while/Identity_5Identity%uazvpibasg/while/gddwjadkgr/add_3:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/while/Identity_5"|
;uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource=uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0"~
<uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource>uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0"z
:uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource<uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0"p
5uazvpibasg_while_gddwjadkgr_readvariableop_1_resource7uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0"p
5uazvpibasg_while_gddwjadkgr_readvariableop_2_resource7uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0"l
3uazvpibasg_while_gddwjadkgr_readvariableop_resource5uazvpibasg_while_gddwjadkgr_readvariableop_resource_0"?
uazvpibasg_while_identity"uazvpibasg/while/Identity:output:0"C
uazvpibasg_while_identity_1$uazvpibasg/while/Identity_1:output:0"C
uazvpibasg_while_identity_2$uazvpibasg/while/Identity_2:output:0"C
uazvpibasg_while_identity_3$uazvpibasg/while/Identity_3:output:0"C
uazvpibasg_while_identity_4$uazvpibasg/while/Identity_4:output:0"C
uazvpibasg_while_identity_5$uazvpibasg/while/Identity_5:output:0"Ô
guazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensoriuazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0"\
+uazvpibasg_while_uazvpibasg_strided_slice_1-uazvpibasg_while_uazvpibasg_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2f
1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp2j
3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp2X
*uazvpibasg/while/gddwjadkgr/ReadVariableOp*uazvpibasg/while/gddwjadkgr/ReadVariableOp2\
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1,uazvpibasg/while/gddwjadkgr/ReadVariableOp_12\
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2: 
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
ßY

while_body_766764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_gddwjadkgr_matmul_readvariableop_resource_0:	 F
3while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 A
2while_gddwjadkgr_biasadd_readvariableop_resource_0:	8
*while_gddwjadkgr_readvariableop_resource_0: :
,while_gddwjadkgr_readvariableop_1_resource_0: :
,while_gddwjadkgr_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_gddwjadkgr_matmul_readvariableop_resource:	 D
1while_gddwjadkgr_matmul_1_readvariableop_resource:	 ?
0while_gddwjadkgr_biasadd_readvariableop_resource:	6
(while_gddwjadkgr_readvariableop_resource: 8
*while_gddwjadkgr_readvariableop_1_resource: 8
*while_gddwjadkgr_readvariableop_2_resource: ¢'while/gddwjadkgr/BiasAdd/ReadVariableOp¢&while/gddwjadkgr/MatMul/ReadVariableOp¢(while/gddwjadkgr/MatMul_1/ReadVariableOp¢while/gddwjadkgr/ReadVariableOp¢!while/gddwjadkgr/ReadVariableOp_1¢!while/gddwjadkgr/ReadVariableOp_2Ã
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
&while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp1while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/gddwjadkgr/MatMul/ReadVariableOpÑ
while/gddwjadkgr/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMulÉ
(while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp3while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/gddwjadkgr/MatMul_1/ReadVariableOpº
while/gddwjadkgr/MatMul_1MatMulwhile_placeholder_20while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMul_1°
while/gddwjadkgr/addAddV2!while/gddwjadkgr/MatMul:product:0#while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/addÂ
'while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp2while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/gddwjadkgr/BiasAdd/ReadVariableOp½
while/gddwjadkgr/BiasAddBiasAddwhile/gddwjadkgr/add:z:0/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/BiasAdd
 while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/gddwjadkgr/split/split_dim
while/gddwjadkgr/splitSplit)while/gddwjadkgr/split/split_dim:output:0!while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gddwjadkgr/split©
while/gddwjadkgr/ReadVariableOpReadVariableOp*while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/gddwjadkgr/ReadVariableOp£
while/gddwjadkgr/mulMul'while/gddwjadkgr/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul¦
while/gddwjadkgr/add_1AddV2while/gddwjadkgr/split:output:0while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_1
while/gddwjadkgr/SigmoidSigmoidwhile/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid¯
!while/gddwjadkgr/ReadVariableOp_1ReadVariableOp,while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_1©
while/gddwjadkgr/mul_1Mul)while/gddwjadkgr/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_1¨
while/gddwjadkgr/add_2AddV2while/gddwjadkgr/split:output:1while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_2
while/gddwjadkgr/Sigmoid_1Sigmoidwhile/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_1
while/gddwjadkgr/mul_2Mulwhile/gddwjadkgr/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_2
while/gddwjadkgr/TanhTanhwhile/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh¢
while/gddwjadkgr/mul_3Mulwhile/gddwjadkgr/Sigmoid:y:0while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_3£
while/gddwjadkgr/add_3AddV2while/gddwjadkgr/mul_2:z:0while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_3¯
!while/gddwjadkgr/ReadVariableOp_2ReadVariableOp,while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_2°
while/gddwjadkgr/mul_4Mul)while/gddwjadkgr/ReadVariableOp_2:value:0while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_4¨
while/gddwjadkgr/add_4AddV2while/gddwjadkgr/split:output:3while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_4
while/gddwjadkgr/Sigmoid_2Sigmoidwhile/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_2
while/gddwjadkgr/Tanh_1Tanhwhile/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh_1¦
while/gddwjadkgr/mul_5Mulwhile/gddwjadkgr/Sigmoid_2:y:0while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gddwjadkgr/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/gddwjadkgr/mul_5:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/gddwjadkgr/add_3:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_gddwjadkgr_biasadd_readvariableop_resource2while_gddwjadkgr_biasadd_readvariableop_resource_0"h
1while_gddwjadkgr_matmul_1_readvariableop_resource3while_gddwjadkgr_matmul_1_readvariableop_resource_0"d
/while_gddwjadkgr_matmul_readvariableop_resource1while_gddwjadkgr_matmul_readvariableop_resource_0"Z
*while_gddwjadkgr_readvariableop_1_resource,while_gddwjadkgr_readvariableop_1_resource_0"Z
*while_gddwjadkgr_readvariableop_2_resource,while_gddwjadkgr_readvariableop_2_resource_0"V
(while_gddwjadkgr_readvariableop_resource*while_gddwjadkgr_readvariableop_resource_0")
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
'while/gddwjadkgr/BiasAdd/ReadVariableOp'while/gddwjadkgr/BiasAdd/ReadVariableOp2P
&while/gddwjadkgr/MatMul/ReadVariableOp&while/gddwjadkgr/MatMul/ReadVariableOp2T
(while/gddwjadkgr/MatMul_1/ReadVariableOp(while/gddwjadkgr/MatMul_1/ReadVariableOp2B
while/gddwjadkgr/ReadVariableOpwhile/gddwjadkgr/ReadVariableOp2F
!while/gddwjadkgr/ReadVariableOp_1!while/gddwjadkgr/ReadVariableOp_12F
!while/gddwjadkgr/ReadVariableOp_2!while/gddwjadkgr/ReadVariableOp_2: 
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
¨0
»
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_768308

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


å
while_cond_769741
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_769741___redundant_placeholder04
0while_while_cond_769741___redundant_placeholder14
0while_while_cond_769741___redundant_placeholder24
0while_while_cond_769741___redundant_placeholder34
0while_while_cond_769741___redundant_placeholder44
0while_while_cond_769741___redundant_placeholder54
0while_while_cond_769741___redundant_placeholder6
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
ÿ
¿
+__inference_gddwjadkgr_layer_call_fn_770198

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

identity_2¢StatefulPartitionedCallë
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
GPU2*0J 8 *O
fJRH
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_7656692
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
p
É
uazvpibasg_while_body_7680902
.uazvpibasg_while_uazvpibasg_while_loop_counter8
4uazvpibasg_while_uazvpibasg_while_maximum_iterations 
uazvpibasg_while_placeholder"
uazvpibasg_while_placeholder_1"
uazvpibasg_while_placeholder_2"
uazvpibasg_while_placeholder_31
-uazvpibasg_while_uazvpibasg_strided_slice_1_0m
iuazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0O
<uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0:	 Q
>uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 L
=uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0:	C
5uazvpibasg_while_gddwjadkgr_readvariableop_resource_0: E
7uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0: E
7uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0: 
uazvpibasg_while_identity
uazvpibasg_while_identity_1
uazvpibasg_while_identity_2
uazvpibasg_while_identity_3
uazvpibasg_while_identity_4
uazvpibasg_while_identity_5/
+uazvpibasg_while_uazvpibasg_strided_slice_1k
guazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensorM
:uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource:	 O
<uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource:	 J
;uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource:	A
3uazvpibasg_while_gddwjadkgr_readvariableop_resource: C
5uazvpibasg_while_gddwjadkgr_readvariableop_1_resource: C
5uazvpibasg_while_gddwjadkgr_readvariableop_2_resource: ¢2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp¢1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp¢3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp¢*uazvpibasg/while/gddwjadkgr/ReadVariableOp¢,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1¢,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2Ù
Buazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Buazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shape
4uazvpibasg/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiuazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0uazvpibasg_while_placeholderKuazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4uazvpibasg/while/TensorArrayV2Read/TensorListGetItemä
1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp<uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOpý
"uazvpibasg/while/gddwjadkgr/MatMulMatMul;uazvpibasg/while/TensorArrayV2Read/TensorListGetItem:item:09uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"uazvpibasg/while/gddwjadkgr/MatMulê
3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp>uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOpæ
$uazvpibasg/while/gddwjadkgr/MatMul_1MatMuluazvpibasg_while_placeholder_2;uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$uazvpibasg/while/gddwjadkgr/MatMul_1Ü
uazvpibasg/while/gddwjadkgr/addAddV2,uazvpibasg/while/gddwjadkgr/MatMul:product:0.uazvpibasg/while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
uazvpibasg/while/gddwjadkgr/addã
2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp=uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOpé
#uazvpibasg/while/gddwjadkgr/BiasAddBiasAdd#uazvpibasg/while/gddwjadkgr/add:z:0:uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#uazvpibasg/while/gddwjadkgr/BiasAdd
+uazvpibasg/while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+uazvpibasg/while/gddwjadkgr/split/split_dim¯
!uazvpibasg/while/gddwjadkgr/splitSplit4uazvpibasg/while/gddwjadkgr/split/split_dim:output:0,uazvpibasg/while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!uazvpibasg/while/gddwjadkgr/splitÊ
*uazvpibasg/while/gddwjadkgr/ReadVariableOpReadVariableOp5uazvpibasg_while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*uazvpibasg/while/gddwjadkgr/ReadVariableOpÏ
uazvpibasg/while/gddwjadkgr/mulMul2uazvpibasg/while/gddwjadkgr/ReadVariableOp:value:0uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
uazvpibasg/while/gddwjadkgr/mulÒ
!uazvpibasg/while/gddwjadkgr/add_1AddV2*uazvpibasg/while/gddwjadkgr/split:output:0#uazvpibasg/while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_1®
#uazvpibasg/while/gddwjadkgr/SigmoidSigmoid%uazvpibasg/while/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#uazvpibasg/while/gddwjadkgr/SigmoidÐ
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1ReadVariableOp7uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1Õ
!uazvpibasg/while/gddwjadkgr/mul_1Mul4uazvpibasg/while/gddwjadkgr/ReadVariableOp_1:value:0uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_1Ô
!uazvpibasg/while/gddwjadkgr/add_2AddV2*uazvpibasg/while/gddwjadkgr/split:output:1%uazvpibasg/while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_2²
%uazvpibasg/while/gddwjadkgr/Sigmoid_1Sigmoid%uazvpibasg/while/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%uazvpibasg/while/gddwjadkgr/Sigmoid_1Ê
!uazvpibasg/while/gddwjadkgr/mul_2Mul)uazvpibasg/while/gddwjadkgr/Sigmoid_1:y:0uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_2ª
 uazvpibasg/while/gddwjadkgr/TanhTanh*uazvpibasg/while/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 uazvpibasg/while/gddwjadkgr/TanhÎ
!uazvpibasg/while/gddwjadkgr/mul_3Mul'uazvpibasg/while/gddwjadkgr/Sigmoid:y:0$uazvpibasg/while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_3Ï
!uazvpibasg/while/gddwjadkgr/add_3AddV2%uazvpibasg/while/gddwjadkgr/mul_2:z:0%uazvpibasg/while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_3Ð
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2ReadVariableOp7uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2Ü
!uazvpibasg/while/gddwjadkgr/mul_4Mul4uazvpibasg/while/gddwjadkgr/ReadVariableOp_2:value:0%uazvpibasg/while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_4Ô
!uazvpibasg/while/gddwjadkgr/add_4AddV2*uazvpibasg/while/gddwjadkgr/split:output:3%uazvpibasg/while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/add_4²
%uazvpibasg/while/gddwjadkgr/Sigmoid_2Sigmoid%uazvpibasg/while/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%uazvpibasg/while/gddwjadkgr/Sigmoid_2©
"uazvpibasg/while/gddwjadkgr/Tanh_1Tanh%uazvpibasg/while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"uazvpibasg/while/gddwjadkgr/Tanh_1Ò
!uazvpibasg/while/gddwjadkgr/mul_5Mul)uazvpibasg/while/gddwjadkgr/Sigmoid_2:y:0&uazvpibasg/while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!uazvpibasg/while/gddwjadkgr/mul_5
5uazvpibasg/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemuazvpibasg_while_placeholder_1uazvpibasg_while_placeholder%uazvpibasg/while/gddwjadkgr/mul_5:z:0*
_output_shapes
: *
element_dtype027
5uazvpibasg/while/TensorArrayV2Write/TensorListSetItemr
uazvpibasg/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
uazvpibasg/while/add/y
uazvpibasg/while/addAddV2uazvpibasg_while_placeholderuazvpibasg/while/add/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/while/addv
uazvpibasg/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
uazvpibasg/while/add_1/y­
uazvpibasg/while/add_1AddV2.uazvpibasg_while_uazvpibasg_while_loop_counter!uazvpibasg/while/add_1/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/while/add_1©
uazvpibasg/while/IdentityIdentityuazvpibasg/while/add_1:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/IdentityÇ
uazvpibasg/while/Identity_1Identity4uazvpibasg_while_uazvpibasg_while_maximum_iterations3^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/Identity_1«
uazvpibasg/while/Identity_2Identityuazvpibasg/while/add:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/Identity_2Ø
uazvpibasg/while/Identity_3IdentityEuazvpibasg/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
uazvpibasg/while/Identity_3É
uazvpibasg/while/Identity_4Identity%uazvpibasg/while/gddwjadkgr/mul_5:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/while/Identity_4É
uazvpibasg/while/Identity_5Identity%uazvpibasg/while/gddwjadkgr/add_3:z:03^uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2^uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp4^uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp+^uazvpibasg/while/gddwjadkgr/ReadVariableOp-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_1-^uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/while/Identity_5"|
;uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource=uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0"~
<uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource>uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0"z
:uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource<uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0"p
5uazvpibasg_while_gddwjadkgr_readvariableop_1_resource7uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0"p
5uazvpibasg_while_gddwjadkgr_readvariableop_2_resource7uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0"l
3uazvpibasg_while_gddwjadkgr_readvariableop_resource5uazvpibasg_while_gddwjadkgr_readvariableop_resource_0"?
uazvpibasg_while_identity"uazvpibasg/while/Identity:output:0"C
uazvpibasg_while_identity_1$uazvpibasg/while/Identity_1:output:0"C
uazvpibasg_while_identity_2$uazvpibasg/while/Identity_2:output:0"C
uazvpibasg_while_identity_3$uazvpibasg/while/Identity_3:output:0"C
uazvpibasg_while_identity_4$uazvpibasg/while/Identity_4:output:0"C
uazvpibasg_while_identity_5$uazvpibasg/while/Identity_5:output:0"Ô
guazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensoriuazvpibasg_while_tensorarrayv2read_tensorlistgetitem_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0"\
+uazvpibasg_while_uazvpibasg_strided_slice_1-uazvpibasg_while_uazvpibasg_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2f
1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp1uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp2j
3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp3uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp2X
*uazvpibasg/while/gddwjadkgr/ReadVariableOp*uazvpibasg/while/gddwjadkgr/ReadVariableOp2\
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_1,uazvpibasg/while/gddwjadkgr/ReadVariableOp_12\
,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2,uazvpibasg/while/gddwjadkgr/ReadVariableOp_2: 
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
ßY

while_body_769202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_gddwjadkgr_matmul_readvariableop_resource_0:	 F
3while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 A
2while_gddwjadkgr_biasadd_readvariableop_resource_0:	8
*while_gddwjadkgr_readvariableop_resource_0: :
,while_gddwjadkgr_readvariableop_1_resource_0: :
,while_gddwjadkgr_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_gddwjadkgr_matmul_readvariableop_resource:	 D
1while_gddwjadkgr_matmul_1_readvariableop_resource:	 ?
0while_gddwjadkgr_biasadd_readvariableop_resource:	6
(while_gddwjadkgr_readvariableop_resource: 8
*while_gddwjadkgr_readvariableop_1_resource: 8
*while_gddwjadkgr_readvariableop_2_resource: ¢'while/gddwjadkgr/BiasAdd/ReadVariableOp¢&while/gddwjadkgr/MatMul/ReadVariableOp¢(while/gddwjadkgr/MatMul_1/ReadVariableOp¢while/gddwjadkgr/ReadVariableOp¢!while/gddwjadkgr/ReadVariableOp_1¢!while/gddwjadkgr/ReadVariableOp_2Ã
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
&while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp1while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/gddwjadkgr/MatMul/ReadVariableOpÑ
while/gddwjadkgr/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMulÉ
(while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp3while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/gddwjadkgr/MatMul_1/ReadVariableOpº
while/gddwjadkgr/MatMul_1MatMulwhile_placeholder_20while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMul_1°
while/gddwjadkgr/addAddV2!while/gddwjadkgr/MatMul:product:0#while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/addÂ
'while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp2while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/gddwjadkgr/BiasAdd/ReadVariableOp½
while/gddwjadkgr/BiasAddBiasAddwhile/gddwjadkgr/add:z:0/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/BiasAdd
 while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/gddwjadkgr/split/split_dim
while/gddwjadkgr/splitSplit)while/gddwjadkgr/split/split_dim:output:0!while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gddwjadkgr/split©
while/gddwjadkgr/ReadVariableOpReadVariableOp*while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/gddwjadkgr/ReadVariableOp£
while/gddwjadkgr/mulMul'while/gddwjadkgr/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul¦
while/gddwjadkgr/add_1AddV2while/gddwjadkgr/split:output:0while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_1
while/gddwjadkgr/SigmoidSigmoidwhile/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid¯
!while/gddwjadkgr/ReadVariableOp_1ReadVariableOp,while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_1©
while/gddwjadkgr/mul_1Mul)while/gddwjadkgr/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_1¨
while/gddwjadkgr/add_2AddV2while/gddwjadkgr/split:output:1while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_2
while/gddwjadkgr/Sigmoid_1Sigmoidwhile/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_1
while/gddwjadkgr/mul_2Mulwhile/gddwjadkgr/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_2
while/gddwjadkgr/TanhTanhwhile/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh¢
while/gddwjadkgr/mul_3Mulwhile/gddwjadkgr/Sigmoid:y:0while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_3£
while/gddwjadkgr/add_3AddV2while/gddwjadkgr/mul_2:z:0while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_3¯
!while/gddwjadkgr/ReadVariableOp_2ReadVariableOp,while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_2°
while/gddwjadkgr/mul_4Mul)while/gddwjadkgr/ReadVariableOp_2:value:0while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_4¨
while/gddwjadkgr/add_4AddV2while/gddwjadkgr/split:output:3while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_4
while/gddwjadkgr/Sigmoid_2Sigmoidwhile/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_2
while/gddwjadkgr/Tanh_1Tanhwhile/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh_1¦
while/gddwjadkgr/mul_5Mulwhile/gddwjadkgr/Sigmoid_2:y:0while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gddwjadkgr/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/gddwjadkgr/mul_5:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/gddwjadkgr/add_3:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_gddwjadkgr_biasadd_readvariableop_resource2while_gddwjadkgr_biasadd_readvariableop_resource_0"h
1while_gddwjadkgr_matmul_1_readvariableop_resource3while_gddwjadkgr_matmul_1_readvariableop_resource_0"d
/while_gddwjadkgr_matmul_readvariableop_resource1while_gddwjadkgr_matmul_readvariableop_resource_0"Z
*while_gddwjadkgr_readvariableop_1_resource,while_gddwjadkgr_readvariableop_1_resource_0"Z
*while_gddwjadkgr_readvariableop_2_resource,while_gddwjadkgr_readvariableop_2_resource_0"V
(while_gddwjadkgr_readvariableop_resource*while_gddwjadkgr_readvariableop_resource_0")
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
'while/gddwjadkgr/BiasAdd/ReadVariableOp'while/gddwjadkgr/BiasAdd/ReadVariableOp2P
&while/gddwjadkgr/MatMul/ReadVariableOp&while/gddwjadkgr/MatMul/ReadVariableOp2T
(while/gddwjadkgr/MatMul_1/ReadVariableOp(while/gddwjadkgr/MatMul_1/ReadVariableOp2B
while/gddwjadkgr/ReadVariableOpwhile/gddwjadkgr/ReadVariableOp2F
!while/gddwjadkgr/ReadVariableOp_1!while/gddwjadkgr/ReadVariableOp_12F
!while/gddwjadkgr/ReadVariableOp_2!while/gddwjadkgr/ReadVariableOp_2: 
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
h

F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_769055

inputs<
)duhsngmesj_matmul_readvariableop_resource:	>
+duhsngmesj_matmul_1_readvariableop_resource:	 9
*duhsngmesj_biasadd_readvariableop_resource:	0
"duhsngmesj_readvariableop_resource: 2
$duhsngmesj_readvariableop_1_resource: 2
$duhsngmesj_readvariableop_2_resource: 
identity¢!duhsngmesj/BiasAdd/ReadVariableOp¢ duhsngmesj/MatMul/ReadVariableOp¢"duhsngmesj/MatMul_1/ReadVariableOp¢duhsngmesj/ReadVariableOp¢duhsngmesj/ReadVariableOp_1¢duhsngmesj/ReadVariableOp_2¢whileD
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
 duhsngmesj/MatMul/ReadVariableOpReadVariableOp)duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 duhsngmesj/MatMul/ReadVariableOp§
duhsngmesj/MatMulMatMulstrided_slice_2:output:0(duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMulµ
"duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp+duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"duhsngmesj/MatMul_1/ReadVariableOp£
duhsngmesj/MatMul_1MatMulzeros:output:0*duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMul_1
duhsngmesj/addAddV2duhsngmesj/MatMul:product:0duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/add®
!duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp*duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!duhsngmesj/BiasAdd/ReadVariableOp¥
duhsngmesj/BiasAddBiasAddduhsngmesj/add:z:0)duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/BiasAddz
duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
duhsngmesj/split/split_dimë
duhsngmesj/splitSplit#duhsngmesj/split/split_dim:output:0duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
duhsngmesj/split
duhsngmesj/ReadVariableOpReadVariableOp"duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp
duhsngmesj/mulMul!duhsngmesj/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul
duhsngmesj/add_1AddV2duhsngmesj/split:output:0duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_1{
duhsngmesj/SigmoidSigmoidduhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid
duhsngmesj/ReadVariableOp_1ReadVariableOp$duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_1
duhsngmesj/mul_1Mul#duhsngmesj/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_1
duhsngmesj/add_2AddV2duhsngmesj/split:output:1duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_2
duhsngmesj/Sigmoid_1Sigmoidduhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_1
duhsngmesj/mul_2Mulduhsngmesj/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_2w
duhsngmesj/TanhTanhduhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh
duhsngmesj/mul_3Mulduhsngmesj/Sigmoid:y:0duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_3
duhsngmesj/add_3AddV2duhsngmesj/mul_2:z:0duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_3
duhsngmesj/ReadVariableOp_2ReadVariableOp$duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_2
duhsngmesj/mul_4Mul#duhsngmesj/ReadVariableOp_2:value:0duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_4
duhsngmesj/add_4AddV2duhsngmesj/split:output:3duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_4
duhsngmesj/Sigmoid_2Sigmoidduhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_2v
duhsngmesj/Tanh_1Tanhduhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh_1
duhsngmesj/mul_5Mulduhsngmesj/Sigmoid_2:y:0duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)duhsngmesj_matmul_readvariableop_resource+duhsngmesj_matmul_1_readvariableop_resource*duhsngmesj_biasadd_readvariableop_resource"duhsngmesj_readvariableop_resource$duhsngmesj_readvariableop_1_resource$duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_768954*
condR
while_cond_768953*Q
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
IdentityIdentitytranspose_1:y:0"^duhsngmesj/BiasAdd/ReadVariableOp!^duhsngmesj/MatMul/ReadVariableOp#^duhsngmesj/MatMul_1/ReadVariableOp^duhsngmesj/ReadVariableOp^duhsngmesj/ReadVariableOp_1^duhsngmesj/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!duhsngmesj/BiasAdd/ReadVariableOp!duhsngmesj/BiasAdd/ReadVariableOp2D
 duhsngmesj/MatMul/ReadVariableOp duhsngmesj/MatMul/ReadVariableOp2H
"duhsngmesj/MatMul_1/ReadVariableOp"duhsngmesj/MatMul_1/ReadVariableOp26
duhsngmesj/ReadVariableOpduhsngmesj/ReadVariableOp2:
duhsngmesj/ReadVariableOp_1duhsngmesj/ReadVariableOp_12:
duhsngmesj/ReadVariableOp_2duhsngmesj/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_768953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_768953___redundant_placeholder04
0while_while_cond_768953___redundant_placeholder14
0while_while_cond_768953___redundant_placeholder24
0while_while_cond_768953___redundant_placeholder34
0while_while_cond_768953___redundant_placeholder44
0while_while_cond_768953___redundant_placeholder54
0while_while_cond_768953___redundant_placeholder6
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


å
while_cond_769201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_769201___redundant_placeholder04
0while_while_cond_769201___redundant_placeholder14
0while_while_cond_769201___redundant_placeholder24
0while_while_cond_769201___redundant_placeholder34
0while_while_cond_769201___redundant_placeholder44
0while_while_cond_769201___redundant_placeholder54
0while_while_cond_769201___redundant_placeholder6
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
£h

F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769843

inputs<
)gddwjadkgr_matmul_readvariableop_resource:	 >
+gddwjadkgr_matmul_1_readvariableop_resource:	 9
*gddwjadkgr_biasadd_readvariableop_resource:	0
"gddwjadkgr_readvariableop_resource: 2
$gddwjadkgr_readvariableop_1_resource: 2
$gddwjadkgr_readvariableop_2_resource: 
identity¢!gddwjadkgr/BiasAdd/ReadVariableOp¢ gddwjadkgr/MatMul/ReadVariableOp¢"gddwjadkgr/MatMul_1/ReadVariableOp¢gddwjadkgr/ReadVariableOp¢gddwjadkgr/ReadVariableOp_1¢gddwjadkgr/ReadVariableOp_2¢whileD
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
 gddwjadkgr/MatMul/ReadVariableOpReadVariableOp)gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 gddwjadkgr/MatMul/ReadVariableOp§
gddwjadkgr/MatMulMatMulstrided_slice_2:output:0(gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMulµ
"gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp+gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"gddwjadkgr/MatMul_1/ReadVariableOp£
gddwjadkgr/MatMul_1MatMulzeros:output:0*gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMul_1
gddwjadkgr/addAddV2gddwjadkgr/MatMul:product:0gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/add®
!gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp*gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!gddwjadkgr/BiasAdd/ReadVariableOp¥
gddwjadkgr/BiasAddBiasAddgddwjadkgr/add:z:0)gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/BiasAddz
gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gddwjadkgr/split/split_dimë
gddwjadkgr/splitSplit#gddwjadkgr/split/split_dim:output:0gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gddwjadkgr/split
gddwjadkgr/ReadVariableOpReadVariableOp"gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp
gddwjadkgr/mulMul!gddwjadkgr/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul
gddwjadkgr/add_1AddV2gddwjadkgr/split:output:0gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_1{
gddwjadkgr/SigmoidSigmoidgddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid
gddwjadkgr/ReadVariableOp_1ReadVariableOp$gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_1
gddwjadkgr/mul_1Mul#gddwjadkgr/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_1
gddwjadkgr/add_2AddV2gddwjadkgr/split:output:1gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_2
gddwjadkgr/Sigmoid_1Sigmoidgddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_1
gddwjadkgr/mul_2Mulgddwjadkgr/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_2w
gddwjadkgr/TanhTanhgddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh
gddwjadkgr/mul_3Mulgddwjadkgr/Sigmoid:y:0gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_3
gddwjadkgr/add_3AddV2gddwjadkgr/mul_2:z:0gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_3
gddwjadkgr/ReadVariableOp_2ReadVariableOp$gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_2
gddwjadkgr/mul_4Mul#gddwjadkgr/ReadVariableOp_2:value:0gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_4
gddwjadkgr/add_4AddV2gddwjadkgr/split:output:3gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_4
gddwjadkgr/Sigmoid_2Sigmoidgddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_2v
gddwjadkgr/Tanh_1Tanhgddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh_1
gddwjadkgr/mul_5Mulgddwjadkgr/Sigmoid_2:y:0gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gddwjadkgr_matmul_readvariableop_resource+gddwjadkgr_matmul_1_readvariableop_resource*gddwjadkgr_biasadd_readvariableop_resource"gddwjadkgr_readvariableop_resource$gddwjadkgr_readvariableop_1_resource$gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_769742*
condR
while_cond_769741*Q
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
IdentityIdentitystrided_slice_3:output:0"^gddwjadkgr/BiasAdd/ReadVariableOp!^gddwjadkgr/MatMul/ReadVariableOp#^gddwjadkgr/MatMul_1/ReadVariableOp^gddwjadkgr/ReadVariableOp^gddwjadkgr/ReadVariableOp_1^gddwjadkgr/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gddwjadkgr/BiasAdd/ReadVariableOp!gddwjadkgr/BiasAdd/ReadVariableOp2D
 gddwjadkgr/MatMul/ReadVariableOp gddwjadkgr/MatMul/ReadVariableOp2H
"gddwjadkgr/MatMul_1/ReadVariableOp"gddwjadkgr/MatMul_1/ReadVariableOp26
gddwjadkgr/ReadVariableOpgddwjadkgr/ReadVariableOp2:
gddwjadkgr/ReadVariableOp_1gddwjadkgr/ReadVariableOp_12:
gddwjadkgr/ReadVariableOp_2gddwjadkgr/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÿ
¿
+__inference_duhsngmesj_layer_call_fn_770041

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

identity_2¢StatefulPartitionedCallë
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
GPU2*0J 8 *O
fJRH
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_7647242
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
ßY

while_body_768774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_duhsngmesj_matmul_readvariableop_resource_0:	F
3while_duhsngmesj_matmul_1_readvariableop_resource_0:	 A
2while_duhsngmesj_biasadd_readvariableop_resource_0:	8
*while_duhsngmesj_readvariableop_resource_0: :
,while_duhsngmesj_readvariableop_1_resource_0: :
,while_duhsngmesj_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_duhsngmesj_matmul_readvariableop_resource:	D
1while_duhsngmesj_matmul_1_readvariableop_resource:	 ?
0while_duhsngmesj_biasadd_readvariableop_resource:	6
(while_duhsngmesj_readvariableop_resource: 8
*while_duhsngmesj_readvariableop_1_resource: 8
*while_duhsngmesj_readvariableop_2_resource: ¢'while/duhsngmesj/BiasAdd/ReadVariableOp¢&while/duhsngmesj/MatMul/ReadVariableOp¢(while/duhsngmesj/MatMul_1/ReadVariableOp¢while/duhsngmesj/ReadVariableOp¢!while/duhsngmesj/ReadVariableOp_1¢!while/duhsngmesj/ReadVariableOp_2Ã
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
&while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp1while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/duhsngmesj/MatMul/ReadVariableOpÑ
while/duhsngmesj/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMulÉ
(while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp3while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/duhsngmesj/MatMul_1/ReadVariableOpº
while/duhsngmesj/MatMul_1MatMulwhile_placeholder_20while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMul_1°
while/duhsngmesj/addAddV2!while/duhsngmesj/MatMul:product:0#while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/addÂ
'while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp2while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/duhsngmesj/BiasAdd/ReadVariableOp½
while/duhsngmesj/BiasAddBiasAddwhile/duhsngmesj/add:z:0/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/BiasAdd
 while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/duhsngmesj/split/split_dim
while/duhsngmesj/splitSplit)while/duhsngmesj/split/split_dim:output:0!while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/duhsngmesj/split©
while/duhsngmesj/ReadVariableOpReadVariableOp*while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/duhsngmesj/ReadVariableOp£
while/duhsngmesj/mulMul'while/duhsngmesj/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul¦
while/duhsngmesj/add_1AddV2while/duhsngmesj/split:output:0while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_1
while/duhsngmesj/SigmoidSigmoidwhile/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid¯
!while/duhsngmesj/ReadVariableOp_1ReadVariableOp,while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_1©
while/duhsngmesj/mul_1Mul)while/duhsngmesj/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_1¨
while/duhsngmesj/add_2AddV2while/duhsngmesj/split:output:1while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_2
while/duhsngmesj/Sigmoid_1Sigmoidwhile/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_1
while/duhsngmesj/mul_2Mulwhile/duhsngmesj/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_2
while/duhsngmesj/TanhTanhwhile/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh¢
while/duhsngmesj/mul_3Mulwhile/duhsngmesj/Sigmoid:y:0while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_3£
while/duhsngmesj/add_3AddV2while/duhsngmesj/mul_2:z:0while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_3¯
!while/duhsngmesj/ReadVariableOp_2ReadVariableOp,while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_2°
while/duhsngmesj/mul_4Mul)while/duhsngmesj/ReadVariableOp_2:value:0while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_4¨
while/duhsngmesj/add_4AddV2while/duhsngmesj/split:output:3while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_4
while/duhsngmesj/Sigmoid_2Sigmoidwhile/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_2
while/duhsngmesj/Tanh_1Tanhwhile/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh_1¦
while/duhsngmesj/mul_5Mulwhile/duhsngmesj/Sigmoid_2:y:0while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/duhsngmesj/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/duhsngmesj/mul_5:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/duhsngmesj/add_3:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_duhsngmesj_biasadd_readvariableop_resource2while_duhsngmesj_biasadd_readvariableop_resource_0"h
1while_duhsngmesj_matmul_1_readvariableop_resource3while_duhsngmesj_matmul_1_readvariableop_resource_0"d
/while_duhsngmesj_matmul_readvariableop_resource1while_duhsngmesj_matmul_readvariableop_resource_0"Z
*while_duhsngmesj_readvariableop_1_resource,while_duhsngmesj_readvariableop_1_resource_0"Z
*while_duhsngmesj_readvariableop_2_resource,while_duhsngmesj_readvariableop_2_resource_0"V
(while_duhsngmesj_readvariableop_resource*while_duhsngmesj_readvariableop_resource_0")
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
'while/duhsngmesj/BiasAdd/ReadVariableOp'while/duhsngmesj/BiasAdd/ReadVariableOp2P
&while/duhsngmesj/MatMul/ReadVariableOp&while/duhsngmesj/MatMul/ReadVariableOp2T
(while/duhsngmesj/MatMul_1/ReadVariableOp(while/duhsngmesj/MatMul_1/ReadVariableOp2B
while/duhsngmesj/ReadVariableOpwhile/duhsngmesj/ReadVariableOp2F
!while/duhsngmesj/ReadVariableOp_1!while/duhsngmesj/ReadVariableOp_12F
!while/duhsngmesj/ReadVariableOp_2!while/duhsngmesj/ReadVariableOp_2: 
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
	

+__inference_gdmdltnblf_layer_call_fn_769089
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¼
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7650872
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
ê

$__inference_signature_wrapper_767389

unopekwlxv
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall
unopekwlxvunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8 **
f%R#
!__inference__wrapped_model_7646372
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
unopekwlxv


+__inference_sequential_layer_call_fn_768271

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
identity¢StatefulPartitionedCall´
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_7671902
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


å
while_cond_765006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_765006___redundant_placeholder04
0while_while_cond_765006___redundant_placeholder14
0while_while_cond_765006___redundant_placeholder24
0while_while_cond_765006___redundant_placeholder34
0while_while_cond_765006___redundant_placeholder44
0while_while_cond_765006___redundant_placeholder54
0while_while_cond_765006___redundant_placeholder6
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


å
while_cond_766977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_766977___redundant_placeholder04
0while_while_cond_766977___redundant_placeholder14
0while_while_cond_766977___redundant_placeholder24
0while_while_cond_766977___redundant_placeholder34
0while_while_cond_766977___redundant_placeholder44
0while_while_cond_766977___redundant_placeholder54
0while_while_cond_766977___redundant_placeholder6
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


å
while_cond_765501
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_765501___redundant_placeholder04
0while_while_cond_765501___redundant_placeholder14
0while_while_cond_765501___redundant_placeholder24
0while_while_cond_765501___redundant_placeholder34
0while_while_cond_765501___redundant_placeholder44
0while_while_cond_765501___redundant_placeholder54
0while_while_cond_765501___redundant_placeholder6
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
ßY

while_body_769562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_gddwjadkgr_matmul_readvariableop_resource_0:	 F
3while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 A
2while_gddwjadkgr_biasadd_readvariableop_resource_0:	8
*while_gddwjadkgr_readvariableop_resource_0: :
,while_gddwjadkgr_readvariableop_1_resource_0: :
,while_gddwjadkgr_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_gddwjadkgr_matmul_readvariableop_resource:	 D
1while_gddwjadkgr_matmul_1_readvariableop_resource:	 ?
0while_gddwjadkgr_biasadd_readvariableop_resource:	6
(while_gddwjadkgr_readvariableop_resource: 8
*while_gddwjadkgr_readvariableop_1_resource: 8
*while_gddwjadkgr_readvariableop_2_resource: ¢'while/gddwjadkgr/BiasAdd/ReadVariableOp¢&while/gddwjadkgr/MatMul/ReadVariableOp¢(while/gddwjadkgr/MatMul_1/ReadVariableOp¢while/gddwjadkgr/ReadVariableOp¢!while/gddwjadkgr/ReadVariableOp_1¢!while/gddwjadkgr/ReadVariableOp_2Ã
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
&while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp1while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/gddwjadkgr/MatMul/ReadVariableOpÑ
while/gddwjadkgr/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMulÉ
(while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp3while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/gddwjadkgr/MatMul_1/ReadVariableOpº
while/gddwjadkgr/MatMul_1MatMulwhile_placeholder_20while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMul_1°
while/gddwjadkgr/addAddV2!while/gddwjadkgr/MatMul:product:0#while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/addÂ
'while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp2while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/gddwjadkgr/BiasAdd/ReadVariableOp½
while/gddwjadkgr/BiasAddBiasAddwhile/gddwjadkgr/add:z:0/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/BiasAdd
 while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/gddwjadkgr/split/split_dim
while/gddwjadkgr/splitSplit)while/gddwjadkgr/split/split_dim:output:0!while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gddwjadkgr/split©
while/gddwjadkgr/ReadVariableOpReadVariableOp*while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/gddwjadkgr/ReadVariableOp£
while/gddwjadkgr/mulMul'while/gddwjadkgr/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul¦
while/gddwjadkgr/add_1AddV2while/gddwjadkgr/split:output:0while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_1
while/gddwjadkgr/SigmoidSigmoidwhile/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid¯
!while/gddwjadkgr/ReadVariableOp_1ReadVariableOp,while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_1©
while/gddwjadkgr/mul_1Mul)while/gddwjadkgr/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_1¨
while/gddwjadkgr/add_2AddV2while/gddwjadkgr/split:output:1while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_2
while/gddwjadkgr/Sigmoid_1Sigmoidwhile/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_1
while/gddwjadkgr/mul_2Mulwhile/gddwjadkgr/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_2
while/gddwjadkgr/TanhTanhwhile/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh¢
while/gddwjadkgr/mul_3Mulwhile/gddwjadkgr/Sigmoid:y:0while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_3£
while/gddwjadkgr/add_3AddV2while/gddwjadkgr/mul_2:z:0while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_3¯
!while/gddwjadkgr/ReadVariableOp_2ReadVariableOp,while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_2°
while/gddwjadkgr/mul_4Mul)while/gddwjadkgr/ReadVariableOp_2:value:0while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_4¨
while/gddwjadkgr/add_4AddV2while/gddwjadkgr/split:output:3while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_4
while/gddwjadkgr/Sigmoid_2Sigmoidwhile/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_2
while/gddwjadkgr/Tanh_1Tanhwhile/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh_1¦
while/gddwjadkgr/mul_5Mulwhile/gddwjadkgr/Sigmoid_2:y:0while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gddwjadkgr/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/gddwjadkgr/mul_5:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/gddwjadkgr/add_3:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_gddwjadkgr_biasadd_readvariableop_resource2while_gddwjadkgr_biasadd_readvariableop_resource_0"h
1while_gddwjadkgr_matmul_1_readvariableop_resource3while_gddwjadkgr_matmul_1_readvariableop_resource_0"d
/while_gddwjadkgr_matmul_readvariableop_resource1while_gddwjadkgr_matmul_readvariableop_resource_0"Z
*while_gddwjadkgr_readvariableop_1_resource,while_gddwjadkgr_readvariableop_1_resource_0"Z
*while_gddwjadkgr_readvariableop_2_resource,while_gddwjadkgr_readvariableop_2_resource_0"V
(while_gddwjadkgr_readvariableop_resource*while_gddwjadkgr_readvariableop_resource_0")
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
'while/gddwjadkgr/BiasAdd/ReadVariableOp'while/gddwjadkgr/BiasAdd/ReadVariableOp2P
&while/gddwjadkgr/MatMul/ReadVariableOp&while/gddwjadkgr/MatMul/ReadVariableOp2T
(while/gddwjadkgr/MatMul_1/ReadVariableOp(while/gddwjadkgr/MatMul_1/ReadVariableOp2B
while/gddwjadkgr/ReadVariableOpwhile/gddwjadkgr/ReadVariableOp2F
!while/gddwjadkgr/ReadVariableOp_1!while/gddwjadkgr/ReadVariableOp_12F
!while/gddwjadkgr/ReadVariableOp_2!while/gddwjadkgr/ReadVariableOp_2: 
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
°'
²
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_765669

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
ÙÊ

F__inference_sequential_layer_call_and_return_conditional_losses_768197

inputsL
6ggbvrcpxtu_conv1d_expanddims_1_readvariableop_resource:K
=ggbvrcpxtu_squeeze_batch_dims_biasadd_readvariableop_resource:G
4gdmdltnblf_duhsngmesj_matmul_readvariableop_resource:	I
6gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource:	 D
5gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource:	;
-gdmdltnblf_duhsngmesj_readvariableop_resource: =
/gdmdltnblf_duhsngmesj_readvariableop_1_resource: =
/gdmdltnblf_duhsngmesj_readvariableop_2_resource: G
4uazvpibasg_gddwjadkgr_matmul_readvariableop_resource:	 I
6uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource:	 D
5uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource:	;
-uazvpibasg_gddwjadkgr_readvariableop_resource: =
/uazvpibasg_gddwjadkgr_readvariableop_1_resource: =
/uazvpibasg_gddwjadkgr_readvariableop_2_resource: ;
)supobtndkp_matmul_readvariableop_resource: 8
*supobtndkp_biasadd_readvariableop_resource:
identity¢,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp¢+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp¢-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp¢$gdmdltnblf/duhsngmesj/ReadVariableOp¢&gdmdltnblf/duhsngmesj/ReadVariableOp_1¢&gdmdltnblf/duhsngmesj/ReadVariableOp_2¢gdmdltnblf/while¢-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp¢4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!supobtndkp/BiasAdd/ReadVariableOp¢ supobtndkp/MatMul/ReadVariableOp¢,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp¢+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp¢-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp¢$uazvpibasg/gddwjadkgr/ReadVariableOp¢&uazvpibasg/gddwjadkgr/ReadVariableOp_1¢&uazvpibasg/gddwjadkgr/ReadVariableOp_2¢uazvpibasg/while
 ggbvrcpxtu/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 ggbvrcpxtu/conv1d/ExpandDims/dim»
ggbvrcpxtu/conv1d/ExpandDims
ExpandDimsinputs)ggbvrcpxtu/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/ExpandDimsÙ
-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6ggbvrcpxtu_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp
"ggbvrcpxtu/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ggbvrcpxtu/conv1d/ExpandDims_1/dimã
ggbvrcpxtu/conv1d/ExpandDims_1
ExpandDims5ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp:value:0+ggbvrcpxtu/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
ggbvrcpxtu/conv1d/ExpandDims_1
ggbvrcpxtu/conv1d/ShapeShape%ggbvrcpxtu/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
ggbvrcpxtu/conv1d/Shape
%ggbvrcpxtu/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%ggbvrcpxtu/conv1d/strided_slice/stack¥
'ggbvrcpxtu/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'ggbvrcpxtu/conv1d/strided_slice/stack_1
'ggbvrcpxtu/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'ggbvrcpxtu/conv1d/strided_slice/stack_2Ì
ggbvrcpxtu/conv1d/strided_sliceStridedSlice ggbvrcpxtu/conv1d/Shape:output:0.ggbvrcpxtu/conv1d/strided_slice/stack:output:00ggbvrcpxtu/conv1d/strided_slice/stack_1:output:00ggbvrcpxtu/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
ggbvrcpxtu/conv1d/strided_slice
ggbvrcpxtu/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
ggbvrcpxtu/conv1d/Reshape/shapeÌ
ggbvrcpxtu/conv1d/ReshapeReshape%ggbvrcpxtu/conv1d/ExpandDims:output:0(ggbvrcpxtu/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/Reshapeî
ggbvrcpxtu/conv1d/Conv2DConv2D"ggbvrcpxtu/conv1d/Reshape:output:0'ggbvrcpxtu/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ggbvrcpxtu/conv1d/Conv2D
!ggbvrcpxtu/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!ggbvrcpxtu/conv1d/concat/values_1
ggbvrcpxtu/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/concat/axisì
ggbvrcpxtu/conv1d/concatConcatV2(ggbvrcpxtu/conv1d/strided_slice:output:0*ggbvrcpxtu/conv1d/concat/values_1:output:0&ggbvrcpxtu/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ggbvrcpxtu/conv1d/concatÉ
ggbvrcpxtu/conv1d/Reshape_1Reshape!ggbvrcpxtu/conv1d/Conv2D:output:0!ggbvrcpxtu/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/Reshape_1Á
ggbvrcpxtu/conv1d/SqueezeSqueeze$ggbvrcpxtu/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/Squeeze
#ggbvrcpxtu/squeeze_batch_dims/ShapeShape"ggbvrcpxtu/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#ggbvrcpxtu/squeeze_batch_dims/Shape°
1ggbvrcpxtu/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack½
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1´
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2
+ggbvrcpxtu/squeeze_batch_dims/strided_sliceStridedSlice,ggbvrcpxtu/squeeze_batch_dims/Shape:output:0:ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack:output:0<ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1:output:0<ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+ggbvrcpxtu/squeeze_batch_dims/strided_slice¯
+ggbvrcpxtu/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+ggbvrcpxtu/squeeze_batch_dims/Reshape/shapeé
%ggbvrcpxtu/squeeze_batch_dims/ReshapeReshape"ggbvrcpxtu/conv1d/Squeeze:output:04ggbvrcpxtu/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ggbvrcpxtu/squeeze_batch_dims/Reshapeæ
4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=ggbvrcpxtu_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%ggbvrcpxtu/squeeze_batch_dims/BiasAddBiasAdd.ggbvrcpxtu/squeeze_batch_dims/Reshape:output:0<ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ggbvrcpxtu/squeeze_batch_dims/BiasAdd¯
-ggbvrcpxtu/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-ggbvrcpxtu/squeeze_batch_dims/concat/values_1¡
)ggbvrcpxtu/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)ggbvrcpxtu/squeeze_batch_dims/concat/axis¨
$ggbvrcpxtu/squeeze_batch_dims/concatConcatV24ggbvrcpxtu/squeeze_batch_dims/strided_slice:output:06ggbvrcpxtu/squeeze_batch_dims/concat/values_1:output:02ggbvrcpxtu/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$ggbvrcpxtu/squeeze_batch_dims/concatö
'ggbvrcpxtu/squeeze_batch_dims/Reshape_1Reshape.ggbvrcpxtu/squeeze_batch_dims/BiasAdd:output:0-ggbvrcpxtu/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'ggbvrcpxtu/squeeze_batch_dims/Reshape_1
vtqejmjhbd/ShapeShape0ggbvrcpxtu/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
vtqejmjhbd/Shape
vtqejmjhbd/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
vtqejmjhbd/strided_slice/stack
 vtqejmjhbd/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 vtqejmjhbd/strided_slice/stack_1
 vtqejmjhbd/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 vtqejmjhbd/strided_slice/stack_2¤
vtqejmjhbd/strided_sliceStridedSlicevtqejmjhbd/Shape:output:0'vtqejmjhbd/strided_slice/stack:output:0)vtqejmjhbd/strided_slice/stack_1:output:0)vtqejmjhbd/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vtqejmjhbd/strided_slicez
vtqejmjhbd/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
vtqejmjhbd/Reshape/shape/1z
vtqejmjhbd/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
vtqejmjhbd/Reshape/shape/2×
vtqejmjhbd/Reshape/shapePack!vtqejmjhbd/strided_slice:output:0#vtqejmjhbd/Reshape/shape/1:output:0#vtqejmjhbd/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
vtqejmjhbd/Reshape/shape¾
vtqejmjhbd/ReshapeReshape0ggbvrcpxtu/squeeze_batch_dims/Reshape_1:output:0!vtqejmjhbd/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vtqejmjhbd/Reshapeo
gdmdltnblf/ShapeShapevtqejmjhbd/Reshape:output:0*
T0*
_output_shapes
:2
gdmdltnblf/Shape
gdmdltnblf/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gdmdltnblf/strided_slice/stack
 gdmdltnblf/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gdmdltnblf/strided_slice/stack_1
 gdmdltnblf/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gdmdltnblf/strided_slice/stack_2¤
gdmdltnblf/strided_sliceStridedSlicegdmdltnblf/Shape:output:0'gdmdltnblf/strided_slice/stack:output:0)gdmdltnblf/strided_slice/stack_1:output:0)gdmdltnblf/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gdmdltnblf/strided_slicer
gdmdltnblf/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros/mul/y
gdmdltnblf/zeros/mulMul!gdmdltnblf/strided_slice:output:0gdmdltnblf/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros/mulu
gdmdltnblf/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gdmdltnblf/zeros/Less/y
gdmdltnblf/zeros/LessLessgdmdltnblf/zeros/mul:z:0 gdmdltnblf/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros/Lessx
gdmdltnblf/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros/packed/1¯
gdmdltnblf/zeros/packedPack!gdmdltnblf/strided_slice:output:0"gdmdltnblf/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gdmdltnblf/zeros/packedu
gdmdltnblf/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gdmdltnblf/zeros/Const¡
gdmdltnblf/zerosFill gdmdltnblf/zeros/packed:output:0gdmdltnblf/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/zerosv
gdmdltnblf/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros_1/mul/y
gdmdltnblf/zeros_1/mulMul!gdmdltnblf/strided_slice:output:0!gdmdltnblf/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros_1/muly
gdmdltnblf/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gdmdltnblf/zeros_1/Less/y
gdmdltnblf/zeros_1/LessLessgdmdltnblf/zeros_1/mul:z:0"gdmdltnblf/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros_1/Less|
gdmdltnblf/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros_1/packed/1µ
gdmdltnblf/zeros_1/packedPack!gdmdltnblf/strided_slice:output:0$gdmdltnblf/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
gdmdltnblf/zeros_1/packedy
gdmdltnblf/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gdmdltnblf/zeros_1/Const©
gdmdltnblf/zeros_1Fill"gdmdltnblf/zeros_1/packed:output:0!gdmdltnblf/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/zeros_1
gdmdltnblf/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gdmdltnblf/transpose/perm°
gdmdltnblf/transpose	Transposevtqejmjhbd/Reshape:output:0"gdmdltnblf/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/transposep
gdmdltnblf/Shape_1Shapegdmdltnblf/transpose:y:0*
T0*
_output_shapes
:2
gdmdltnblf/Shape_1
 gdmdltnblf/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gdmdltnblf/strided_slice_1/stack
"gdmdltnblf/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_1/stack_1
"gdmdltnblf/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_1/stack_2°
gdmdltnblf/strided_slice_1StridedSlicegdmdltnblf/Shape_1:output:0)gdmdltnblf/strided_slice_1/stack:output:0+gdmdltnblf/strided_slice_1/stack_1:output:0+gdmdltnblf/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gdmdltnblf/strided_slice_1
&gdmdltnblf/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&gdmdltnblf/TensorArrayV2/element_shapeÞ
gdmdltnblf/TensorArrayV2TensorListReserve/gdmdltnblf/TensorArrayV2/element_shape:output:0#gdmdltnblf/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gdmdltnblf/TensorArrayV2Õ
@gdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@gdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2gdmdltnblf/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgdmdltnblf/transpose:y:0Igdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2gdmdltnblf/TensorArrayUnstack/TensorListFromTensor
 gdmdltnblf/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gdmdltnblf/strided_slice_2/stack
"gdmdltnblf/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_2/stack_1
"gdmdltnblf/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_2/stack_2¾
gdmdltnblf/strided_slice_2StridedSlicegdmdltnblf/transpose:y:0)gdmdltnblf/strided_slice_2/stack:output:0+gdmdltnblf/strided_slice_2/stack_1:output:0+gdmdltnblf/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
gdmdltnblf/strided_slice_2Ð
+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOpReadVariableOp4gdmdltnblf_duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOpÓ
gdmdltnblf/duhsngmesj/MatMulMatMul#gdmdltnblf/strided_slice_2:output:03gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/duhsngmesj/MatMulÖ
-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp6gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOpÏ
gdmdltnblf/duhsngmesj/MatMul_1MatMulgdmdltnblf/zeros:output:05gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
gdmdltnblf/duhsngmesj/MatMul_1Ä
gdmdltnblf/duhsngmesj/addAddV2&gdmdltnblf/duhsngmesj/MatMul:product:0(gdmdltnblf/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/duhsngmesj/addÏ
,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp5gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOpÑ
gdmdltnblf/duhsngmesj/BiasAddBiasAddgdmdltnblf/duhsngmesj/add:z:04gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/duhsngmesj/BiasAdd
%gdmdltnblf/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%gdmdltnblf/duhsngmesj/split/split_dim
gdmdltnblf/duhsngmesj/splitSplit.gdmdltnblf/duhsngmesj/split/split_dim:output:0&gdmdltnblf/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gdmdltnblf/duhsngmesj/split¶
$gdmdltnblf/duhsngmesj/ReadVariableOpReadVariableOp-gdmdltnblf_duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02&
$gdmdltnblf/duhsngmesj/ReadVariableOpº
gdmdltnblf/duhsngmesj/mulMul,gdmdltnblf/duhsngmesj/ReadVariableOp:value:0gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mulº
gdmdltnblf/duhsngmesj/add_1AddV2$gdmdltnblf/duhsngmesj/split:output:0gdmdltnblf/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_1
gdmdltnblf/duhsngmesj/SigmoidSigmoidgdmdltnblf/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/Sigmoid¼
&gdmdltnblf/duhsngmesj/ReadVariableOp_1ReadVariableOp/gdmdltnblf_duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&gdmdltnblf/duhsngmesj/ReadVariableOp_1À
gdmdltnblf/duhsngmesj/mul_1Mul.gdmdltnblf/duhsngmesj/ReadVariableOp_1:value:0gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_1¼
gdmdltnblf/duhsngmesj/add_2AddV2$gdmdltnblf/duhsngmesj/split:output:1gdmdltnblf/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_2 
gdmdltnblf/duhsngmesj/Sigmoid_1Sigmoidgdmdltnblf/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gdmdltnblf/duhsngmesj/Sigmoid_1µ
gdmdltnblf/duhsngmesj/mul_2Mul#gdmdltnblf/duhsngmesj/Sigmoid_1:y:0gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_2
gdmdltnblf/duhsngmesj/TanhTanh$gdmdltnblf/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/Tanh¶
gdmdltnblf/duhsngmesj/mul_3Mul!gdmdltnblf/duhsngmesj/Sigmoid:y:0gdmdltnblf/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_3·
gdmdltnblf/duhsngmesj/add_3AddV2gdmdltnblf/duhsngmesj/mul_2:z:0gdmdltnblf/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_3¼
&gdmdltnblf/duhsngmesj/ReadVariableOp_2ReadVariableOp/gdmdltnblf_duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&gdmdltnblf/duhsngmesj/ReadVariableOp_2Ä
gdmdltnblf/duhsngmesj/mul_4Mul.gdmdltnblf/duhsngmesj/ReadVariableOp_2:value:0gdmdltnblf/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_4¼
gdmdltnblf/duhsngmesj/add_4AddV2$gdmdltnblf/duhsngmesj/split:output:3gdmdltnblf/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_4 
gdmdltnblf/duhsngmesj/Sigmoid_2Sigmoidgdmdltnblf/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gdmdltnblf/duhsngmesj/Sigmoid_2
gdmdltnblf/duhsngmesj/Tanh_1Tanhgdmdltnblf/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/Tanh_1º
gdmdltnblf/duhsngmesj/mul_5Mul#gdmdltnblf/duhsngmesj/Sigmoid_2:y:0 gdmdltnblf/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_5¥
(gdmdltnblf/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(gdmdltnblf/TensorArrayV2_1/element_shapeä
gdmdltnblf/TensorArrayV2_1TensorListReserve1gdmdltnblf/TensorArrayV2_1/element_shape:output:0#gdmdltnblf/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gdmdltnblf/TensorArrayV2_1d
gdmdltnblf/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/time
#gdmdltnblf/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#gdmdltnblf/while/maximum_iterations
gdmdltnblf/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/while/loop_counter°
gdmdltnblf/whileWhile&gdmdltnblf/while/loop_counter:output:0,gdmdltnblf/while/maximum_iterations:output:0gdmdltnblf/time:output:0#gdmdltnblf/TensorArrayV2_1:handle:0gdmdltnblf/zeros:output:0gdmdltnblf/zeros_1:output:0#gdmdltnblf/strided_slice_1:output:0Bgdmdltnblf/TensorArrayUnstack/TensorListFromTensor:output_handle:04gdmdltnblf_duhsngmesj_matmul_readvariableop_resource6gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource5gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource-gdmdltnblf_duhsngmesj_readvariableop_resource/gdmdltnblf_duhsngmesj_readvariableop_1_resource/gdmdltnblf_duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*(
body R
gdmdltnblf_while_body_767914*(
cond R
gdmdltnblf_while_cond_767913*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
gdmdltnblf/whileË
;gdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;gdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shape
-gdmdltnblf/TensorArrayV2Stack/TensorListStackTensorListStackgdmdltnblf/while:output:3Dgdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-gdmdltnblf/TensorArrayV2Stack/TensorListStack
 gdmdltnblf/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 gdmdltnblf/strided_slice_3/stack
"gdmdltnblf/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gdmdltnblf/strided_slice_3/stack_1
"gdmdltnblf/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_3/stack_2Ü
gdmdltnblf/strided_slice_3StridedSlice6gdmdltnblf/TensorArrayV2Stack/TensorListStack:tensor:0)gdmdltnblf/strided_slice_3/stack:output:0+gdmdltnblf/strided_slice_3/stack_1:output:0+gdmdltnblf/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gdmdltnblf/strided_slice_3
gdmdltnblf/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gdmdltnblf/transpose_1/permÑ
gdmdltnblf/transpose_1	Transpose6gdmdltnblf/TensorArrayV2Stack/TensorListStack:tensor:0$gdmdltnblf/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/transpose_1n
uazvpibasg/ShapeShapegdmdltnblf/transpose_1:y:0*
T0*
_output_shapes
:2
uazvpibasg/Shape
uazvpibasg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
uazvpibasg/strided_slice/stack
 uazvpibasg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 uazvpibasg/strided_slice/stack_1
 uazvpibasg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 uazvpibasg/strided_slice/stack_2¤
uazvpibasg/strided_sliceStridedSliceuazvpibasg/Shape:output:0'uazvpibasg/strided_slice/stack:output:0)uazvpibasg/strided_slice/stack_1:output:0)uazvpibasg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
uazvpibasg/strided_slicer
uazvpibasg/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros/mul/y
uazvpibasg/zeros/mulMul!uazvpibasg/strided_slice:output:0uazvpibasg/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros/mulu
uazvpibasg/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
uazvpibasg/zeros/Less/y
uazvpibasg/zeros/LessLessuazvpibasg/zeros/mul:z:0 uazvpibasg/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros/Lessx
uazvpibasg/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros/packed/1¯
uazvpibasg/zeros/packedPack!uazvpibasg/strided_slice:output:0"uazvpibasg/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
uazvpibasg/zeros/packedu
uazvpibasg/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
uazvpibasg/zeros/Const¡
uazvpibasg/zerosFill uazvpibasg/zeros/packed:output:0uazvpibasg/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/zerosv
uazvpibasg/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros_1/mul/y
uazvpibasg/zeros_1/mulMul!uazvpibasg/strided_slice:output:0!uazvpibasg/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros_1/muly
uazvpibasg/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
uazvpibasg/zeros_1/Less/y
uazvpibasg/zeros_1/LessLessuazvpibasg/zeros_1/mul:z:0"uazvpibasg/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros_1/Less|
uazvpibasg/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros_1/packed/1µ
uazvpibasg/zeros_1/packedPack!uazvpibasg/strided_slice:output:0$uazvpibasg/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
uazvpibasg/zeros_1/packedy
uazvpibasg/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
uazvpibasg/zeros_1/Const©
uazvpibasg/zeros_1Fill"uazvpibasg/zeros_1/packed:output:0!uazvpibasg/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/zeros_1
uazvpibasg/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
uazvpibasg/transpose/perm¯
uazvpibasg/transpose	Transposegdmdltnblf/transpose_1:y:0"uazvpibasg/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/transposep
uazvpibasg/Shape_1Shapeuazvpibasg/transpose:y:0*
T0*
_output_shapes
:2
uazvpibasg/Shape_1
 uazvpibasg/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 uazvpibasg/strided_slice_1/stack
"uazvpibasg/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_1/stack_1
"uazvpibasg/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_1/stack_2°
uazvpibasg/strided_slice_1StridedSliceuazvpibasg/Shape_1:output:0)uazvpibasg/strided_slice_1/stack:output:0+uazvpibasg/strided_slice_1/stack_1:output:0+uazvpibasg/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
uazvpibasg/strided_slice_1
&uazvpibasg/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&uazvpibasg/TensorArrayV2/element_shapeÞ
uazvpibasg/TensorArrayV2TensorListReserve/uazvpibasg/TensorArrayV2/element_shape:output:0#uazvpibasg/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
uazvpibasg/TensorArrayV2Õ
@uazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@uazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2uazvpibasg/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoruazvpibasg/transpose:y:0Iuazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2uazvpibasg/TensorArrayUnstack/TensorListFromTensor
 uazvpibasg/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 uazvpibasg/strided_slice_2/stack
"uazvpibasg/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_2/stack_1
"uazvpibasg/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_2/stack_2¾
uazvpibasg/strided_slice_2StridedSliceuazvpibasg/transpose:y:0)uazvpibasg/strided_slice_2/stack:output:0+uazvpibasg/strided_slice_2/stack_1:output:0+uazvpibasg/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
uazvpibasg/strided_slice_2Ð
+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp4uazvpibasg_gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOpÓ
uazvpibasg/gddwjadkgr/MatMulMatMul#uazvpibasg/strided_slice_2:output:03uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uazvpibasg/gddwjadkgr/MatMulÖ
-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp6uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOpÏ
uazvpibasg/gddwjadkgr/MatMul_1MatMuluazvpibasg/zeros:output:05uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
uazvpibasg/gddwjadkgr/MatMul_1Ä
uazvpibasg/gddwjadkgr/addAddV2&uazvpibasg/gddwjadkgr/MatMul:product:0(uazvpibasg/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uazvpibasg/gddwjadkgr/addÏ
,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp5uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOpÑ
uazvpibasg/gddwjadkgr/BiasAddBiasAdduazvpibasg/gddwjadkgr/add:z:04uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uazvpibasg/gddwjadkgr/BiasAdd
%uazvpibasg/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%uazvpibasg/gddwjadkgr/split/split_dim
uazvpibasg/gddwjadkgr/splitSplit.uazvpibasg/gddwjadkgr/split/split_dim:output:0&uazvpibasg/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
uazvpibasg/gddwjadkgr/split¶
$uazvpibasg/gddwjadkgr/ReadVariableOpReadVariableOp-uazvpibasg_gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02&
$uazvpibasg/gddwjadkgr/ReadVariableOpº
uazvpibasg/gddwjadkgr/mulMul,uazvpibasg/gddwjadkgr/ReadVariableOp:value:0uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mulº
uazvpibasg/gddwjadkgr/add_1AddV2$uazvpibasg/gddwjadkgr/split:output:0uazvpibasg/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_1
uazvpibasg/gddwjadkgr/SigmoidSigmoiduazvpibasg/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/Sigmoid¼
&uazvpibasg/gddwjadkgr/ReadVariableOp_1ReadVariableOp/uazvpibasg_gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&uazvpibasg/gddwjadkgr/ReadVariableOp_1À
uazvpibasg/gddwjadkgr/mul_1Mul.uazvpibasg/gddwjadkgr/ReadVariableOp_1:value:0uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_1¼
uazvpibasg/gddwjadkgr/add_2AddV2$uazvpibasg/gddwjadkgr/split:output:1uazvpibasg/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_2 
uazvpibasg/gddwjadkgr/Sigmoid_1Sigmoiduazvpibasg/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
uazvpibasg/gddwjadkgr/Sigmoid_1µ
uazvpibasg/gddwjadkgr/mul_2Mul#uazvpibasg/gddwjadkgr/Sigmoid_1:y:0uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_2
uazvpibasg/gddwjadkgr/TanhTanh$uazvpibasg/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/Tanh¶
uazvpibasg/gddwjadkgr/mul_3Mul!uazvpibasg/gddwjadkgr/Sigmoid:y:0uazvpibasg/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_3·
uazvpibasg/gddwjadkgr/add_3AddV2uazvpibasg/gddwjadkgr/mul_2:z:0uazvpibasg/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_3¼
&uazvpibasg/gddwjadkgr/ReadVariableOp_2ReadVariableOp/uazvpibasg_gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&uazvpibasg/gddwjadkgr/ReadVariableOp_2Ä
uazvpibasg/gddwjadkgr/mul_4Mul.uazvpibasg/gddwjadkgr/ReadVariableOp_2:value:0uazvpibasg/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_4¼
uazvpibasg/gddwjadkgr/add_4AddV2$uazvpibasg/gddwjadkgr/split:output:3uazvpibasg/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_4 
uazvpibasg/gddwjadkgr/Sigmoid_2Sigmoiduazvpibasg/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
uazvpibasg/gddwjadkgr/Sigmoid_2
uazvpibasg/gddwjadkgr/Tanh_1Tanhuazvpibasg/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/Tanh_1º
uazvpibasg/gddwjadkgr/mul_5Mul#uazvpibasg/gddwjadkgr/Sigmoid_2:y:0 uazvpibasg/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_5¥
(uazvpibasg/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(uazvpibasg/TensorArrayV2_1/element_shapeä
uazvpibasg/TensorArrayV2_1TensorListReserve1uazvpibasg/TensorArrayV2_1/element_shape:output:0#uazvpibasg/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
uazvpibasg/TensorArrayV2_1d
uazvpibasg/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/time
#uazvpibasg/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#uazvpibasg/while/maximum_iterations
uazvpibasg/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/while/loop_counter°
uazvpibasg/whileWhile&uazvpibasg/while/loop_counter:output:0,uazvpibasg/while/maximum_iterations:output:0uazvpibasg/time:output:0#uazvpibasg/TensorArrayV2_1:handle:0uazvpibasg/zeros:output:0uazvpibasg/zeros_1:output:0#uazvpibasg/strided_slice_1:output:0Buazvpibasg/TensorArrayUnstack/TensorListFromTensor:output_handle:04uazvpibasg_gddwjadkgr_matmul_readvariableop_resource6uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource5uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource-uazvpibasg_gddwjadkgr_readvariableop_resource/uazvpibasg_gddwjadkgr_readvariableop_1_resource/uazvpibasg_gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*(
body R
uazvpibasg_while_body_768090*(
cond R
uazvpibasg_while_cond_768089*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
uazvpibasg/whileË
;uazvpibasg/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;uazvpibasg/TensorArrayV2Stack/TensorListStack/element_shape
-uazvpibasg/TensorArrayV2Stack/TensorListStackTensorListStackuazvpibasg/while:output:3Duazvpibasg/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-uazvpibasg/TensorArrayV2Stack/TensorListStack
 uazvpibasg/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 uazvpibasg/strided_slice_3/stack
"uazvpibasg/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"uazvpibasg/strided_slice_3/stack_1
"uazvpibasg/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_3/stack_2Ü
uazvpibasg/strided_slice_3StridedSlice6uazvpibasg/TensorArrayV2Stack/TensorListStack:tensor:0)uazvpibasg/strided_slice_3/stack:output:0+uazvpibasg/strided_slice_3/stack_1:output:0+uazvpibasg/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
uazvpibasg/strided_slice_3
uazvpibasg/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
uazvpibasg/transpose_1/permÑ
uazvpibasg/transpose_1	Transpose6uazvpibasg/TensorArrayV2Stack/TensorListStack:tensor:0$uazvpibasg/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/transpose_1®
 supobtndkp/MatMul/ReadVariableOpReadVariableOp)supobtndkp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 supobtndkp/MatMul/ReadVariableOp±
supobtndkp/MatMulMatMul#uazvpibasg/strided_slice_3:output:0(supobtndkp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
supobtndkp/MatMul­
!supobtndkp/BiasAdd/ReadVariableOpReadVariableOp*supobtndkp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!supobtndkp/BiasAdd/ReadVariableOp­
supobtndkp/BiasAddBiasAddsupobtndkp/MatMul:product:0)supobtndkp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
supobtndkp/BiasAddÏ
IdentityIdentitysupobtndkp/BiasAdd:output:0-^gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp,^gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp.^gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp%^gdmdltnblf/duhsngmesj/ReadVariableOp'^gdmdltnblf/duhsngmesj/ReadVariableOp_1'^gdmdltnblf/duhsngmesj/ReadVariableOp_2^gdmdltnblf/while.^ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp5^ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp"^supobtndkp/BiasAdd/ReadVariableOp!^supobtndkp/MatMul/ReadVariableOp-^uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp,^uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp.^uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp%^uazvpibasg/gddwjadkgr/ReadVariableOp'^uazvpibasg/gddwjadkgr/ReadVariableOp_1'^uazvpibasg/gddwjadkgr/ReadVariableOp_2^uazvpibasg/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp2Z
+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp2^
-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp2L
$gdmdltnblf/duhsngmesj/ReadVariableOp$gdmdltnblf/duhsngmesj/ReadVariableOp2P
&gdmdltnblf/duhsngmesj/ReadVariableOp_1&gdmdltnblf/duhsngmesj/ReadVariableOp_12P
&gdmdltnblf/duhsngmesj/ReadVariableOp_2&gdmdltnblf/duhsngmesj/ReadVariableOp_22$
gdmdltnblf/whilegdmdltnblf/while2^
-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp2l
4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!supobtndkp/BiasAdd/ReadVariableOp!supobtndkp/BiasAdd/ReadVariableOp2D
 supobtndkp/MatMul/ReadVariableOp supobtndkp/MatMul/ReadVariableOp2\
,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp2Z
+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp2^
-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp2L
$uazvpibasg/gddwjadkgr/ReadVariableOp$uazvpibasg/gddwjadkgr/ReadVariableOp2P
&uazvpibasg/gddwjadkgr/ReadVariableOp_1&uazvpibasg/gddwjadkgr/ReadVariableOp_12P
&uazvpibasg/gddwjadkgr/ReadVariableOp_2&uazvpibasg/gddwjadkgr/ReadVariableOp_22$
uazvpibasg/whileuazvpibasg/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù[
Ê
__inference__traced_save_770338
file_prefix0
,savev2_ggbvrcpxtu_kernel_read_readvariableop.
*savev2_ggbvrcpxtu_bias_read_readvariableop0
,savev2_supobtndkp_kernel_read_readvariableop.
*savev2_supobtndkp_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_gdmdltnblf_duhsngmesj_kernel_read_readvariableopE
Asavev2_gdmdltnblf_duhsngmesj_recurrent_kernel_read_readvariableop9
5savev2_gdmdltnblf_duhsngmesj_bias_read_readvariableopP
Lsavev2_gdmdltnblf_duhsngmesj_input_gate_peephole_weights_read_readvariableopQ
Msavev2_gdmdltnblf_duhsngmesj_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_gdmdltnblf_duhsngmesj_output_gate_peephole_weights_read_readvariableop;
7savev2_uazvpibasg_gddwjadkgr_kernel_read_readvariableopE
Asavev2_uazvpibasg_gddwjadkgr_recurrent_kernel_read_readvariableop9
5savev2_uazvpibasg_gddwjadkgr_bias_read_readvariableopP
Lsavev2_uazvpibasg_gddwjadkgr_input_gate_peephole_weights_read_readvariableopQ
Msavev2_uazvpibasg_gddwjadkgr_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_uazvpibasg_gddwjadkgr_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_ggbvrcpxtu_kernel_rms_read_readvariableop:
6savev2_rmsprop_ggbvrcpxtu_bias_rms_read_readvariableop<
8savev2_rmsprop_supobtndkp_kernel_rms_read_readvariableop:
6savev2_rmsprop_supobtndkp_bias_rms_read_readvariableopG
Csavev2_rmsprop_gdmdltnblf_duhsngmesj_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_gdmdltnblf_duhsngmesj_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_gdmdltnblf_duhsngmesj_bias_rms_read_readvariableop\
Xsavev2_rmsprop_gdmdltnblf_duhsngmesj_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_gdmdltnblf_duhsngmesj_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_gdmdltnblf_duhsngmesj_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_uazvpibasg_gddwjadkgr_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_uazvpibasg_gddwjadkgr_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_uazvpibasg_gddwjadkgr_bias_rms_read_readvariableop\
Xsavev2_rmsprop_uazvpibasg_gddwjadkgr_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_uazvpibasg_gddwjadkgr_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_uazvpibasg_gddwjadkgr_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_ggbvrcpxtu_kernel_read_readvariableop*savev2_ggbvrcpxtu_bias_read_readvariableop,savev2_supobtndkp_kernel_read_readvariableop*savev2_supobtndkp_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_gdmdltnblf_duhsngmesj_kernel_read_readvariableopAsavev2_gdmdltnblf_duhsngmesj_recurrent_kernel_read_readvariableop5savev2_gdmdltnblf_duhsngmesj_bias_read_readvariableopLsavev2_gdmdltnblf_duhsngmesj_input_gate_peephole_weights_read_readvariableopMsavev2_gdmdltnblf_duhsngmesj_forget_gate_peephole_weights_read_readvariableopMsavev2_gdmdltnblf_duhsngmesj_output_gate_peephole_weights_read_readvariableop7savev2_uazvpibasg_gddwjadkgr_kernel_read_readvariableopAsavev2_uazvpibasg_gddwjadkgr_recurrent_kernel_read_readvariableop5savev2_uazvpibasg_gddwjadkgr_bias_read_readvariableopLsavev2_uazvpibasg_gddwjadkgr_input_gate_peephole_weights_read_readvariableopMsavev2_uazvpibasg_gddwjadkgr_forget_gate_peephole_weights_read_readvariableopMsavev2_uazvpibasg_gddwjadkgr_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_ggbvrcpxtu_kernel_rms_read_readvariableop6savev2_rmsprop_ggbvrcpxtu_bias_rms_read_readvariableop8savev2_rmsprop_supobtndkp_kernel_rms_read_readvariableop6savev2_rmsprop_supobtndkp_bias_rms_read_readvariableopCsavev2_rmsprop_gdmdltnblf_duhsngmesj_kernel_rms_read_readvariableopMsavev2_rmsprop_gdmdltnblf_duhsngmesj_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_gdmdltnblf_duhsngmesj_bias_rms_read_readvariableopXsavev2_rmsprop_gdmdltnblf_duhsngmesj_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_gdmdltnblf_duhsngmesj_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_gdmdltnblf_duhsngmesj_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_uazvpibasg_gddwjadkgr_kernel_rms_read_readvariableopMsavev2_rmsprop_uazvpibasg_gddwjadkgr_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_uazvpibasg_gddwjadkgr_bias_rms_read_readvariableopXsavev2_rmsprop_uazvpibasg_gddwjadkgr_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_uazvpibasg_gddwjadkgr_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_uazvpibasg_gddwjadkgr_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
ÿ
¿
+__inference_gddwjadkgr_layer_call_fn_770175

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

identity_2¢StatefulPartitionedCallë
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
GPU2*0J 8 *O
fJRH
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_7654822
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
Ò	
÷
F__inference_supobtndkp_layer_call_and_return_conditional_losses_766614

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
Ñ

+__inference_uazvpibasg_layer_call_fn_769894

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall­
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7665902
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
¸'
´
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_769974

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
ÿ
¿
+__inference_duhsngmesj_layer_call_fn_770064

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

identity_2¢StatefulPartitionedCallë
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
GPU2*0J 8 *O
fJRH
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_7649112
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
ßY

while_body_768594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_duhsngmesj_matmul_readvariableop_resource_0:	F
3while_duhsngmesj_matmul_1_readvariableop_resource_0:	 A
2while_duhsngmesj_biasadd_readvariableop_resource_0:	8
*while_duhsngmesj_readvariableop_resource_0: :
,while_duhsngmesj_readvariableop_1_resource_0: :
,while_duhsngmesj_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_duhsngmesj_matmul_readvariableop_resource:	D
1while_duhsngmesj_matmul_1_readvariableop_resource:	 ?
0while_duhsngmesj_biasadd_readvariableop_resource:	6
(while_duhsngmesj_readvariableop_resource: 8
*while_duhsngmesj_readvariableop_1_resource: 8
*while_duhsngmesj_readvariableop_2_resource: ¢'while/duhsngmesj/BiasAdd/ReadVariableOp¢&while/duhsngmesj/MatMul/ReadVariableOp¢(while/duhsngmesj/MatMul_1/ReadVariableOp¢while/duhsngmesj/ReadVariableOp¢!while/duhsngmesj/ReadVariableOp_1¢!while/duhsngmesj/ReadVariableOp_2Ã
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
&while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp1while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/duhsngmesj/MatMul/ReadVariableOpÑ
while/duhsngmesj/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMulÉ
(while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp3while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/duhsngmesj/MatMul_1/ReadVariableOpº
while/duhsngmesj/MatMul_1MatMulwhile_placeholder_20while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMul_1°
while/duhsngmesj/addAddV2!while/duhsngmesj/MatMul:product:0#while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/addÂ
'while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp2while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/duhsngmesj/BiasAdd/ReadVariableOp½
while/duhsngmesj/BiasAddBiasAddwhile/duhsngmesj/add:z:0/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/BiasAdd
 while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/duhsngmesj/split/split_dim
while/duhsngmesj/splitSplit)while/duhsngmesj/split/split_dim:output:0!while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/duhsngmesj/split©
while/duhsngmesj/ReadVariableOpReadVariableOp*while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/duhsngmesj/ReadVariableOp£
while/duhsngmesj/mulMul'while/duhsngmesj/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul¦
while/duhsngmesj/add_1AddV2while/duhsngmesj/split:output:0while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_1
while/duhsngmesj/SigmoidSigmoidwhile/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid¯
!while/duhsngmesj/ReadVariableOp_1ReadVariableOp,while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_1©
while/duhsngmesj/mul_1Mul)while/duhsngmesj/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_1¨
while/duhsngmesj/add_2AddV2while/duhsngmesj/split:output:1while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_2
while/duhsngmesj/Sigmoid_1Sigmoidwhile/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_1
while/duhsngmesj/mul_2Mulwhile/duhsngmesj/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_2
while/duhsngmesj/TanhTanhwhile/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh¢
while/duhsngmesj/mul_3Mulwhile/duhsngmesj/Sigmoid:y:0while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_3£
while/duhsngmesj/add_3AddV2while/duhsngmesj/mul_2:z:0while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_3¯
!while/duhsngmesj/ReadVariableOp_2ReadVariableOp,while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_2°
while/duhsngmesj/mul_4Mul)while/duhsngmesj/ReadVariableOp_2:value:0while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_4¨
while/duhsngmesj/add_4AddV2while/duhsngmesj/split:output:3while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_4
while/duhsngmesj/Sigmoid_2Sigmoidwhile/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_2
while/duhsngmesj/Tanh_1Tanhwhile/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh_1¦
while/duhsngmesj/mul_5Mulwhile/duhsngmesj/Sigmoid_2:y:0while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/duhsngmesj/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/duhsngmesj/mul_5:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/duhsngmesj/add_3:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_duhsngmesj_biasadd_readvariableop_resource2while_duhsngmesj_biasadd_readvariableop_resource_0"h
1while_duhsngmesj_matmul_1_readvariableop_resource3while_duhsngmesj_matmul_1_readvariableop_resource_0"d
/while_duhsngmesj_matmul_readvariableop_resource1while_duhsngmesj_matmul_readvariableop_resource_0"Z
*while_duhsngmesj_readvariableop_1_resource,while_duhsngmesj_readvariableop_1_resource_0"Z
*while_duhsngmesj_readvariableop_2_resource,while_duhsngmesj_readvariableop_2_resource_0"V
(while_duhsngmesj_readvariableop_resource*while_duhsngmesj_readvariableop_resource_0")
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
'while/duhsngmesj/BiasAdd/ReadVariableOp'while/duhsngmesj/BiasAdd/ReadVariableOp2P
&while/duhsngmesj/MatMul/ReadVariableOp&while/duhsngmesj/MatMul/ReadVariableOp2T
(while/duhsngmesj/MatMul_1/ReadVariableOp(while/duhsngmesj/MatMul_1/ReadVariableOp2B
while/duhsngmesj/ReadVariableOpwhile/duhsngmesj/ReadVariableOp2F
!while/duhsngmesj/ReadVariableOp_1!while/duhsngmesj/ReadVariableOp_12F
!while/duhsngmesj/ReadVariableOp_2!while/duhsngmesj/ReadVariableOp_2: 
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
h

F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_766397

inputs<
)duhsngmesj_matmul_readvariableop_resource:	>
+duhsngmesj_matmul_1_readvariableop_resource:	 9
*duhsngmesj_biasadd_readvariableop_resource:	0
"duhsngmesj_readvariableop_resource: 2
$duhsngmesj_readvariableop_1_resource: 2
$duhsngmesj_readvariableop_2_resource: 
identity¢!duhsngmesj/BiasAdd/ReadVariableOp¢ duhsngmesj/MatMul/ReadVariableOp¢"duhsngmesj/MatMul_1/ReadVariableOp¢duhsngmesj/ReadVariableOp¢duhsngmesj/ReadVariableOp_1¢duhsngmesj/ReadVariableOp_2¢whileD
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
 duhsngmesj/MatMul/ReadVariableOpReadVariableOp)duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 duhsngmesj/MatMul/ReadVariableOp§
duhsngmesj/MatMulMatMulstrided_slice_2:output:0(duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMulµ
"duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp+duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"duhsngmesj/MatMul_1/ReadVariableOp£
duhsngmesj/MatMul_1MatMulzeros:output:0*duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMul_1
duhsngmesj/addAddV2duhsngmesj/MatMul:product:0duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/add®
!duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp*duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!duhsngmesj/BiasAdd/ReadVariableOp¥
duhsngmesj/BiasAddBiasAddduhsngmesj/add:z:0)duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/BiasAddz
duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
duhsngmesj/split/split_dimë
duhsngmesj/splitSplit#duhsngmesj/split/split_dim:output:0duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
duhsngmesj/split
duhsngmesj/ReadVariableOpReadVariableOp"duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp
duhsngmesj/mulMul!duhsngmesj/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul
duhsngmesj/add_1AddV2duhsngmesj/split:output:0duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_1{
duhsngmesj/SigmoidSigmoidduhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid
duhsngmesj/ReadVariableOp_1ReadVariableOp$duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_1
duhsngmesj/mul_1Mul#duhsngmesj/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_1
duhsngmesj/add_2AddV2duhsngmesj/split:output:1duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_2
duhsngmesj/Sigmoid_1Sigmoidduhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_1
duhsngmesj/mul_2Mulduhsngmesj/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_2w
duhsngmesj/TanhTanhduhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh
duhsngmesj/mul_3Mulduhsngmesj/Sigmoid:y:0duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_3
duhsngmesj/add_3AddV2duhsngmesj/mul_2:z:0duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_3
duhsngmesj/ReadVariableOp_2ReadVariableOp$duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_2
duhsngmesj/mul_4Mul#duhsngmesj/ReadVariableOp_2:value:0duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_4
duhsngmesj/add_4AddV2duhsngmesj/split:output:3duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_4
duhsngmesj/Sigmoid_2Sigmoidduhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_2v
duhsngmesj/Tanh_1Tanhduhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh_1
duhsngmesj/mul_5Mulduhsngmesj/Sigmoid_2:y:0duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)duhsngmesj_matmul_readvariableop_resource+duhsngmesj_matmul_1_readvariableop_resource*duhsngmesj_biasadd_readvariableop_resource"duhsngmesj_readvariableop_resource$duhsngmesj_readvariableop_1_resource$duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_766296*
condR
while_cond_766295*Q
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
IdentityIdentitytranspose_1:y:0"^duhsngmesj/BiasAdd/ReadVariableOp!^duhsngmesj/MatMul/ReadVariableOp#^duhsngmesj/MatMul_1/ReadVariableOp^duhsngmesj/ReadVariableOp^duhsngmesj/ReadVariableOp_1^duhsngmesj/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!duhsngmesj/BiasAdd/ReadVariableOp!duhsngmesj/BiasAdd/ReadVariableOp2D
 duhsngmesj/MatMul/ReadVariableOp duhsngmesj/MatMul/ReadVariableOp2H
"duhsngmesj/MatMul_1/ReadVariableOp"duhsngmesj/MatMul_1/ReadVariableOp26
duhsngmesj/ReadVariableOpduhsngmesj/ReadVariableOp2:
duhsngmesj/ReadVariableOp_1duhsngmesj/ReadVariableOp_12:
duhsngmesj/ReadVariableOp_2duhsngmesj/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_768413
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_768413___redundant_placeholder04
0while_while_cond_768413___redundant_placeholder14
0while_while_cond_768413___redundant_placeholder24
0while_while_cond_768413___redundant_placeholder34
0while_while_cond_768413___redundant_placeholder44
0while_while_cond_768413___redundant_placeholder54
0while_while_cond_768413___redundant_placeholder6
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
Æ

+__inference_ggbvrcpxtu_layer_call_fn_768317

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
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
GPU2*0J 8 *O
fJRH
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_7661972
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
p
É
gdmdltnblf_while_body_7675102
.gdmdltnblf_while_gdmdltnblf_while_loop_counter8
4gdmdltnblf_while_gdmdltnblf_while_maximum_iterations 
gdmdltnblf_while_placeholder"
gdmdltnblf_while_placeholder_1"
gdmdltnblf_while_placeholder_2"
gdmdltnblf_while_placeholder_31
-gdmdltnblf_while_gdmdltnblf_strided_slice_1_0m
igdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0O
<gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0:	Q
>gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0:	 L
=gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0:	C
5gdmdltnblf_while_duhsngmesj_readvariableop_resource_0: E
7gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0: E
7gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0: 
gdmdltnblf_while_identity
gdmdltnblf_while_identity_1
gdmdltnblf_while_identity_2
gdmdltnblf_while_identity_3
gdmdltnblf_while_identity_4
gdmdltnblf_while_identity_5/
+gdmdltnblf_while_gdmdltnblf_strided_slice_1k
ggdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensorM
:gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource:	O
<gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource:	 J
;gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource:	A
3gdmdltnblf_while_duhsngmesj_readvariableop_resource: C
5gdmdltnblf_while_duhsngmesj_readvariableop_1_resource: C
5gdmdltnblf_while_duhsngmesj_readvariableop_2_resource: ¢2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp¢1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp¢3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp¢*gdmdltnblf/while/duhsngmesj/ReadVariableOp¢,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1¢,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2Ù
Bgdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bgdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shape
4gdmdltnblf/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemigdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0gdmdltnblf_while_placeholderKgdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4gdmdltnblf/while/TensorArrayV2Read/TensorListGetItemä
1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp<gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOpý
"gdmdltnblf/while/duhsngmesj/MatMulMatMul;gdmdltnblf/while/TensorArrayV2Read/TensorListGetItem:item:09gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"gdmdltnblf/while/duhsngmesj/MatMulê
3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp>gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOpæ
$gdmdltnblf/while/duhsngmesj/MatMul_1MatMulgdmdltnblf_while_placeholder_2;gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$gdmdltnblf/while/duhsngmesj/MatMul_1Ü
gdmdltnblf/while/duhsngmesj/addAddV2,gdmdltnblf/while/duhsngmesj/MatMul:product:0.gdmdltnblf/while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
gdmdltnblf/while/duhsngmesj/addã
2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp=gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOpé
#gdmdltnblf/while/duhsngmesj/BiasAddBiasAdd#gdmdltnblf/while/duhsngmesj/add:z:0:gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#gdmdltnblf/while/duhsngmesj/BiasAdd
+gdmdltnblf/while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+gdmdltnblf/while/duhsngmesj/split/split_dim¯
!gdmdltnblf/while/duhsngmesj/splitSplit4gdmdltnblf/while/duhsngmesj/split/split_dim:output:0,gdmdltnblf/while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!gdmdltnblf/while/duhsngmesj/splitÊ
*gdmdltnblf/while/duhsngmesj/ReadVariableOpReadVariableOp5gdmdltnblf_while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*gdmdltnblf/while/duhsngmesj/ReadVariableOpÏ
gdmdltnblf/while/duhsngmesj/mulMul2gdmdltnblf/while/duhsngmesj/ReadVariableOp:value:0gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gdmdltnblf/while/duhsngmesj/mulÒ
!gdmdltnblf/while/duhsngmesj/add_1AddV2*gdmdltnblf/while/duhsngmesj/split:output:0#gdmdltnblf/while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_1®
#gdmdltnblf/while/duhsngmesj/SigmoidSigmoid%gdmdltnblf/while/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#gdmdltnblf/while/duhsngmesj/SigmoidÐ
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1ReadVariableOp7gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1Õ
!gdmdltnblf/while/duhsngmesj/mul_1Mul4gdmdltnblf/while/duhsngmesj/ReadVariableOp_1:value:0gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_1Ô
!gdmdltnblf/while/duhsngmesj/add_2AddV2*gdmdltnblf/while/duhsngmesj/split:output:1%gdmdltnblf/while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_2²
%gdmdltnblf/while/duhsngmesj/Sigmoid_1Sigmoid%gdmdltnblf/while/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%gdmdltnblf/while/duhsngmesj/Sigmoid_1Ê
!gdmdltnblf/while/duhsngmesj/mul_2Mul)gdmdltnblf/while/duhsngmesj/Sigmoid_1:y:0gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_2ª
 gdmdltnblf/while/duhsngmesj/TanhTanh*gdmdltnblf/while/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 gdmdltnblf/while/duhsngmesj/TanhÎ
!gdmdltnblf/while/duhsngmesj/mul_3Mul'gdmdltnblf/while/duhsngmesj/Sigmoid:y:0$gdmdltnblf/while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_3Ï
!gdmdltnblf/while/duhsngmesj/add_3AddV2%gdmdltnblf/while/duhsngmesj/mul_2:z:0%gdmdltnblf/while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_3Ð
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2ReadVariableOp7gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2Ü
!gdmdltnblf/while/duhsngmesj/mul_4Mul4gdmdltnblf/while/duhsngmesj/ReadVariableOp_2:value:0%gdmdltnblf/while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_4Ô
!gdmdltnblf/while/duhsngmesj/add_4AddV2*gdmdltnblf/while/duhsngmesj/split:output:3%gdmdltnblf/while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/add_4²
%gdmdltnblf/while/duhsngmesj/Sigmoid_2Sigmoid%gdmdltnblf/while/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%gdmdltnblf/while/duhsngmesj/Sigmoid_2©
"gdmdltnblf/while/duhsngmesj/Tanh_1Tanh%gdmdltnblf/while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"gdmdltnblf/while/duhsngmesj/Tanh_1Ò
!gdmdltnblf/while/duhsngmesj/mul_5Mul)gdmdltnblf/while/duhsngmesj/Sigmoid_2:y:0&gdmdltnblf/while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!gdmdltnblf/while/duhsngmesj/mul_5
5gdmdltnblf/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgdmdltnblf_while_placeholder_1gdmdltnblf_while_placeholder%gdmdltnblf/while/duhsngmesj/mul_5:z:0*
_output_shapes
: *
element_dtype027
5gdmdltnblf/while/TensorArrayV2Write/TensorListSetItemr
gdmdltnblf/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gdmdltnblf/while/add/y
gdmdltnblf/while/addAddV2gdmdltnblf_while_placeholdergdmdltnblf/while/add/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/while/addv
gdmdltnblf/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gdmdltnblf/while/add_1/y­
gdmdltnblf/while/add_1AddV2.gdmdltnblf_while_gdmdltnblf_while_loop_counter!gdmdltnblf/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/while/add_1©
gdmdltnblf/while/IdentityIdentitygdmdltnblf/while/add_1:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/IdentityÇ
gdmdltnblf/while/Identity_1Identity4gdmdltnblf_while_gdmdltnblf_while_maximum_iterations3^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/Identity_1«
gdmdltnblf/while/Identity_2Identitygdmdltnblf/while/add:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/Identity_2Ø
gdmdltnblf/while/Identity_3IdentityEgdmdltnblf/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
gdmdltnblf/while/Identity_3É
gdmdltnblf/while/Identity_4Identity%gdmdltnblf/while/duhsngmesj/mul_5:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/while/Identity_4É
gdmdltnblf/while/Identity_5Identity%gdmdltnblf/while/duhsngmesj/add_3:z:03^gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2^gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp4^gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp+^gdmdltnblf/while/duhsngmesj/ReadVariableOp-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_1-^gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/while/Identity_5"|
;gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource=gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0"~
<gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource>gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0"z
:gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource<gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0"p
5gdmdltnblf_while_duhsngmesj_readvariableop_1_resource7gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0"p
5gdmdltnblf_while_duhsngmesj_readvariableop_2_resource7gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0"l
3gdmdltnblf_while_duhsngmesj_readvariableop_resource5gdmdltnblf_while_duhsngmesj_readvariableop_resource_0"\
+gdmdltnblf_while_gdmdltnblf_strided_slice_1-gdmdltnblf_while_gdmdltnblf_strided_slice_1_0"?
gdmdltnblf_while_identity"gdmdltnblf/while/Identity:output:0"C
gdmdltnblf_while_identity_1$gdmdltnblf/while/Identity_1:output:0"C
gdmdltnblf_while_identity_2$gdmdltnblf/while/Identity_2:output:0"C
gdmdltnblf_while_identity_3$gdmdltnblf/while/Identity_3:output:0"C
gdmdltnblf_while_identity_4$gdmdltnblf/while/Identity_4:output:0"C
gdmdltnblf_while_identity_5$gdmdltnblf/while/Identity_5:output:0"Ô
ggdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensorigdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2f
1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp1gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp2j
3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp3gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp2X
*gdmdltnblf/while/duhsngmesj/ReadVariableOp*gdmdltnblf/while/duhsngmesj/ReadVariableOp2\
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_1,gdmdltnblf/while/duhsngmesj/ReadVariableOp_12\
,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2,gdmdltnblf/while/duhsngmesj/ReadVariableOp_2: 
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
h

F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768875

inputs<
)duhsngmesj_matmul_readvariableop_resource:	>
+duhsngmesj_matmul_1_readvariableop_resource:	 9
*duhsngmesj_biasadd_readvariableop_resource:	0
"duhsngmesj_readvariableop_resource: 2
$duhsngmesj_readvariableop_1_resource: 2
$duhsngmesj_readvariableop_2_resource: 
identity¢!duhsngmesj/BiasAdd/ReadVariableOp¢ duhsngmesj/MatMul/ReadVariableOp¢"duhsngmesj/MatMul_1/ReadVariableOp¢duhsngmesj/ReadVariableOp¢duhsngmesj/ReadVariableOp_1¢duhsngmesj/ReadVariableOp_2¢whileD
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
 duhsngmesj/MatMul/ReadVariableOpReadVariableOp)duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 duhsngmesj/MatMul/ReadVariableOp§
duhsngmesj/MatMulMatMulstrided_slice_2:output:0(duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMulµ
"duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp+duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"duhsngmesj/MatMul_1/ReadVariableOp£
duhsngmesj/MatMul_1MatMulzeros:output:0*duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMul_1
duhsngmesj/addAddV2duhsngmesj/MatMul:product:0duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/add®
!duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp*duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!duhsngmesj/BiasAdd/ReadVariableOp¥
duhsngmesj/BiasAddBiasAddduhsngmesj/add:z:0)duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/BiasAddz
duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
duhsngmesj/split/split_dimë
duhsngmesj/splitSplit#duhsngmesj/split/split_dim:output:0duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
duhsngmesj/split
duhsngmesj/ReadVariableOpReadVariableOp"duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp
duhsngmesj/mulMul!duhsngmesj/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul
duhsngmesj/add_1AddV2duhsngmesj/split:output:0duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_1{
duhsngmesj/SigmoidSigmoidduhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid
duhsngmesj/ReadVariableOp_1ReadVariableOp$duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_1
duhsngmesj/mul_1Mul#duhsngmesj/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_1
duhsngmesj/add_2AddV2duhsngmesj/split:output:1duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_2
duhsngmesj/Sigmoid_1Sigmoidduhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_1
duhsngmesj/mul_2Mulduhsngmesj/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_2w
duhsngmesj/TanhTanhduhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh
duhsngmesj/mul_3Mulduhsngmesj/Sigmoid:y:0duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_3
duhsngmesj/add_3AddV2duhsngmesj/mul_2:z:0duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_3
duhsngmesj/ReadVariableOp_2ReadVariableOp$duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_2
duhsngmesj/mul_4Mul#duhsngmesj/ReadVariableOp_2:value:0duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_4
duhsngmesj/add_4AddV2duhsngmesj/split:output:3duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_4
duhsngmesj/Sigmoid_2Sigmoidduhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_2v
duhsngmesj/Tanh_1Tanhduhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh_1
duhsngmesj/mul_5Mulduhsngmesj/Sigmoid_2:y:0duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)duhsngmesj_matmul_readvariableop_resource+duhsngmesj_matmul_1_readvariableop_resource*duhsngmesj_biasadd_readvariableop_resource"duhsngmesj_readvariableop_resource$duhsngmesj_readvariableop_1_resource$duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_768774*
condR
while_cond_768773*Q
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
IdentityIdentitytranspose_1:y:0"^duhsngmesj/BiasAdd/ReadVariableOp!^duhsngmesj/MatMul/ReadVariableOp#^duhsngmesj/MatMul_1/ReadVariableOp^duhsngmesj/ReadVariableOp^duhsngmesj/ReadVariableOp_1^duhsngmesj/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!duhsngmesj/BiasAdd/ReadVariableOp!duhsngmesj/BiasAdd/ReadVariableOp2D
 duhsngmesj/MatMul/ReadVariableOp duhsngmesj/MatMul/ReadVariableOp2H
"duhsngmesj/MatMul_1/ReadVariableOp"duhsngmesj/MatMul_1/ReadVariableOp26
duhsngmesj/ReadVariableOpduhsngmesj/ReadVariableOp2:
duhsngmesj/ReadVariableOp_1duhsngmesj/ReadVariableOp_12:
duhsngmesj/ReadVariableOp_2duhsngmesj/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

+__inference_gdmdltnblf_layer_call_fn_769072
inputs_0
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¼
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7648242
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
Ù

+__inference_gdmdltnblf_layer_call_fn_769106

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall±
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7663972
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
Û

'sequential_uazvpibasg_while_body_764530H
Dsequential_uazvpibasg_while_sequential_uazvpibasg_while_loop_counterN
Jsequential_uazvpibasg_while_sequential_uazvpibasg_while_maximum_iterations+
'sequential_uazvpibasg_while_placeholder-
)sequential_uazvpibasg_while_placeholder_1-
)sequential_uazvpibasg_while_placeholder_2-
)sequential_uazvpibasg_while_placeholder_3G
Csequential_uazvpibasg_while_sequential_uazvpibasg_strided_slice_1_0
sequential_uazvpibasg_while_tensorarrayv2read_tensorlistgetitem_sequential_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0:	 \
Isequential_uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 W
Hsequential_uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0:	N
@sequential_uazvpibasg_while_gddwjadkgr_readvariableop_resource_0: P
Bsequential_uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0: P
Bsequential_uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0: (
$sequential_uazvpibasg_while_identity*
&sequential_uazvpibasg_while_identity_1*
&sequential_uazvpibasg_while_identity_2*
&sequential_uazvpibasg_while_identity_3*
&sequential_uazvpibasg_while_identity_4*
&sequential_uazvpibasg_while_identity_5E
Asequential_uazvpibasg_while_sequential_uazvpibasg_strided_slice_1
}sequential_uazvpibasg_while_tensorarrayv2read_tensorlistgetitem_sequential_uazvpibasg_tensorarrayunstack_tensorlistfromtensorX
Esequential_uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource:	 Z
Gsequential_uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource:	 U
Fsequential_uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource:	L
>sequential_uazvpibasg_while_gddwjadkgr_readvariableop_resource: N
@sequential_uazvpibasg_while_gddwjadkgr_readvariableop_1_resource: N
@sequential_uazvpibasg_while_gddwjadkgr_readvariableop_2_resource: ¢=sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp¢<sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp¢>sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp¢5sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp¢7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_1¢7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2ï
Msequential/uazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/uazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/uazvpibasg/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_uazvpibasg_while_tensorarrayv2read_tensorlistgetitem_sequential_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0'sequential_uazvpibasg_while_placeholderVsequential/uazvpibasg/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/uazvpibasg/while/TensorArrayV2Read/TensorListGetItem
<sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOpGsequential_uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp©
-sequential/uazvpibasg/while/gddwjadkgr/MatMulMatMulFsequential/uazvpibasg/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/uazvpibasg/while/gddwjadkgr/MatMul
>sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOpIsequential_uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp
/sequential/uazvpibasg/while/gddwjadkgr/MatMul_1MatMul)sequential_uazvpibasg_while_placeholder_2Fsequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/uazvpibasg/while/gddwjadkgr/MatMul_1
*sequential/uazvpibasg/while/gddwjadkgr/addAddV27sequential/uazvpibasg/while/gddwjadkgr/MatMul:product:09sequential/uazvpibasg/while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/uazvpibasg/while/gddwjadkgr/add
=sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOpHsequential_uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp
.sequential/uazvpibasg/while/gddwjadkgr/BiasAddBiasAdd.sequential/uazvpibasg/while/gddwjadkgr/add:z:0Esequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/uazvpibasg/while/gddwjadkgr/BiasAdd²
6sequential/uazvpibasg/while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/uazvpibasg/while/gddwjadkgr/split/split_dimÛ
,sequential/uazvpibasg/while/gddwjadkgr/splitSplit?sequential/uazvpibasg/while/gddwjadkgr/split/split_dim:output:07sequential/uazvpibasg/while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/uazvpibasg/while/gddwjadkgr/splitë
5sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOpReadVariableOp@sequential_uazvpibasg_while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOpû
*sequential/uazvpibasg/while/gddwjadkgr/mulMul=sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp:value:0)sequential_uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/uazvpibasg/while/gddwjadkgr/mulþ
,sequential/uazvpibasg/while/gddwjadkgr/add_1AddV25sequential/uazvpibasg/while/gddwjadkgr/split:output:0.sequential/uazvpibasg/while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/add_1Ï
.sequential/uazvpibasg/while/gddwjadkgr/SigmoidSigmoid0sequential/uazvpibasg/while/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/uazvpibasg/while/gddwjadkgr/Sigmoidñ
7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_1ReadVariableOpBsequential_uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_1
,sequential/uazvpibasg/while/gddwjadkgr/mul_1Mul?sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_1:value:0)sequential_uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/mul_1
,sequential/uazvpibasg/while/gddwjadkgr/add_2AddV25sequential/uazvpibasg/while/gddwjadkgr/split:output:10sequential/uazvpibasg/while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/add_2Ó
0sequential/uazvpibasg/while/gddwjadkgr/Sigmoid_1Sigmoid0sequential/uazvpibasg/while/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/uazvpibasg/while/gddwjadkgr/Sigmoid_1ö
,sequential/uazvpibasg/while/gddwjadkgr/mul_2Mul4sequential/uazvpibasg/while/gddwjadkgr/Sigmoid_1:y:0)sequential_uazvpibasg_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/mul_2Ë
+sequential/uazvpibasg/while/gddwjadkgr/TanhTanh5sequential/uazvpibasg/while/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/uazvpibasg/while/gddwjadkgr/Tanhú
,sequential/uazvpibasg/while/gddwjadkgr/mul_3Mul2sequential/uazvpibasg/while/gddwjadkgr/Sigmoid:y:0/sequential/uazvpibasg/while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/mul_3û
,sequential/uazvpibasg/while/gddwjadkgr/add_3AddV20sequential/uazvpibasg/while/gddwjadkgr/mul_2:z:00sequential/uazvpibasg/while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/add_3ñ
7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2ReadVariableOpBsequential_uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2
,sequential/uazvpibasg/while/gddwjadkgr/mul_4Mul?sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2:value:00sequential/uazvpibasg/while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/mul_4
,sequential/uazvpibasg/while/gddwjadkgr/add_4AddV25sequential/uazvpibasg/while/gddwjadkgr/split:output:30sequential/uazvpibasg/while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/add_4Ó
0sequential/uazvpibasg/while/gddwjadkgr/Sigmoid_2Sigmoid0sequential/uazvpibasg/while/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/uazvpibasg/while/gddwjadkgr/Sigmoid_2Ê
-sequential/uazvpibasg/while/gddwjadkgr/Tanh_1Tanh0sequential/uazvpibasg/while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/uazvpibasg/while/gddwjadkgr/Tanh_1þ
,sequential/uazvpibasg/while/gddwjadkgr/mul_5Mul4sequential/uazvpibasg/while/gddwjadkgr/Sigmoid_2:y:01sequential/uazvpibasg/while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/uazvpibasg/while/gddwjadkgr/mul_5Ì
@sequential/uazvpibasg/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_uazvpibasg_while_placeholder_1'sequential_uazvpibasg_while_placeholder0sequential/uazvpibasg/while/gddwjadkgr/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/uazvpibasg/while/TensorArrayV2Write/TensorListSetItem
!sequential/uazvpibasg/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/uazvpibasg/while/add/yÁ
sequential/uazvpibasg/while/addAddV2'sequential_uazvpibasg_while_placeholder*sequential/uazvpibasg/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/uazvpibasg/while/add
#sequential/uazvpibasg/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/uazvpibasg/while/add_1/yä
!sequential/uazvpibasg/while/add_1AddV2Dsequential_uazvpibasg_while_sequential_uazvpibasg_while_loop_counter,sequential/uazvpibasg/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/uazvpibasg/while/add_1
$sequential/uazvpibasg/while/IdentityIdentity%sequential/uazvpibasg/while/add_1:z:0>^sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp=^sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp?^sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp6^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp8^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_18^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/uazvpibasg/while/Identityµ
&sequential/uazvpibasg/while/Identity_1IdentityJsequential_uazvpibasg_while_sequential_uazvpibasg_while_maximum_iterations>^sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp=^sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp?^sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp6^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp8^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_18^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/uazvpibasg/while/Identity_1
&sequential/uazvpibasg/while/Identity_2Identity#sequential/uazvpibasg/while/add:z:0>^sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp=^sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp?^sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp6^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp8^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_18^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/uazvpibasg/while/Identity_2»
&sequential/uazvpibasg/while/Identity_3IdentityPsequential/uazvpibasg/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp=^sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp?^sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp6^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp8^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_18^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/uazvpibasg/while/Identity_3¬
&sequential/uazvpibasg/while/Identity_4Identity0sequential/uazvpibasg/while/gddwjadkgr/mul_5:z:0>^sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp=^sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp?^sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp6^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp8^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_18^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/while/Identity_4¬
&sequential/uazvpibasg/while/Identity_5Identity0sequential/uazvpibasg/while/gddwjadkgr/add_3:z:0>^sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp=^sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp?^sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp6^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp8^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_18^sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/while/Identity_5"
Fsequential_uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resourceHsequential_uazvpibasg_while_gddwjadkgr_biasadd_readvariableop_resource_0"
Gsequential_uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resourceIsequential_uazvpibasg_while_gddwjadkgr_matmul_1_readvariableop_resource_0"
Esequential_uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resourceGsequential_uazvpibasg_while_gddwjadkgr_matmul_readvariableop_resource_0"
@sequential_uazvpibasg_while_gddwjadkgr_readvariableop_1_resourceBsequential_uazvpibasg_while_gddwjadkgr_readvariableop_1_resource_0"
@sequential_uazvpibasg_while_gddwjadkgr_readvariableop_2_resourceBsequential_uazvpibasg_while_gddwjadkgr_readvariableop_2_resource_0"
>sequential_uazvpibasg_while_gddwjadkgr_readvariableop_resource@sequential_uazvpibasg_while_gddwjadkgr_readvariableop_resource_0"U
$sequential_uazvpibasg_while_identity-sequential/uazvpibasg/while/Identity:output:0"Y
&sequential_uazvpibasg_while_identity_1/sequential/uazvpibasg/while/Identity_1:output:0"Y
&sequential_uazvpibasg_while_identity_2/sequential/uazvpibasg/while/Identity_2:output:0"Y
&sequential_uazvpibasg_while_identity_3/sequential/uazvpibasg/while/Identity_3:output:0"Y
&sequential_uazvpibasg_while_identity_4/sequential/uazvpibasg/while/Identity_4:output:0"Y
&sequential_uazvpibasg_while_identity_5/sequential/uazvpibasg/while/Identity_5:output:0"
Asequential_uazvpibasg_while_sequential_uazvpibasg_strided_slice_1Csequential_uazvpibasg_while_sequential_uazvpibasg_strided_slice_1_0"
}sequential_uazvpibasg_while_tensorarrayv2read_tensorlistgetitem_sequential_uazvpibasg_tensorarrayunstack_tensorlistfromtensorsequential_uazvpibasg_while_tensorarrayv2read_tensorlistgetitem_sequential_uazvpibasg_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp=sequential/uazvpibasg/while/gddwjadkgr/BiasAdd/ReadVariableOp2|
<sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp<sequential/uazvpibasg/while/gddwjadkgr/MatMul/ReadVariableOp2
>sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp>sequential/uazvpibasg/while/gddwjadkgr/MatMul_1/ReadVariableOp2n
5sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp5sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp2r
7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_17sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_12r
7sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_27sequential/uazvpibasg/while/gddwjadkgr/ReadVariableOp_2: 
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
é

+__inference_uazvpibasg_layer_call_fn_769877
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¯
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7658452
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
Ýh

F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768695
inputs_0<
)duhsngmesj_matmul_readvariableop_resource:	>
+duhsngmesj_matmul_1_readvariableop_resource:	 9
*duhsngmesj_biasadd_readvariableop_resource:	0
"duhsngmesj_readvariableop_resource: 2
$duhsngmesj_readvariableop_1_resource: 2
$duhsngmesj_readvariableop_2_resource: 
identity¢!duhsngmesj/BiasAdd/ReadVariableOp¢ duhsngmesj/MatMul/ReadVariableOp¢"duhsngmesj/MatMul_1/ReadVariableOp¢duhsngmesj/ReadVariableOp¢duhsngmesj/ReadVariableOp_1¢duhsngmesj/ReadVariableOp_2¢whileF
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
 duhsngmesj/MatMul/ReadVariableOpReadVariableOp)duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 duhsngmesj/MatMul/ReadVariableOp§
duhsngmesj/MatMulMatMulstrided_slice_2:output:0(duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMulµ
"duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp+duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"duhsngmesj/MatMul_1/ReadVariableOp£
duhsngmesj/MatMul_1MatMulzeros:output:0*duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMul_1
duhsngmesj/addAddV2duhsngmesj/MatMul:product:0duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/add®
!duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp*duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!duhsngmesj/BiasAdd/ReadVariableOp¥
duhsngmesj/BiasAddBiasAddduhsngmesj/add:z:0)duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/BiasAddz
duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
duhsngmesj/split/split_dimë
duhsngmesj/splitSplit#duhsngmesj/split/split_dim:output:0duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
duhsngmesj/split
duhsngmesj/ReadVariableOpReadVariableOp"duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp
duhsngmesj/mulMul!duhsngmesj/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul
duhsngmesj/add_1AddV2duhsngmesj/split:output:0duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_1{
duhsngmesj/SigmoidSigmoidduhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid
duhsngmesj/ReadVariableOp_1ReadVariableOp$duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_1
duhsngmesj/mul_1Mul#duhsngmesj/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_1
duhsngmesj/add_2AddV2duhsngmesj/split:output:1duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_2
duhsngmesj/Sigmoid_1Sigmoidduhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_1
duhsngmesj/mul_2Mulduhsngmesj/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_2w
duhsngmesj/TanhTanhduhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh
duhsngmesj/mul_3Mulduhsngmesj/Sigmoid:y:0duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_3
duhsngmesj/add_3AddV2duhsngmesj/mul_2:z:0duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_3
duhsngmesj/ReadVariableOp_2ReadVariableOp$duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_2
duhsngmesj/mul_4Mul#duhsngmesj/ReadVariableOp_2:value:0duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_4
duhsngmesj/add_4AddV2duhsngmesj/split:output:3duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_4
duhsngmesj/Sigmoid_2Sigmoidduhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_2v
duhsngmesj/Tanh_1Tanhduhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh_1
duhsngmesj/mul_5Mulduhsngmesj/Sigmoid_2:y:0duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)duhsngmesj_matmul_readvariableop_resource+duhsngmesj_matmul_1_readvariableop_resource*duhsngmesj_biasadd_readvariableop_resource"duhsngmesj_readvariableop_resource$duhsngmesj_readvariableop_1_resource$duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_768594*
condR
while_cond_768593*Q
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
IdentityIdentitytranspose_1:y:0"^duhsngmesj/BiasAdd/ReadVariableOp!^duhsngmesj/MatMul/ReadVariableOp#^duhsngmesj/MatMul_1/ReadVariableOp^duhsngmesj/ReadVariableOp^duhsngmesj/ReadVariableOp_1^duhsngmesj/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!duhsngmesj/BiasAdd/ReadVariableOp!duhsngmesj/BiasAdd/ReadVariableOp2D
 duhsngmesj/MatMul/ReadVariableOp duhsngmesj/MatMul/ReadVariableOp2H
"duhsngmesj/MatMul_1/ReadVariableOp"duhsngmesj/MatMul_1/ReadVariableOp26
duhsngmesj/ReadVariableOpduhsngmesj/ReadVariableOp2:
duhsngmesj/ReadVariableOp_1duhsngmesj/ReadVariableOp_12:
duhsngmesj/ReadVariableOp_2duhsngmesj/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
F
ã
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_765087

inputs$
duhsngmesj_764988:	$
duhsngmesj_764990:	  
duhsngmesj_764992:	
duhsngmesj_764994: 
duhsngmesj_764996: 
duhsngmesj_764998: 
identity¢"duhsngmesj/StatefulPartitionedCall¢whileD
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
strided_slice_2Ó
"duhsngmesj/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0duhsngmesj_764988duhsngmesj_764990duhsngmesj_764992duhsngmesj_764994duhsngmesj_764996duhsngmesj_764998*
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
GPU2*0J 8 *O
fJRH
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_7649112$
"duhsngmesj/StatefulPartitionedCall
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
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0duhsngmesj_764988duhsngmesj_764990duhsngmesj_764992duhsngmesj_764994duhsngmesj_764996duhsngmesj_764998*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_765007*
condR
while_cond_765006*Q
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
IdentityIdentitytranspose_1:y:0#^duhsngmesj/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"duhsngmesj/StatefulPartitionedCall"duhsngmesj/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

'sequential_gdmdltnblf_while_body_764354H
Dsequential_gdmdltnblf_while_sequential_gdmdltnblf_while_loop_counterN
Jsequential_gdmdltnblf_while_sequential_gdmdltnblf_while_maximum_iterations+
'sequential_gdmdltnblf_while_placeholder-
)sequential_gdmdltnblf_while_placeholder_1-
)sequential_gdmdltnblf_while_placeholder_2-
)sequential_gdmdltnblf_while_placeholder_3G
Csequential_gdmdltnblf_while_sequential_gdmdltnblf_strided_slice_1_0
sequential_gdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_sequential_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0:	\
Isequential_gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0:	 W
Hsequential_gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0:	N
@sequential_gdmdltnblf_while_duhsngmesj_readvariableop_resource_0: P
Bsequential_gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0: P
Bsequential_gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0: (
$sequential_gdmdltnblf_while_identity*
&sequential_gdmdltnblf_while_identity_1*
&sequential_gdmdltnblf_while_identity_2*
&sequential_gdmdltnblf_while_identity_3*
&sequential_gdmdltnblf_while_identity_4*
&sequential_gdmdltnblf_while_identity_5E
Asequential_gdmdltnblf_while_sequential_gdmdltnblf_strided_slice_1
}sequential_gdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_sequential_gdmdltnblf_tensorarrayunstack_tensorlistfromtensorX
Esequential_gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource:	Z
Gsequential_gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource:	 U
Fsequential_gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource:	L
>sequential_gdmdltnblf_while_duhsngmesj_readvariableop_resource: N
@sequential_gdmdltnblf_while_duhsngmesj_readvariableop_1_resource: N
@sequential_gdmdltnblf_while_duhsngmesj_readvariableop_2_resource: ¢=sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp¢<sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp¢>sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp¢5sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp¢7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_1¢7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2ï
Msequential/gdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/gdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/gdmdltnblf/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_gdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_sequential_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0'sequential_gdmdltnblf_while_placeholderVsequential/gdmdltnblf/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/gdmdltnblf/while/TensorArrayV2Read/TensorListGetItem
<sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOpReadVariableOpGsequential_gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp©
-sequential/gdmdltnblf/while/duhsngmesj/MatMulMatMulFsequential/gdmdltnblf/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/gdmdltnblf/while/duhsngmesj/MatMul
>sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOpIsequential_gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp
/sequential/gdmdltnblf/while/duhsngmesj/MatMul_1MatMul)sequential_gdmdltnblf_while_placeholder_2Fsequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/gdmdltnblf/while/duhsngmesj/MatMul_1
*sequential/gdmdltnblf/while/duhsngmesj/addAddV27sequential/gdmdltnblf/while/duhsngmesj/MatMul:product:09sequential/gdmdltnblf/while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/gdmdltnblf/while/duhsngmesj/add
=sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOpHsequential_gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp
.sequential/gdmdltnblf/while/duhsngmesj/BiasAddBiasAdd.sequential/gdmdltnblf/while/duhsngmesj/add:z:0Esequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/gdmdltnblf/while/duhsngmesj/BiasAdd²
6sequential/gdmdltnblf/while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/gdmdltnblf/while/duhsngmesj/split/split_dimÛ
,sequential/gdmdltnblf/while/duhsngmesj/splitSplit?sequential/gdmdltnblf/while/duhsngmesj/split/split_dim:output:07sequential/gdmdltnblf/while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/gdmdltnblf/while/duhsngmesj/splitë
5sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOpReadVariableOp@sequential_gdmdltnblf_while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOpû
*sequential/gdmdltnblf/while/duhsngmesj/mulMul=sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp:value:0)sequential_gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/gdmdltnblf/while/duhsngmesj/mulþ
,sequential/gdmdltnblf/while/duhsngmesj/add_1AddV25sequential/gdmdltnblf/while/duhsngmesj/split:output:0.sequential/gdmdltnblf/while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/add_1Ï
.sequential/gdmdltnblf/while/duhsngmesj/SigmoidSigmoid0sequential/gdmdltnblf/while/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/gdmdltnblf/while/duhsngmesj/Sigmoidñ
7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_1ReadVariableOpBsequential_gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_1
,sequential/gdmdltnblf/while/duhsngmesj/mul_1Mul?sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_1:value:0)sequential_gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/mul_1
,sequential/gdmdltnblf/while/duhsngmesj/add_2AddV25sequential/gdmdltnblf/while/duhsngmesj/split:output:10sequential/gdmdltnblf/while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/add_2Ó
0sequential/gdmdltnblf/while/duhsngmesj/Sigmoid_1Sigmoid0sequential/gdmdltnblf/while/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/gdmdltnblf/while/duhsngmesj/Sigmoid_1ö
,sequential/gdmdltnblf/while/duhsngmesj/mul_2Mul4sequential/gdmdltnblf/while/duhsngmesj/Sigmoid_1:y:0)sequential_gdmdltnblf_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/mul_2Ë
+sequential/gdmdltnblf/while/duhsngmesj/TanhTanh5sequential/gdmdltnblf/while/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/gdmdltnblf/while/duhsngmesj/Tanhú
,sequential/gdmdltnblf/while/duhsngmesj/mul_3Mul2sequential/gdmdltnblf/while/duhsngmesj/Sigmoid:y:0/sequential/gdmdltnblf/while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/mul_3û
,sequential/gdmdltnblf/while/duhsngmesj/add_3AddV20sequential/gdmdltnblf/while/duhsngmesj/mul_2:z:00sequential/gdmdltnblf/while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/add_3ñ
7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2ReadVariableOpBsequential_gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2
,sequential/gdmdltnblf/while/duhsngmesj/mul_4Mul?sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2:value:00sequential/gdmdltnblf/while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/mul_4
,sequential/gdmdltnblf/while/duhsngmesj/add_4AddV25sequential/gdmdltnblf/while/duhsngmesj/split:output:30sequential/gdmdltnblf/while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/add_4Ó
0sequential/gdmdltnblf/while/duhsngmesj/Sigmoid_2Sigmoid0sequential/gdmdltnblf/while/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/gdmdltnblf/while/duhsngmesj/Sigmoid_2Ê
-sequential/gdmdltnblf/while/duhsngmesj/Tanh_1Tanh0sequential/gdmdltnblf/while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/gdmdltnblf/while/duhsngmesj/Tanh_1þ
,sequential/gdmdltnblf/while/duhsngmesj/mul_5Mul4sequential/gdmdltnblf/while/duhsngmesj/Sigmoid_2:y:01sequential/gdmdltnblf/while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/gdmdltnblf/while/duhsngmesj/mul_5Ì
@sequential/gdmdltnblf/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_gdmdltnblf_while_placeholder_1'sequential_gdmdltnblf_while_placeholder0sequential/gdmdltnblf/while/duhsngmesj/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/gdmdltnblf/while/TensorArrayV2Write/TensorListSetItem
!sequential/gdmdltnblf/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/gdmdltnblf/while/add/yÁ
sequential/gdmdltnblf/while/addAddV2'sequential_gdmdltnblf_while_placeholder*sequential/gdmdltnblf/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/gdmdltnblf/while/add
#sequential/gdmdltnblf/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/gdmdltnblf/while/add_1/yä
!sequential/gdmdltnblf/while/add_1AddV2Dsequential_gdmdltnblf_while_sequential_gdmdltnblf_while_loop_counter,sequential/gdmdltnblf/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/gdmdltnblf/while/add_1
$sequential/gdmdltnblf/while/IdentityIdentity%sequential/gdmdltnblf/while/add_1:z:0>^sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp=^sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp?^sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp6^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp8^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_18^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/gdmdltnblf/while/Identityµ
&sequential/gdmdltnblf/while/Identity_1IdentityJsequential_gdmdltnblf_while_sequential_gdmdltnblf_while_maximum_iterations>^sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp=^sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp?^sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp6^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp8^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_18^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/gdmdltnblf/while/Identity_1
&sequential/gdmdltnblf/while/Identity_2Identity#sequential/gdmdltnblf/while/add:z:0>^sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp=^sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp?^sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp6^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp8^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_18^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/gdmdltnblf/while/Identity_2»
&sequential/gdmdltnblf/while/Identity_3IdentityPsequential/gdmdltnblf/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp=^sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp?^sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp6^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp8^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_18^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/gdmdltnblf/while/Identity_3¬
&sequential/gdmdltnblf/while/Identity_4Identity0sequential/gdmdltnblf/while/duhsngmesj/mul_5:z:0>^sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp=^sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp?^sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp6^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp8^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_18^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/while/Identity_4¬
&sequential/gdmdltnblf/while/Identity_5Identity0sequential/gdmdltnblf/while/duhsngmesj/add_3:z:0>^sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp=^sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp?^sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp6^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp8^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_18^sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/while/Identity_5"
Fsequential_gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resourceHsequential_gdmdltnblf_while_duhsngmesj_biasadd_readvariableop_resource_0"
Gsequential_gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resourceIsequential_gdmdltnblf_while_duhsngmesj_matmul_1_readvariableop_resource_0"
Esequential_gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resourceGsequential_gdmdltnblf_while_duhsngmesj_matmul_readvariableop_resource_0"
@sequential_gdmdltnblf_while_duhsngmesj_readvariableop_1_resourceBsequential_gdmdltnblf_while_duhsngmesj_readvariableop_1_resource_0"
@sequential_gdmdltnblf_while_duhsngmesj_readvariableop_2_resourceBsequential_gdmdltnblf_while_duhsngmesj_readvariableop_2_resource_0"
>sequential_gdmdltnblf_while_duhsngmesj_readvariableop_resource@sequential_gdmdltnblf_while_duhsngmesj_readvariableop_resource_0"U
$sequential_gdmdltnblf_while_identity-sequential/gdmdltnblf/while/Identity:output:0"Y
&sequential_gdmdltnblf_while_identity_1/sequential/gdmdltnblf/while/Identity_1:output:0"Y
&sequential_gdmdltnblf_while_identity_2/sequential/gdmdltnblf/while/Identity_2:output:0"Y
&sequential_gdmdltnblf_while_identity_3/sequential/gdmdltnblf/while/Identity_3:output:0"Y
&sequential_gdmdltnblf_while_identity_4/sequential/gdmdltnblf/while/Identity_4:output:0"Y
&sequential_gdmdltnblf_while_identity_5/sequential/gdmdltnblf/while/Identity_5:output:0"
Asequential_gdmdltnblf_while_sequential_gdmdltnblf_strided_slice_1Csequential_gdmdltnblf_while_sequential_gdmdltnblf_strided_slice_1_0"
}sequential_gdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_sequential_gdmdltnblf_tensorarrayunstack_tensorlistfromtensorsequential_gdmdltnblf_while_tensorarrayv2read_tensorlistgetitem_sequential_gdmdltnblf_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp=sequential/gdmdltnblf/while/duhsngmesj/BiasAdd/ReadVariableOp2|
<sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp<sequential/gdmdltnblf/while/duhsngmesj/MatMul/ReadVariableOp2
>sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp>sequential/gdmdltnblf/while/duhsngmesj/MatMul_1/ReadVariableOp2n
5sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp5sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp2r
7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_17sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_12r
7sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_27sequential/gdmdltnblf/while/duhsngmesj/ReadVariableOp_2: 
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

¡	
'sequential_uazvpibasg_while_cond_764529H
Dsequential_uazvpibasg_while_sequential_uazvpibasg_while_loop_counterN
Jsequential_uazvpibasg_while_sequential_uazvpibasg_while_maximum_iterations+
'sequential_uazvpibasg_while_placeholder-
)sequential_uazvpibasg_while_placeholder_1-
)sequential_uazvpibasg_while_placeholder_2-
)sequential_uazvpibasg_while_placeholder_3J
Fsequential_uazvpibasg_while_less_sequential_uazvpibasg_strided_slice_1`
\sequential_uazvpibasg_while_sequential_uazvpibasg_while_cond_764529___redundant_placeholder0`
\sequential_uazvpibasg_while_sequential_uazvpibasg_while_cond_764529___redundant_placeholder1`
\sequential_uazvpibasg_while_sequential_uazvpibasg_while_cond_764529___redundant_placeholder2`
\sequential_uazvpibasg_while_sequential_uazvpibasg_while_cond_764529___redundant_placeholder3`
\sequential_uazvpibasg_while_sequential_uazvpibasg_while_cond_764529___redundant_placeholder4`
\sequential_uazvpibasg_while_sequential_uazvpibasg_while_cond_764529___redundant_placeholder5`
\sequential_uazvpibasg_while_sequential_uazvpibasg_while_cond_764529___redundant_placeholder6(
$sequential_uazvpibasg_while_identity
Þ
 sequential/uazvpibasg/while/LessLess'sequential_uazvpibasg_while_placeholderFsequential_uazvpibasg_while_less_sequential_uazvpibasg_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/uazvpibasg/while/Less
$sequential/uazvpibasg/while/IdentityIdentity$sequential/uazvpibasg/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/uazvpibasg/while/Identity"U
$sequential_uazvpibasg_while_identity-sequential/uazvpibasg/while/Identity:output:0*(
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


å
while_cond_765764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_765764___redundant_placeholder04
0while_while_cond_765764___redundant_placeholder14
0while_while_cond_765764___redundant_placeholder24
0while_while_cond_765764___redundant_placeholder34
0while_while_cond_765764___redundant_placeholder44
0while_while_cond_765764___redundant_placeholder54
0while_while_cond_765764___redundant_placeholder6
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
±

"__inference__traced_restore_770465
file_prefix8
"assignvariableop_ggbvrcpxtu_kernel:0
"assignvariableop_1_ggbvrcpxtu_bias:6
$assignvariableop_2_supobtndkp_kernel: 0
"assignvariableop_3_supobtndkp_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_gdmdltnblf_duhsngmesj_kernel:	M
:assignvariableop_10_gdmdltnblf_duhsngmesj_recurrent_kernel:	 =
.assignvariableop_11_gdmdltnblf_duhsngmesj_bias:	S
Eassignvariableop_12_gdmdltnblf_duhsngmesj_input_gate_peephole_weights: T
Fassignvariableop_13_gdmdltnblf_duhsngmesj_forget_gate_peephole_weights: T
Fassignvariableop_14_gdmdltnblf_duhsngmesj_output_gate_peephole_weights: C
0assignvariableop_15_uazvpibasg_gddwjadkgr_kernel:	 M
:assignvariableop_16_uazvpibasg_gddwjadkgr_recurrent_kernel:	 =
.assignvariableop_17_uazvpibasg_gddwjadkgr_bias:	S
Eassignvariableop_18_uazvpibasg_gddwjadkgr_input_gate_peephole_weights: T
Fassignvariableop_19_uazvpibasg_gddwjadkgr_forget_gate_peephole_weights: T
Fassignvariableop_20_uazvpibasg_gddwjadkgr_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_ggbvrcpxtu_kernel_rms:=
/assignvariableop_24_rmsprop_ggbvrcpxtu_bias_rms:C
1assignvariableop_25_rmsprop_supobtndkp_kernel_rms: =
/assignvariableop_26_rmsprop_supobtndkp_bias_rms:O
<assignvariableop_27_rmsprop_gdmdltnblf_duhsngmesj_kernel_rms:	Y
Fassignvariableop_28_rmsprop_gdmdltnblf_duhsngmesj_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_gdmdltnblf_duhsngmesj_bias_rms:	_
Qassignvariableop_30_rmsprop_gdmdltnblf_duhsngmesj_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_gdmdltnblf_duhsngmesj_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_gdmdltnblf_duhsngmesj_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_uazvpibasg_gddwjadkgr_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_uazvpibasg_gddwjadkgr_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_uazvpibasg_gddwjadkgr_bias_rms:	_
Qassignvariableop_36_rmsprop_uazvpibasg_gddwjadkgr_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_uazvpibasg_gddwjadkgr_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_uazvpibasg_gddwjadkgr_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_ggbvrcpxtu_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_ggbvrcpxtu_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_supobtndkp_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_supobtndkp_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_gdmdltnblf_duhsngmesj_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_gdmdltnblf_duhsngmesj_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_gdmdltnblf_duhsngmesj_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_gdmdltnblf_duhsngmesj_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_gdmdltnblf_duhsngmesj_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_gdmdltnblf_duhsngmesj_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_uazvpibasg_gddwjadkgr_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_uazvpibasg_gddwjadkgr_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_uazvpibasg_gddwjadkgr_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_uazvpibasg_gddwjadkgr_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_uazvpibasg_gddwjadkgr_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_uazvpibasg_gddwjadkgr_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_ggbvrcpxtu_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_ggbvrcpxtu_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_supobtndkp_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_supobtndkp_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_gdmdltnblf_duhsngmesj_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_gdmdltnblf_duhsngmesj_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_gdmdltnblf_duhsngmesj_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_gdmdltnblf_duhsngmesj_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_gdmdltnblf_duhsngmesj_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_gdmdltnblf_duhsngmesj_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_uazvpibasg_gddwjadkgr_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_uazvpibasg_gddwjadkgr_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_uazvpibasg_gddwjadkgr_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_uazvpibasg_gddwjadkgr_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_uazvpibasg_gddwjadkgr_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_uazvpibasg_gddwjadkgr_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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
ßY

while_body_769382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_gddwjadkgr_matmul_readvariableop_resource_0:	 F
3while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 A
2while_gddwjadkgr_biasadd_readvariableop_resource_0:	8
*while_gddwjadkgr_readvariableop_resource_0: :
,while_gddwjadkgr_readvariableop_1_resource_0: :
,while_gddwjadkgr_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_gddwjadkgr_matmul_readvariableop_resource:	 D
1while_gddwjadkgr_matmul_1_readvariableop_resource:	 ?
0while_gddwjadkgr_biasadd_readvariableop_resource:	6
(while_gddwjadkgr_readvariableop_resource: 8
*while_gddwjadkgr_readvariableop_1_resource: 8
*while_gddwjadkgr_readvariableop_2_resource: ¢'while/gddwjadkgr/BiasAdd/ReadVariableOp¢&while/gddwjadkgr/MatMul/ReadVariableOp¢(while/gddwjadkgr/MatMul_1/ReadVariableOp¢while/gddwjadkgr/ReadVariableOp¢!while/gddwjadkgr/ReadVariableOp_1¢!while/gddwjadkgr/ReadVariableOp_2Ã
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
&while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp1while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/gddwjadkgr/MatMul/ReadVariableOpÑ
while/gddwjadkgr/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMulÉ
(while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp3while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/gddwjadkgr/MatMul_1/ReadVariableOpº
while/gddwjadkgr/MatMul_1MatMulwhile_placeholder_20while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMul_1°
while/gddwjadkgr/addAddV2!while/gddwjadkgr/MatMul:product:0#while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/addÂ
'while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp2while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/gddwjadkgr/BiasAdd/ReadVariableOp½
while/gddwjadkgr/BiasAddBiasAddwhile/gddwjadkgr/add:z:0/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/BiasAdd
 while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/gddwjadkgr/split/split_dim
while/gddwjadkgr/splitSplit)while/gddwjadkgr/split/split_dim:output:0!while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gddwjadkgr/split©
while/gddwjadkgr/ReadVariableOpReadVariableOp*while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/gddwjadkgr/ReadVariableOp£
while/gddwjadkgr/mulMul'while/gddwjadkgr/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul¦
while/gddwjadkgr/add_1AddV2while/gddwjadkgr/split:output:0while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_1
while/gddwjadkgr/SigmoidSigmoidwhile/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid¯
!while/gddwjadkgr/ReadVariableOp_1ReadVariableOp,while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_1©
while/gddwjadkgr/mul_1Mul)while/gddwjadkgr/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_1¨
while/gddwjadkgr/add_2AddV2while/gddwjadkgr/split:output:1while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_2
while/gddwjadkgr/Sigmoid_1Sigmoidwhile/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_1
while/gddwjadkgr/mul_2Mulwhile/gddwjadkgr/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_2
while/gddwjadkgr/TanhTanhwhile/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh¢
while/gddwjadkgr/mul_3Mulwhile/gddwjadkgr/Sigmoid:y:0while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_3£
while/gddwjadkgr/add_3AddV2while/gddwjadkgr/mul_2:z:0while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_3¯
!while/gddwjadkgr/ReadVariableOp_2ReadVariableOp,while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_2°
while/gddwjadkgr/mul_4Mul)while/gddwjadkgr/ReadVariableOp_2:value:0while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_4¨
while/gddwjadkgr/add_4AddV2while/gddwjadkgr/split:output:3while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_4
while/gddwjadkgr/Sigmoid_2Sigmoidwhile/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_2
while/gddwjadkgr/Tanh_1Tanhwhile/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh_1¦
while/gddwjadkgr/mul_5Mulwhile/gddwjadkgr/Sigmoid_2:y:0while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gddwjadkgr/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/gddwjadkgr/mul_5:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/gddwjadkgr/add_3:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_gddwjadkgr_biasadd_readvariableop_resource2while_gddwjadkgr_biasadd_readvariableop_resource_0"h
1while_gddwjadkgr_matmul_1_readvariableop_resource3while_gddwjadkgr_matmul_1_readvariableop_resource_0"d
/while_gddwjadkgr_matmul_readvariableop_resource1while_gddwjadkgr_matmul_readvariableop_resource_0"Z
*while_gddwjadkgr_readvariableop_1_resource,while_gddwjadkgr_readvariableop_1_resource_0"Z
*while_gddwjadkgr_readvariableop_2_resource,while_gddwjadkgr_readvariableop_2_resource_0"V
(while_gddwjadkgr_readvariableop_resource*while_gddwjadkgr_readvariableop_resource_0")
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
'while/gddwjadkgr/BiasAdd/ReadVariableOp'while/gddwjadkgr/BiasAdd/ReadVariableOp2P
&while/gddwjadkgr/MatMul/ReadVariableOp&while/gddwjadkgr/MatMul/ReadVariableOp2T
(while/gddwjadkgr/MatMul_1/ReadVariableOp(while/gddwjadkgr/MatMul_1/ReadVariableOp2B
while/gddwjadkgr/ReadVariableOpwhile/gddwjadkgr/ReadVariableOp2F
!while/gddwjadkgr/ReadVariableOp_1!while/gddwjadkgr/ReadVariableOp_12F
!while/gddwjadkgr/ReadVariableOp_2!while/gddwjadkgr/ReadVariableOp_2: 
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


å
while_cond_769561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_769561___redundant_placeholder04
0while_while_cond_769561___redundant_placeholder14
0while_while_cond_769561___redundant_placeholder24
0while_while_cond_769561___redundant_placeholder34
0while_while_cond_769561___redundant_placeholder44
0while_while_cond_769561___redundant_placeholder54
0while_while_cond_769561___redundant_placeholder6
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


å
while_cond_768593
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_768593___redundant_placeholder04
0while_while_cond_768593___redundant_placeholder14
0while_while_cond_768593___redundant_placeholder24
0while_while_cond_768593___redundant_placeholder34
0while_while_cond_768593___redundant_placeholder44
0while_while_cond_768593___redundant_placeholder54
0while_while_cond_768593___redundant_placeholder6
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
Û
G
+__inference_vtqejmjhbd_layer_call_fn_768335

inputs
identityË
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
GPU2*0J 8 *O
fJRH
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_7662162
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
Ùh

F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769303
inputs_0<
)gddwjadkgr_matmul_readvariableop_resource:	 >
+gddwjadkgr_matmul_1_readvariableop_resource:	 9
*gddwjadkgr_biasadd_readvariableop_resource:	0
"gddwjadkgr_readvariableop_resource: 2
$gddwjadkgr_readvariableop_1_resource: 2
$gddwjadkgr_readvariableop_2_resource: 
identity¢!gddwjadkgr/BiasAdd/ReadVariableOp¢ gddwjadkgr/MatMul/ReadVariableOp¢"gddwjadkgr/MatMul_1/ReadVariableOp¢gddwjadkgr/ReadVariableOp¢gddwjadkgr/ReadVariableOp_1¢gddwjadkgr/ReadVariableOp_2¢whileF
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
 gddwjadkgr/MatMul/ReadVariableOpReadVariableOp)gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 gddwjadkgr/MatMul/ReadVariableOp§
gddwjadkgr/MatMulMatMulstrided_slice_2:output:0(gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMulµ
"gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp+gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"gddwjadkgr/MatMul_1/ReadVariableOp£
gddwjadkgr/MatMul_1MatMulzeros:output:0*gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMul_1
gddwjadkgr/addAddV2gddwjadkgr/MatMul:product:0gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/add®
!gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp*gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!gddwjadkgr/BiasAdd/ReadVariableOp¥
gddwjadkgr/BiasAddBiasAddgddwjadkgr/add:z:0)gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/BiasAddz
gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gddwjadkgr/split/split_dimë
gddwjadkgr/splitSplit#gddwjadkgr/split/split_dim:output:0gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gddwjadkgr/split
gddwjadkgr/ReadVariableOpReadVariableOp"gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp
gddwjadkgr/mulMul!gddwjadkgr/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul
gddwjadkgr/add_1AddV2gddwjadkgr/split:output:0gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_1{
gddwjadkgr/SigmoidSigmoidgddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid
gddwjadkgr/ReadVariableOp_1ReadVariableOp$gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_1
gddwjadkgr/mul_1Mul#gddwjadkgr/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_1
gddwjadkgr/add_2AddV2gddwjadkgr/split:output:1gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_2
gddwjadkgr/Sigmoid_1Sigmoidgddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_1
gddwjadkgr/mul_2Mulgddwjadkgr/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_2w
gddwjadkgr/TanhTanhgddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh
gddwjadkgr/mul_3Mulgddwjadkgr/Sigmoid:y:0gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_3
gddwjadkgr/add_3AddV2gddwjadkgr/mul_2:z:0gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_3
gddwjadkgr/ReadVariableOp_2ReadVariableOp$gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_2
gddwjadkgr/mul_4Mul#gddwjadkgr/ReadVariableOp_2:value:0gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_4
gddwjadkgr/add_4AddV2gddwjadkgr/split:output:3gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_4
gddwjadkgr/Sigmoid_2Sigmoidgddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_2v
gddwjadkgr/Tanh_1Tanhgddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh_1
gddwjadkgr/mul_5Mulgddwjadkgr/Sigmoid_2:y:0gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gddwjadkgr_matmul_readvariableop_resource+gddwjadkgr_matmul_1_readvariableop_resource*gddwjadkgr_biasadd_readvariableop_resource"gddwjadkgr_readvariableop_resource$gddwjadkgr_readvariableop_1_resource$gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_769202*
condR
while_cond_769201*Q
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
IdentityIdentitystrided_slice_3:output:0"^gddwjadkgr/BiasAdd/ReadVariableOp!^gddwjadkgr/MatMul/ReadVariableOp#^gddwjadkgr/MatMul_1/ReadVariableOp^gddwjadkgr/ReadVariableOp^gddwjadkgr/ReadVariableOp_1^gddwjadkgr/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gddwjadkgr/BiasAdd/ReadVariableOp!gddwjadkgr/BiasAdd/ReadVariableOp2D
 gddwjadkgr/MatMul/ReadVariableOp gddwjadkgr/MatMul/ReadVariableOp2H
"gddwjadkgr/MatMul_1/ReadVariableOp"gddwjadkgr/MatMul_1/ReadVariableOp26
gddwjadkgr/ReadVariableOpgddwjadkgr/ReadVariableOp2:
gddwjadkgr/ReadVariableOp_1gddwjadkgr/ReadVariableOp_12:
gddwjadkgr/ReadVariableOp_2gddwjadkgr/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


å
while_cond_769381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_769381___redundant_placeholder04
0while_while_cond_769381___redundant_placeholder14
0while_while_cond_769381___redundant_placeholder24
0while_while_cond_769381___redundant_placeholder34
0while_while_cond_769381___redundant_placeholder44
0while_while_cond_769381___redundant_placeholder54
0while_while_cond_769381___redundant_placeholder6
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


å
while_cond_766763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_766763___redundant_placeholder04
0while_while_cond_766763___redundant_placeholder14
0while_while_cond_766763___redundant_placeholder24
0while_while_cond_766763___redundant_placeholder34
0while_while_cond_766763___redundant_placeholder44
0while_while_cond_766763___redundant_placeholder54
0while_while_cond_766763___redundant_placeholder6
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
Ç)
Å
while_body_765502
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gddwjadkgr_765526_0:	 ,
while_gddwjadkgr_765528_0:	 (
while_gddwjadkgr_765530_0:	'
while_gddwjadkgr_765532_0: '
while_gddwjadkgr_765534_0: '
while_gddwjadkgr_765536_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gddwjadkgr_765526:	 *
while_gddwjadkgr_765528:	 &
while_gddwjadkgr_765530:	%
while_gddwjadkgr_765532: %
while_gddwjadkgr_765534: %
while_gddwjadkgr_765536: ¢(while/gddwjadkgr/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¯
(while/gddwjadkgr/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_gddwjadkgr_765526_0while_gddwjadkgr_765528_0while_gddwjadkgr_765530_0while_gddwjadkgr_765532_0while_gddwjadkgr_765534_0while_gddwjadkgr_765536_0*
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
GPU2*0J 8 *O
fJRH
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_7654822*
(while/gddwjadkgr/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gddwjadkgr/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/gddwjadkgr/StatefulPartitionedCall:output:1)^while/gddwjadkgr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/gddwjadkgr/StatefulPartitionedCall:output:2)^while/gddwjadkgr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_gddwjadkgr_765526while_gddwjadkgr_765526_0"4
while_gddwjadkgr_765528while_gddwjadkgr_765528_0"4
while_gddwjadkgr_765530while_gddwjadkgr_765530_0"4
while_gddwjadkgr_765532while_gddwjadkgr_765532_0"4
while_gddwjadkgr_765534while_gddwjadkgr_765534_0"4
while_gddwjadkgr_765536while_gddwjadkgr_765536_0")
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
(while/gddwjadkgr/StatefulPartitionedCall(while/gddwjadkgr/StatefulPartitionedCall: 
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
£h

F__inference_uazvpibasg_layer_call_and_return_conditional_losses_766590

inputs<
)gddwjadkgr_matmul_readvariableop_resource:	 >
+gddwjadkgr_matmul_1_readvariableop_resource:	 9
*gddwjadkgr_biasadd_readvariableop_resource:	0
"gddwjadkgr_readvariableop_resource: 2
$gddwjadkgr_readvariableop_1_resource: 2
$gddwjadkgr_readvariableop_2_resource: 
identity¢!gddwjadkgr/BiasAdd/ReadVariableOp¢ gddwjadkgr/MatMul/ReadVariableOp¢"gddwjadkgr/MatMul_1/ReadVariableOp¢gddwjadkgr/ReadVariableOp¢gddwjadkgr/ReadVariableOp_1¢gddwjadkgr/ReadVariableOp_2¢whileD
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
 gddwjadkgr/MatMul/ReadVariableOpReadVariableOp)gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 gddwjadkgr/MatMul/ReadVariableOp§
gddwjadkgr/MatMulMatMulstrided_slice_2:output:0(gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMulµ
"gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp+gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"gddwjadkgr/MatMul_1/ReadVariableOp£
gddwjadkgr/MatMul_1MatMulzeros:output:0*gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMul_1
gddwjadkgr/addAddV2gddwjadkgr/MatMul:product:0gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/add®
!gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp*gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!gddwjadkgr/BiasAdd/ReadVariableOp¥
gddwjadkgr/BiasAddBiasAddgddwjadkgr/add:z:0)gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/BiasAddz
gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gddwjadkgr/split/split_dimë
gddwjadkgr/splitSplit#gddwjadkgr/split/split_dim:output:0gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gddwjadkgr/split
gddwjadkgr/ReadVariableOpReadVariableOp"gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp
gddwjadkgr/mulMul!gddwjadkgr/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul
gddwjadkgr/add_1AddV2gddwjadkgr/split:output:0gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_1{
gddwjadkgr/SigmoidSigmoidgddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid
gddwjadkgr/ReadVariableOp_1ReadVariableOp$gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_1
gddwjadkgr/mul_1Mul#gddwjadkgr/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_1
gddwjadkgr/add_2AddV2gddwjadkgr/split:output:1gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_2
gddwjadkgr/Sigmoid_1Sigmoidgddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_1
gddwjadkgr/mul_2Mulgddwjadkgr/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_2w
gddwjadkgr/TanhTanhgddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh
gddwjadkgr/mul_3Mulgddwjadkgr/Sigmoid:y:0gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_3
gddwjadkgr/add_3AddV2gddwjadkgr/mul_2:z:0gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_3
gddwjadkgr/ReadVariableOp_2ReadVariableOp$gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_2
gddwjadkgr/mul_4Mul#gddwjadkgr/ReadVariableOp_2:value:0gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_4
gddwjadkgr/add_4AddV2gddwjadkgr/split:output:3gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_4
gddwjadkgr/Sigmoid_2Sigmoidgddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_2v
gddwjadkgr/Tanh_1Tanhgddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh_1
gddwjadkgr/mul_5Mulgddwjadkgr/Sigmoid_2:y:0gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gddwjadkgr_matmul_readvariableop_resource+gddwjadkgr_matmul_1_readvariableop_resource*gddwjadkgr_biasadd_readvariableop_resource"gddwjadkgr_readvariableop_resource$gddwjadkgr_readvariableop_1_resource$gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_766489*
condR
while_cond_766488*Q
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
IdentityIdentitystrided_slice_3:output:0"^gddwjadkgr/BiasAdd/ReadVariableOp!^gddwjadkgr/MatMul/ReadVariableOp#^gddwjadkgr/MatMul_1/ReadVariableOp^gddwjadkgr/ReadVariableOp^gddwjadkgr/ReadVariableOp_1^gddwjadkgr/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gddwjadkgr/BiasAdd/ReadVariableOp!gddwjadkgr/BiasAdd/ReadVariableOp2D
 gddwjadkgr/MatMul/ReadVariableOp gddwjadkgr/MatMul/ReadVariableOp2H
"gddwjadkgr/MatMul_1/ReadVariableOp"gddwjadkgr/MatMul_1/ReadVariableOp26
gddwjadkgr/ReadVariableOpgddwjadkgr/ReadVariableOp2:
gddwjadkgr/ReadVariableOp_1gddwjadkgr/ReadVariableOp_12:
gddwjadkgr/ReadVariableOp_2gddwjadkgr/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ

+__inference_uazvpibasg_layer_call_fn_769911

inputs
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall­
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7668652
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


å
while_cond_768773
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_768773___redundant_placeholder04
0while_while_cond_768773___redundant_placeholder14
0while_while_cond_768773___redundant_placeholder24
0while_while_cond_768773___redundant_placeholder34
0while_while_cond_768773___redundant_placeholder44
0while_while_cond_768773___redundant_placeholder54
0while_while_cond_768773___redundant_placeholder6
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
ßY

while_body_766296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_duhsngmesj_matmul_readvariableop_resource_0:	F
3while_duhsngmesj_matmul_1_readvariableop_resource_0:	 A
2while_duhsngmesj_biasadd_readvariableop_resource_0:	8
*while_duhsngmesj_readvariableop_resource_0: :
,while_duhsngmesj_readvariableop_1_resource_0: :
,while_duhsngmesj_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_duhsngmesj_matmul_readvariableop_resource:	D
1while_duhsngmesj_matmul_1_readvariableop_resource:	 ?
0while_duhsngmesj_biasadd_readvariableop_resource:	6
(while_duhsngmesj_readvariableop_resource: 8
*while_duhsngmesj_readvariableop_1_resource: 8
*while_duhsngmesj_readvariableop_2_resource: ¢'while/duhsngmesj/BiasAdd/ReadVariableOp¢&while/duhsngmesj/MatMul/ReadVariableOp¢(while/duhsngmesj/MatMul_1/ReadVariableOp¢while/duhsngmesj/ReadVariableOp¢!while/duhsngmesj/ReadVariableOp_1¢!while/duhsngmesj/ReadVariableOp_2Ã
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
&while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp1while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/duhsngmesj/MatMul/ReadVariableOpÑ
while/duhsngmesj/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMulÉ
(while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp3while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/duhsngmesj/MatMul_1/ReadVariableOpº
while/duhsngmesj/MatMul_1MatMulwhile_placeholder_20while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMul_1°
while/duhsngmesj/addAddV2!while/duhsngmesj/MatMul:product:0#while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/addÂ
'while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp2while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/duhsngmesj/BiasAdd/ReadVariableOp½
while/duhsngmesj/BiasAddBiasAddwhile/duhsngmesj/add:z:0/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/BiasAdd
 while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/duhsngmesj/split/split_dim
while/duhsngmesj/splitSplit)while/duhsngmesj/split/split_dim:output:0!while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/duhsngmesj/split©
while/duhsngmesj/ReadVariableOpReadVariableOp*while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/duhsngmesj/ReadVariableOp£
while/duhsngmesj/mulMul'while/duhsngmesj/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul¦
while/duhsngmesj/add_1AddV2while/duhsngmesj/split:output:0while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_1
while/duhsngmesj/SigmoidSigmoidwhile/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid¯
!while/duhsngmesj/ReadVariableOp_1ReadVariableOp,while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_1©
while/duhsngmesj/mul_1Mul)while/duhsngmesj/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_1¨
while/duhsngmesj/add_2AddV2while/duhsngmesj/split:output:1while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_2
while/duhsngmesj/Sigmoid_1Sigmoidwhile/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_1
while/duhsngmesj/mul_2Mulwhile/duhsngmesj/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_2
while/duhsngmesj/TanhTanhwhile/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh¢
while/duhsngmesj/mul_3Mulwhile/duhsngmesj/Sigmoid:y:0while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_3£
while/duhsngmesj/add_3AddV2while/duhsngmesj/mul_2:z:0while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_3¯
!while/duhsngmesj/ReadVariableOp_2ReadVariableOp,while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_2°
while/duhsngmesj/mul_4Mul)while/duhsngmesj/ReadVariableOp_2:value:0while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_4¨
while/duhsngmesj/add_4AddV2while/duhsngmesj/split:output:3while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_4
while/duhsngmesj/Sigmoid_2Sigmoidwhile/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_2
while/duhsngmesj/Tanh_1Tanhwhile/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh_1¦
while/duhsngmesj/mul_5Mulwhile/duhsngmesj/Sigmoid_2:y:0while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/duhsngmesj/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/duhsngmesj/mul_5:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/duhsngmesj/add_3:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_duhsngmesj_biasadd_readvariableop_resource2while_duhsngmesj_biasadd_readvariableop_resource_0"h
1while_duhsngmesj_matmul_1_readvariableop_resource3while_duhsngmesj_matmul_1_readvariableop_resource_0"d
/while_duhsngmesj_matmul_readvariableop_resource1while_duhsngmesj_matmul_readvariableop_resource_0"Z
*while_duhsngmesj_readvariableop_1_resource,while_duhsngmesj_readvariableop_1_resource_0"Z
*while_duhsngmesj_readvariableop_2_resource,while_duhsngmesj_readvariableop_2_resource_0"V
(while_duhsngmesj_readvariableop_resource*while_duhsngmesj_readvariableop_resource_0")
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
'while/duhsngmesj/BiasAdd/ReadVariableOp'while/duhsngmesj/BiasAdd/ReadVariableOp2P
&while/duhsngmesj/MatMul/ReadVariableOp&while/duhsngmesj/MatMul/ReadVariableOp2T
(while/duhsngmesj/MatMul_1/ReadVariableOp(while/duhsngmesj/MatMul_1/ReadVariableOp2B
while/duhsngmesj/ReadVariableOpwhile/duhsngmesj/ReadVariableOp2F
!while/duhsngmesj/ReadVariableOp_1!while/duhsngmesj/ReadVariableOp_12F
!while/duhsngmesj/ReadVariableOp_2!while/duhsngmesj/ReadVariableOp_2: 
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
ßY

while_body_766489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_gddwjadkgr_matmul_readvariableop_resource_0:	 F
3while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 A
2while_gddwjadkgr_biasadd_readvariableop_resource_0:	8
*while_gddwjadkgr_readvariableop_resource_0: :
,while_gddwjadkgr_readvariableop_1_resource_0: :
,while_gddwjadkgr_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_gddwjadkgr_matmul_readvariableop_resource:	 D
1while_gddwjadkgr_matmul_1_readvariableop_resource:	 ?
0while_gddwjadkgr_biasadd_readvariableop_resource:	6
(while_gddwjadkgr_readvariableop_resource: 8
*while_gddwjadkgr_readvariableop_1_resource: 8
*while_gddwjadkgr_readvariableop_2_resource: ¢'while/gddwjadkgr/BiasAdd/ReadVariableOp¢&while/gddwjadkgr/MatMul/ReadVariableOp¢(while/gddwjadkgr/MatMul_1/ReadVariableOp¢while/gddwjadkgr/ReadVariableOp¢!while/gddwjadkgr/ReadVariableOp_1¢!while/gddwjadkgr/ReadVariableOp_2Ã
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
&while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp1while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/gddwjadkgr/MatMul/ReadVariableOpÑ
while/gddwjadkgr/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMulÉ
(while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp3while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/gddwjadkgr/MatMul_1/ReadVariableOpº
while/gddwjadkgr/MatMul_1MatMulwhile_placeholder_20while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMul_1°
while/gddwjadkgr/addAddV2!while/gddwjadkgr/MatMul:product:0#while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/addÂ
'while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp2while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/gddwjadkgr/BiasAdd/ReadVariableOp½
while/gddwjadkgr/BiasAddBiasAddwhile/gddwjadkgr/add:z:0/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/BiasAdd
 while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/gddwjadkgr/split/split_dim
while/gddwjadkgr/splitSplit)while/gddwjadkgr/split/split_dim:output:0!while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gddwjadkgr/split©
while/gddwjadkgr/ReadVariableOpReadVariableOp*while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/gddwjadkgr/ReadVariableOp£
while/gddwjadkgr/mulMul'while/gddwjadkgr/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul¦
while/gddwjadkgr/add_1AddV2while/gddwjadkgr/split:output:0while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_1
while/gddwjadkgr/SigmoidSigmoidwhile/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid¯
!while/gddwjadkgr/ReadVariableOp_1ReadVariableOp,while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_1©
while/gddwjadkgr/mul_1Mul)while/gddwjadkgr/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_1¨
while/gddwjadkgr/add_2AddV2while/gddwjadkgr/split:output:1while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_2
while/gddwjadkgr/Sigmoid_1Sigmoidwhile/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_1
while/gddwjadkgr/mul_2Mulwhile/gddwjadkgr/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_2
while/gddwjadkgr/TanhTanhwhile/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh¢
while/gddwjadkgr/mul_3Mulwhile/gddwjadkgr/Sigmoid:y:0while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_3£
while/gddwjadkgr/add_3AddV2while/gddwjadkgr/mul_2:z:0while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_3¯
!while/gddwjadkgr/ReadVariableOp_2ReadVariableOp,while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_2°
while/gddwjadkgr/mul_4Mul)while/gddwjadkgr/ReadVariableOp_2:value:0while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_4¨
while/gddwjadkgr/add_4AddV2while/gddwjadkgr/split:output:3while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_4
while/gddwjadkgr/Sigmoid_2Sigmoidwhile/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_2
while/gddwjadkgr/Tanh_1Tanhwhile/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh_1¦
while/gddwjadkgr/mul_5Mulwhile/gddwjadkgr/Sigmoid_2:y:0while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gddwjadkgr/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/gddwjadkgr/mul_5:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/gddwjadkgr/add_3:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_gddwjadkgr_biasadd_readvariableop_resource2while_gddwjadkgr_biasadd_readvariableop_resource_0"h
1while_gddwjadkgr_matmul_1_readvariableop_resource3while_gddwjadkgr_matmul_1_readvariableop_resource_0"d
/while_gddwjadkgr_matmul_readvariableop_resource1while_gddwjadkgr_matmul_readvariableop_resource_0"Z
*while_gddwjadkgr_readvariableop_1_resource,while_gddwjadkgr_readvariableop_1_resource_0"Z
*while_gddwjadkgr_readvariableop_2_resource,while_gddwjadkgr_readvariableop_2_resource_0"V
(while_gddwjadkgr_readvariableop_resource*while_gddwjadkgr_readvariableop_resource_0")
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
'while/gddwjadkgr/BiasAdd/ReadVariableOp'while/gddwjadkgr/BiasAdd/ReadVariableOp2P
&while/gddwjadkgr/MatMul/ReadVariableOp&while/gddwjadkgr/MatMul/ReadVariableOp2T
(while/gddwjadkgr/MatMul_1/ReadVariableOp(while/gddwjadkgr/MatMul_1/ReadVariableOp2B
while/gddwjadkgr/ReadVariableOpwhile/gddwjadkgr/ReadVariableOp2F
!while/gddwjadkgr/ReadVariableOp_1!while/gddwjadkgr/ReadVariableOp_12F
!while/gddwjadkgr/ReadVariableOp_2!while/gddwjadkgr/ReadVariableOp_2: 
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
F
ã
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_765582

inputs$
gddwjadkgr_765483:	 $
gddwjadkgr_765485:	  
gddwjadkgr_765487:	
gddwjadkgr_765489: 
gddwjadkgr_765491: 
gddwjadkgr_765493: 
identity¢"gddwjadkgr/StatefulPartitionedCall¢whileD
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
strided_slice_2Ó
"gddwjadkgr/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0gddwjadkgr_765483gddwjadkgr_765485gddwjadkgr_765487gddwjadkgr_765489gddwjadkgr_765491gddwjadkgr_765493*
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
GPU2*0J 8 *O
fJRH
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_7654822$
"gddwjadkgr/StatefulPartitionedCall
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
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gddwjadkgr_765483gddwjadkgr_765485gddwjadkgr_765487gddwjadkgr_765489gddwjadkgr_765491gddwjadkgr_765493*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_765502*
condR
while_cond_765501*Q
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
IdentityIdentitystrided_slice_3:output:0#^gddwjadkgr/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"gddwjadkgr/StatefulPartitionedCall"gddwjadkgr/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°'
²
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_764911

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
Ò	
÷
F__inference_supobtndkp_layer_call_and_return_conditional_losses_769921

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
¸'
´
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_770152

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


+__inference_sequential_layer_call_fn_766656

unopekwlxv
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
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCall
unopekwlxvunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_7666212
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
unopekwlxv

«
F__inference_sequential_layer_call_and_return_conditional_losses_766621

inputs'
ggbvrcpxtu_766198:
ggbvrcpxtu_766200:$
gdmdltnblf_766398:	$
gdmdltnblf_766400:	  
gdmdltnblf_766402:	
gdmdltnblf_766404: 
gdmdltnblf_766406: 
gdmdltnblf_766408: $
uazvpibasg_766591:	 $
uazvpibasg_766593:	  
uazvpibasg_766595:	
uazvpibasg_766597: 
uazvpibasg_766599: 
uazvpibasg_766601: #
supobtndkp_766615: 
supobtndkp_766617:
identity¢"gdmdltnblf/StatefulPartitionedCall¢"ggbvrcpxtu/StatefulPartitionedCall¢"supobtndkp/StatefulPartitionedCall¢"uazvpibasg/StatefulPartitionedCall©
"ggbvrcpxtu/StatefulPartitionedCallStatefulPartitionedCallinputsggbvrcpxtu_766198ggbvrcpxtu_766200*
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
GPU2*0J 8 *O
fJRH
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_7661972$
"ggbvrcpxtu/StatefulPartitionedCall
vtqejmjhbd/PartitionedCallPartitionedCall+ggbvrcpxtu/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_7662162
vtqejmjhbd/PartitionedCall
"gdmdltnblf/StatefulPartitionedCallStatefulPartitionedCall#vtqejmjhbd/PartitionedCall:output:0gdmdltnblf_766398gdmdltnblf_766400gdmdltnblf_766402gdmdltnblf_766404gdmdltnblf_766406gdmdltnblf_766408*
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7663972$
"gdmdltnblf/StatefulPartitionedCall
"uazvpibasg/StatefulPartitionedCallStatefulPartitionedCall+gdmdltnblf/StatefulPartitionedCall:output:0uazvpibasg_766591uazvpibasg_766593uazvpibasg_766595uazvpibasg_766597uazvpibasg_766599uazvpibasg_766601*
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7665902$
"uazvpibasg/StatefulPartitionedCallÆ
"supobtndkp/StatefulPartitionedCallStatefulPartitionedCall+uazvpibasg/StatefulPartitionedCall:output:0supobtndkp_766615supobtndkp_766617*
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
GPU2*0J 8 *O
fJRH
F__inference_supobtndkp_layer_call_and_return_conditional_losses_7666142$
"supobtndkp/StatefulPartitionedCall
IdentityIdentity+supobtndkp/StatefulPartitionedCall:output:0#^gdmdltnblf/StatefulPartitionedCall#^ggbvrcpxtu/StatefulPartitionedCall#^supobtndkp/StatefulPartitionedCall#^uazvpibasg/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"gdmdltnblf/StatefulPartitionedCall"gdmdltnblf/StatefulPartitionedCall2H
"ggbvrcpxtu/StatefulPartitionedCall"ggbvrcpxtu/StatefulPartitionedCall2H
"supobtndkp/StatefulPartitionedCall"supobtndkp/StatefulPartitionedCall2H
"uazvpibasg/StatefulPartitionedCall"uazvpibasg/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ýh

F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768515
inputs_0<
)duhsngmesj_matmul_readvariableop_resource:	>
+duhsngmesj_matmul_1_readvariableop_resource:	 9
*duhsngmesj_biasadd_readvariableop_resource:	0
"duhsngmesj_readvariableop_resource: 2
$duhsngmesj_readvariableop_1_resource: 2
$duhsngmesj_readvariableop_2_resource: 
identity¢!duhsngmesj/BiasAdd/ReadVariableOp¢ duhsngmesj/MatMul/ReadVariableOp¢"duhsngmesj/MatMul_1/ReadVariableOp¢duhsngmesj/ReadVariableOp¢duhsngmesj/ReadVariableOp_1¢duhsngmesj/ReadVariableOp_2¢whileF
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
 duhsngmesj/MatMul/ReadVariableOpReadVariableOp)duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 duhsngmesj/MatMul/ReadVariableOp§
duhsngmesj/MatMulMatMulstrided_slice_2:output:0(duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMulµ
"duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp+duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"duhsngmesj/MatMul_1/ReadVariableOp£
duhsngmesj/MatMul_1MatMulzeros:output:0*duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMul_1
duhsngmesj/addAddV2duhsngmesj/MatMul:product:0duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/add®
!duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp*duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!duhsngmesj/BiasAdd/ReadVariableOp¥
duhsngmesj/BiasAddBiasAddduhsngmesj/add:z:0)duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/BiasAddz
duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
duhsngmesj/split/split_dimë
duhsngmesj/splitSplit#duhsngmesj/split/split_dim:output:0duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
duhsngmesj/split
duhsngmesj/ReadVariableOpReadVariableOp"duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp
duhsngmesj/mulMul!duhsngmesj/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul
duhsngmesj/add_1AddV2duhsngmesj/split:output:0duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_1{
duhsngmesj/SigmoidSigmoidduhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid
duhsngmesj/ReadVariableOp_1ReadVariableOp$duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_1
duhsngmesj/mul_1Mul#duhsngmesj/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_1
duhsngmesj/add_2AddV2duhsngmesj/split:output:1duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_2
duhsngmesj/Sigmoid_1Sigmoidduhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_1
duhsngmesj/mul_2Mulduhsngmesj/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_2w
duhsngmesj/TanhTanhduhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh
duhsngmesj/mul_3Mulduhsngmesj/Sigmoid:y:0duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_3
duhsngmesj/add_3AddV2duhsngmesj/mul_2:z:0duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_3
duhsngmesj/ReadVariableOp_2ReadVariableOp$duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_2
duhsngmesj/mul_4Mul#duhsngmesj/ReadVariableOp_2:value:0duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_4
duhsngmesj/add_4AddV2duhsngmesj/split:output:3duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_4
duhsngmesj/Sigmoid_2Sigmoidduhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_2v
duhsngmesj/Tanh_1Tanhduhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh_1
duhsngmesj/mul_5Mulduhsngmesj/Sigmoid_2:y:0duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)duhsngmesj_matmul_readvariableop_resource+duhsngmesj_matmul_1_readvariableop_resource*duhsngmesj_biasadd_readvariableop_resource"duhsngmesj_readvariableop_resource$duhsngmesj_readvariableop_1_resource$duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_768414*
condR
while_cond_768413*Q
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
IdentityIdentitytranspose_1:y:0"^duhsngmesj/BiasAdd/ReadVariableOp!^duhsngmesj/MatMul/ReadVariableOp#^duhsngmesj/MatMul_1/ReadVariableOp^duhsngmesj/ReadVariableOp^duhsngmesj/ReadVariableOp_1^duhsngmesj/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!duhsngmesj/BiasAdd/ReadVariableOp!duhsngmesj/BiasAdd/ReadVariableOp2D
 duhsngmesj/MatMul/ReadVariableOp duhsngmesj/MatMul/ReadVariableOp2H
"duhsngmesj/MatMul_1/ReadVariableOp"duhsngmesj/MatMul_1/ReadVariableOp26
duhsngmesj/ReadVariableOpduhsngmesj/ReadVariableOp2:
duhsngmesj/ReadVariableOp_1duhsngmesj/ReadVariableOp_12:
duhsngmesj/ReadVariableOp_2duhsngmesj/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

¡	
'sequential_gdmdltnblf_while_cond_764353H
Dsequential_gdmdltnblf_while_sequential_gdmdltnblf_while_loop_counterN
Jsequential_gdmdltnblf_while_sequential_gdmdltnblf_while_maximum_iterations+
'sequential_gdmdltnblf_while_placeholder-
)sequential_gdmdltnblf_while_placeholder_1-
)sequential_gdmdltnblf_while_placeholder_2-
)sequential_gdmdltnblf_while_placeholder_3J
Fsequential_gdmdltnblf_while_less_sequential_gdmdltnblf_strided_slice_1`
\sequential_gdmdltnblf_while_sequential_gdmdltnblf_while_cond_764353___redundant_placeholder0`
\sequential_gdmdltnblf_while_sequential_gdmdltnblf_while_cond_764353___redundant_placeholder1`
\sequential_gdmdltnblf_while_sequential_gdmdltnblf_while_cond_764353___redundant_placeholder2`
\sequential_gdmdltnblf_while_sequential_gdmdltnblf_while_cond_764353___redundant_placeholder3`
\sequential_gdmdltnblf_while_sequential_gdmdltnblf_while_cond_764353___redundant_placeholder4`
\sequential_gdmdltnblf_while_sequential_gdmdltnblf_while_cond_764353___redundant_placeholder5`
\sequential_gdmdltnblf_while_sequential_gdmdltnblf_while_cond_764353___redundant_placeholder6(
$sequential_gdmdltnblf_while_identity
Þ
 sequential/gdmdltnblf/while/LessLess'sequential_gdmdltnblf_while_placeholderFsequential_gdmdltnblf_while_less_sequential_gdmdltnblf_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/gdmdltnblf/while/Less
$sequential/gdmdltnblf/while/IdentityIdentity$sequential/gdmdltnblf/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/gdmdltnblf/while/Identity"U
$sequential_gdmdltnblf_while_identity-sequential/gdmdltnblf/while/Identity:output:0*(
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


uazvpibasg_while_cond_7676852
.uazvpibasg_while_uazvpibasg_while_loop_counter8
4uazvpibasg_while_uazvpibasg_while_maximum_iterations 
uazvpibasg_while_placeholder"
uazvpibasg_while_placeholder_1"
uazvpibasg_while_placeholder_2"
uazvpibasg_while_placeholder_34
0uazvpibasg_while_less_uazvpibasg_strided_slice_1J
Fuazvpibasg_while_uazvpibasg_while_cond_767685___redundant_placeholder0J
Fuazvpibasg_while_uazvpibasg_while_cond_767685___redundant_placeholder1J
Fuazvpibasg_while_uazvpibasg_while_cond_767685___redundant_placeholder2J
Fuazvpibasg_while_uazvpibasg_while_cond_767685___redundant_placeholder3J
Fuazvpibasg_while_uazvpibasg_while_cond_767685___redundant_placeholder4J
Fuazvpibasg_while_uazvpibasg_while_cond_767685___redundant_placeholder5J
Fuazvpibasg_while_uazvpibasg_while_cond_767685___redundant_placeholder6
uazvpibasg_while_identity
§
uazvpibasg/while/LessLessuazvpibasg_while_placeholder0uazvpibasg_while_less_uazvpibasg_strided_slice_1*
T0*
_output_shapes
: 2
uazvpibasg/while/Less~
uazvpibasg/while/IdentityIdentityuazvpibasg/while/Less:z:0*
T0
*
_output_shapes
: 2
uazvpibasg/while/Identity"?
uazvpibasg_while_identity"uazvpibasg/while/Identity:output:0*(
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
ßY

while_body_769742
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_gddwjadkgr_matmul_readvariableop_resource_0:	 F
3while_gddwjadkgr_matmul_1_readvariableop_resource_0:	 A
2while_gddwjadkgr_biasadd_readvariableop_resource_0:	8
*while_gddwjadkgr_readvariableop_resource_0: :
,while_gddwjadkgr_readvariableop_1_resource_0: :
,while_gddwjadkgr_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_gddwjadkgr_matmul_readvariableop_resource:	 D
1while_gddwjadkgr_matmul_1_readvariableop_resource:	 ?
0while_gddwjadkgr_biasadd_readvariableop_resource:	6
(while_gddwjadkgr_readvariableop_resource: 8
*while_gddwjadkgr_readvariableop_1_resource: 8
*while_gddwjadkgr_readvariableop_2_resource: ¢'while/gddwjadkgr/BiasAdd/ReadVariableOp¢&while/gddwjadkgr/MatMul/ReadVariableOp¢(while/gddwjadkgr/MatMul_1/ReadVariableOp¢while/gddwjadkgr/ReadVariableOp¢!while/gddwjadkgr/ReadVariableOp_1¢!while/gddwjadkgr/ReadVariableOp_2Ã
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
&while/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp1while_gddwjadkgr_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/gddwjadkgr/MatMul/ReadVariableOpÑ
while/gddwjadkgr/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMulÉ
(while/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp3while_gddwjadkgr_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/gddwjadkgr/MatMul_1/ReadVariableOpº
while/gddwjadkgr/MatMul_1MatMulwhile_placeholder_20while/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/MatMul_1°
while/gddwjadkgr/addAddV2!while/gddwjadkgr/MatMul:product:0#while/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/addÂ
'while/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp2while_gddwjadkgr_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/gddwjadkgr/BiasAdd/ReadVariableOp½
while/gddwjadkgr/BiasAddBiasAddwhile/gddwjadkgr/add:z:0/while/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gddwjadkgr/BiasAdd
 while/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/gddwjadkgr/split/split_dim
while/gddwjadkgr/splitSplit)while/gddwjadkgr/split/split_dim:output:0!while/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gddwjadkgr/split©
while/gddwjadkgr/ReadVariableOpReadVariableOp*while_gddwjadkgr_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/gddwjadkgr/ReadVariableOp£
while/gddwjadkgr/mulMul'while/gddwjadkgr/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul¦
while/gddwjadkgr/add_1AddV2while/gddwjadkgr/split:output:0while/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_1
while/gddwjadkgr/SigmoidSigmoidwhile/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid¯
!while/gddwjadkgr/ReadVariableOp_1ReadVariableOp,while_gddwjadkgr_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_1©
while/gddwjadkgr/mul_1Mul)while/gddwjadkgr/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_1¨
while/gddwjadkgr/add_2AddV2while/gddwjadkgr/split:output:1while/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_2
while/gddwjadkgr/Sigmoid_1Sigmoidwhile/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_1
while/gddwjadkgr/mul_2Mulwhile/gddwjadkgr/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_2
while/gddwjadkgr/TanhTanhwhile/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh¢
while/gddwjadkgr/mul_3Mulwhile/gddwjadkgr/Sigmoid:y:0while/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_3£
while/gddwjadkgr/add_3AddV2while/gddwjadkgr/mul_2:z:0while/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_3¯
!while/gddwjadkgr/ReadVariableOp_2ReadVariableOp,while_gddwjadkgr_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/gddwjadkgr/ReadVariableOp_2°
while/gddwjadkgr/mul_4Mul)while/gddwjadkgr/ReadVariableOp_2:value:0while/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_4¨
while/gddwjadkgr/add_4AddV2while/gddwjadkgr/split:output:3while/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/add_4
while/gddwjadkgr/Sigmoid_2Sigmoidwhile/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Sigmoid_2
while/gddwjadkgr/Tanh_1Tanhwhile/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/Tanh_1¦
while/gddwjadkgr/mul_5Mulwhile/gddwjadkgr/Sigmoid_2:y:0while/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gddwjadkgr/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gddwjadkgr/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/gddwjadkgr/mul_5:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/gddwjadkgr/add_3:z:0(^while/gddwjadkgr/BiasAdd/ReadVariableOp'^while/gddwjadkgr/MatMul/ReadVariableOp)^while/gddwjadkgr/MatMul_1/ReadVariableOp ^while/gddwjadkgr/ReadVariableOp"^while/gddwjadkgr/ReadVariableOp_1"^while/gddwjadkgr/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_gddwjadkgr_biasadd_readvariableop_resource2while_gddwjadkgr_biasadd_readvariableop_resource_0"h
1while_gddwjadkgr_matmul_1_readvariableop_resource3while_gddwjadkgr_matmul_1_readvariableop_resource_0"d
/while_gddwjadkgr_matmul_readvariableop_resource1while_gddwjadkgr_matmul_readvariableop_resource_0"Z
*while_gddwjadkgr_readvariableop_1_resource,while_gddwjadkgr_readvariableop_1_resource_0"Z
*while_gddwjadkgr_readvariableop_2_resource,while_gddwjadkgr_readvariableop_2_resource_0"V
(while_gddwjadkgr_readvariableop_resource*while_gddwjadkgr_readvariableop_resource_0")
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
'while/gddwjadkgr/BiasAdd/ReadVariableOp'while/gddwjadkgr/BiasAdd/ReadVariableOp2P
&while/gddwjadkgr/MatMul/ReadVariableOp&while/gddwjadkgr/MatMul/ReadVariableOp2T
(while/gddwjadkgr/MatMul_1/ReadVariableOp(while/gddwjadkgr/MatMul_1/ReadVariableOp2B
while/gddwjadkgr/ReadVariableOpwhile/gddwjadkgr/ReadVariableOp2F
!while/gddwjadkgr/ReadVariableOp_1!while/gddwjadkgr/ReadVariableOp_12F
!while/gddwjadkgr/ReadVariableOp_2!while/gddwjadkgr/ReadVariableOp_2: 
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


gdmdltnblf_while_cond_7679132
.gdmdltnblf_while_gdmdltnblf_while_loop_counter8
4gdmdltnblf_while_gdmdltnblf_while_maximum_iterations 
gdmdltnblf_while_placeholder"
gdmdltnblf_while_placeholder_1"
gdmdltnblf_while_placeholder_2"
gdmdltnblf_while_placeholder_34
0gdmdltnblf_while_less_gdmdltnblf_strided_slice_1J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767913___redundant_placeholder0J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767913___redundant_placeholder1J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767913___redundant_placeholder2J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767913___redundant_placeholder3J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767913___redundant_placeholder4J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767913___redundant_placeholder5J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767913___redundant_placeholder6
gdmdltnblf_while_identity
§
gdmdltnblf/while/LessLessgdmdltnblf_while_placeholder0gdmdltnblf_while_less_gdmdltnblf_strided_slice_1*
T0*
_output_shapes
: 2
gdmdltnblf/while/Less~
gdmdltnblf/while/IdentityIdentitygdmdltnblf/while/Less:z:0*
T0
*
_output_shapes
: 2
gdmdltnblf/while/Identity"?
gdmdltnblf_while_identity"gdmdltnblf/while/Identity:output:0*(
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
ßY

while_body_766978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_duhsngmesj_matmul_readvariableop_resource_0:	F
3while_duhsngmesj_matmul_1_readvariableop_resource_0:	 A
2while_duhsngmesj_biasadd_readvariableop_resource_0:	8
*while_duhsngmesj_readvariableop_resource_0: :
,while_duhsngmesj_readvariableop_1_resource_0: :
,while_duhsngmesj_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_duhsngmesj_matmul_readvariableop_resource:	D
1while_duhsngmesj_matmul_1_readvariableop_resource:	 ?
0while_duhsngmesj_biasadd_readvariableop_resource:	6
(while_duhsngmesj_readvariableop_resource: 8
*while_duhsngmesj_readvariableop_1_resource: 8
*while_duhsngmesj_readvariableop_2_resource: ¢'while/duhsngmesj/BiasAdd/ReadVariableOp¢&while/duhsngmesj/MatMul/ReadVariableOp¢(while/duhsngmesj/MatMul_1/ReadVariableOp¢while/duhsngmesj/ReadVariableOp¢!while/duhsngmesj/ReadVariableOp_1¢!while/duhsngmesj/ReadVariableOp_2Ã
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
&while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp1while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/duhsngmesj/MatMul/ReadVariableOpÑ
while/duhsngmesj/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMulÉ
(while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp3while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/duhsngmesj/MatMul_1/ReadVariableOpº
while/duhsngmesj/MatMul_1MatMulwhile_placeholder_20while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMul_1°
while/duhsngmesj/addAddV2!while/duhsngmesj/MatMul:product:0#while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/addÂ
'while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp2while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/duhsngmesj/BiasAdd/ReadVariableOp½
while/duhsngmesj/BiasAddBiasAddwhile/duhsngmesj/add:z:0/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/BiasAdd
 while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/duhsngmesj/split/split_dim
while/duhsngmesj/splitSplit)while/duhsngmesj/split/split_dim:output:0!while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/duhsngmesj/split©
while/duhsngmesj/ReadVariableOpReadVariableOp*while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/duhsngmesj/ReadVariableOp£
while/duhsngmesj/mulMul'while/duhsngmesj/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul¦
while/duhsngmesj/add_1AddV2while/duhsngmesj/split:output:0while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_1
while/duhsngmesj/SigmoidSigmoidwhile/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid¯
!while/duhsngmesj/ReadVariableOp_1ReadVariableOp,while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_1©
while/duhsngmesj/mul_1Mul)while/duhsngmesj/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_1¨
while/duhsngmesj/add_2AddV2while/duhsngmesj/split:output:1while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_2
while/duhsngmesj/Sigmoid_1Sigmoidwhile/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_1
while/duhsngmesj/mul_2Mulwhile/duhsngmesj/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_2
while/duhsngmesj/TanhTanhwhile/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh¢
while/duhsngmesj/mul_3Mulwhile/duhsngmesj/Sigmoid:y:0while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_3£
while/duhsngmesj/add_3AddV2while/duhsngmesj/mul_2:z:0while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_3¯
!while/duhsngmesj/ReadVariableOp_2ReadVariableOp,while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_2°
while/duhsngmesj/mul_4Mul)while/duhsngmesj/ReadVariableOp_2:value:0while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_4¨
while/duhsngmesj/add_4AddV2while/duhsngmesj/split:output:3while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_4
while/duhsngmesj/Sigmoid_2Sigmoidwhile/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_2
while/duhsngmesj/Tanh_1Tanhwhile/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh_1¦
while/duhsngmesj/mul_5Mulwhile/duhsngmesj/Sigmoid_2:y:0while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/duhsngmesj/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/duhsngmesj/mul_5:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/duhsngmesj/add_3:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_duhsngmesj_biasadd_readvariableop_resource2while_duhsngmesj_biasadd_readvariableop_resource_0"h
1while_duhsngmesj_matmul_1_readvariableop_resource3while_duhsngmesj_matmul_1_readvariableop_resource_0"d
/while_duhsngmesj_matmul_readvariableop_resource1while_duhsngmesj_matmul_readvariableop_resource_0"Z
*while_duhsngmesj_readvariableop_1_resource,while_duhsngmesj_readvariableop_1_resource_0"Z
*while_duhsngmesj_readvariableop_2_resource,while_duhsngmesj_readvariableop_2_resource_0"V
(while_duhsngmesj_readvariableop_resource*while_duhsngmesj_readvariableop_resource_0")
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
'while/duhsngmesj/BiasAdd/ReadVariableOp'while/duhsngmesj/BiasAdd/ReadVariableOp2P
&while/duhsngmesj/MatMul/ReadVariableOp&while/duhsngmesj/MatMul/ReadVariableOp2T
(while/duhsngmesj/MatMul_1/ReadVariableOp(while/duhsngmesj/MatMul_1/ReadVariableOp2B
while/duhsngmesj/ReadVariableOpwhile/duhsngmesj/ReadVariableOp2F
!while/duhsngmesj/ReadVariableOp_1!while/duhsngmesj/ReadVariableOp_12F
!while/duhsngmesj/ReadVariableOp_2!while/duhsngmesj/ReadVariableOp_2: 
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


+__inference_sequential_layer_call_fn_768234

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
identity¢StatefulPartitionedCall´
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_7666212
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


gdmdltnblf_while_cond_7675092
.gdmdltnblf_while_gdmdltnblf_while_loop_counter8
4gdmdltnblf_while_gdmdltnblf_while_maximum_iterations 
gdmdltnblf_while_placeholder"
gdmdltnblf_while_placeholder_1"
gdmdltnblf_while_placeholder_2"
gdmdltnblf_while_placeholder_34
0gdmdltnblf_while_less_gdmdltnblf_strided_slice_1J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767509___redundant_placeholder0J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767509___redundant_placeholder1J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767509___redundant_placeholder2J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767509___redundant_placeholder3J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767509___redundant_placeholder4J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767509___redundant_placeholder5J
Fgdmdltnblf_while_gdmdltnblf_while_cond_767509___redundant_placeholder6
gdmdltnblf_while_identity
§
gdmdltnblf/while/LessLessgdmdltnblf_while_placeholder0gdmdltnblf_while_less_gdmdltnblf_strided_slice_1*
T0*
_output_shapes
: 2
gdmdltnblf/while/Less~
gdmdltnblf/while/IdentityIdentitygdmdltnblf/while/Less:z:0*
T0
*
_output_shapes
: 2
gdmdltnblf/while/Identity"?
gdmdltnblf_while_identity"gdmdltnblf/while/Identity:output:0*(
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


uazvpibasg_while_cond_7680892
.uazvpibasg_while_uazvpibasg_while_loop_counter8
4uazvpibasg_while_uazvpibasg_while_maximum_iterations 
uazvpibasg_while_placeholder"
uazvpibasg_while_placeholder_1"
uazvpibasg_while_placeholder_2"
uazvpibasg_while_placeholder_34
0uazvpibasg_while_less_uazvpibasg_strided_slice_1J
Fuazvpibasg_while_uazvpibasg_while_cond_768089___redundant_placeholder0J
Fuazvpibasg_while_uazvpibasg_while_cond_768089___redundant_placeholder1J
Fuazvpibasg_while_uazvpibasg_while_cond_768089___redundant_placeholder2J
Fuazvpibasg_while_uazvpibasg_while_cond_768089___redundant_placeholder3J
Fuazvpibasg_while_uazvpibasg_while_cond_768089___redundant_placeholder4J
Fuazvpibasg_while_uazvpibasg_while_cond_768089___redundant_placeholder5J
Fuazvpibasg_while_uazvpibasg_while_cond_768089___redundant_placeholder6
uazvpibasg_while_identity
§
uazvpibasg/while/LessLessuazvpibasg_while_placeholder0uazvpibasg_while_less_uazvpibasg_strided_slice_1*
T0*
_output_shapes
: 2
uazvpibasg/while/Less~
uazvpibasg/while/IdentityIdentityuazvpibasg/while/Less:z:0*
T0
*
_output_shapes
: 2
uazvpibasg/while/Identity"?
uazvpibasg_while_identity"uazvpibasg/while/Identity:output:0*(
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

«
F__inference_sequential_layer_call_and_return_conditional_losses_767190

inputs'
ggbvrcpxtu_767152:
ggbvrcpxtu_767154:$
gdmdltnblf_767158:	$
gdmdltnblf_767160:	  
gdmdltnblf_767162:	
gdmdltnblf_767164: 
gdmdltnblf_767166: 
gdmdltnblf_767168: $
uazvpibasg_767171:	 $
uazvpibasg_767173:	  
uazvpibasg_767175:	
uazvpibasg_767177: 
uazvpibasg_767179: 
uazvpibasg_767181: #
supobtndkp_767184: 
supobtndkp_767186:
identity¢"gdmdltnblf/StatefulPartitionedCall¢"ggbvrcpxtu/StatefulPartitionedCall¢"supobtndkp/StatefulPartitionedCall¢"uazvpibasg/StatefulPartitionedCall©
"ggbvrcpxtu/StatefulPartitionedCallStatefulPartitionedCallinputsggbvrcpxtu_767152ggbvrcpxtu_767154*
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
GPU2*0J 8 *O
fJRH
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_7661972$
"ggbvrcpxtu/StatefulPartitionedCall
vtqejmjhbd/PartitionedCallPartitionedCall+ggbvrcpxtu/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_7662162
vtqejmjhbd/PartitionedCall
"gdmdltnblf/StatefulPartitionedCallStatefulPartitionedCall#vtqejmjhbd/PartitionedCall:output:0gdmdltnblf_767158gdmdltnblf_767160gdmdltnblf_767162gdmdltnblf_767164gdmdltnblf_767166gdmdltnblf_767168*
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7670792$
"gdmdltnblf/StatefulPartitionedCall
"uazvpibasg/StatefulPartitionedCallStatefulPartitionedCall+gdmdltnblf/StatefulPartitionedCall:output:0uazvpibasg_767171uazvpibasg_767173uazvpibasg_767175uazvpibasg_767177uazvpibasg_767179uazvpibasg_767181*
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7668652$
"uazvpibasg/StatefulPartitionedCallÆ
"supobtndkp/StatefulPartitionedCallStatefulPartitionedCall+uazvpibasg/StatefulPartitionedCall:output:0supobtndkp_767184supobtndkp_767186*
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
GPU2*0J 8 *O
fJRH
F__inference_supobtndkp_layer_call_and_return_conditional_losses_7666142$
"supobtndkp/StatefulPartitionedCall
IdentityIdentity+supobtndkp/StatefulPartitionedCall:output:0#^gdmdltnblf/StatefulPartitionedCall#^ggbvrcpxtu/StatefulPartitionedCall#^supobtndkp/StatefulPartitionedCall#^uazvpibasg/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"gdmdltnblf/StatefulPartitionedCall"gdmdltnblf/StatefulPartitionedCall2H
"ggbvrcpxtu/StatefulPartitionedCall"ggbvrcpxtu/StatefulPartitionedCall2H
"supobtndkp/StatefulPartitionedCall"supobtndkp/StatefulPartitionedCall2H
"uazvpibasg/StatefulPartitionedCall"uazvpibasg/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

+__inference_supobtndkp_layer_call_fn_769930

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallù
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
GPU2*0J 8 *O
fJRH
F__inference_supobtndkp_layer_call_and_return_conditional_losses_7666142
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
ÙÊ

F__inference_sequential_layer_call_and_return_conditional_losses_767793

inputsL
6ggbvrcpxtu_conv1d_expanddims_1_readvariableop_resource:K
=ggbvrcpxtu_squeeze_batch_dims_biasadd_readvariableop_resource:G
4gdmdltnblf_duhsngmesj_matmul_readvariableop_resource:	I
6gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource:	 D
5gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource:	;
-gdmdltnblf_duhsngmesj_readvariableop_resource: =
/gdmdltnblf_duhsngmesj_readvariableop_1_resource: =
/gdmdltnblf_duhsngmesj_readvariableop_2_resource: G
4uazvpibasg_gddwjadkgr_matmul_readvariableop_resource:	 I
6uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource:	 D
5uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource:	;
-uazvpibasg_gddwjadkgr_readvariableop_resource: =
/uazvpibasg_gddwjadkgr_readvariableop_1_resource: =
/uazvpibasg_gddwjadkgr_readvariableop_2_resource: ;
)supobtndkp_matmul_readvariableop_resource: 8
*supobtndkp_biasadd_readvariableop_resource:
identity¢,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp¢+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp¢-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp¢$gdmdltnblf/duhsngmesj/ReadVariableOp¢&gdmdltnblf/duhsngmesj/ReadVariableOp_1¢&gdmdltnblf/duhsngmesj/ReadVariableOp_2¢gdmdltnblf/while¢-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp¢4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!supobtndkp/BiasAdd/ReadVariableOp¢ supobtndkp/MatMul/ReadVariableOp¢,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp¢+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp¢-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp¢$uazvpibasg/gddwjadkgr/ReadVariableOp¢&uazvpibasg/gddwjadkgr/ReadVariableOp_1¢&uazvpibasg/gddwjadkgr/ReadVariableOp_2¢uazvpibasg/while
 ggbvrcpxtu/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 ggbvrcpxtu/conv1d/ExpandDims/dim»
ggbvrcpxtu/conv1d/ExpandDims
ExpandDimsinputs)ggbvrcpxtu/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/ExpandDimsÙ
-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6ggbvrcpxtu_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp
"ggbvrcpxtu/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ggbvrcpxtu/conv1d/ExpandDims_1/dimã
ggbvrcpxtu/conv1d/ExpandDims_1
ExpandDims5ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp:value:0+ggbvrcpxtu/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
ggbvrcpxtu/conv1d/ExpandDims_1
ggbvrcpxtu/conv1d/ShapeShape%ggbvrcpxtu/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
ggbvrcpxtu/conv1d/Shape
%ggbvrcpxtu/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%ggbvrcpxtu/conv1d/strided_slice/stack¥
'ggbvrcpxtu/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'ggbvrcpxtu/conv1d/strided_slice/stack_1
'ggbvrcpxtu/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'ggbvrcpxtu/conv1d/strided_slice/stack_2Ì
ggbvrcpxtu/conv1d/strided_sliceStridedSlice ggbvrcpxtu/conv1d/Shape:output:0.ggbvrcpxtu/conv1d/strided_slice/stack:output:00ggbvrcpxtu/conv1d/strided_slice/stack_1:output:00ggbvrcpxtu/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
ggbvrcpxtu/conv1d/strided_slice
ggbvrcpxtu/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
ggbvrcpxtu/conv1d/Reshape/shapeÌ
ggbvrcpxtu/conv1d/ReshapeReshape%ggbvrcpxtu/conv1d/ExpandDims:output:0(ggbvrcpxtu/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/Reshapeî
ggbvrcpxtu/conv1d/Conv2DConv2D"ggbvrcpxtu/conv1d/Reshape:output:0'ggbvrcpxtu/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ggbvrcpxtu/conv1d/Conv2D
!ggbvrcpxtu/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!ggbvrcpxtu/conv1d/concat/values_1
ggbvrcpxtu/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/concat/axisì
ggbvrcpxtu/conv1d/concatConcatV2(ggbvrcpxtu/conv1d/strided_slice:output:0*ggbvrcpxtu/conv1d/concat/values_1:output:0&ggbvrcpxtu/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ggbvrcpxtu/conv1d/concatÉ
ggbvrcpxtu/conv1d/Reshape_1Reshape!ggbvrcpxtu/conv1d/Conv2D:output:0!ggbvrcpxtu/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/Reshape_1Á
ggbvrcpxtu/conv1d/SqueezeSqueeze$ggbvrcpxtu/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
ggbvrcpxtu/conv1d/Squeeze
#ggbvrcpxtu/squeeze_batch_dims/ShapeShape"ggbvrcpxtu/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#ggbvrcpxtu/squeeze_batch_dims/Shape°
1ggbvrcpxtu/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack½
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1´
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2
+ggbvrcpxtu/squeeze_batch_dims/strided_sliceStridedSlice,ggbvrcpxtu/squeeze_batch_dims/Shape:output:0:ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack:output:0<ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1:output:0<ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+ggbvrcpxtu/squeeze_batch_dims/strided_slice¯
+ggbvrcpxtu/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+ggbvrcpxtu/squeeze_batch_dims/Reshape/shapeé
%ggbvrcpxtu/squeeze_batch_dims/ReshapeReshape"ggbvrcpxtu/conv1d/Squeeze:output:04ggbvrcpxtu/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ggbvrcpxtu/squeeze_batch_dims/Reshapeæ
4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=ggbvrcpxtu_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%ggbvrcpxtu/squeeze_batch_dims/BiasAddBiasAdd.ggbvrcpxtu/squeeze_batch_dims/Reshape:output:0<ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ggbvrcpxtu/squeeze_batch_dims/BiasAdd¯
-ggbvrcpxtu/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-ggbvrcpxtu/squeeze_batch_dims/concat/values_1¡
)ggbvrcpxtu/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)ggbvrcpxtu/squeeze_batch_dims/concat/axis¨
$ggbvrcpxtu/squeeze_batch_dims/concatConcatV24ggbvrcpxtu/squeeze_batch_dims/strided_slice:output:06ggbvrcpxtu/squeeze_batch_dims/concat/values_1:output:02ggbvrcpxtu/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$ggbvrcpxtu/squeeze_batch_dims/concatö
'ggbvrcpxtu/squeeze_batch_dims/Reshape_1Reshape.ggbvrcpxtu/squeeze_batch_dims/BiasAdd:output:0-ggbvrcpxtu/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'ggbvrcpxtu/squeeze_batch_dims/Reshape_1
vtqejmjhbd/ShapeShape0ggbvrcpxtu/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
vtqejmjhbd/Shape
vtqejmjhbd/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
vtqejmjhbd/strided_slice/stack
 vtqejmjhbd/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 vtqejmjhbd/strided_slice/stack_1
 vtqejmjhbd/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 vtqejmjhbd/strided_slice/stack_2¤
vtqejmjhbd/strided_sliceStridedSlicevtqejmjhbd/Shape:output:0'vtqejmjhbd/strided_slice/stack:output:0)vtqejmjhbd/strided_slice/stack_1:output:0)vtqejmjhbd/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vtqejmjhbd/strided_slicez
vtqejmjhbd/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
vtqejmjhbd/Reshape/shape/1z
vtqejmjhbd/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
vtqejmjhbd/Reshape/shape/2×
vtqejmjhbd/Reshape/shapePack!vtqejmjhbd/strided_slice:output:0#vtqejmjhbd/Reshape/shape/1:output:0#vtqejmjhbd/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
vtqejmjhbd/Reshape/shape¾
vtqejmjhbd/ReshapeReshape0ggbvrcpxtu/squeeze_batch_dims/Reshape_1:output:0!vtqejmjhbd/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vtqejmjhbd/Reshapeo
gdmdltnblf/ShapeShapevtqejmjhbd/Reshape:output:0*
T0*
_output_shapes
:2
gdmdltnblf/Shape
gdmdltnblf/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gdmdltnblf/strided_slice/stack
 gdmdltnblf/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gdmdltnblf/strided_slice/stack_1
 gdmdltnblf/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gdmdltnblf/strided_slice/stack_2¤
gdmdltnblf/strided_sliceStridedSlicegdmdltnblf/Shape:output:0'gdmdltnblf/strided_slice/stack:output:0)gdmdltnblf/strided_slice/stack_1:output:0)gdmdltnblf/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gdmdltnblf/strided_slicer
gdmdltnblf/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros/mul/y
gdmdltnblf/zeros/mulMul!gdmdltnblf/strided_slice:output:0gdmdltnblf/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros/mulu
gdmdltnblf/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gdmdltnblf/zeros/Less/y
gdmdltnblf/zeros/LessLessgdmdltnblf/zeros/mul:z:0 gdmdltnblf/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros/Lessx
gdmdltnblf/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros/packed/1¯
gdmdltnblf/zeros/packedPack!gdmdltnblf/strided_slice:output:0"gdmdltnblf/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gdmdltnblf/zeros/packedu
gdmdltnblf/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gdmdltnblf/zeros/Const¡
gdmdltnblf/zerosFill gdmdltnblf/zeros/packed:output:0gdmdltnblf/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/zerosv
gdmdltnblf/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros_1/mul/y
gdmdltnblf/zeros_1/mulMul!gdmdltnblf/strided_slice:output:0!gdmdltnblf/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros_1/muly
gdmdltnblf/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gdmdltnblf/zeros_1/Less/y
gdmdltnblf/zeros_1/LessLessgdmdltnblf/zeros_1/mul:z:0"gdmdltnblf/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
gdmdltnblf/zeros_1/Less|
gdmdltnblf/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/zeros_1/packed/1µ
gdmdltnblf/zeros_1/packedPack!gdmdltnblf/strided_slice:output:0$gdmdltnblf/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
gdmdltnblf/zeros_1/packedy
gdmdltnblf/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gdmdltnblf/zeros_1/Const©
gdmdltnblf/zeros_1Fill"gdmdltnblf/zeros_1/packed:output:0!gdmdltnblf/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/zeros_1
gdmdltnblf/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gdmdltnblf/transpose/perm°
gdmdltnblf/transpose	Transposevtqejmjhbd/Reshape:output:0"gdmdltnblf/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/transposep
gdmdltnblf/Shape_1Shapegdmdltnblf/transpose:y:0*
T0*
_output_shapes
:2
gdmdltnblf/Shape_1
 gdmdltnblf/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gdmdltnblf/strided_slice_1/stack
"gdmdltnblf/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_1/stack_1
"gdmdltnblf/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_1/stack_2°
gdmdltnblf/strided_slice_1StridedSlicegdmdltnblf/Shape_1:output:0)gdmdltnblf/strided_slice_1/stack:output:0+gdmdltnblf/strided_slice_1/stack_1:output:0+gdmdltnblf/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gdmdltnblf/strided_slice_1
&gdmdltnblf/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&gdmdltnblf/TensorArrayV2/element_shapeÞ
gdmdltnblf/TensorArrayV2TensorListReserve/gdmdltnblf/TensorArrayV2/element_shape:output:0#gdmdltnblf/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gdmdltnblf/TensorArrayV2Õ
@gdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@gdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2gdmdltnblf/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgdmdltnblf/transpose:y:0Igdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2gdmdltnblf/TensorArrayUnstack/TensorListFromTensor
 gdmdltnblf/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gdmdltnblf/strided_slice_2/stack
"gdmdltnblf/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_2/stack_1
"gdmdltnblf/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_2/stack_2¾
gdmdltnblf/strided_slice_2StridedSlicegdmdltnblf/transpose:y:0)gdmdltnblf/strided_slice_2/stack:output:0+gdmdltnblf/strided_slice_2/stack_1:output:0+gdmdltnblf/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
gdmdltnblf/strided_slice_2Ð
+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOpReadVariableOp4gdmdltnblf_duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOpÓ
gdmdltnblf/duhsngmesj/MatMulMatMul#gdmdltnblf/strided_slice_2:output:03gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/duhsngmesj/MatMulÖ
-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp6gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOpÏ
gdmdltnblf/duhsngmesj/MatMul_1MatMulgdmdltnblf/zeros:output:05gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
gdmdltnblf/duhsngmesj/MatMul_1Ä
gdmdltnblf/duhsngmesj/addAddV2&gdmdltnblf/duhsngmesj/MatMul:product:0(gdmdltnblf/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/duhsngmesj/addÏ
,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp5gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOpÑ
gdmdltnblf/duhsngmesj/BiasAddBiasAddgdmdltnblf/duhsngmesj/add:z:04gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gdmdltnblf/duhsngmesj/BiasAdd
%gdmdltnblf/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%gdmdltnblf/duhsngmesj/split/split_dim
gdmdltnblf/duhsngmesj/splitSplit.gdmdltnblf/duhsngmesj/split/split_dim:output:0&gdmdltnblf/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gdmdltnblf/duhsngmesj/split¶
$gdmdltnblf/duhsngmesj/ReadVariableOpReadVariableOp-gdmdltnblf_duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02&
$gdmdltnblf/duhsngmesj/ReadVariableOpº
gdmdltnblf/duhsngmesj/mulMul,gdmdltnblf/duhsngmesj/ReadVariableOp:value:0gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mulº
gdmdltnblf/duhsngmesj/add_1AddV2$gdmdltnblf/duhsngmesj/split:output:0gdmdltnblf/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_1
gdmdltnblf/duhsngmesj/SigmoidSigmoidgdmdltnblf/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/Sigmoid¼
&gdmdltnblf/duhsngmesj/ReadVariableOp_1ReadVariableOp/gdmdltnblf_duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&gdmdltnblf/duhsngmesj/ReadVariableOp_1À
gdmdltnblf/duhsngmesj/mul_1Mul.gdmdltnblf/duhsngmesj/ReadVariableOp_1:value:0gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_1¼
gdmdltnblf/duhsngmesj/add_2AddV2$gdmdltnblf/duhsngmesj/split:output:1gdmdltnblf/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_2 
gdmdltnblf/duhsngmesj/Sigmoid_1Sigmoidgdmdltnblf/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gdmdltnblf/duhsngmesj/Sigmoid_1µ
gdmdltnblf/duhsngmesj/mul_2Mul#gdmdltnblf/duhsngmesj/Sigmoid_1:y:0gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_2
gdmdltnblf/duhsngmesj/TanhTanh$gdmdltnblf/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/Tanh¶
gdmdltnblf/duhsngmesj/mul_3Mul!gdmdltnblf/duhsngmesj/Sigmoid:y:0gdmdltnblf/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_3·
gdmdltnblf/duhsngmesj/add_3AddV2gdmdltnblf/duhsngmesj/mul_2:z:0gdmdltnblf/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_3¼
&gdmdltnblf/duhsngmesj/ReadVariableOp_2ReadVariableOp/gdmdltnblf_duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&gdmdltnblf/duhsngmesj/ReadVariableOp_2Ä
gdmdltnblf/duhsngmesj/mul_4Mul.gdmdltnblf/duhsngmesj/ReadVariableOp_2:value:0gdmdltnblf/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_4¼
gdmdltnblf/duhsngmesj/add_4AddV2$gdmdltnblf/duhsngmesj/split:output:3gdmdltnblf/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/add_4 
gdmdltnblf/duhsngmesj/Sigmoid_2Sigmoidgdmdltnblf/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
gdmdltnblf/duhsngmesj/Sigmoid_2
gdmdltnblf/duhsngmesj/Tanh_1Tanhgdmdltnblf/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/Tanh_1º
gdmdltnblf/duhsngmesj/mul_5Mul#gdmdltnblf/duhsngmesj/Sigmoid_2:y:0 gdmdltnblf/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/duhsngmesj/mul_5¥
(gdmdltnblf/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(gdmdltnblf/TensorArrayV2_1/element_shapeä
gdmdltnblf/TensorArrayV2_1TensorListReserve1gdmdltnblf/TensorArrayV2_1/element_shape:output:0#gdmdltnblf/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gdmdltnblf/TensorArrayV2_1d
gdmdltnblf/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/time
#gdmdltnblf/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#gdmdltnblf/while/maximum_iterations
gdmdltnblf/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gdmdltnblf/while/loop_counter°
gdmdltnblf/whileWhile&gdmdltnblf/while/loop_counter:output:0,gdmdltnblf/while/maximum_iterations:output:0gdmdltnblf/time:output:0#gdmdltnblf/TensorArrayV2_1:handle:0gdmdltnblf/zeros:output:0gdmdltnblf/zeros_1:output:0#gdmdltnblf/strided_slice_1:output:0Bgdmdltnblf/TensorArrayUnstack/TensorListFromTensor:output_handle:04gdmdltnblf_duhsngmesj_matmul_readvariableop_resource6gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource5gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource-gdmdltnblf_duhsngmesj_readvariableop_resource/gdmdltnblf_duhsngmesj_readvariableop_1_resource/gdmdltnblf_duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*(
body R
gdmdltnblf_while_body_767510*(
cond R
gdmdltnblf_while_cond_767509*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
gdmdltnblf/whileË
;gdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;gdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shape
-gdmdltnblf/TensorArrayV2Stack/TensorListStackTensorListStackgdmdltnblf/while:output:3Dgdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-gdmdltnblf/TensorArrayV2Stack/TensorListStack
 gdmdltnblf/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 gdmdltnblf/strided_slice_3/stack
"gdmdltnblf/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"gdmdltnblf/strided_slice_3/stack_1
"gdmdltnblf/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gdmdltnblf/strided_slice_3/stack_2Ü
gdmdltnblf/strided_slice_3StridedSlice6gdmdltnblf/TensorArrayV2Stack/TensorListStack:tensor:0)gdmdltnblf/strided_slice_3/stack:output:0+gdmdltnblf/strided_slice_3/stack_1:output:0+gdmdltnblf/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gdmdltnblf/strided_slice_3
gdmdltnblf/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gdmdltnblf/transpose_1/permÑ
gdmdltnblf/transpose_1	Transpose6gdmdltnblf/TensorArrayV2Stack/TensorListStack:tensor:0$gdmdltnblf/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gdmdltnblf/transpose_1n
uazvpibasg/ShapeShapegdmdltnblf/transpose_1:y:0*
T0*
_output_shapes
:2
uazvpibasg/Shape
uazvpibasg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
uazvpibasg/strided_slice/stack
 uazvpibasg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 uazvpibasg/strided_slice/stack_1
 uazvpibasg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 uazvpibasg/strided_slice/stack_2¤
uazvpibasg/strided_sliceStridedSliceuazvpibasg/Shape:output:0'uazvpibasg/strided_slice/stack:output:0)uazvpibasg/strided_slice/stack_1:output:0)uazvpibasg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
uazvpibasg/strided_slicer
uazvpibasg/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros/mul/y
uazvpibasg/zeros/mulMul!uazvpibasg/strided_slice:output:0uazvpibasg/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros/mulu
uazvpibasg/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
uazvpibasg/zeros/Less/y
uazvpibasg/zeros/LessLessuazvpibasg/zeros/mul:z:0 uazvpibasg/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros/Lessx
uazvpibasg/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros/packed/1¯
uazvpibasg/zeros/packedPack!uazvpibasg/strided_slice:output:0"uazvpibasg/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
uazvpibasg/zeros/packedu
uazvpibasg/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
uazvpibasg/zeros/Const¡
uazvpibasg/zerosFill uazvpibasg/zeros/packed:output:0uazvpibasg/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/zerosv
uazvpibasg/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros_1/mul/y
uazvpibasg/zeros_1/mulMul!uazvpibasg/strided_slice:output:0!uazvpibasg/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros_1/muly
uazvpibasg/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
uazvpibasg/zeros_1/Less/y
uazvpibasg/zeros_1/LessLessuazvpibasg/zeros_1/mul:z:0"uazvpibasg/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
uazvpibasg/zeros_1/Less|
uazvpibasg/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/zeros_1/packed/1µ
uazvpibasg/zeros_1/packedPack!uazvpibasg/strided_slice:output:0$uazvpibasg/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
uazvpibasg/zeros_1/packedy
uazvpibasg/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
uazvpibasg/zeros_1/Const©
uazvpibasg/zeros_1Fill"uazvpibasg/zeros_1/packed:output:0!uazvpibasg/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/zeros_1
uazvpibasg/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
uazvpibasg/transpose/perm¯
uazvpibasg/transpose	Transposegdmdltnblf/transpose_1:y:0"uazvpibasg/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/transposep
uazvpibasg/Shape_1Shapeuazvpibasg/transpose:y:0*
T0*
_output_shapes
:2
uazvpibasg/Shape_1
 uazvpibasg/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 uazvpibasg/strided_slice_1/stack
"uazvpibasg/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_1/stack_1
"uazvpibasg/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_1/stack_2°
uazvpibasg/strided_slice_1StridedSliceuazvpibasg/Shape_1:output:0)uazvpibasg/strided_slice_1/stack:output:0+uazvpibasg/strided_slice_1/stack_1:output:0+uazvpibasg/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
uazvpibasg/strided_slice_1
&uazvpibasg/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&uazvpibasg/TensorArrayV2/element_shapeÞ
uazvpibasg/TensorArrayV2TensorListReserve/uazvpibasg/TensorArrayV2/element_shape:output:0#uazvpibasg/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
uazvpibasg/TensorArrayV2Õ
@uazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@uazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2uazvpibasg/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoruazvpibasg/transpose:y:0Iuazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2uazvpibasg/TensorArrayUnstack/TensorListFromTensor
 uazvpibasg/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 uazvpibasg/strided_slice_2/stack
"uazvpibasg/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_2/stack_1
"uazvpibasg/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_2/stack_2¾
uazvpibasg/strided_slice_2StridedSliceuazvpibasg/transpose:y:0)uazvpibasg/strided_slice_2/stack:output:0+uazvpibasg/strided_slice_2/stack_1:output:0+uazvpibasg/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
uazvpibasg/strided_slice_2Ð
+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp4uazvpibasg_gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOpÓ
uazvpibasg/gddwjadkgr/MatMulMatMul#uazvpibasg/strided_slice_2:output:03uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uazvpibasg/gddwjadkgr/MatMulÖ
-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp6uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOpÏ
uazvpibasg/gddwjadkgr/MatMul_1MatMuluazvpibasg/zeros:output:05uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
uazvpibasg/gddwjadkgr/MatMul_1Ä
uazvpibasg/gddwjadkgr/addAddV2&uazvpibasg/gddwjadkgr/MatMul:product:0(uazvpibasg/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uazvpibasg/gddwjadkgr/addÏ
,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp5uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOpÑ
uazvpibasg/gddwjadkgr/BiasAddBiasAdduazvpibasg/gddwjadkgr/add:z:04uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
uazvpibasg/gddwjadkgr/BiasAdd
%uazvpibasg/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%uazvpibasg/gddwjadkgr/split/split_dim
uazvpibasg/gddwjadkgr/splitSplit.uazvpibasg/gddwjadkgr/split/split_dim:output:0&uazvpibasg/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
uazvpibasg/gddwjadkgr/split¶
$uazvpibasg/gddwjadkgr/ReadVariableOpReadVariableOp-uazvpibasg_gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02&
$uazvpibasg/gddwjadkgr/ReadVariableOpº
uazvpibasg/gddwjadkgr/mulMul,uazvpibasg/gddwjadkgr/ReadVariableOp:value:0uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mulº
uazvpibasg/gddwjadkgr/add_1AddV2$uazvpibasg/gddwjadkgr/split:output:0uazvpibasg/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_1
uazvpibasg/gddwjadkgr/SigmoidSigmoiduazvpibasg/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/Sigmoid¼
&uazvpibasg/gddwjadkgr/ReadVariableOp_1ReadVariableOp/uazvpibasg_gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&uazvpibasg/gddwjadkgr/ReadVariableOp_1À
uazvpibasg/gddwjadkgr/mul_1Mul.uazvpibasg/gddwjadkgr/ReadVariableOp_1:value:0uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_1¼
uazvpibasg/gddwjadkgr/add_2AddV2$uazvpibasg/gddwjadkgr/split:output:1uazvpibasg/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_2 
uazvpibasg/gddwjadkgr/Sigmoid_1Sigmoiduazvpibasg/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
uazvpibasg/gddwjadkgr/Sigmoid_1µ
uazvpibasg/gddwjadkgr/mul_2Mul#uazvpibasg/gddwjadkgr/Sigmoid_1:y:0uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_2
uazvpibasg/gddwjadkgr/TanhTanh$uazvpibasg/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/Tanh¶
uazvpibasg/gddwjadkgr/mul_3Mul!uazvpibasg/gddwjadkgr/Sigmoid:y:0uazvpibasg/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_3·
uazvpibasg/gddwjadkgr/add_3AddV2uazvpibasg/gddwjadkgr/mul_2:z:0uazvpibasg/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_3¼
&uazvpibasg/gddwjadkgr/ReadVariableOp_2ReadVariableOp/uazvpibasg_gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&uazvpibasg/gddwjadkgr/ReadVariableOp_2Ä
uazvpibasg/gddwjadkgr/mul_4Mul.uazvpibasg/gddwjadkgr/ReadVariableOp_2:value:0uazvpibasg/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_4¼
uazvpibasg/gddwjadkgr/add_4AddV2$uazvpibasg/gddwjadkgr/split:output:3uazvpibasg/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/add_4 
uazvpibasg/gddwjadkgr/Sigmoid_2Sigmoiduazvpibasg/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
uazvpibasg/gddwjadkgr/Sigmoid_2
uazvpibasg/gddwjadkgr/Tanh_1Tanhuazvpibasg/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/Tanh_1º
uazvpibasg/gddwjadkgr/mul_5Mul#uazvpibasg/gddwjadkgr/Sigmoid_2:y:0 uazvpibasg/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/gddwjadkgr/mul_5¥
(uazvpibasg/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(uazvpibasg/TensorArrayV2_1/element_shapeä
uazvpibasg/TensorArrayV2_1TensorListReserve1uazvpibasg/TensorArrayV2_1/element_shape:output:0#uazvpibasg/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
uazvpibasg/TensorArrayV2_1d
uazvpibasg/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/time
#uazvpibasg/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#uazvpibasg/while/maximum_iterations
uazvpibasg/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
uazvpibasg/while/loop_counter°
uazvpibasg/whileWhile&uazvpibasg/while/loop_counter:output:0,uazvpibasg/while/maximum_iterations:output:0uazvpibasg/time:output:0#uazvpibasg/TensorArrayV2_1:handle:0uazvpibasg/zeros:output:0uazvpibasg/zeros_1:output:0#uazvpibasg/strided_slice_1:output:0Buazvpibasg/TensorArrayUnstack/TensorListFromTensor:output_handle:04uazvpibasg_gddwjadkgr_matmul_readvariableop_resource6uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource5uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource-uazvpibasg_gddwjadkgr_readvariableop_resource/uazvpibasg_gddwjadkgr_readvariableop_1_resource/uazvpibasg_gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*(
body R
uazvpibasg_while_body_767686*(
cond R
uazvpibasg_while_cond_767685*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
uazvpibasg/whileË
;uazvpibasg/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;uazvpibasg/TensorArrayV2Stack/TensorListStack/element_shape
-uazvpibasg/TensorArrayV2Stack/TensorListStackTensorListStackuazvpibasg/while:output:3Duazvpibasg/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-uazvpibasg/TensorArrayV2Stack/TensorListStack
 uazvpibasg/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 uazvpibasg/strided_slice_3/stack
"uazvpibasg/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"uazvpibasg/strided_slice_3/stack_1
"uazvpibasg/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"uazvpibasg/strided_slice_3/stack_2Ü
uazvpibasg/strided_slice_3StridedSlice6uazvpibasg/TensorArrayV2Stack/TensorListStack:tensor:0)uazvpibasg/strided_slice_3/stack:output:0+uazvpibasg/strided_slice_3/stack_1:output:0+uazvpibasg/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
uazvpibasg/strided_slice_3
uazvpibasg/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
uazvpibasg/transpose_1/permÑ
uazvpibasg/transpose_1	Transpose6uazvpibasg/TensorArrayV2Stack/TensorListStack:tensor:0$uazvpibasg/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
uazvpibasg/transpose_1®
 supobtndkp/MatMul/ReadVariableOpReadVariableOp)supobtndkp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 supobtndkp/MatMul/ReadVariableOp±
supobtndkp/MatMulMatMul#uazvpibasg/strided_slice_3:output:0(supobtndkp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
supobtndkp/MatMul­
!supobtndkp/BiasAdd/ReadVariableOpReadVariableOp*supobtndkp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!supobtndkp/BiasAdd/ReadVariableOp­
supobtndkp/BiasAddBiasAddsupobtndkp/MatMul:product:0)supobtndkp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
supobtndkp/BiasAddÏ
IdentityIdentitysupobtndkp/BiasAdd:output:0-^gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp,^gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp.^gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp%^gdmdltnblf/duhsngmesj/ReadVariableOp'^gdmdltnblf/duhsngmesj/ReadVariableOp_1'^gdmdltnblf/duhsngmesj/ReadVariableOp_2^gdmdltnblf/while.^ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp5^ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp"^supobtndkp/BiasAdd/ReadVariableOp!^supobtndkp/MatMul/ReadVariableOp-^uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp,^uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp.^uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp%^uazvpibasg/gddwjadkgr/ReadVariableOp'^uazvpibasg/gddwjadkgr/ReadVariableOp_1'^uazvpibasg/gddwjadkgr/ReadVariableOp_2^uazvpibasg/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp,gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp2Z
+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp+gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp2^
-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp-gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp2L
$gdmdltnblf/duhsngmesj/ReadVariableOp$gdmdltnblf/duhsngmesj/ReadVariableOp2P
&gdmdltnblf/duhsngmesj/ReadVariableOp_1&gdmdltnblf/duhsngmesj/ReadVariableOp_12P
&gdmdltnblf/duhsngmesj/ReadVariableOp_2&gdmdltnblf/duhsngmesj/ReadVariableOp_22$
gdmdltnblf/whilegdmdltnblf/while2^
-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp-ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp2l
4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp4ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!supobtndkp/BiasAdd/ReadVariableOp!supobtndkp/BiasAdd/ReadVariableOp2D
 supobtndkp/MatMul/ReadVariableOp supobtndkp/MatMul/ReadVariableOp2\
,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp,uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp2Z
+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp+uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp2^
-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp-uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp2L
$uazvpibasg/gddwjadkgr/ReadVariableOp$uazvpibasg/gddwjadkgr/ReadVariableOp2P
&uazvpibasg/gddwjadkgr/ReadVariableOp_1&uazvpibasg/gddwjadkgr/ReadVariableOp_12P
&uazvpibasg/gddwjadkgr/ReadVariableOp_2&uazvpibasg/gddwjadkgr/ReadVariableOp_22$
uazvpibasg/whileuazvpibasg/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ö
!__inference__wrapped_model_764637

unopekwlxvW
Asequential_ggbvrcpxtu_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_ggbvrcpxtu_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_gdmdltnblf_duhsngmesj_matmul_readvariableop_resource:	T
Asequential_gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource:	 O
@sequential_gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource:	F
8sequential_gdmdltnblf_duhsngmesj_readvariableop_resource: H
:sequential_gdmdltnblf_duhsngmesj_readvariableop_1_resource: H
:sequential_gdmdltnblf_duhsngmesj_readvariableop_2_resource: R
?sequential_uazvpibasg_gddwjadkgr_matmul_readvariableop_resource:	 T
Asequential_uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource:	 O
@sequential_uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource:	F
8sequential_uazvpibasg_gddwjadkgr_readvariableop_resource: H
:sequential_uazvpibasg_gddwjadkgr_readvariableop_1_resource: H
:sequential_uazvpibasg_gddwjadkgr_readvariableop_2_resource: F
4sequential_supobtndkp_matmul_readvariableop_resource: C
5sequential_supobtndkp_biasadd_readvariableop_resource:
identity¢7sequential/gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp¢6sequential/gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp¢8sequential/gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp¢/sequential/gdmdltnblf/duhsngmesj/ReadVariableOp¢1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_1¢1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_2¢sequential/gdmdltnblf/while¢8sequential/ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,sequential/supobtndkp/BiasAdd/ReadVariableOp¢+sequential/supobtndkp/MatMul/ReadVariableOp¢7sequential/uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp¢6sequential/uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp¢8sequential/uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp¢/sequential/uazvpibasg/gddwjadkgr/ReadVariableOp¢1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_1¢1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_2¢sequential/uazvpibasg/while¥
+sequential/ggbvrcpxtu/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/ggbvrcpxtu/conv1d/ExpandDims/dimà
'sequential/ggbvrcpxtu/conv1d/ExpandDims
ExpandDims
unopekwlxv4sequential/ggbvrcpxtu/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/ggbvrcpxtu/conv1d/ExpandDimsú
8sequential/ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_ggbvrcpxtu_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/ggbvrcpxtu/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/ggbvrcpxtu/conv1d/ExpandDims_1/dim
)sequential/ggbvrcpxtu/conv1d/ExpandDims_1
ExpandDims@sequential/ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/ggbvrcpxtu/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/ggbvrcpxtu/conv1d/ExpandDims_1¨
"sequential/ggbvrcpxtu/conv1d/ShapeShape0sequential/ggbvrcpxtu/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/ggbvrcpxtu/conv1d/Shape®
0sequential/ggbvrcpxtu/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/ggbvrcpxtu/conv1d/strided_slice/stack»
2sequential/ggbvrcpxtu/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/ggbvrcpxtu/conv1d/strided_slice/stack_1²
2sequential/ggbvrcpxtu/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/ggbvrcpxtu/conv1d/strided_slice/stack_2
*sequential/ggbvrcpxtu/conv1d/strided_sliceStridedSlice+sequential/ggbvrcpxtu/conv1d/Shape:output:09sequential/ggbvrcpxtu/conv1d/strided_slice/stack:output:0;sequential/ggbvrcpxtu/conv1d/strided_slice/stack_1:output:0;sequential/ggbvrcpxtu/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/ggbvrcpxtu/conv1d/strided_slice±
*sequential/ggbvrcpxtu/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/ggbvrcpxtu/conv1d/Reshape/shapeø
$sequential/ggbvrcpxtu/conv1d/ReshapeReshape0sequential/ggbvrcpxtu/conv1d/ExpandDims:output:03sequential/ggbvrcpxtu/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/ggbvrcpxtu/conv1d/Reshape
#sequential/ggbvrcpxtu/conv1d/Conv2DConv2D-sequential/ggbvrcpxtu/conv1d/Reshape:output:02sequential/ggbvrcpxtu/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/ggbvrcpxtu/conv1d/Conv2D±
,sequential/ggbvrcpxtu/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/ggbvrcpxtu/conv1d/concat/values_1
(sequential/ggbvrcpxtu/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/ggbvrcpxtu/conv1d/concat/axis£
#sequential/ggbvrcpxtu/conv1d/concatConcatV23sequential/ggbvrcpxtu/conv1d/strided_slice:output:05sequential/ggbvrcpxtu/conv1d/concat/values_1:output:01sequential/ggbvrcpxtu/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/ggbvrcpxtu/conv1d/concatõ
&sequential/ggbvrcpxtu/conv1d/Reshape_1Reshape,sequential/ggbvrcpxtu/conv1d/Conv2D:output:0,sequential/ggbvrcpxtu/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/ggbvrcpxtu/conv1d/Reshape_1â
$sequential/ggbvrcpxtu/conv1d/SqueezeSqueeze/sequential/ggbvrcpxtu/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/ggbvrcpxtu/conv1d/Squeeze½
.sequential/ggbvrcpxtu/squeeze_batch_dims/ShapeShape-sequential/ggbvrcpxtu/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/ggbvrcpxtu/squeeze_batch_dims/ShapeÆ
<sequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stackÓ
>sequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/ggbvrcpxtu/squeeze_batch_dims/strided_sliceStridedSlice7sequential/ggbvrcpxtu/squeeze_batch_dims/Shape:output:0Esequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/ggbvrcpxtu/squeeze_batch_dims/strided_sliceÅ
6sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape/shape
0sequential/ggbvrcpxtu/squeeze_batch_dims/ReshapeReshape-sequential/ggbvrcpxtu/conv1d/Squeeze:output:0?sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape
?sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_ggbvrcpxtu_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAddBiasAdd9sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape:output:0Gsequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAddÅ
8sequential/ggbvrcpxtu/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/ggbvrcpxtu/squeeze_batch_dims/concat/values_1·
4sequential/ggbvrcpxtu/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/ggbvrcpxtu/squeeze_batch_dims/concat/axisß
/sequential/ggbvrcpxtu/squeeze_batch_dims/concatConcatV2?sequential/ggbvrcpxtu/squeeze_batch_dims/strided_slice:output:0Asequential/ggbvrcpxtu/squeeze_batch_dims/concat/values_1:output:0=sequential/ggbvrcpxtu/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/ggbvrcpxtu/squeeze_batch_dims/concat¢
2sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape_1Reshape9sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd:output:08sequential/ggbvrcpxtu/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape_1¥
sequential/vtqejmjhbd/ShapeShape;sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/vtqejmjhbd/Shape 
)sequential/vtqejmjhbd/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/vtqejmjhbd/strided_slice/stack¤
+sequential/vtqejmjhbd/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/vtqejmjhbd/strided_slice/stack_1¤
+sequential/vtqejmjhbd/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/vtqejmjhbd/strided_slice/stack_2æ
#sequential/vtqejmjhbd/strided_sliceStridedSlice$sequential/vtqejmjhbd/Shape:output:02sequential/vtqejmjhbd/strided_slice/stack:output:04sequential/vtqejmjhbd/strided_slice/stack_1:output:04sequential/vtqejmjhbd/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/vtqejmjhbd/strided_slice
%sequential/vtqejmjhbd/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/vtqejmjhbd/Reshape/shape/1
%sequential/vtqejmjhbd/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/vtqejmjhbd/Reshape/shape/2
#sequential/vtqejmjhbd/Reshape/shapePack,sequential/vtqejmjhbd/strided_slice:output:0.sequential/vtqejmjhbd/Reshape/shape/1:output:0.sequential/vtqejmjhbd/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/vtqejmjhbd/Reshape/shapeê
sequential/vtqejmjhbd/ReshapeReshape;sequential/ggbvrcpxtu/squeeze_batch_dims/Reshape_1:output:0,sequential/vtqejmjhbd/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/vtqejmjhbd/Reshape
sequential/gdmdltnblf/ShapeShape&sequential/vtqejmjhbd/Reshape:output:0*
T0*
_output_shapes
:2
sequential/gdmdltnblf/Shape 
)sequential/gdmdltnblf/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/gdmdltnblf/strided_slice/stack¤
+sequential/gdmdltnblf/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/gdmdltnblf/strided_slice/stack_1¤
+sequential/gdmdltnblf/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/gdmdltnblf/strided_slice/stack_2æ
#sequential/gdmdltnblf/strided_sliceStridedSlice$sequential/gdmdltnblf/Shape:output:02sequential/gdmdltnblf/strided_slice/stack:output:04sequential/gdmdltnblf/strided_slice/stack_1:output:04sequential/gdmdltnblf/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/gdmdltnblf/strided_slice
!sequential/gdmdltnblf/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/gdmdltnblf/zeros/mul/yÄ
sequential/gdmdltnblf/zeros/mulMul,sequential/gdmdltnblf/strided_slice:output:0*sequential/gdmdltnblf/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/gdmdltnblf/zeros/mul
"sequential/gdmdltnblf/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/gdmdltnblf/zeros/Less/y¿
 sequential/gdmdltnblf/zeros/LessLess#sequential/gdmdltnblf/zeros/mul:z:0+sequential/gdmdltnblf/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/gdmdltnblf/zeros/Less
$sequential/gdmdltnblf/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/gdmdltnblf/zeros/packed/1Û
"sequential/gdmdltnblf/zeros/packedPack,sequential/gdmdltnblf/strided_slice:output:0-sequential/gdmdltnblf/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/gdmdltnblf/zeros/packed
!sequential/gdmdltnblf/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/gdmdltnblf/zeros/ConstÍ
sequential/gdmdltnblf/zerosFill+sequential/gdmdltnblf/zeros/packed:output:0*sequential/gdmdltnblf/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gdmdltnblf/zeros
#sequential/gdmdltnblf/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/gdmdltnblf/zeros_1/mul/yÊ
!sequential/gdmdltnblf/zeros_1/mulMul,sequential/gdmdltnblf/strided_slice:output:0,sequential/gdmdltnblf/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/gdmdltnblf/zeros_1/mul
$sequential/gdmdltnblf/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/gdmdltnblf/zeros_1/Less/yÇ
"sequential/gdmdltnblf/zeros_1/LessLess%sequential/gdmdltnblf/zeros_1/mul:z:0-sequential/gdmdltnblf/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/gdmdltnblf/zeros_1/Less
&sequential/gdmdltnblf/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/gdmdltnblf/zeros_1/packed/1á
$sequential/gdmdltnblf/zeros_1/packedPack,sequential/gdmdltnblf/strided_slice:output:0/sequential/gdmdltnblf/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/gdmdltnblf/zeros_1/packed
#sequential/gdmdltnblf/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/gdmdltnblf/zeros_1/ConstÕ
sequential/gdmdltnblf/zeros_1Fill-sequential/gdmdltnblf/zeros_1/packed:output:0,sequential/gdmdltnblf/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gdmdltnblf/zeros_1¡
$sequential/gdmdltnblf/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/gdmdltnblf/transpose/permÜ
sequential/gdmdltnblf/transpose	Transpose&sequential/vtqejmjhbd/Reshape:output:0-sequential/gdmdltnblf/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/gdmdltnblf/transpose
sequential/gdmdltnblf/Shape_1Shape#sequential/gdmdltnblf/transpose:y:0*
T0*
_output_shapes
:2
sequential/gdmdltnblf/Shape_1¤
+sequential/gdmdltnblf/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/gdmdltnblf/strided_slice_1/stack¨
-sequential/gdmdltnblf/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/gdmdltnblf/strided_slice_1/stack_1¨
-sequential/gdmdltnblf/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/gdmdltnblf/strided_slice_1/stack_2ò
%sequential/gdmdltnblf/strided_slice_1StridedSlice&sequential/gdmdltnblf/Shape_1:output:04sequential/gdmdltnblf/strided_slice_1/stack:output:06sequential/gdmdltnblf/strided_slice_1/stack_1:output:06sequential/gdmdltnblf/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/gdmdltnblf/strided_slice_1±
1sequential/gdmdltnblf/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/gdmdltnblf/TensorArrayV2/element_shape
#sequential/gdmdltnblf/TensorArrayV2TensorListReserve:sequential/gdmdltnblf/TensorArrayV2/element_shape:output:0.sequential/gdmdltnblf/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/gdmdltnblf/TensorArrayV2ë
Ksequential/gdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/gdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/gdmdltnblf/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/gdmdltnblf/transpose:y:0Tsequential/gdmdltnblf/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/gdmdltnblf/TensorArrayUnstack/TensorListFromTensor¤
+sequential/gdmdltnblf/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/gdmdltnblf/strided_slice_2/stack¨
-sequential/gdmdltnblf/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/gdmdltnblf/strided_slice_2/stack_1¨
-sequential/gdmdltnblf/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/gdmdltnblf/strided_slice_2/stack_2
%sequential/gdmdltnblf/strided_slice_2StridedSlice#sequential/gdmdltnblf/transpose:y:04sequential/gdmdltnblf/strided_slice_2/stack:output:06sequential/gdmdltnblf/strided_slice_2/stack_1:output:06sequential/gdmdltnblf/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/gdmdltnblf/strided_slice_2ñ
6sequential/gdmdltnblf/duhsngmesj/MatMul/ReadVariableOpReadVariableOp?sequential_gdmdltnblf_duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/gdmdltnblf/duhsngmesj/MatMul/ReadVariableOpÿ
'sequential/gdmdltnblf/duhsngmesj/MatMulMatMul.sequential/gdmdltnblf/strided_slice_2:output:0>sequential/gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/gdmdltnblf/duhsngmesj/MatMul÷
8sequential/gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOpAsequential_gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOpû
)sequential/gdmdltnblf/duhsngmesj/MatMul_1MatMul$sequential/gdmdltnblf/zeros:output:0@sequential/gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/gdmdltnblf/duhsngmesj/MatMul_1ð
$sequential/gdmdltnblf/duhsngmesj/addAddV21sequential/gdmdltnblf/duhsngmesj/MatMul:product:03sequential/gdmdltnblf/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/gdmdltnblf/duhsngmesj/addð
7sequential/gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp@sequential_gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOpý
(sequential/gdmdltnblf/duhsngmesj/BiasAddBiasAdd(sequential/gdmdltnblf/duhsngmesj/add:z:0?sequential/gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/gdmdltnblf/duhsngmesj/BiasAdd¦
0sequential/gdmdltnblf/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/gdmdltnblf/duhsngmesj/split/split_dimÃ
&sequential/gdmdltnblf/duhsngmesj/splitSplit9sequential/gdmdltnblf/duhsngmesj/split/split_dim:output:01sequential/gdmdltnblf/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/gdmdltnblf/duhsngmesj/split×
/sequential/gdmdltnblf/duhsngmesj/ReadVariableOpReadVariableOp8sequential_gdmdltnblf_duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/gdmdltnblf/duhsngmesj/ReadVariableOpæ
$sequential/gdmdltnblf/duhsngmesj/mulMul7sequential/gdmdltnblf/duhsngmesj/ReadVariableOp:value:0&sequential/gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/gdmdltnblf/duhsngmesj/mulæ
&sequential/gdmdltnblf/duhsngmesj/add_1AddV2/sequential/gdmdltnblf/duhsngmesj/split:output:0(sequential/gdmdltnblf/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/add_1½
(sequential/gdmdltnblf/duhsngmesj/SigmoidSigmoid*sequential/gdmdltnblf/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/gdmdltnblf/duhsngmesj/SigmoidÝ
1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_1ReadVariableOp:sequential_gdmdltnblf_duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_1ì
&sequential/gdmdltnblf/duhsngmesj/mul_1Mul9sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_1:value:0&sequential/gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/mul_1è
&sequential/gdmdltnblf/duhsngmesj/add_2AddV2/sequential/gdmdltnblf/duhsngmesj/split:output:1*sequential/gdmdltnblf/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/add_2Á
*sequential/gdmdltnblf/duhsngmesj/Sigmoid_1Sigmoid*sequential/gdmdltnblf/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/gdmdltnblf/duhsngmesj/Sigmoid_1á
&sequential/gdmdltnblf/duhsngmesj/mul_2Mul.sequential/gdmdltnblf/duhsngmesj/Sigmoid_1:y:0&sequential/gdmdltnblf/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/mul_2¹
%sequential/gdmdltnblf/duhsngmesj/TanhTanh/sequential/gdmdltnblf/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/gdmdltnblf/duhsngmesj/Tanhâ
&sequential/gdmdltnblf/duhsngmesj/mul_3Mul,sequential/gdmdltnblf/duhsngmesj/Sigmoid:y:0)sequential/gdmdltnblf/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/mul_3ã
&sequential/gdmdltnblf/duhsngmesj/add_3AddV2*sequential/gdmdltnblf/duhsngmesj/mul_2:z:0*sequential/gdmdltnblf/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/add_3Ý
1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_2ReadVariableOp:sequential_gdmdltnblf_duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_2ð
&sequential/gdmdltnblf/duhsngmesj/mul_4Mul9sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_2:value:0*sequential/gdmdltnblf/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/mul_4è
&sequential/gdmdltnblf/duhsngmesj/add_4AddV2/sequential/gdmdltnblf/duhsngmesj/split:output:3*sequential/gdmdltnblf/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/add_4Á
*sequential/gdmdltnblf/duhsngmesj/Sigmoid_2Sigmoid*sequential/gdmdltnblf/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/gdmdltnblf/duhsngmesj/Sigmoid_2¸
'sequential/gdmdltnblf/duhsngmesj/Tanh_1Tanh*sequential/gdmdltnblf/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/gdmdltnblf/duhsngmesj/Tanh_1æ
&sequential/gdmdltnblf/duhsngmesj/mul_5Mul.sequential/gdmdltnblf/duhsngmesj/Sigmoid_2:y:0+sequential/gdmdltnblf/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/gdmdltnblf/duhsngmesj/mul_5»
3sequential/gdmdltnblf/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/gdmdltnblf/TensorArrayV2_1/element_shape
%sequential/gdmdltnblf/TensorArrayV2_1TensorListReserve<sequential/gdmdltnblf/TensorArrayV2_1/element_shape:output:0.sequential/gdmdltnblf/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/gdmdltnblf/TensorArrayV2_1z
sequential/gdmdltnblf/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gdmdltnblf/time«
.sequential/gdmdltnblf/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/gdmdltnblf/while/maximum_iterations
(sequential/gdmdltnblf/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/gdmdltnblf/while/loop_counterö	
sequential/gdmdltnblf/whileWhile1sequential/gdmdltnblf/while/loop_counter:output:07sequential/gdmdltnblf/while/maximum_iterations:output:0#sequential/gdmdltnblf/time:output:0.sequential/gdmdltnblf/TensorArrayV2_1:handle:0$sequential/gdmdltnblf/zeros:output:0&sequential/gdmdltnblf/zeros_1:output:0.sequential/gdmdltnblf/strided_slice_1:output:0Msequential/gdmdltnblf/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_gdmdltnblf_duhsngmesj_matmul_readvariableop_resourceAsequential_gdmdltnblf_duhsngmesj_matmul_1_readvariableop_resource@sequential_gdmdltnblf_duhsngmesj_biasadd_readvariableop_resource8sequential_gdmdltnblf_duhsngmesj_readvariableop_resource:sequential_gdmdltnblf_duhsngmesj_readvariableop_1_resource:sequential_gdmdltnblf_duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*3
body+R)
'sequential_gdmdltnblf_while_body_764354*3
cond+R)
'sequential_gdmdltnblf_while_cond_764353*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/gdmdltnblf/whileá
Fsequential/gdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/gdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/gdmdltnblf/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/gdmdltnblf/while:output:3Osequential/gdmdltnblf/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/gdmdltnblf/TensorArrayV2Stack/TensorListStack­
+sequential/gdmdltnblf/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/gdmdltnblf/strided_slice_3/stack¨
-sequential/gdmdltnblf/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/gdmdltnblf/strided_slice_3/stack_1¨
-sequential/gdmdltnblf/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/gdmdltnblf/strided_slice_3/stack_2
%sequential/gdmdltnblf/strided_slice_3StridedSliceAsequential/gdmdltnblf/TensorArrayV2Stack/TensorListStack:tensor:04sequential/gdmdltnblf/strided_slice_3/stack:output:06sequential/gdmdltnblf/strided_slice_3/stack_1:output:06sequential/gdmdltnblf/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/gdmdltnblf/strided_slice_3¥
&sequential/gdmdltnblf/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/gdmdltnblf/transpose_1/permý
!sequential/gdmdltnblf/transpose_1	TransposeAsequential/gdmdltnblf/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/gdmdltnblf/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/gdmdltnblf/transpose_1
sequential/uazvpibasg/ShapeShape%sequential/gdmdltnblf/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/uazvpibasg/Shape 
)sequential/uazvpibasg/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/uazvpibasg/strided_slice/stack¤
+sequential/uazvpibasg/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/uazvpibasg/strided_slice/stack_1¤
+sequential/uazvpibasg/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/uazvpibasg/strided_slice/stack_2æ
#sequential/uazvpibasg/strided_sliceStridedSlice$sequential/uazvpibasg/Shape:output:02sequential/uazvpibasg/strided_slice/stack:output:04sequential/uazvpibasg/strided_slice/stack_1:output:04sequential/uazvpibasg/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/uazvpibasg/strided_slice
!sequential/uazvpibasg/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/uazvpibasg/zeros/mul/yÄ
sequential/uazvpibasg/zeros/mulMul,sequential/uazvpibasg/strided_slice:output:0*sequential/uazvpibasg/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/uazvpibasg/zeros/mul
"sequential/uazvpibasg/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/uazvpibasg/zeros/Less/y¿
 sequential/uazvpibasg/zeros/LessLess#sequential/uazvpibasg/zeros/mul:z:0+sequential/uazvpibasg/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/uazvpibasg/zeros/Less
$sequential/uazvpibasg/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/uazvpibasg/zeros/packed/1Û
"sequential/uazvpibasg/zeros/packedPack,sequential/uazvpibasg/strided_slice:output:0-sequential/uazvpibasg/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/uazvpibasg/zeros/packed
!sequential/uazvpibasg/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/uazvpibasg/zeros/ConstÍ
sequential/uazvpibasg/zerosFill+sequential/uazvpibasg/zeros/packed:output:0*sequential/uazvpibasg/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/uazvpibasg/zeros
#sequential/uazvpibasg/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/uazvpibasg/zeros_1/mul/yÊ
!sequential/uazvpibasg/zeros_1/mulMul,sequential/uazvpibasg/strided_slice:output:0,sequential/uazvpibasg/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/uazvpibasg/zeros_1/mul
$sequential/uazvpibasg/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/uazvpibasg/zeros_1/Less/yÇ
"sequential/uazvpibasg/zeros_1/LessLess%sequential/uazvpibasg/zeros_1/mul:z:0-sequential/uazvpibasg/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/uazvpibasg/zeros_1/Less
&sequential/uazvpibasg/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/uazvpibasg/zeros_1/packed/1á
$sequential/uazvpibasg/zeros_1/packedPack,sequential/uazvpibasg/strided_slice:output:0/sequential/uazvpibasg/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/uazvpibasg/zeros_1/packed
#sequential/uazvpibasg/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/uazvpibasg/zeros_1/ConstÕ
sequential/uazvpibasg/zeros_1Fill-sequential/uazvpibasg/zeros_1/packed:output:0,sequential/uazvpibasg/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/uazvpibasg/zeros_1¡
$sequential/uazvpibasg/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/uazvpibasg/transpose/permÛ
sequential/uazvpibasg/transpose	Transpose%sequential/gdmdltnblf/transpose_1:y:0-sequential/uazvpibasg/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/uazvpibasg/transpose
sequential/uazvpibasg/Shape_1Shape#sequential/uazvpibasg/transpose:y:0*
T0*
_output_shapes
:2
sequential/uazvpibasg/Shape_1¤
+sequential/uazvpibasg/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/uazvpibasg/strided_slice_1/stack¨
-sequential/uazvpibasg/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/uazvpibasg/strided_slice_1/stack_1¨
-sequential/uazvpibasg/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/uazvpibasg/strided_slice_1/stack_2ò
%sequential/uazvpibasg/strided_slice_1StridedSlice&sequential/uazvpibasg/Shape_1:output:04sequential/uazvpibasg/strided_slice_1/stack:output:06sequential/uazvpibasg/strided_slice_1/stack_1:output:06sequential/uazvpibasg/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/uazvpibasg/strided_slice_1±
1sequential/uazvpibasg/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/uazvpibasg/TensorArrayV2/element_shape
#sequential/uazvpibasg/TensorArrayV2TensorListReserve:sequential/uazvpibasg/TensorArrayV2/element_shape:output:0.sequential/uazvpibasg/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/uazvpibasg/TensorArrayV2ë
Ksequential/uazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/uazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/uazvpibasg/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/uazvpibasg/transpose:y:0Tsequential/uazvpibasg/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/uazvpibasg/TensorArrayUnstack/TensorListFromTensor¤
+sequential/uazvpibasg/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/uazvpibasg/strided_slice_2/stack¨
-sequential/uazvpibasg/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/uazvpibasg/strided_slice_2/stack_1¨
-sequential/uazvpibasg/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/uazvpibasg/strided_slice_2/stack_2
%sequential/uazvpibasg/strided_slice_2StridedSlice#sequential/uazvpibasg/transpose:y:04sequential/uazvpibasg/strided_slice_2/stack:output:06sequential/uazvpibasg/strided_slice_2/stack_1:output:06sequential/uazvpibasg/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/uazvpibasg/strided_slice_2ñ
6sequential/uazvpibasg/gddwjadkgr/MatMul/ReadVariableOpReadVariableOp?sequential_uazvpibasg_gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/uazvpibasg/gddwjadkgr/MatMul/ReadVariableOpÿ
'sequential/uazvpibasg/gddwjadkgr/MatMulMatMul.sequential/uazvpibasg/strided_slice_2:output:0>sequential/uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/uazvpibasg/gddwjadkgr/MatMul÷
8sequential/uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOpAsequential_uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOpû
)sequential/uazvpibasg/gddwjadkgr/MatMul_1MatMul$sequential/uazvpibasg/zeros:output:0@sequential/uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/uazvpibasg/gddwjadkgr/MatMul_1ð
$sequential/uazvpibasg/gddwjadkgr/addAddV21sequential/uazvpibasg/gddwjadkgr/MatMul:product:03sequential/uazvpibasg/gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/uazvpibasg/gddwjadkgr/addð
7sequential/uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp@sequential_uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOpý
(sequential/uazvpibasg/gddwjadkgr/BiasAddBiasAdd(sequential/uazvpibasg/gddwjadkgr/add:z:0?sequential/uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/uazvpibasg/gddwjadkgr/BiasAdd¦
0sequential/uazvpibasg/gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/uazvpibasg/gddwjadkgr/split/split_dimÃ
&sequential/uazvpibasg/gddwjadkgr/splitSplit9sequential/uazvpibasg/gddwjadkgr/split/split_dim:output:01sequential/uazvpibasg/gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/uazvpibasg/gddwjadkgr/split×
/sequential/uazvpibasg/gddwjadkgr/ReadVariableOpReadVariableOp8sequential_uazvpibasg_gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/uazvpibasg/gddwjadkgr/ReadVariableOpæ
$sequential/uazvpibasg/gddwjadkgr/mulMul7sequential/uazvpibasg/gddwjadkgr/ReadVariableOp:value:0&sequential/uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/uazvpibasg/gddwjadkgr/mulæ
&sequential/uazvpibasg/gddwjadkgr/add_1AddV2/sequential/uazvpibasg/gddwjadkgr/split:output:0(sequential/uazvpibasg/gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/add_1½
(sequential/uazvpibasg/gddwjadkgr/SigmoidSigmoid*sequential/uazvpibasg/gddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/uazvpibasg/gddwjadkgr/SigmoidÝ
1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_1ReadVariableOp:sequential_uazvpibasg_gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_1ì
&sequential/uazvpibasg/gddwjadkgr/mul_1Mul9sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_1:value:0&sequential/uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/mul_1è
&sequential/uazvpibasg/gddwjadkgr/add_2AddV2/sequential/uazvpibasg/gddwjadkgr/split:output:1*sequential/uazvpibasg/gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/add_2Á
*sequential/uazvpibasg/gddwjadkgr/Sigmoid_1Sigmoid*sequential/uazvpibasg/gddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/uazvpibasg/gddwjadkgr/Sigmoid_1á
&sequential/uazvpibasg/gddwjadkgr/mul_2Mul.sequential/uazvpibasg/gddwjadkgr/Sigmoid_1:y:0&sequential/uazvpibasg/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/mul_2¹
%sequential/uazvpibasg/gddwjadkgr/TanhTanh/sequential/uazvpibasg/gddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/uazvpibasg/gddwjadkgr/Tanhâ
&sequential/uazvpibasg/gddwjadkgr/mul_3Mul,sequential/uazvpibasg/gddwjadkgr/Sigmoid:y:0)sequential/uazvpibasg/gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/mul_3ã
&sequential/uazvpibasg/gddwjadkgr/add_3AddV2*sequential/uazvpibasg/gddwjadkgr/mul_2:z:0*sequential/uazvpibasg/gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/add_3Ý
1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_2ReadVariableOp:sequential_uazvpibasg_gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_2ð
&sequential/uazvpibasg/gddwjadkgr/mul_4Mul9sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_2:value:0*sequential/uazvpibasg/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/mul_4è
&sequential/uazvpibasg/gddwjadkgr/add_4AddV2/sequential/uazvpibasg/gddwjadkgr/split:output:3*sequential/uazvpibasg/gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/add_4Á
*sequential/uazvpibasg/gddwjadkgr/Sigmoid_2Sigmoid*sequential/uazvpibasg/gddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/uazvpibasg/gddwjadkgr/Sigmoid_2¸
'sequential/uazvpibasg/gddwjadkgr/Tanh_1Tanh*sequential/uazvpibasg/gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/uazvpibasg/gddwjadkgr/Tanh_1æ
&sequential/uazvpibasg/gddwjadkgr/mul_5Mul.sequential/uazvpibasg/gddwjadkgr/Sigmoid_2:y:0+sequential/uazvpibasg/gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/uazvpibasg/gddwjadkgr/mul_5»
3sequential/uazvpibasg/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/uazvpibasg/TensorArrayV2_1/element_shape
%sequential/uazvpibasg/TensorArrayV2_1TensorListReserve<sequential/uazvpibasg/TensorArrayV2_1/element_shape:output:0.sequential/uazvpibasg/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/uazvpibasg/TensorArrayV2_1z
sequential/uazvpibasg/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/uazvpibasg/time«
.sequential/uazvpibasg/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/uazvpibasg/while/maximum_iterations
(sequential/uazvpibasg/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/uazvpibasg/while/loop_counterö	
sequential/uazvpibasg/whileWhile1sequential/uazvpibasg/while/loop_counter:output:07sequential/uazvpibasg/while/maximum_iterations:output:0#sequential/uazvpibasg/time:output:0.sequential/uazvpibasg/TensorArrayV2_1:handle:0$sequential/uazvpibasg/zeros:output:0&sequential/uazvpibasg/zeros_1:output:0.sequential/uazvpibasg/strided_slice_1:output:0Msequential/uazvpibasg/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_uazvpibasg_gddwjadkgr_matmul_readvariableop_resourceAsequential_uazvpibasg_gddwjadkgr_matmul_1_readvariableop_resource@sequential_uazvpibasg_gddwjadkgr_biasadd_readvariableop_resource8sequential_uazvpibasg_gddwjadkgr_readvariableop_resource:sequential_uazvpibasg_gddwjadkgr_readvariableop_1_resource:sequential_uazvpibasg_gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*3
body+R)
'sequential_uazvpibasg_while_body_764530*3
cond+R)
'sequential_uazvpibasg_while_cond_764529*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/uazvpibasg/whileá
Fsequential/uazvpibasg/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/uazvpibasg/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/uazvpibasg/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/uazvpibasg/while:output:3Osequential/uazvpibasg/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/uazvpibasg/TensorArrayV2Stack/TensorListStack­
+sequential/uazvpibasg/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/uazvpibasg/strided_slice_3/stack¨
-sequential/uazvpibasg/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/uazvpibasg/strided_slice_3/stack_1¨
-sequential/uazvpibasg/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/uazvpibasg/strided_slice_3/stack_2
%sequential/uazvpibasg/strided_slice_3StridedSliceAsequential/uazvpibasg/TensorArrayV2Stack/TensorListStack:tensor:04sequential/uazvpibasg/strided_slice_3/stack:output:06sequential/uazvpibasg/strided_slice_3/stack_1:output:06sequential/uazvpibasg/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/uazvpibasg/strided_slice_3¥
&sequential/uazvpibasg/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/uazvpibasg/transpose_1/permý
!sequential/uazvpibasg/transpose_1	TransposeAsequential/uazvpibasg/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/uazvpibasg/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/uazvpibasg/transpose_1Ï
+sequential/supobtndkp/MatMul/ReadVariableOpReadVariableOp4sequential_supobtndkp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/supobtndkp/MatMul/ReadVariableOpÝ
sequential/supobtndkp/MatMulMatMul.sequential/uazvpibasg/strided_slice_3:output:03sequential/supobtndkp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/supobtndkp/MatMulÎ
,sequential/supobtndkp/BiasAdd/ReadVariableOpReadVariableOp5sequential_supobtndkp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/supobtndkp/BiasAdd/ReadVariableOpÙ
sequential/supobtndkp/BiasAddBiasAdd&sequential/supobtndkp/MatMul:product:04sequential/supobtndkp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/supobtndkp/BiasAdd 
IdentityIdentity&sequential/supobtndkp/BiasAdd:output:08^sequential/gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp7^sequential/gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp9^sequential/gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp0^sequential/gdmdltnblf/duhsngmesj/ReadVariableOp2^sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_12^sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_2^sequential/gdmdltnblf/while9^sequential/ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp@^sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp-^sequential/supobtndkp/BiasAdd/ReadVariableOp,^sequential/supobtndkp/MatMul/ReadVariableOp8^sequential/uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp7^sequential/uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp9^sequential/uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp0^sequential/uazvpibasg/gddwjadkgr/ReadVariableOp2^sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_12^sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_2^sequential/uazvpibasg/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2r
7sequential/gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp7sequential/gdmdltnblf/duhsngmesj/BiasAdd/ReadVariableOp2p
6sequential/gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp6sequential/gdmdltnblf/duhsngmesj/MatMul/ReadVariableOp2t
8sequential/gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp8sequential/gdmdltnblf/duhsngmesj/MatMul_1/ReadVariableOp2b
/sequential/gdmdltnblf/duhsngmesj/ReadVariableOp/sequential/gdmdltnblf/duhsngmesj/ReadVariableOp2f
1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_11sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_12f
1sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_21sequential/gdmdltnblf/duhsngmesj/ReadVariableOp_22:
sequential/gdmdltnblf/whilesequential/gdmdltnblf/while2t
8sequential/ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp8sequential/ggbvrcpxtu/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/ggbvrcpxtu/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,sequential/supobtndkp/BiasAdd/ReadVariableOp,sequential/supobtndkp/BiasAdd/ReadVariableOp2Z
+sequential/supobtndkp/MatMul/ReadVariableOp+sequential/supobtndkp/MatMul/ReadVariableOp2r
7sequential/uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp7sequential/uazvpibasg/gddwjadkgr/BiasAdd/ReadVariableOp2p
6sequential/uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp6sequential/uazvpibasg/gddwjadkgr/MatMul/ReadVariableOp2t
8sequential/uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp8sequential/uazvpibasg/gddwjadkgr/MatMul_1/ReadVariableOp2b
/sequential/uazvpibasg/gddwjadkgr/ReadVariableOp/sequential/uazvpibasg/gddwjadkgr/ReadVariableOp2f
1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_11sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_12f
1sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_21sequential/uazvpibasg/gddwjadkgr/ReadVariableOp_22:
sequential/uazvpibasg/whilesequential/uazvpibasg/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
unopekwlxv

¯
F__inference_sequential_layer_call_and_return_conditional_losses_767344

unopekwlxv'
ggbvrcpxtu_767306:
ggbvrcpxtu_767308:$
gdmdltnblf_767312:	$
gdmdltnblf_767314:	  
gdmdltnblf_767316:	
gdmdltnblf_767318: 
gdmdltnblf_767320: 
gdmdltnblf_767322: $
uazvpibasg_767325:	 $
uazvpibasg_767327:	  
uazvpibasg_767329:	
uazvpibasg_767331: 
uazvpibasg_767333: 
uazvpibasg_767335: #
supobtndkp_767338: 
supobtndkp_767340:
identity¢"gdmdltnblf/StatefulPartitionedCall¢"ggbvrcpxtu/StatefulPartitionedCall¢"supobtndkp/StatefulPartitionedCall¢"uazvpibasg/StatefulPartitionedCall­
"ggbvrcpxtu/StatefulPartitionedCallStatefulPartitionedCall
unopekwlxvggbvrcpxtu_767306ggbvrcpxtu_767308*
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
GPU2*0J 8 *O
fJRH
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_7661972$
"ggbvrcpxtu/StatefulPartitionedCall
vtqejmjhbd/PartitionedCallPartitionedCall+ggbvrcpxtu/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *O
fJRH
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_7662162
vtqejmjhbd/PartitionedCall
"gdmdltnblf/StatefulPartitionedCallStatefulPartitionedCall#vtqejmjhbd/PartitionedCall:output:0gdmdltnblf_767312gdmdltnblf_767314gdmdltnblf_767316gdmdltnblf_767318gdmdltnblf_767320gdmdltnblf_767322*
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7670792$
"gdmdltnblf/StatefulPartitionedCall
"uazvpibasg/StatefulPartitionedCallStatefulPartitionedCall+gdmdltnblf/StatefulPartitionedCall:output:0uazvpibasg_767325uazvpibasg_767327uazvpibasg_767329uazvpibasg_767331uazvpibasg_767333uazvpibasg_767335*
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7668652$
"uazvpibasg/StatefulPartitionedCallÆ
"supobtndkp/StatefulPartitionedCallStatefulPartitionedCall+uazvpibasg/StatefulPartitionedCall:output:0supobtndkp_767338supobtndkp_767340*
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
GPU2*0J 8 *O
fJRH
F__inference_supobtndkp_layer_call_and_return_conditional_losses_7666142$
"supobtndkp/StatefulPartitionedCall
IdentityIdentity+supobtndkp/StatefulPartitionedCall:output:0#^gdmdltnblf/StatefulPartitionedCall#^ggbvrcpxtu/StatefulPartitionedCall#^supobtndkp/StatefulPartitionedCall#^uazvpibasg/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"gdmdltnblf/StatefulPartitionedCall"gdmdltnblf/StatefulPartitionedCall2H
"ggbvrcpxtu/StatefulPartitionedCall"ggbvrcpxtu/StatefulPartitionedCall2H
"supobtndkp/StatefulPartitionedCall"supobtndkp/StatefulPartitionedCall2H
"uazvpibasg/StatefulPartitionedCall"uazvpibasg/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
unopekwlxv
ßY

while_body_768954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_duhsngmesj_matmul_readvariableop_resource_0:	F
3while_duhsngmesj_matmul_1_readvariableop_resource_0:	 A
2while_duhsngmesj_biasadd_readvariableop_resource_0:	8
*while_duhsngmesj_readvariableop_resource_0: :
,while_duhsngmesj_readvariableop_1_resource_0: :
,while_duhsngmesj_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_duhsngmesj_matmul_readvariableop_resource:	D
1while_duhsngmesj_matmul_1_readvariableop_resource:	 ?
0while_duhsngmesj_biasadd_readvariableop_resource:	6
(while_duhsngmesj_readvariableop_resource: 8
*while_duhsngmesj_readvariableop_1_resource: 8
*while_duhsngmesj_readvariableop_2_resource: ¢'while/duhsngmesj/BiasAdd/ReadVariableOp¢&while/duhsngmesj/MatMul/ReadVariableOp¢(while/duhsngmesj/MatMul_1/ReadVariableOp¢while/duhsngmesj/ReadVariableOp¢!while/duhsngmesj/ReadVariableOp_1¢!while/duhsngmesj/ReadVariableOp_2Ã
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
&while/duhsngmesj/MatMul/ReadVariableOpReadVariableOp1while_duhsngmesj_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/duhsngmesj/MatMul/ReadVariableOpÑ
while/duhsngmesj/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMulÉ
(while/duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp3while_duhsngmesj_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/duhsngmesj/MatMul_1/ReadVariableOpº
while/duhsngmesj/MatMul_1MatMulwhile_placeholder_20while/duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/MatMul_1°
while/duhsngmesj/addAddV2!while/duhsngmesj/MatMul:product:0#while/duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/addÂ
'while/duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp2while_duhsngmesj_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/duhsngmesj/BiasAdd/ReadVariableOp½
while/duhsngmesj/BiasAddBiasAddwhile/duhsngmesj/add:z:0/while/duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/duhsngmesj/BiasAdd
 while/duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/duhsngmesj/split/split_dim
while/duhsngmesj/splitSplit)while/duhsngmesj/split/split_dim:output:0!while/duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/duhsngmesj/split©
while/duhsngmesj/ReadVariableOpReadVariableOp*while_duhsngmesj_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/duhsngmesj/ReadVariableOp£
while/duhsngmesj/mulMul'while/duhsngmesj/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul¦
while/duhsngmesj/add_1AddV2while/duhsngmesj/split:output:0while/duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_1
while/duhsngmesj/SigmoidSigmoidwhile/duhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid¯
!while/duhsngmesj/ReadVariableOp_1ReadVariableOp,while_duhsngmesj_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_1©
while/duhsngmesj/mul_1Mul)while/duhsngmesj/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_1¨
while/duhsngmesj/add_2AddV2while/duhsngmesj/split:output:1while/duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_2
while/duhsngmesj/Sigmoid_1Sigmoidwhile/duhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_1
while/duhsngmesj/mul_2Mulwhile/duhsngmesj/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_2
while/duhsngmesj/TanhTanhwhile/duhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh¢
while/duhsngmesj/mul_3Mulwhile/duhsngmesj/Sigmoid:y:0while/duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_3£
while/duhsngmesj/add_3AddV2while/duhsngmesj/mul_2:z:0while/duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_3¯
!while/duhsngmesj/ReadVariableOp_2ReadVariableOp,while_duhsngmesj_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/duhsngmesj/ReadVariableOp_2°
while/duhsngmesj/mul_4Mul)while/duhsngmesj/ReadVariableOp_2:value:0while/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_4¨
while/duhsngmesj/add_4AddV2while/duhsngmesj/split:output:3while/duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/add_4
while/duhsngmesj/Sigmoid_2Sigmoidwhile/duhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Sigmoid_2
while/duhsngmesj/Tanh_1Tanhwhile/duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/Tanh_1¦
while/duhsngmesj/mul_5Mulwhile/duhsngmesj/Sigmoid_2:y:0while/duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/duhsngmesj/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/duhsngmesj/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/duhsngmesj/mul_5:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/duhsngmesj/add_3:z:0(^while/duhsngmesj/BiasAdd/ReadVariableOp'^while/duhsngmesj/MatMul/ReadVariableOp)^while/duhsngmesj/MatMul_1/ReadVariableOp ^while/duhsngmesj/ReadVariableOp"^while/duhsngmesj/ReadVariableOp_1"^while/duhsngmesj/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_duhsngmesj_biasadd_readvariableop_resource2while_duhsngmesj_biasadd_readvariableop_resource_0"h
1while_duhsngmesj_matmul_1_readvariableop_resource3while_duhsngmesj_matmul_1_readvariableop_resource_0"d
/while_duhsngmesj_matmul_readvariableop_resource1while_duhsngmesj_matmul_readvariableop_resource_0"Z
*while_duhsngmesj_readvariableop_1_resource,while_duhsngmesj_readvariableop_1_resource_0"Z
*while_duhsngmesj_readvariableop_2_resource,while_duhsngmesj_readvariableop_2_resource_0"V
(while_duhsngmesj_readvariableop_resource*while_duhsngmesj_readvariableop_resource_0")
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
'while/duhsngmesj/BiasAdd/ReadVariableOp'while/duhsngmesj/BiasAdd/ReadVariableOp2P
&while/duhsngmesj/MatMul/ReadVariableOp&while/duhsngmesj/MatMul/ReadVariableOp2T
(while/duhsngmesj/MatMul_1/ReadVariableOp(while/duhsngmesj/MatMul_1/ReadVariableOp2B
while/duhsngmesj/ReadVariableOpwhile/duhsngmesj/ReadVariableOp2F
!while/duhsngmesj/ReadVariableOp_1!while/duhsngmesj/ReadVariableOp_12F
!while/duhsngmesj/ReadVariableOp_2!while/duhsngmesj/ReadVariableOp_2: 
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
F
ã
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_764824

inputs$
duhsngmesj_764725:	$
duhsngmesj_764727:	  
duhsngmesj_764729:	
duhsngmesj_764731: 
duhsngmesj_764733: 
duhsngmesj_764735: 
identity¢"duhsngmesj/StatefulPartitionedCall¢whileD
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
strided_slice_2Ó
"duhsngmesj/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0duhsngmesj_764725duhsngmesj_764727duhsngmesj_764729duhsngmesj_764731duhsngmesj_764733duhsngmesj_764735*
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
GPU2*0J 8 *O
fJRH
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_7647242$
"duhsngmesj/StatefulPartitionedCall
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
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0duhsngmesj_764725duhsngmesj_764727duhsngmesj_764729duhsngmesj_764731duhsngmesj_764733duhsngmesj_764735*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_764744*
condR
while_cond_764743*Q
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
IdentityIdentitytranspose_1:y:0#^duhsngmesj/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"duhsngmesj/StatefulPartitionedCall"duhsngmesj/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é

+__inference_uazvpibasg_layer_call_fn_769860
inputs_0
unknown:	 
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall¯
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
GPU2*0J 8 *O
fJRH
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_7655822
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
h

F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_767079

inputs<
)duhsngmesj_matmul_readvariableop_resource:	>
+duhsngmesj_matmul_1_readvariableop_resource:	 9
*duhsngmesj_biasadd_readvariableop_resource:	0
"duhsngmesj_readvariableop_resource: 2
$duhsngmesj_readvariableop_1_resource: 2
$duhsngmesj_readvariableop_2_resource: 
identity¢!duhsngmesj/BiasAdd/ReadVariableOp¢ duhsngmesj/MatMul/ReadVariableOp¢"duhsngmesj/MatMul_1/ReadVariableOp¢duhsngmesj/ReadVariableOp¢duhsngmesj/ReadVariableOp_1¢duhsngmesj/ReadVariableOp_2¢whileD
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
 duhsngmesj/MatMul/ReadVariableOpReadVariableOp)duhsngmesj_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 duhsngmesj/MatMul/ReadVariableOp§
duhsngmesj/MatMulMatMulstrided_slice_2:output:0(duhsngmesj/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMulµ
"duhsngmesj/MatMul_1/ReadVariableOpReadVariableOp+duhsngmesj_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"duhsngmesj/MatMul_1/ReadVariableOp£
duhsngmesj/MatMul_1MatMulzeros:output:0*duhsngmesj/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/MatMul_1
duhsngmesj/addAddV2duhsngmesj/MatMul:product:0duhsngmesj/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/add®
!duhsngmesj/BiasAdd/ReadVariableOpReadVariableOp*duhsngmesj_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!duhsngmesj/BiasAdd/ReadVariableOp¥
duhsngmesj/BiasAddBiasAddduhsngmesj/add:z:0)duhsngmesj/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
duhsngmesj/BiasAddz
duhsngmesj/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
duhsngmesj/split/split_dimë
duhsngmesj/splitSplit#duhsngmesj/split/split_dim:output:0duhsngmesj/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
duhsngmesj/split
duhsngmesj/ReadVariableOpReadVariableOp"duhsngmesj_readvariableop_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp
duhsngmesj/mulMul!duhsngmesj/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul
duhsngmesj/add_1AddV2duhsngmesj/split:output:0duhsngmesj/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_1{
duhsngmesj/SigmoidSigmoidduhsngmesj/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid
duhsngmesj/ReadVariableOp_1ReadVariableOp$duhsngmesj_readvariableop_1_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_1
duhsngmesj/mul_1Mul#duhsngmesj/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_1
duhsngmesj/add_2AddV2duhsngmesj/split:output:1duhsngmesj/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_2
duhsngmesj/Sigmoid_1Sigmoidduhsngmesj/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_1
duhsngmesj/mul_2Mulduhsngmesj/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_2w
duhsngmesj/TanhTanhduhsngmesj/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh
duhsngmesj/mul_3Mulduhsngmesj/Sigmoid:y:0duhsngmesj/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_3
duhsngmesj/add_3AddV2duhsngmesj/mul_2:z:0duhsngmesj/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_3
duhsngmesj/ReadVariableOp_2ReadVariableOp$duhsngmesj_readvariableop_2_resource*
_output_shapes
: *
dtype02
duhsngmesj/ReadVariableOp_2
duhsngmesj/mul_4Mul#duhsngmesj/ReadVariableOp_2:value:0duhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_4
duhsngmesj/add_4AddV2duhsngmesj/split:output:3duhsngmesj/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/add_4
duhsngmesj/Sigmoid_2Sigmoidduhsngmesj/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Sigmoid_2v
duhsngmesj/Tanh_1Tanhduhsngmesj/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/Tanh_1
duhsngmesj/mul_5Mulduhsngmesj/Sigmoid_2:y:0duhsngmesj/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
duhsngmesj/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)duhsngmesj_matmul_readvariableop_resource+duhsngmesj_matmul_1_readvariableop_resource*duhsngmesj_biasadd_readvariableop_resource"duhsngmesj_readvariableop_resource$duhsngmesj_readvariableop_1_resource$duhsngmesj_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_766978*
condR
while_cond_766977*Q
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
IdentityIdentitytranspose_1:y:0"^duhsngmesj/BiasAdd/ReadVariableOp!^duhsngmesj/MatMul/ReadVariableOp#^duhsngmesj/MatMul_1/ReadVariableOp^duhsngmesj/ReadVariableOp^duhsngmesj/ReadVariableOp_1^duhsngmesj/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!duhsngmesj/BiasAdd/ReadVariableOp!duhsngmesj/BiasAdd/ReadVariableOp2D
 duhsngmesj/MatMul/ReadVariableOp duhsngmesj/MatMul/ReadVariableOp2H
"duhsngmesj/MatMul_1/ReadVariableOp"duhsngmesj/MatMul_1/ReadVariableOp26
duhsngmesj/ReadVariableOpduhsngmesj/ReadVariableOp2:
duhsngmesj/ReadVariableOp_1duhsngmesj/ReadVariableOp_12:
duhsngmesj/ReadVariableOp_2duhsngmesj/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç)
Å
while_body_764744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_duhsngmesj_764768_0:	,
while_duhsngmesj_764770_0:	 (
while_duhsngmesj_764772_0:	'
while_duhsngmesj_764774_0: '
while_duhsngmesj_764776_0: '
while_duhsngmesj_764778_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_duhsngmesj_764768:	*
while_duhsngmesj_764770:	 &
while_duhsngmesj_764772:	%
while_duhsngmesj_764774: %
while_duhsngmesj_764776: %
while_duhsngmesj_764778: ¢(while/duhsngmesj/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¯
(while/duhsngmesj/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_duhsngmesj_764768_0while_duhsngmesj_764770_0while_duhsngmesj_764772_0while_duhsngmesj_764774_0while_duhsngmesj_764776_0while_duhsngmesj_764778_0*
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
GPU2*0J 8 *O
fJRH
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_7647242*
(while/duhsngmesj/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/duhsngmesj/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/duhsngmesj/StatefulPartitionedCall:output:1)^while/duhsngmesj/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/duhsngmesj/StatefulPartitionedCall:output:2)^while/duhsngmesj/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_duhsngmesj_764768while_duhsngmesj_764768_0"4
while_duhsngmesj_764770while_duhsngmesj_764770_0"4
while_duhsngmesj_764772while_duhsngmesj_764772_0"4
while_duhsngmesj_764774while_duhsngmesj_764774_0"4
while_duhsngmesj_764776while_duhsngmesj_764776_0"4
while_duhsngmesj_764778while_duhsngmesj_764778_0")
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
(while/duhsngmesj/StatefulPartitionedCall(while/duhsngmesj/StatefulPartitionedCall: 
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


å
while_cond_766488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_766488___redundant_placeholder04
0while_while_cond_766488___redundant_placeholder14
0while_while_cond_766488___redundant_placeholder24
0while_while_cond_766488___redundant_placeholder34
0while_while_cond_766488___redundant_placeholder44
0while_while_cond_766488___redundant_placeholder54
0while_while_cond_766488___redundant_placeholder6
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
Ù

+__inference_gdmdltnblf_layer_call_fn_769123

inputs
unknown:	
	unknown_0:	 
	unknown_1:	
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity¢StatefulPartitionedCall±
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
GPU2*0J 8 *O
fJRH
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_7670792
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
F
ã
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_765845

inputs$
gddwjadkgr_765746:	 $
gddwjadkgr_765748:	  
gddwjadkgr_765750:	
gddwjadkgr_765752: 
gddwjadkgr_765754: 
gddwjadkgr_765756: 
identity¢"gddwjadkgr/StatefulPartitionedCall¢whileD
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
strided_slice_2Ó
"gddwjadkgr/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0gddwjadkgr_765746gddwjadkgr_765748gddwjadkgr_765750gddwjadkgr_765752gddwjadkgr_765754gddwjadkgr_765756*
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
GPU2*0J 8 *O
fJRH
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_7656692$
"gddwjadkgr/StatefulPartitionedCall
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
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gddwjadkgr_765746gddwjadkgr_765748gddwjadkgr_765750gddwjadkgr_765752gddwjadkgr_765754gddwjadkgr_765756*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_765765*
condR
while_cond_765764*Q
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
IdentityIdentitystrided_slice_3:output:0#^gddwjadkgr/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"gddwjadkgr/StatefulPartitionedCall"gddwjadkgr/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£h

F__inference_uazvpibasg_layer_call_and_return_conditional_losses_766865

inputs<
)gddwjadkgr_matmul_readvariableop_resource:	 >
+gddwjadkgr_matmul_1_readvariableop_resource:	 9
*gddwjadkgr_biasadd_readvariableop_resource:	0
"gddwjadkgr_readvariableop_resource: 2
$gddwjadkgr_readvariableop_1_resource: 2
$gddwjadkgr_readvariableop_2_resource: 
identity¢!gddwjadkgr/BiasAdd/ReadVariableOp¢ gddwjadkgr/MatMul/ReadVariableOp¢"gddwjadkgr/MatMul_1/ReadVariableOp¢gddwjadkgr/ReadVariableOp¢gddwjadkgr/ReadVariableOp_1¢gddwjadkgr/ReadVariableOp_2¢whileD
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
 gddwjadkgr/MatMul/ReadVariableOpReadVariableOp)gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 gddwjadkgr/MatMul/ReadVariableOp§
gddwjadkgr/MatMulMatMulstrided_slice_2:output:0(gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMulµ
"gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp+gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"gddwjadkgr/MatMul_1/ReadVariableOp£
gddwjadkgr/MatMul_1MatMulzeros:output:0*gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMul_1
gddwjadkgr/addAddV2gddwjadkgr/MatMul:product:0gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/add®
!gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp*gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!gddwjadkgr/BiasAdd/ReadVariableOp¥
gddwjadkgr/BiasAddBiasAddgddwjadkgr/add:z:0)gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/BiasAddz
gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gddwjadkgr/split/split_dimë
gddwjadkgr/splitSplit#gddwjadkgr/split/split_dim:output:0gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gddwjadkgr/split
gddwjadkgr/ReadVariableOpReadVariableOp"gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp
gddwjadkgr/mulMul!gddwjadkgr/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul
gddwjadkgr/add_1AddV2gddwjadkgr/split:output:0gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_1{
gddwjadkgr/SigmoidSigmoidgddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid
gddwjadkgr/ReadVariableOp_1ReadVariableOp$gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_1
gddwjadkgr/mul_1Mul#gddwjadkgr/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_1
gddwjadkgr/add_2AddV2gddwjadkgr/split:output:1gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_2
gddwjadkgr/Sigmoid_1Sigmoidgddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_1
gddwjadkgr/mul_2Mulgddwjadkgr/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_2w
gddwjadkgr/TanhTanhgddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh
gddwjadkgr/mul_3Mulgddwjadkgr/Sigmoid:y:0gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_3
gddwjadkgr/add_3AddV2gddwjadkgr/mul_2:z:0gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_3
gddwjadkgr/ReadVariableOp_2ReadVariableOp$gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_2
gddwjadkgr/mul_4Mul#gddwjadkgr/ReadVariableOp_2:value:0gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_4
gddwjadkgr/add_4AddV2gddwjadkgr/split:output:3gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_4
gddwjadkgr/Sigmoid_2Sigmoidgddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_2v
gddwjadkgr/Tanh_1Tanhgddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh_1
gddwjadkgr/mul_5Mulgddwjadkgr/Sigmoid_2:y:0gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gddwjadkgr_matmul_readvariableop_resource+gddwjadkgr_matmul_1_readvariableop_resource*gddwjadkgr_biasadd_readvariableop_resource"gddwjadkgr_readvariableop_resource$gddwjadkgr_readvariableop_1_resource$gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_766764*
condR
while_cond_766763*Q
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
IdentityIdentitystrided_slice_3:output:0"^gddwjadkgr/BiasAdd/ReadVariableOp!^gddwjadkgr/MatMul/ReadVariableOp#^gddwjadkgr/MatMul_1/ReadVariableOp^gddwjadkgr/ReadVariableOp^gddwjadkgr/ReadVariableOp_1^gddwjadkgr/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gddwjadkgr/BiasAdd/ReadVariableOp!gddwjadkgr/BiasAdd/ReadVariableOp2D
 gddwjadkgr/MatMul/ReadVariableOp gddwjadkgr/MatMul/ReadVariableOp2H
"gddwjadkgr/MatMul_1/ReadVariableOp"gddwjadkgr/MatMul_1/ReadVariableOp26
gddwjadkgr/ReadVariableOpgddwjadkgr/ReadVariableOp2:
gddwjadkgr/ReadVariableOp_1gddwjadkgr/ReadVariableOp_12:
gddwjadkgr/ReadVariableOp_2gddwjadkgr/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ç)
Å
while_body_765765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gddwjadkgr_765789_0:	 ,
while_gddwjadkgr_765791_0:	 (
while_gddwjadkgr_765793_0:	'
while_gddwjadkgr_765795_0: '
while_gddwjadkgr_765797_0: '
while_gddwjadkgr_765799_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gddwjadkgr_765789:	 *
while_gddwjadkgr_765791:	 &
while_gddwjadkgr_765793:	%
while_gddwjadkgr_765795: %
while_gddwjadkgr_765797: %
while_gddwjadkgr_765799: ¢(while/gddwjadkgr/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¯
(while/gddwjadkgr/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_gddwjadkgr_765789_0while_gddwjadkgr_765791_0while_gddwjadkgr_765793_0while_gddwjadkgr_765795_0while_gddwjadkgr_765797_0while_gddwjadkgr_765799_0*
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
GPU2*0J 8 *O
fJRH
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_7656692*
(while/gddwjadkgr/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gddwjadkgr/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gddwjadkgr/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/gddwjadkgr/StatefulPartitionedCall:output:1)^while/gddwjadkgr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/gddwjadkgr/StatefulPartitionedCall:output:2)^while/gddwjadkgr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_gddwjadkgr_765789while_gddwjadkgr_765789_0"4
while_gddwjadkgr_765791while_gddwjadkgr_765791_0"4
while_gddwjadkgr_765793while_gddwjadkgr_765793_0"4
while_gddwjadkgr_765795while_gddwjadkgr_765795_0"4
while_gddwjadkgr_765797while_gddwjadkgr_765797_0"4
while_gddwjadkgr_765799while_gddwjadkgr_765799_0")
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
(while/gddwjadkgr/StatefulPartitionedCall(while/gddwjadkgr/StatefulPartitionedCall: 
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
Ùh

F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769483
inputs_0<
)gddwjadkgr_matmul_readvariableop_resource:	 >
+gddwjadkgr_matmul_1_readvariableop_resource:	 9
*gddwjadkgr_biasadd_readvariableop_resource:	0
"gddwjadkgr_readvariableop_resource: 2
$gddwjadkgr_readvariableop_1_resource: 2
$gddwjadkgr_readvariableop_2_resource: 
identity¢!gddwjadkgr/BiasAdd/ReadVariableOp¢ gddwjadkgr/MatMul/ReadVariableOp¢"gddwjadkgr/MatMul_1/ReadVariableOp¢gddwjadkgr/ReadVariableOp¢gddwjadkgr/ReadVariableOp_1¢gddwjadkgr/ReadVariableOp_2¢whileF
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
 gddwjadkgr/MatMul/ReadVariableOpReadVariableOp)gddwjadkgr_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 gddwjadkgr/MatMul/ReadVariableOp§
gddwjadkgr/MatMulMatMulstrided_slice_2:output:0(gddwjadkgr/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMulµ
"gddwjadkgr/MatMul_1/ReadVariableOpReadVariableOp+gddwjadkgr_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"gddwjadkgr/MatMul_1/ReadVariableOp£
gddwjadkgr/MatMul_1MatMulzeros:output:0*gddwjadkgr/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/MatMul_1
gddwjadkgr/addAddV2gddwjadkgr/MatMul:product:0gddwjadkgr/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/add®
!gddwjadkgr/BiasAdd/ReadVariableOpReadVariableOp*gddwjadkgr_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!gddwjadkgr/BiasAdd/ReadVariableOp¥
gddwjadkgr/BiasAddBiasAddgddwjadkgr/add:z:0)gddwjadkgr/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gddwjadkgr/BiasAddz
gddwjadkgr/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
gddwjadkgr/split/split_dimë
gddwjadkgr/splitSplit#gddwjadkgr/split/split_dim:output:0gddwjadkgr/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gddwjadkgr/split
gddwjadkgr/ReadVariableOpReadVariableOp"gddwjadkgr_readvariableop_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp
gddwjadkgr/mulMul!gddwjadkgr/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul
gddwjadkgr/add_1AddV2gddwjadkgr/split:output:0gddwjadkgr/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_1{
gddwjadkgr/SigmoidSigmoidgddwjadkgr/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid
gddwjadkgr/ReadVariableOp_1ReadVariableOp$gddwjadkgr_readvariableop_1_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_1
gddwjadkgr/mul_1Mul#gddwjadkgr/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_1
gddwjadkgr/add_2AddV2gddwjadkgr/split:output:1gddwjadkgr/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_2
gddwjadkgr/Sigmoid_1Sigmoidgddwjadkgr/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_1
gddwjadkgr/mul_2Mulgddwjadkgr/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_2w
gddwjadkgr/TanhTanhgddwjadkgr/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh
gddwjadkgr/mul_3Mulgddwjadkgr/Sigmoid:y:0gddwjadkgr/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_3
gddwjadkgr/add_3AddV2gddwjadkgr/mul_2:z:0gddwjadkgr/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_3
gddwjadkgr/ReadVariableOp_2ReadVariableOp$gddwjadkgr_readvariableop_2_resource*
_output_shapes
: *
dtype02
gddwjadkgr/ReadVariableOp_2
gddwjadkgr/mul_4Mul#gddwjadkgr/ReadVariableOp_2:value:0gddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_4
gddwjadkgr/add_4AddV2gddwjadkgr/split:output:3gddwjadkgr/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/add_4
gddwjadkgr/Sigmoid_2Sigmoidgddwjadkgr/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Sigmoid_2v
gddwjadkgr/Tanh_1Tanhgddwjadkgr/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/Tanh_1
gddwjadkgr/mul_5Mulgddwjadkgr/Sigmoid_2:y:0gddwjadkgr/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gddwjadkgr/mul_5
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
while/loop_counterê
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)gddwjadkgr_matmul_readvariableop_resource+gddwjadkgr_matmul_1_readvariableop_resource*gddwjadkgr_biasadd_readvariableop_resource"gddwjadkgr_readvariableop_resource$gddwjadkgr_readvariableop_1_resource$gddwjadkgr_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_769382*
condR
while_cond_769381*Q
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
IdentityIdentitystrided_slice_3:output:0"^gddwjadkgr/BiasAdd/ReadVariableOp!^gddwjadkgr/MatMul/ReadVariableOp#^gddwjadkgr/MatMul_1/ReadVariableOp^gddwjadkgr/ReadVariableOp^gddwjadkgr/ReadVariableOp_1^gddwjadkgr/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!gddwjadkgr/BiasAdd/ReadVariableOp!gddwjadkgr/BiasAdd/ReadVariableOp2D
 gddwjadkgr/MatMul/ReadVariableOp gddwjadkgr/MatMul/ReadVariableOp2H
"gddwjadkgr/MatMul_1/ReadVariableOp"gddwjadkgr/MatMul_1/ReadVariableOp26
gddwjadkgr/ReadVariableOpgddwjadkgr/ReadVariableOp2:
gddwjadkgr/ReadVariableOp_1gddwjadkgr/ReadVariableOp_12:
gddwjadkgr/ReadVariableOp_2gddwjadkgr/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


å
while_cond_764743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_764743___redundant_placeholder04
0while_while_cond_764743___redundant_placeholder14
0while_while_cond_764743___redundant_placeholder24
0while_while_cond_764743___redundant_placeholder34
0while_while_cond_764743___redundant_placeholder44
0while_while_cond_764743___redundant_placeholder54
0while_while_cond_764743___redundant_placeholder6
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
¸'
´
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_770018

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
¨0
»
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_766197

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

b
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_766216

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


å
while_cond_766295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_766295___redundant_placeholder04
0while_while_cond_766295___redundant_placeholder14
0while_while_cond_766295___redundant_placeholder24
0while_while_cond_766295___redundant_placeholder34
0while_while_cond_766295___redundant_placeholder44
0while_while_cond_766295___redundant_placeholder54
0while_while_cond_766295___redundant_placeholder6
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
Ç)
Å
while_body_765007
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_duhsngmesj_765031_0:	,
while_duhsngmesj_765033_0:	 (
while_duhsngmesj_765035_0:	'
while_duhsngmesj_765037_0: '
while_duhsngmesj_765039_0: '
while_duhsngmesj_765041_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_duhsngmesj_765031:	*
while_duhsngmesj_765033:	 &
while_duhsngmesj_765035:	%
while_duhsngmesj_765037: %
while_duhsngmesj_765039: %
while_duhsngmesj_765041: ¢(while/duhsngmesj/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItem¯
(while/duhsngmesj/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_duhsngmesj_765031_0while_duhsngmesj_765033_0while_duhsngmesj_765035_0while_duhsngmesj_765037_0while_duhsngmesj_765039_0while_duhsngmesj_765041_0*
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
GPU2*0J 8 *O
fJRH
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_7649112*
(while/duhsngmesj/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/duhsngmesj/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/duhsngmesj/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/duhsngmesj/StatefulPartitionedCall:output:1)^while/duhsngmesj/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/duhsngmesj/StatefulPartitionedCall:output:2)^while/duhsngmesj/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_duhsngmesj_765031while_duhsngmesj_765031_0"4
while_duhsngmesj_765033while_duhsngmesj_765033_0"4
while_duhsngmesj_765035while_duhsngmesj_765035_0"4
while_duhsngmesj_765037while_duhsngmesj_765037_0"4
while_duhsngmesj_765039while_duhsngmesj_765039_0"4
while_duhsngmesj_765041while_duhsngmesj_765041_0")
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
(while/duhsngmesj/StatefulPartitionedCall(while/duhsngmesj/StatefulPartitionedCall: 
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

unopekwlxv;
serving_default_unopekwlxv:0ÿÿÿÿÿÿÿÿÿ>

supobtndkp0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:º
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
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "unopekwlxv"}}, {"class_name": "Conv1D", "config": {"name": "ggbvrcpxtu", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "vtqejmjhbd", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "gdmdltnblf", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "duhsngmesj", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "uazvpibasg", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "gddwjadkgr", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "supobtndkp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "unopekwlxv"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "unopekwlxv"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "ggbvrcpxtu", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "vtqejmjhbd", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "gdmdltnblf", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "duhsngmesj", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "uazvpibasg", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "gddwjadkgr", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "supobtndkp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
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
{"name": "ggbvrcpxtu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "ggbvrcpxtu", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"name": "vtqejmjhbd", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "vtqejmjhbd", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
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
_tf_keras_rnn_layerä{"name": "gdmdltnblf", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "gdmdltnblf", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "duhsngmesj", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
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
_tf_keras_rnn_layerê{"name": "uazvpibasg", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "uazvpibasg", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "gddwjadkgr", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "supobtndkp", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "supobtndkp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
':%2ggbvrcpxtu/kernel
:2ggbvrcpxtu/bias
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
_tf_keras_layer¼{"name": "duhsngmesj", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "duhsngmesj", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
_tf_keras_layerÀ{"name": "gddwjadkgr", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "gddwjadkgr", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
#:! 2supobtndkp/kernel
:2supobtndkp/bias
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
/:-	2gdmdltnblf/duhsngmesj/kernel
9:7	 2&gdmdltnblf/duhsngmesj/recurrent_kernel
):'2gdmdltnblf/duhsngmesj/bias
?:= 21gdmdltnblf/duhsngmesj/input_gate_peephole_weights
@:> 22gdmdltnblf/duhsngmesj/forget_gate_peephole_weights
@:> 22gdmdltnblf/duhsngmesj/output_gate_peephole_weights
/:-	 2uazvpibasg/gddwjadkgr/kernel
9:7	 2&uazvpibasg/gddwjadkgr/recurrent_kernel
):'2uazvpibasg/gddwjadkgr/bias
?:= 21uazvpibasg/gddwjadkgr/input_gate_peephole_weights
@:> 22uazvpibasg/gddwjadkgr/forget_gate_peephole_weights
@:> 22uazvpibasg/gddwjadkgr/output_gate_peephole_weights
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
1:/2RMSprop/ggbvrcpxtu/kernel/rms
':%2RMSprop/ggbvrcpxtu/bias/rms
-:+ 2RMSprop/supobtndkp/kernel/rms
':%2RMSprop/supobtndkp/bias/rms
9:7	2(RMSprop/gdmdltnblf/duhsngmesj/kernel/rms
C:A	 22RMSprop/gdmdltnblf/duhsngmesj/recurrent_kernel/rms
3:12&RMSprop/gdmdltnblf/duhsngmesj/bias/rms
I:G 2=RMSprop/gdmdltnblf/duhsngmesj/input_gate_peephole_weights/rms
J:H 2>RMSprop/gdmdltnblf/duhsngmesj/forget_gate_peephole_weights/rms
J:H 2>RMSprop/gdmdltnblf/duhsngmesj/output_gate_peephole_weights/rms
9:7	 2(RMSprop/uazvpibasg/gddwjadkgr/kernel/rms
C:A	 22RMSprop/uazvpibasg/gddwjadkgr/recurrent_kernel/rms
3:12&RMSprop/uazvpibasg/gddwjadkgr/bias/rms
I:G 2=RMSprop/uazvpibasg/gddwjadkgr/input_gate_peephole_weights/rms
J:H 2>RMSprop/uazvpibasg/gddwjadkgr/forget_gate_peephole_weights/rms
J:H 2>RMSprop/uazvpibasg/gddwjadkgr/output_gate_peephole_weights/rms
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_767793
F__inference_sequential_layer_call_and_return_conditional_losses_768197
F__inference_sequential_layer_call_and_return_conditional_losses_767303
F__inference_sequential_layer_call_and_return_conditional_losses_767344À
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
ú2÷
+__inference_sequential_layer_call_fn_766656
+__inference_sequential_layer_call_fn_768234
+__inference_sequential_layer_call_fn_768271
+__inference_sequential_layer_call_fn_767262À
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
ê2ç
!__inference__wrapped_model_764637Á
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

unopekwlxvÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_768308¢
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
Õ2Ò
+__inference_ggbvrcpxtu_layer_call_fn_768317¢
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
ð2í
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_768330¢
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
Õ2Ò
+__inference_vtqejmjhbd_layer_call_fn_768335¢
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
2
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768515
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768695
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768875
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_769055æ
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
 2
+__inference_gdmdltnblf_layer_call_fn_769072
+__inference_gdmdltnblf_layer_call_fn_769089
+__inference_gdmdltnblf_layer_call_fn_769106
+__inference_gdmdltnblf_layer_call_fn_769123æ
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
2
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769303
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769483
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769663
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769843æ
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
 2
+__inference_uazvpibasg_layer_call_fn_769860
+__inference_uazvpibasg_layer_call_fn_769877
+__inference_uazvpibasg_layer_call_fn_769894
+__inference_uazvpibasg_layer_call_fn_769911æ
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
ð2í
F__inference_supobtndkp_layer_call_and_return_conditional_losses_769921¢
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
Õ2Ò
+__inference_supobtndkp_layer_call_fn_769930¢
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
ÎBË
$__inference_signature_wrapper_767389
unopekwlxv"
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
Ô2Ñ
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_769974
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_770018¾
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
2
+__inference_duhsngmesj_layer_call_fn_770041
+__inference_duhsngmesj_layer_call_fn_770064¾
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
Ô2Ñ
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_770108
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_770152¾
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
2
+__inference_gddwjadkgr_layer_call_fn_770175
+__inference_gddwjadkgr_layer_call_fn_770198¾
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
 ®
!__inference__wrapped_model_764637-./012345678"#;¢8
1¢.
,)

unopekwlxvÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

supobtndkp$!

supobtndkpÿÿÿÿÿÿÿÿÿË
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_769974-./012¢}
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
 Ë
F__inference_duhsngmesj_layer_call_and_return_conditional_losses_770018-./012¢}
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
  
+__inference_duhsngmesj_layer_call_fn_770041ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ  
+__inference_duhsngmesj_layer_call_fn_770064ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ë
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_770108345678¢}
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
 Ë
F__inference_gddwjadkgr_layer_call_and_return_conditional_losses_770152345678¢}
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
  
+__inference_gddwjadkgr_layer_call_fn_770175ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ  
+__inference_gddwjadkgr_layer_call_fn_770198ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ü
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768515-./012S¢P
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
 Ü
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768695-./012S¢P
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
 Â
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_768875x-./012C¢@
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
 Â
F__inference_gdmdltnblf_layer_call_and_return_conditional_losses_769055x-./012C¢@
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
 ´
+__inference_gdmdltnblf_layer_call_fn_769072-./012S¢P
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
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ´
+__inference_gdmdltnblf_layer_call_fn_769089-./012S¢P
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
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
+__inference_gdmdltnblf_layer_call_fn_769106k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_gdmdltnblf_layer_call_fn_769123k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ ¶
F__inference_ggbvrcpxtu_layer_call_and_return_conditional_losses_768308l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_ggbvrcpxtu_layer_call_fn_768317_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÈ
F__inference_sequential_layer_call_and_return_conditional_losses_767303~-./012345678"#C¢@
9¢6
,)

unopekwlxvÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
F__inference_sequential_layer_call_and_return_conditional_losses_767344~-./012345678"#C¢@
9¢6
,)

unopekwlxvÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_767793z-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_768197z-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
+__inference_sequential_layer_call_fn_766656q-./012345678"#C¢@
9¢6
,)

unopekwlxvÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_sequential_layer_call_fn_767262q-./012345678"#C¢@
9¢6
,)

unopekwlxvÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_768234m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_768271m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
$__inference_signature_wrapper_767389-./012345678"#I¢F
¢ 
?ª<
:

unopekwlxv,)

unopekwlxvÿÿÿÿÿÿÿÿÿ"7ª4
2

supobtndkp$!

supobtndkpÿÿÿÿÿÿÿÿÿ¦
F__inference_supobtndkp_layer_call_and_return_conditional_losses_769921\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_supobtndkp_layer_call_fn_769930O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÏ
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769303345678S¢P
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
 Ï
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769483345678S¢P
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
 ¾
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769663t345678C¢@
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
 ¾
F__inference_uazvpibasg_layer_call_and_return_conditional_losses_769843t345678C¢@
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
 ¦
+__inference_uazvpibasg_layer_call_fn_769860w345678S¢P
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
ª "ÿÿÿÿÿÿÿÿÿ ¦
+__inference_uazvpibasg_layer_call_fn_769877w345678S¢P
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
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_uazvpibasg_layer_call_fn_769894g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_uazvpibasg_layer_call_fn_769911g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ ®
F__inference_vtqejmjhbd_layer_call_and_return_conditional_losses_768330d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_vtqejmjhbd_layer_call_fn_768335W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ