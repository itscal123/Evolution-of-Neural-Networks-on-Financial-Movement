2
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
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718â¨/

iigfihrkup/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameiigfihrkup/kernel
{
%iigfihrkup/kernel/Read/ReadVariableOpReadVariableOpiigfihrkup/kernel*"
_output_shapes
:*
dtype0
v
iigfihrkup/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameiigfihrkup/bias
o
#iigfihrkup/bias/Read/ReadVariableOpReadVariableOpiigfihrkup/bias*
_output_shapes
:*
dtype0
~
iktogmlrmp/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameiktogmlrmp/kernel
w
%iktogmlrmp/kernel/Read/ReadVariableOpReadVariableOpiktogmlrmp/kernel*
_output_shapes

: *
dtype0
v
iktogmlrmp/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameiktogmlrmp/bias
o
#iktogmlrmp/bias/Read/ReadVariableOpReadVariableOpiktogmlrmp/bias*
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
omilqdycns/ddlymsxapn/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameomilqdycns/ddlymsxapn/kernel

0omilqdycns/ddlymsxapn/kernel/Read/ReadVariableOpReadVariableOpomilqdycns/ddlymsxapn/kernel*
_output_shapes
:	*
dtype0
©
&omilqdycns/ddlymsxapn/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&omilqdycns/ddlymsxapn/recurrent_kernel
¢
:omilqdycns/ddlymsxapn/recurrent_kernel/Read/ReadVariableOpReadVariableOp&omilqdycns/ddlymsxapn/recurrent_kernel*
_output_shapes
:	 *
dtype0

omilqdycns/ddlymsxapn/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameomilqdycns/ddlymsxapn/bias

.omilqdycns/ddlymsxapn/bias/Read/ReadVariableOpReadVariableOpomilqdycns/ddlymsxapn/bias*
_output_shapes	
:*
dtype0
º
1omilqdycns/ddlymsxapn/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31omilqdycns/ddlymsxapn/input_gate_peephole_weights
³
Eomilqdycns/ddlymsxapn/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1omilqdycns/ddlymsxapn/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2omilqdycns/ddlymsxapn/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42omilqdycns/ddlymsxapn/forget_gate_peephole_weights
µ
Fomilqdycns/ddlymsxapn/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2omilqdycns/ddlymsxapn/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2omilqdycns/ddlymsxapn/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42omilqdycns/ddlymsxapn/output_gate_peephole_weights
µ
Fomilqdycns/ddlymsxapn/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2omilqdycns/ddlymsxapn/output_gate_peephole_weights*
_output_shapes
: *
dtype0

vlxoswgdqw/vdaevhnmja/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namevlxoswgdqw/vdaevhnmja/kernel

0vlxoswgdqw/vdaevhnmja/kernel/Read/ReadVariableOpReadVariableOpvlxoswgdqw/vdaevhnmja/kernel*
_output_shapes
:	 *
dtype0
©
&vlxoswgdqw/vdaevhnmja/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&vlxoswgdqw/vdaevhnmja/recurrent_kernel
¢
:vlxoswgdqw/vdaevhnmja/recurrent_kernel/Read/ReadVariableOpReadVariableOp&vlxoswgdqw/vdaevhnmja/recurrent_kernel*
_output_shapes
:	 *
dtype0

vlxoswgdqw/vdaevhnmja/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namevlxoswgdqw/vdaevhnmja/bias

.vlxoswgdqw/vdaevhnmja/bias/Read/ReadVariableOpReadVariableOpvlxoswgdqw/vdaevhnmja/bias*
_output_shapes	
:*
dtype0
º
1vlxoswgdqw/vdaevhnmja/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights
³
Evlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights
µ
Fvlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2vlxoswgdqw/vdaevhnmja/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights
µ
Fvlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights*
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
RMSprop/iigfihrkup/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/iigfihrkup/kernel/rms

1RMSprop/iigfihrkup/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/iigfihrkup/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/iigfihrkup/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/iigfihrkup/bias/rms

/RMSprop/iigfihrkup/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/iigfihrkup/bias/rms*
_output_shapes
:*
dtype0

RMSprop/iktogmlrmp/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/iktogmlrmp/kernel/rms

1RMSprop/iktogmlrmp/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/iktogmlrmp/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/iktogmlrmp/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/iktogmlrmp/bias/rms

/RMSprop/iktogmlrmp/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/iktogmlrmp/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/omilqdycns/ddlymsxapn/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/omilqdycns/ddlymsxapn/kernel/rms
¦
<RMSprop/omilqdycns/ddlymsxapn/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/omilqdycns/ddlymsxapn/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rms
º
FRMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/omilqdycns/ddlymsxapn/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/omilqdycns/ddlymsxapn/bias/rms

:RMSprop/omilqdycns/ddlymsxapn/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/omilqdycns/ddlymsxapn/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rms
Ë
QRMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rms
Í
RRMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rms
Í
RRMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rms
¦
<RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rms
º
FRMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/vlxoswgdqw/vdaevhnmja/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/vlxoswgdqw/vdaevhnmja/bias/rms

:RMSprop/vlxoswgdqw/vdaevhnmja/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/vlxoswgdqw/vdaevhnmja/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rms
Ë
QRMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rms
Í
RRMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rms
Í
RRMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rms*
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
9metrics
	variables
:layer_regularization_losses
;layer_metrics

<layers
regularization_losses
=non_trainable_variables
	trainable_variables
 
][
VARIABLE_VALUEiigfihrkup/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEiigfihrkup/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
>metrics
	variables
?layer_regularization_losses
@layer_metrics

Alayers
regularization_losses
Bnon_trainable_variables
trainable_variables
 
 
 
­
Cmetrics
	variables
Dlayer_regularization_losses
Elayer_metrics

Flayers
regularization_losses
Gnon_trainable_variables
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
Mmetrics
	variables

Nstates
Olayer_regularization_losses
Player_metrics

Qlayers
regularization_losses
Rnon_trainable_variables
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
Xmetrics
	variables

Ystates
Zlayer_regularization_losses
[layer_metrics

\layers
regularization_losses
]non_trainable_variables
 trainable_variables
][
VARIABLE_VALUEiktogmlrmp/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEiktogmlrmp/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
^metrics
$	variables
_layer_regularization_losses
`layer_metrics

alayers
%regularization_losses
bnon_trainable_variables
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
XV
VARIABLE_VALUEomilqdycns/ddlymsxapn/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&omilqdycns/ddlymsxapn/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEomilqdycns/ddlymsxapn/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1omilqdycns/ddlymsxapn/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2omilqdycns/ddlymsxapn/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2omilqdycns/ddlymsxapn/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEvlxoswgdqw/vdaevhnmja/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&vlxoswgdqw/vdaevhnmja/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEvlxoswgdqw/vdaevhnmja/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE

c0
 
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
dmetrics
I	variables
elayer_regularization_losses
flayer_metrics

glayers
Jregularization_losses
hnon_trainable_variables
Ktrainable_variables
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
 
*
30
41
52
63
74
85
­
imetrics
T	variables
jlayer_regularization_losses
klayer_metrics

llayers
Uregularization_losses
mnon_trainable_variables
Vtrainable_variables
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
VARIABLE_VALUERMSprop/iigfihrkup/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/iigfihrkup/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/iktogmlrmp/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/iktogmlrmp/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/omilqdycns/ddlymsxapn/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/omilqdycns/ddlymsxapn/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/vlxoswgdqw/vdaevhnmja/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_ahzwxypkrhPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_ahzwxypkrhiigfihrkup/kerneliigfihrkup/biasomilqdycns/ddlymsxapn/kernel&omilqdycns/ddlymsxapn/recurrent_kernelomilqdycns/ddlymsxapn/bias1omilqdycns/ddlymsxapn/input_gate_peephole_weights2omilqdycns/ddlymsxapn/forget_gate_peephole_weights2omilqdycns/ddlymsxapn/output_gate_peephole_weightsvlxoswgdqw/vdaevhnmja/kernel&vlxoswgdqw/vdaevhnmja/recurrent_kernelvlxoswgdqw/vdaevhnmja/bias1vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights2vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights2vlxoswgdqw/vdaevhnmja/output_gate_peephole_weightsiktogmlrmp/kerneliktogmlrmp/bias*
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
$__inference_signature_wrapper_223138
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%iigfihrkup/kernel/Read/ReadVariableOp#iigfihrkup/bias/Read/ReadVariableOp%iktogmlrmp/kernel/Read/ReadVariableOp#iktogmlrmp/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0omilqdycns/ddlymsxapn/kernel/Read/ReadVariableOp:omilqdycns/ddlymsxapn/recurrent_kernel/Read/ReadVariableOp.omilqdycns/ddlymsxapn/bias/Read/ReadVariableOpEomilqdycns/ddlymsxapn/input_gate_peephole_weights/Read/ReadVariableOpFomilqdycns/ddlymsxapn/forget_gate_peephole_weights/Read/ReadVariableOpFomilqdycns/ddlymsxapn/output_gate_peephole_weights/Read/ReadVariableOp0vlxoswgdqw/vdaevhnmja/kernel/Read/ReadVariableOp:vlxoswgdqw/vdaevhnmja/recurrent_kernel/Read/ReadVariableOp.vlxoswgdqw/vdaevhnmja/bias/Read/ReadVariableOpEvlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/Read/ReadVariableOpFvlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/Read/ReadVariableOpFvlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/iigfihrkup/kernel/rms/Read/ReadVariableOp/RMSprop/iigfihrkup/bias/rms/Read/ReadVariableOp1RMSprop/iktogmlrmp/kernel/rms/Read/ReadVariableOp/RMSprop/iktogmlrmp/bias/rms/Read/ReadVariableOp<RMSprop/omilqdycns/ddlymsxapn/kernel/rms/Read/ReadVariableOpFRMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/omilqdycns/ddlymsxapn/bias/rms/Read/ReadVariableOpQRMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rms/Read/ReadVariableOpFRMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/vlxoswgdqw/vdaevhnmja/bias/rms/Read/ReadVariableOpQRMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
__inference__traced_save_226087
å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameiigfihrkup/kerneliigfihrkup/biasiktogmlrmp/kerneliktogmlrmp/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoomilqdycns/ddlymsxapn/kernel&omilqdycns/ddlymsxapn/recurrent_kernelomilqdycns/ddlymsxapn/bias1omilqdycns/ddlymsxapn/input_gate_peephole_weights2omilqdycns/ddlymsxapn/forget_gate_peephole_weights2omilqdycns/ddlymsxapn/output_gate_peephole_weightsvlxoswgdqw/vdaevhnmja/kernel&vlxoswgdqw/vdaevhnmja/recurrent_kernelvlxoswgdqw/vdaevhnmja/bias1vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights2vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights2vlxoswgdqw/vdaevhnmja/output_gate_peephole_weightstotalcountRMSprop/iigfihrkup/kernel/rmsRMSprop/iigfihrkup/bias/rmsRMSprop/iktogmlrmp/kernel/rmsRMSprop/iktogmlrmp/bias/rms(RMSprop/omilqdycns/ddlymsxapn/kernel/rms2RMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rms&RMSprop/omilqdycns/ddlymsxapn/bias/rms=RMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rms>RMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rms>RMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rms(RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rms2RMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rms&RMSprop/vlxoswgdqw/vdaevhnmja/bias/rms=RMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rms>RMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rms>RMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rms*3
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
"__inference__traced_restore_226214¥Û-
£h

F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_222614

inputs<
)vdaevhnmja_matmul_readvariableop_resource:	 >
+vdaevhnmja_matmul_1_readvariableop_resource:	 9
*vdaevhnmja_biasadd_readvariableop_resource:	0
"vdaevhnmja_readvariableop_resource: 2
$vdaevhnmja_readvariableop_1_resource: 2
$vdaevhnmja_readvariableop_2_resource: 
identity¢!vdaevhnmja/BiasAdd/ReadVariableOp¢ vdaevhnmja/MatMul/ReadVariableOp¢"vdaevhnmja/MatMul_1/ReadVariableOp¢vdaevhnmja/ReadVariableOp¢vdaevhnmja/ReadVariableOp_1¢vdaevhnmja/ReadVariableOp_2¢whileD
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
 vdaevhnmja/MatMul/ReadVariableOpReadVariableOp)vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 vdaevhnmja/MatMul/ReadVariableOp§
vdaevhnmja/MatMulMatMulstrided_slice_2:output:0(vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMulµ
"vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp+vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"vdaevhnmja/MatMul_1/ReadVariableOp£
vdaevhnmja/MatMul_1MatMulzeros:output:0*vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMul_1
vdaevhnmja/addAddV2vdaevhnmja/MatMul:product:0vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/add®
!vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp*vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!vdaevhnmja/BiasAdd/ReadVariableOp¥
vdaevhnmja/BiasAddBiasAddvdaevhnmja/add:z:0)vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/BiasAddz
vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vdaevhnmja/split/split_dimë
vdaevhnmja/splitSplit#vdaevhnmja/split/split_dim:output:0vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vdaevhnmja/split
vdaevhnmja/ReadVariableOpReadVariableOp"vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp
vdaevhnmja/mulMul!vdaevhnmja/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul
vdaevhnmja/add_1AddV2vdaevhnmja/split:output:0vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_1{
vdaevhnmja/SigmoidSigmoidvdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid
vdaevhnmja/ReadVariableOp_1ReadVariableOp$vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_1
vdaevhnmja/mul_1Mul#vdaevhnmja/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_1
vdaevhnmja/add_2AddV2vdaevhnmja/split:output:1vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_2
vdaevhnmja/Sigmoid_1Sigmoidvdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_1
vdaevhnmja/mul_2Mulvdaevhnmja/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_2w
vdaevhnmja/TanhTanhvdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh
vdaevhnmja/mul_3Mulvdaevhnmja/Sigmoid:y:0vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_3
vdaevhnmja/add_3AddV2vdaevhnmja/mul_2:z:0vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_3
vdaevhnmja/ReadVariableOp_2ReadVariableOp$vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_2
vdaevhnmja/mul_4Mul#vdaevhnmja/ReadVariableOp_2:value:0vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_4
vdaevhnmja/add_4AddV2vdaevhnmja/split:output:3vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_4
vdaevhnmja/Sigmoid_2Sigmoidvdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_2v
vdaevhnmja/Tanh_1Tanhvdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh_1
vdaevhnmja/mul_5Mulvdaevhnmja/Sigmoid_2:y:0vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)vdaevhnmja_matmul_readvariableop_resource+vdaevhnmja_matmul_1_readvariableop_resource*vdaevhnmja_biasadd_readvariableop_resource"vdaevhnmja_readvariableop_resource$vdaevhnmja_readvariableop_1_resource$vdaevhnmja_readvariableop_2_resource*
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
while_body_222513*
condR
while_cond_222512*Q
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
IdentityIdentitystrided_slice_3:output:0"^vdaevhnmja/BiasAdd/ReadVariableOp!^vdaevhnmja/MatMul/ReadVariableOp#^vdaevhnmja/MatMul_1/ReadVariableOp^vdaevhnmja/ReadVariableOp^vdaevhnmja/ReadVariableOp_1^vdaevhnmja/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!vdaevhnmja/BiasAdd/ReadVariableOp!vdaevhnmja/BiasAdd/ReadVariableOp2D
 vdaevhnmja/MatMul/ReadVariableOp vdaevhnmja/MatMul/ReadVariableOp2H
"vdaevhnmja/MatMul_1/ReadVariableOp"vdaevhnmja/MatMul_1/ReadVariableOp26
vdaevhnmja/ReadVariableOpvdaevhnmja/ReadVariableOp2:
vdaevhnmja/ReadVariableOp_1vdaevhnmja/ReadVariableOp_12:
vdaevhnmja/ReadVariableOp_2vdaevhnmja/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
p
É
omilqdycns_while_body_2237372
.omilqdycns_while_omilqdycns_while_loop_counter8
4omilqdycns_while_omilqdycns_while_maximum_iterations 
omilqdycns_while_placeholder"
omilqdycns_while_placeholder_1"
omilqdycns_while_placeholder_2"
omilqdycns_while_placeholder_31
-omilqdycns_while_omilqdycns_strided_slice_1_0m
iomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0O
<omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0:	Q
>omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 L
=omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0:	C
5omilqdycns_while_ddlymsxapn_readvariableop_resource_0: E
7omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0: E
7omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0: 
omilqdycns_while_identity
omilqdycns_while_identity_1
omilqdycns_while_identity_2
omilqdycns_while_identity_3
omilqdycns_while_identity_4
omilqdycns_while_identity_5/
+omilqdycns_while_omilqdycns_strided_slice_1k
gomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensorM
:omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource:	O
<omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource:	 J
;omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource:	A
3omilqdycns_while_ddlymsxapn_readvariableop_resource: C
5omilqdycns_while_ddlymsxapn_readvariableop_1_resource: C
5omilqdycns_while_ddlymsxapn_readvariableop_2_resource: ¢2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp¢1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp¢3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp¢*omilqdycns/while/ddlymsxapn/ReadVariableOp¢,omilqdycns/while/ddlymsxapn/ReadVariableOp_1¢,omilqdycns/while/ddlymsxapn/ReadVariableOp_2Ù
Bomilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bomilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shape
4omilqdycns/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0omilqdycns_while_placeholderKomilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4omilqdycns/while/TensorArrayV2Read/TensorListGetItemä
1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp<omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOpý
"omilqdycns/while/ddlymsxapn/MatMulMatMul;omilqdycns/while/TensorArrayV2Read/TensorListGetItem:item:09omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"omilqdycns/while/ddlymsxapn/MatMulê
3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp>omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOpæ
$omilqdycns/while/ddlymsxapn/MatMul_1MatMulomilqdycns_while_placeholder_2;omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$omilqdycns/while/ddlymsxapn/MatMul_1Ü
omilqdycns/while/ddlymsxapn/addAddV2,omilqdycns/while/ddlymsxapn/MatMul:product:0.omilqdycns/while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
omilqdycns/while/ddlymsxapn/addã
2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp=omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOpé
#omilqdycns/while/ddlymsxapn/BiasAddBiasAdd#omilqdycns/while/ddlymsxapn/add:z:0:omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#omilqdycns/while/ddlymsxapn/BiasAdd
+omilqdycns/while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+omilqdycns/while/ddlymsxapn/split/split_dim¯
!omilqdycns/while/ddlymsxapn/splitSplit4omilqdycns/while/ddlymsxapn/split/split_dim:output:0,omilqdycns/while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!omilqdycns/while/ddlymsxapn/splitÊ
*omilqdycns/while/ddlymsxapn/ReadVariableOpReadVariableOp5omilqdycns_while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*omilqdycns/while/ddlymsxapn/ReadVariableOpÏ
omilqdycns/while/ddlymsxapn/mulMul2omilqdycns/while/ddlymsxapn/ReadVariableOp:value:0omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
omilqdycns/while/ddlymsxapn/mulÒ
!omilqdycns/while/ddlymsxapn/add_1AddV2*omilqdycns/while/ddlymsxapn/split:output:0#omilqdycns/while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_1®
#omilqdycns/while/ddlymsxapn/SigmoidSigmoid%omilqdycns/while/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#omilqdycns/while/ddlymsxapn/SigmoidÐ
,omilqdycns/while/ddlymsxapn/ReadVariableOp_1ReadVariableOp7omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,omilqdycns/while/ddlymsxapn/ReadVariableOp_1Õ
!omilqdycns/while/ddlymsxapn/mul_1Mul4omilqdycns/while/ddlymsxapn/ReadVariableOp_1:value:0omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_1Ô
!omilqdycns/while/ddlymsxapn/add_2AddV2*omilqdycns/while/ddlymsxapn/split:output:1%omilqdycns/while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_2²
%omilqdycns/while/ddlymsxapn/Sigmoid_1Sigmoid%omilqdycns/while/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%omilqdycns/while/ddlymsxapn/Sigmoid_1Ê
!omilqdycns/while/ddlymsxapn/mul_2Mul)omilqdycns/while/ddlymsxapn/Sigmoid_1:y:0omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_2ª
 omilqdycns/while/ddlymsxapn/TanhTanh*omilqdycns/while/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 omilqdycns/while/ddlymsxapn/TanhÎ
!omilqdycns/while/ddlymsxapn/mul_3Mul'omilqdycns/while/ddlymsxapn/Sigmoid:y:0$omilqdycns/while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_3Ï
!omilqdycns/while/ddlymsxapn/add_3AddV2%omilqdycns/while/ddlymsxapn/mul_2:z:0%omilqdycns/while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_3Ð
,omilqdycns/while/ddlymsxapn/ReadVariableOp_2ReadVariableOp7omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,omilqdycns/while/ddlymsxapn/ReadVariableOp_2Ü
!omilqdycns/while/ddlymsxapn/mul_4Mul4omilqdycns/while/ddlymsxapn/ReadVariableOp_2:value:0%omilqdycns/while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_4Ô
!omilqdycns/while/ddlymsxapn/add_4AddV2*omilqdycns/while/ddlymsxapn/split:output:3%omilqdycns/while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_4²
%omilqdycns/while/ddlymsxapn/Sigmoid_2Sigmoid%omilqdycns/while/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%omilqdycns/while/ddlymsxapn/Sigmoid_2©
"omilqdycns/while/ddlymsxapn/Tanh_1Tanh%omilqdycns/while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"omilqdycns/while/ddlymsxapn/Tanh_1Ò
!omilqdycns/while/ddlymsxapn/mul_5Mul)omilqdycns/while/ddlymsxapn/Sigmoid_2:y:0&omilqdycns/while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_5
5omilqdycns/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemomilqdycns_while_placeholder_1omilqdycns_while_placeholder%omilqdycns/while/ddlymsxapn/mul_5:z:0*
_output_shapes
: *
element_dtype027
5omilqdycns/while/TensorArrayV2Write/TensorListSetItemr
omilqdycns/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
omilqdycns/while/add/y
omilqdycns/while/addAddV2omilqdycns_while_placeholderomilqdycns/while/add/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/while/addv
omilqdycns/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
omilqdycns/while/add_1/y­
omilqdycns/while/add_1AddV2.omilqdycns_while_omilqdycns_while_loop_counter!omilqdycns/while/add_1/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/while/add_1©
omilqdycns/while/IdentityIdentityomilqdycns/while/add_1:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/IdentityÇ
omilqdycns/while/Identity_1Identity4omilqdycns_while_omilqdycns_while_maximum_iterations3^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/Identity_1«
omilqdycns/while/Identity_2Identityomilqdycns/while/add:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/Identity_2Ø
omilqdycns/while/Identity_3IdentityEomilqdycns/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/Identity_3É
omilqdycns/while/Identity_4Identity%omilqdycns/while/ddlymsxapn/mul_5:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/while/Identity_4É
omilqdycns/while/Identity_5Identity%omilqdycns/while/ddlymsxapn/add_3:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/while/Identity_5"|
;omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource=omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0"~
<omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource>omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0"z
:omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource<omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0"p
5omilqdycns_while_ddlymsxapn_readvariableop_1_resource7omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0"p
5omilqdycns_while_ddlymsxapn_readvariableop_2_resource7omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0"l
3omilqdycns_while_ddlymsxapn_readvariableop_resource5omilqdycns_while_ddlymsxapn_readvariableop_resource_0"?
omilqdycns_while_identity"omilqdycns/while/Identity:output:0"C
omilqdycns_while_identity_1$omilqdycns/while/Identity_1:output:0"C
omilqdycns_while_identity_2$omilqdycns/while/Identity_2:output:0"C
omilqdycns_while_identity_3$omilqdycns/while/Identity_3:output:0"C
omilqdycns_while_identity_4$omilqdycns/while/Identity_4:output:0"C
omilqdycns_while_identity_5$omilqdycns/while/Identity_5:output:0"\
+omilqdycns_while_omilqdycns_strided_slice_1-omilqdycns_while_omilqdycns_strided_slice_1_0"Ô
gomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensoriomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2f
1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp2j
3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp2X
*omilqdycns/while/ddlymsxapn/ReadVariableOp*omilqdycns/while/ddlymsxapn/ReadVariableOp2\
,omilqdycns/while/ddlymsxapn/ReadVariableOp_1,omilqdycns/while/ddlymsxapn/ReadVariableOp_12\
,omilqdycns/while/ddlymsxapn/ReadVariableOp_2,omilqdycns/while/ddlymsxapn/ReadVariableOp_2: 
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
while_body_222045
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ddlymsxapn_matmul_readvariableop_resource_0:	F
3while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 A
2while_ddlymsxapn_biasadd_readvariableop_resource_0:	8
*while_ddlymsxapn_readvariableop_resource_0: :
,while_ddlymsxapn_readvariableop_1_resource_0: :
,while_ddlymsxapn_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ddlymsxapn_matmul_readvariableop_resource:	D
1while_ddlymsxapn_matmul_1_readvariableop_resource:	 ?
0while_ddlymsxapn_biasadd_readvariableop_resource:	6
(while_ddlymsxapn_readvariableop_resource: 8
*while_ddlymsxapn_readvariableop_1_resource: 8
*while_ddlymsxapn_readvariableop_2_resource: ¢'while/ddlymsxapn/BiasAdd/ReadVariableOp¢&while/ddlymsxapn/MatMul/ReadVariableOp¢(while/ddlymsxapn/MatMul_1/ReadVariableOp¢while/ddlymsxapn/ReadVariableOp¢!while/ddlymsxapn/ReadVariableOp_1¢!while/ddlymsxapn/ReadVariableOp_2Ã
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
&while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp1while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ddlymsxapn/MatMul/ReadVariableOpÑ
while/ddlymsxapn/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMulÉ
(while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp3while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ddlymsxapn/MatMul_1/ReadVariableOpº
while/ddlymsxapn/MatMul_1MatMulwhile_placeholder_20while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMul_1°
while/ddlymsxapn/addAddV2!while/ddlymsxapn/MatMul:product:0#while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/addÂ
'while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp2while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ddlymsxapn/BiasAdd/ReadVariableOp½
while/ddlymsxapn/BiasAddBiasAddwhile/ddlymsxapn/add:z:0/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/BiasAdd
 while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ddlymsxapn/split/split_dim
while/ddlymsxapn/splitSplit)while/ddlymsxapn/split/split_dim:output:0!while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ddlymsxapn/split©
while/ddlymsxapn/ReadVariableOpReadVariableOp*while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ddlymsxapn/ReadVariableOp£
while/ddlymsxapn/mulMul'while/ddlymsxapn/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul¦
while/ddlymsxapn/add_1AddV2while/ddlymsxapn/split:output:0while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_1
while/ddlymsxapn/SigmoidSigmoidwhile/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid¯
!while/ddlymsxapn/ReadVariableOp_1ReadVariableOp,while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_1©
while/ddlymsxapn/mul_1Mul)while/ddlymsxapn/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_1¨
while/ddlymsxapn/add_2AddV2while/ddlymsxapn/split:output:1while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_2
while/ddlymsxapn/Sigmoid_1Sigmoidwhile/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_1
while/ddlymsxapn/mul_2Mulwhile/ddlymsxapn/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_2
while/ddlymsxapn/TanhTanhwhile/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh¢
while/ddlymsxapn/mul_3Mulwhile/ddlymsxapn/Sigmoid:y:0while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_3£
while/ddlymsxapn/add_3AddV2while/ddlymsxapn/mul_2:z:0while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_3¯
!while/ddlymsxapn/ReadVariableOp_2ReadVariableOp,while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_2°
while/ddlymsxapn/mul_4Mul)while/ddlymsxapn/ReadVariableOp_2:value:0while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_4¨
while/ddlymsxapn/add_4AddV2while/ddlymsxapn/split:output:3while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_4
while/ddlymsxapn/Sigmoid_2Sigmoidwhile/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_2
while/ddlymsxapn/Tanh_1Tanhwhile/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh_1¦
while/ddlymsxapn/mul_5Mulwhile/ddlymsxapn/Sigmoid_2:y:0while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ddlymsxapn/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ddlymsxapn/mul_5:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ddlymsxapn/add_3:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ddlymsxapn_biasadd_readvariableop_resource2while_ddlymsxapn_biasadd_readvariableop_resource_0"h
1while_ddlymsxapn_matmul_1_readvariableop_resource3while_ddlymsxapn_matmul_1_readvariableop_resource_0"d
/while_ddlymsxapn_matmul_readvariableop_resource1while_ddlymsxapn_matmul_readvariableop_resource_0"Z
*while_ddlymsxapn_readvariableop_1_resource,while_ddlymsxapn_readvariableop_1_resource_0"Z
*while_ddlymsxapn_readvariableop_2_resource,while_ddlymsxapn_readvariableop_2_resource_0"V
(while_ddlymsxapn_readvariableop_resource*while_ddlymsxapn_readvariableop_resource_0")
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
'while/ddlymsxapn/BiasAdd/ReadVariableOp'while/ddlymsxapn/BiasAdd/ReadVariableOp2P
&while/ddlymsxapn/MatMul/ReadVariableOp&while/ddlymsxapn/MatMul/ReadVariableOp2T
(while/ddlymsxapn/MatMul_1/ReadVariableOp(while/ddlymsxapn/MatMul_1/ReadVariableOp2B
while/ddlymsxapn/ReadVariableOpwhile/ddlymsxapn/ReadVariableOp2F
!while/ddlymsxapn/ReadVariableOp_1!while/ddlymsxapn/ReadVariableOp_12F
!while/ddlymsxapn/ReadVariableOp_2!while/ddlymsxapn/ReadVariableOp_2: 
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
+__inference_vlxoswgdqw_layer_call_fn_224906
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2215942
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_222828

inputs<
)ddlymsxapn_matmul_readvariableop_resource:	>
+ddlymsxapn_matmul_1_readvariableop_resource:	 9
*ddlymsxapn_biasadd_readvariableop_resource:	0
"ddlymsxapn_readvariableop_resource: 2
$ddlymsxapn_readvariableop_1_resource: 2
$ddlymsxapn_readvariableop_2_resource: 
identity¢!ddlymsxapn/BiasAdd/ReadVariableOp¢ ddlymsxapn/MatMul/ReadVariableOp¢"ddlymsxapn/MatMul_1/ReadVariableOp¢ddlymsxapn/ReadVariableOp¢ddlymsxapn/ReadVariableOp_1¢ddlymsxapn/ReadVariableOp_2¢whileD
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
 ddlymsxapn/MatMul/ReadVariableOpReadVariableOp)ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ddlymsxapn/MatMul/ReadVariableOp§
ddlymsxapn/MatMulMatMulstrided_slice_2:output:0(ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMulµ
"ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp+ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ddlymsxapn/MatMul_1/ReadVariableOp£
ddlymsxapn/MatMul_1MatMulzeros:output:0*ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMul_1
ddlymsxapn/addAddV2ddlymsxapn/MatMul:product:0ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/add®
!ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp*ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ddlymsxapn/BiasAdd/ReadVariableOp¥
ddlymsxapn/BiasAddBiasAddddlymsxapn/add:z:0)ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/BiasAddz
ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ddlymsxapn/split/split_dimë
ddlymsxapn/splitSplit#ddlymsxapn/split/split_dim:output:0ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ddlymsxapn/split
ddlymsxapn/ReadVariableOpReadVariableOp"ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp
ddlymsxapn/mulMul!ddlymsxapn/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul
ddlymsxapn/add_1AddV2ddlymsxapn/split:output:0ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_1{
ddlymsxapn/SigmoidSigmoidddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid
ddlymsxapn/ReadVariableOp_1ReadVariableOp$ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_1
ddlymsxapn/mul_1Mul#ddlymsxapn/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_1
ddlymsxapn/add_2AddV2ddlymsxapn/split:output:1ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_2
ddlymsxapn/Sigmoid_1Sigmoidddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_1
ddlymsxapn/mul_2Mulddlymsxapn/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_2w
ddlymsxapn/TanhTanhddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh
ddlymsxapn/mul_3Mulddlymsxapn/Sigmoid:y:0ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_3
ddlymsxapn/add_3AddV2ddlymsxapn/mul_2:z:0ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_3
ddlymsxapn/ReadVariableOp_2ReadVariableOp$ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_2
ddlymsxapn/mul_4Mul#ddlymsxapn/ReadVariableOp_2:value:0ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_4
ddlymsxapn/add_4AddV2ddlymsxapn/split:output:3ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_4
ddlymsxapn/Sigmoid_2Sigmoidddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_2v
ddlymsxapn/Tanh_1Tanhddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh_1
ddlymsxapn/mul_5Mulddlymsxapn/Sigmoid_2:y:0ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ddlymsxapn_matmul_readvariableop_resource+ddlymsxapn_matmul_1_readvariableop_resource*ddlymsxapn_biasadd_readvariableop_resource"ddlymsxapn_readvariableop_resource$ddlymsxapn_readvariableop_1_resource$ddlymsxapn_readvariableop_2_resource*
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
while_body_222727*
condR
while_cond_222726*Q
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
IdentityIdentitytranspose_1:y:0"^ddlymsxapn/BiasAdd/ReadVariableOp!^ddlymsxapn/MatMul/ReadVariableOp#^ddlymsxapn/MatMul_1/ReadVariableOp^ddlymsxapn/ReadVariableOp^ddlymsxapn/ReadVariableOp_1^ddlymsxapn/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ddlymsxapn/BiasAdd/ReadVariableOp!ddlymsxapn/BiasAdd/ReadVariableOp2D
 ddlymsxapn/MatMul/ReadVariableOp ddlymsxapn/MatMul/ReadVariableOp2H
"ddlymsxapn/MatMul_1/ReadVariableOp"ddlymsxapn/MatMul_1/ReadVariableOp26
ddlymsxapn/ReadVariableOpddlymsxapn/ReadVariableOp2:
ddlymsxapn/ReadVariableOp_1ddlymsxapn/ReadVariableOp_12:
ddlymsxapn/ReadVariableOp_2ddlymsxapn/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸'
´
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_225769

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

b
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_221965

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
while_cond_220492
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_220492___redundant_placeholder04
0while_while_cond_220492___redundant_placeholder14
0while_while_cond_220492___redundant_placeholder24
0while_while_cond_220492___redundant_placeholder34
0while_while_cond_220492___redundant_placeholder44
0while_while_cond_220492___redundant_placeholder54
0while_while_cond_220492___redundant_placeholder6
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
Ñ

+__inference_vlxoswgdqw_layer_call_fn_224940

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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2226142
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
Ç)
Å
while_body_221514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_vdaevhnmja_221538_0:	 ,
while_vdaevhnmja_221540_0:	 (
while_vdaevhnmja_221542_0:	'
while_vdaevhnmja_221544_0: '
while_vdaevhnmja_221546_0: '
while_vdaevhnmja_221548_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_vdaevhnmja_221538:	 *
while_vdaevhnmja_221540:	 &
while_vdaevhnmja_221542:	%
while_vdaevhnmja_221544: %
while_vdaevhnmja_221546: %
while_vdaevhnmja_221548: ¢(while/vdaevhnmja/StatefulPartitionedCallÃ
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
(while/vdaevhnmja/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_vdaevhnmja_221538_0while_vdaevhnmja_221540_0while_vdaevhnmja_221542_0while_vdaevhnmja_221544_0while_vdaevhnmja_221546_0while_vdaevhnmja_221548_0*
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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_2214182*
(while/vdaevhnmja/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/vdaevhnmja/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/vdaevhnmja/StatefulPartitionedCall:output:1)^while/vdaevhnmja/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/vdaevhnmja/StatefulPartitionedCall:output:2)^while/vdaevhnmja/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"4
while_vdaevhnmja_221538while_vdaevhnmja_221538_0"4
while_vdaevhnmja_221540while_vdaevhnmja_221540_0"4
while_vdaevhnmja_221542while_vdaevhnmja_221542_0"4
while_vdaevhnmja_221544while_vdaevhnmja_221544_0"4
while_vdaevhnmja_221546while_vdaevhnmja_221546_0"4
while_vdaevhnmja_221548while_vdaevhnmja_221548_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/vdaevhnmja/StatefulPartitionedCall(while/vdaevhnmja/StatefulPartitionedCall: 
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
®
Ö
!__inference__wrapped_model_220386

ahzwxypkrhW
Asequential_iigfihrkup_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_iigfihrkup_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_omilqdycns_ddlymsxapn_matmul_readvariableop_resource:	T
Asequential_omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource:	 O
@sequential_omilqdycns_ddlymsxapn_biasadd_readvariableop_resource:	F
8sequential_omilqdycns_ddlymsxapn_readvariableop_resource: H
:sequential_omilqdycns_ddlymsxapn_readvariableop_1_resource: H
:sequential_omilqdycns_ddlymsxapn_readvariableop_2_resource: R
?sequential_vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource:	 T
Asequential_vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource:	 O
@sequential_vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource:	F
8sequential_vlxoswgdqw_vdaevhnmja_readvariableop_resource: H
:sequential_vlxoswgdqw_vdaevhnmja_readvariableop_1_resource: H
:sequential_vlxoswgdqw_vdaevhnmja_readvariableop_2_resource: F
4sequential_iktogmlrmp_matmul_readvariableop_resource: C
5sequential_iktogmlrmp_biasadd_readvariableop_resource:
identity¢8sequential/iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,sequential/iktogmlrmp/BiasAdd/ReadVariableOp¢+sequential/iktogmlrmp/MatMul/ReadVariableOp¢7sequential/omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp¢6sequential/omilqdycns/ddlymsxapn/MatMul/ReadVariableOp¢8sequential/omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp¢/sequential/omilqdycns/ddlymsxapn/ReadVariableOp¢1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_1¢1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_2¢sequential/omilqdycns/while¢7sequential/vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp¢6sequential/vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp¢8sequential/vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp¢/sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp¢1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_1¢1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_2¢sequential/vlxoswgdqw/while¥
+sequential/iigfihrkup/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/iigfihrkup/conv1d/ExpandDims/dimà
'sequential/iigfihrkup/conv1d/ExpandDims
ExpandDims
ahzwxypkrh4sequential/iigfihrkup/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/iigfihrkup/conv1d/ExpandDimsú
8sequential/iigfihrkup/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_iigfihrkup_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/iigfihrkup/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/iigfihrkup/conv1d/ExpandDims_1/dim
)sequential/iigfihrkup/conv1d/ExpandDims_1
ExpandDims@sequential/iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/iigfihrkup/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/iigfihrkup/conv1d/ExpandDims_1¨
"sequential/iigfihrkup/conv1d/ShapeShape0sequential/iigfihrkup/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/iigfihrkup/conv1d/Shape®
0sequential/iigfihrkup/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/iigfihrkup/conv1d/strided_slice/stack»
2sequential/iigfihrkup/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/iigfihrkup/conv1d/strided_slice/stack_1²
2sequential/iigfihrkup/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/iigfihrkup/conv1d/strided_slice/stack_2
*sequential/iigfihrkup/conv1d/strided_sliceStridedSlice+sequential/iigfihrkup/conv1d/Shape:output:09sequential/iigfihrkup/conv1d/strided_slice/stack:output:0;sequential/iigfihrkup/conv1d/strided_slice/stack_1:output:0;sequential/iigfihrkup/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/iigfihrkup/conv1d/strided_slice±
*sequential/iigfihrkup/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/iigfihrkup/conv1d/Reshape/shapeø
$sequential/iigfihrkup/conv1d/ReshapeReshape0sequential/iigfihrkup/conv1d/ExpandDims:output:03sequential/iigfihrkup/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/iigfihrkup/conv1d/Reshape
#sequential/iigfihrkup/conv1d/Conv2DConv2D-sequential/iigfihrkup/conv1d/Reshape:output:02sequential/iigfihrkup/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/iigfihrkup/conv1d/Conv2D±
,sequential/iigfihrkup/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/iigfihrkup/conv1d/concat/values_1
(sequential/iigfihrkup/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/iigfihrkup/conv1d/concat/axis£
#sequential/iigfihrkup/conv1d/concatConcatV23sequential/iigfihrkup/conv1d/strided_slice:output:05sequential/iigfihrkup/conv1d/concat/values_1:output:01sequential/iigfihrkup/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/iigfihrkup/conv1d/concatõ
&sequential/iigfihrkup/conv1d/Reshape_1Reshape,sequential/iigfihrkup/conv1d/Conv2D:output:0,sequential/iigfihrkup/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/iigfihrkup/conv1d/Reshape_1â
$sequential/iigfihrkup/conv1d/SqueezeSqueeze/sequential/iigfihrkup/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/iigfihrkup/conv1d/Squeeze½
.sequential/iigfihrkup/squeeze_batch_dims/ShapeShape-sequential/iigfihrkup/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/iigfihrkup/squeeze_batch_dims/ShapeÆ
<sequential/iigfihrkup/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/iigfihrkup/squeeze_batch_dims/strided_slice/stackÓ
>sequential/iigfihrkup/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/iigfihrkup/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/iigfihrkup/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/iigfihrkup/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/iigfihrkup/squeeze_batch_dims/strided_sliceStridedSlice7sequential/iigfihrkup/squeeze_batch_dims/Shape:output:0Esequential/iigfihrkup/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/iigfihrkup/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/iigfihrkup/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/iigfihrkup/squeeze_batch_dims/strided_sliceÅ
6sequential/iigfihrkup/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/iigfihrkup/squeeze_batch_dims/Reshape/shape
0sequential/iigfihrkup/squeeze_batch_dims/ReshapeReshape-sequential/iigfihrkup/conv1d/Squeeze:output:0?sequential/iigfihrkup/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/iigfihrkup/squeeze_batch_dims/Reshape
?sequential/iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_iigfihrkup_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/iigfihrkup/squeeze_batch_dims/BiasAddBiasAdd9sequential/iigfihrkup/squeeze_batch_dims/Reshape:output:0Gsequential/iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/iigfihrkup/squeeze_batch_dims/BiasAddÅ
8sequential/iigfihrkup/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/iigfihrkup/squeeze_batch_dims/concat/values_1·
4sequential/iigfihrkup/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/iigfihrkup/squeeze_batch_dims/concat/axisß
/sequential/iigfihrkup/squeeze_batch_dims/concatConcatV2?sequential/iigfihrkup/squeeze_batch_dims/strided_slice:output:0Asequential/iigfihrkup/squeeze_batch_dims/concat/values_1:output:0=sequential/iigfihrkup/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/iigfihrkup/squeeze_batch_dims/concat¢
2sequential/iigfihrkup/squeeze_batch_dims/Reshape_1Reshape9sequential/iigfihrkup/squeeze_batch_dims/BiasAdd:output:08sequential/iigfihrkup/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/iigfihrkup/squeeze_batch_dims/Reshape_1¥
sequential/xfbsciqeco/ShapeShape;sequential/iigfihrkup/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/xfbsciqeco/Shape 
)sequential/xfbsciqeco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/xfbsciqeco/strided_slice/stack¤
+sequential/xfbsciqeco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xfbsciqeco/strided_slice/stack_1¤
+sequential/xfbsciqeco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xfbsciqeco/strided_slice/stack_2æ
#sequential/xfbsciqeco/strided_sliceStridedSlice$sequential/xfbsciqeco/Shape:output:02sequential/xfbsciqeco/strided_slice/stack:output:04sequential/xfbsciqeco/strided_slice/stack_1:output:04sequential/xfbsciqeco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/xfbsciqeco/strided_slice
%sequential/xfbsciqeco/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/xfbsciqeco/Reshape/shape/1
%sequential/xfbsciqeco/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/xfbsciqeco/Reshape/shape/2
#sequential/xfbsciqeco/Reshape/shapePack,sequential/xfbsciqeco/strided_slice:output:0.sequential/xfbsciqeco/Reshape/shape/1:output:0.sequential/xfbsciqeco/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/xfbsciqeco/Reshape/shapeê
sequential/xfbsciqeco/ReshapeReshape;sequential/iigfihrkup/squeeze_batch_dims/Reshape_1:output:0,sequential/xfbsciqeco/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/xfbsciqeco/Reshape
sequential/omilqdycns/ShapeShape&sequential/xfbsciqeco/Reshape:output:0*
T0*
_output_shapes
:2
sequential/omilqdycns/Shape 
)sequential/omilqdycns/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/omilqdycns/strided_slice/stack¤
+sequential/omilqdycns/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/omilqdycns/strided_slice/stack_1¤
+sequential/omilqdycns/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/omilqdycns/strided_slice/stack_2æ
#sequential/omilqdycns/strided_sliceStridedSlice$sequential/omilqdycns/Shape:output:02sequential/omilqdycns/strided_slice/stack:output:04sequential/omilqdycns/strided_slice/stack_1:output:04sequential/omilqdycns/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/omilqdycns/strided_slice
!sequential/omilqdycns/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/omilqdycns/zeros/mul/yÄ
sequential/omilqdycns/zeros/mulMul,sequential/omilqdycns/strided_slice:output:0*sequential/omilqdycns/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/omilqdycns/zeros/mul
"sequential/omilqdycns/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/omilqdycns/zeros/Less/y¿
 sequential/omilqdycns/zeros/LessLess#sequential/omilqdycns/zeros/mul:z:0+sequential/omilqdycns/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/omilqdycns/zeros/Less
$sequential/omilqdycns/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/omilqdycns/zeros/packed/1Û
"sequential/omilqdycns/zeros/packedPack,sequential/omilqdycns/strided_slice:output:0-sequential/omilqdycns/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/omilqdycns/zeros/packed
!sequential/omilqdycns/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/omilqdycns/zeros/ConstÍ
sequential/omilqdycns/zerosFill+sequential/omilqdycns/zeros/packed:output:0*sequential/omilqdycns/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/omilqdycns/zeros
#sequential/omilqdycns/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/omilqdycns/zeros_1/mul/yÊ
!sequential/omilqdycns/zeros_1/mulMul,sequential/omilqdycns/strided_slice:output:0,sequential/omilqdycns/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/omilqdycns/zeros_1/mul
$sequential/omilqdycns/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/omilqdycns/zeros_1/Less/yÇ
"sequential/omilqdycns/zeros_1/LessLess%sequential/omilqdycns/zeros_1/mul:z:0-sequential/omilqdycns/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/omilqdycns/zeros_1/Less
&sequential/omilqdycns/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/omilqdycns/zeros_1/packed/1á
$sequential/omilqdycns/zeros_1/packedPack,sequential/omilqdycns/strided_slice:output:0/sequential/omilqdycns/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/omilqdycns/zeros_1/packed
#sequential/omilqdycns/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/omilqdycns/zeros_1/ConstÕ
sequential/omilqdycns/zeros_1Fill-sequential/omilqdycns/zeros_1/packed:output:0,sequential/omilqdycns/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/omilqdycns/zeros_1¡
$sequential/omilqdycns/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/omilqdycns/transpose/permÜ
sequential/omilqdycns/transpose	Transpose&sequential/xfbsciqeco/Reshape:output:0-sequential/omilqdycns/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/omilqdycns/transpose
sequential/omilqdycns/Shape_1Shape#sequential/omilqdycns/transpose:y:0*
T0*
_output_shapes
:2
sequential/omilqdycns/Shape_1¤
+sequential/omilqdycns/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/omilqdycns/strided_slice_1/stack¨
-sequential/omilqdycns/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/omilqdycns/strided_slice_1/stack_1¨
-sequential/omilqdycns/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/omilqdycns/strided_slice_1/stack_2ò
%sequential/omilqdycns/strided_slice_1StridedSlice&sequential/omilqdycns/Shape_1:output:04sequential/omilqdycns/strided_slice_1/stack:output:06sequential/omilqdycns/strided_slice_1/stack_1:output:06sequential/omilqdycns/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/omilqdycns/strided_slice_1±
1sequential/omilqdycns/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/omilqdycns/TensorArrayV2/element_shape
#sequential/omilqdycns/TensorArrayV2TensorListReserve:sequential/omilqdycns/TensorArrayV2/element_shape:output:0.sequential/omilqdycns/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/omilqdycns/TensorArrayV2ë
Ksequential/omilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/omilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/omilqdycns/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/omilqdycns/transpose:y:0Tsequential/omilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/omilqdycns/TensorArrayUnstack/TensorListFromTensor¤
+sequential/omilqdycns/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/omilqdycns/strided_slice_2/stack¨
-sequential/omilqdycns/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/omilqdycns/strided_slice_2/stack_1¨
-sequential/omilqdycns/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/omilqdycns/strided_slice_2/stack_2
%sequential/omilqdycns/strided_slice_2StridedSlice#sequential/omilqdycns/transpose:y:04sequential/omilqdycns/strided_slice_2/stack:output:06sequential/omilqdycns/strided_slice_2/stack_1:output:06sequential/omilqdycns/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/omilqdycns/strided_slice_2ñ
6sequential/omilqdycns/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp?sequential_omilqdycns_ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/omilqdycns/ddlymsxapn/MatMul/ReadVariableOpÿ
'sequential/omilqdycns/ddlymsxapn/MatMulMatMul.sequential/omilqdycns/strided_slice_2:output:0>sequential/omilqdycns/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/omilqdycns/ddlymsxapn/MatMul÷
8sequential/omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOpAsequential_omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOpû
)sequential/omilqdycns/ddlymsxapn/MatMul_1MatMul$sequential/omilqdycns/zeros:output:0@sequential/omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/omilqdycns/ddlymsxapn/MatMul_1ð
$sequential/omilqdycns/ddlymsxapn/addAddV21sequential/omilqdycns/ddlymsxapn/MatMul:product:03sequential/omilqdycns/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/omilqdycns/ddlymsxapn/addð
7sequential/omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp@sequential_omilqdycns_ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOpý
(sequential/omilqdycns/ddlymsxapn/BiasAddBiasAdd(sequential/omilqdycns/ddlymsxapn/add:z:0?sequential/omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/omilqdycns/ddlymsxapn/BiasAdd¦
0sequential/omilqdycns/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/omilqdycns/ddlymsxapn/split/split_dimÃ
&sequential/omilqdycns/ddlymsxapn/splitSplit9sequential/omilqdycns/ddlymsxapn/split/split_dim:output:01sequential/omilqdycns/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/omilqdycns/ddlymsxapn/split×
/sequential/omilqdycns/ddlymsxapn/ReadVariableOpReadVariableOp8sequential_omilqdycns_ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/omilqdycns/ddlymsxapn/ReadVariableOpæ
$sequential/omilqdycns/ddlymsxapn/mulMul7sequential/omilqdycns/ddlymsxapn/ReadVariableOp:value:0&sequential/omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/omilqdycns/ddlymsxapn/mulæ
&sequential/omilqdycns/ddlymsxapn/add_1AddV2/sequential/omilqdycns/ddlymsxapn/split:output:0(sequential/omilqdycns/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/add_1½
(sequential/omilqdycns/ddlymsxapn/SigmoidSigmoid*sequential/omilqdycns/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/omilqdycns/ddlymsxapn/SigmoidÝ
1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_1ReadVariableOp:sequential_omilqdycns_ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_1ì
&sequential/omilqdycns/ddlymsxapn/mul_1Mul9sequential/omilqdycns/ddlymsxapn/ReadVariableOp_1:value:0&sequential/omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/mul_1è
&sequential/omilqdycns/ddlymsxapn/add_2AddV2/sequential/omilqdycns/ddlymsxapn/split:output:1*sequential/omilqdycns/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/add_2Á
*sequential/omilqdycns/ddlymsxapn/Sigmoid_1Sigmoid*sequential/omilqdycns/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/omilqdycns/ddlymsxapn/Sigmoid_1á
&sequential/omilqdycns/ddlymsxapn/mul_2Mul.sequential/omilqdycns/ddlymsxapn/Sigmoid_1:y:0&sequential/omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/mul_2¹
%sequential/omilqdycns/ddlymsxapn/TanhTanh/sequential/omilqdycns/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/omilqdycns/ddlymsxapn/Tanhâ
&sequential/omilqdycns/ddlymsxapn/mul_3Mul,sequential/omilqdycns/ddlymsxapn/Sigmoid:y:0)sequential/omilqdycns/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/mul_3ã
&sequential/omilqdycns/ddlymsxapn/add_3AddV2*sequential/omilqdycns/ddlymsxapn/mul_2:z:0*sequential/omilqdycns/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/add_3Ý
1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_2ReadVariableOp:sequential_omilqdycns_ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_2ð
&sequential/omilqdycns/ddlymsxapn/mul_4Mul9sequential/omilqdycns/ddlymsxapn/ReadVariableOp_2:value:0*sequential/omilqdycns/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/mul_4è
&sequential/omilqdycns/ddlymsxapn/add_4AddV2/sequential/omilqdycns/ddlymsxapn/split:output:3*sequential/omilqdycns/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/add_4Á
*sequential/omilqdycns/ddlymsxapn/Sigmoid_2Sigmoid*sequential/omilqdycns/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/omilqdycns/ddlymsxapn/Sigmoid_2¸
'sequential/omilqdycns/ddlymsxapn/Tanh_1Tanh*sequential/omilqdycns/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/omilqdycns/ddlymsxapn/Tanh_1æ
&sequential/omilqdycns/ddlymsxapn/mul_5Mul.sequential/omilqdycns/ddlymsxapn/Sigmoid_2:y:0+sequential/omilqdycns/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/ddlymsxapn/mul_5»
3sequential/omilqdycns/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/omilqdycns/TensorArrayV2_1/element_shape
%sequential/omilqdycns/TensorArrayV2_1TensorListReserve<sequential/omilqdycns/TensorArrayV2_1/element_shape:output:0.sequential/omilqdycns/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/omilqdycns/TensorArrayV2_1z
sequential/omilqdycns/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/omilqdycns/time«
.sequential/omilqdycns/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/omilqdycns/while/maximum_iterations
(sequential/omilqdycns/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/omilqdycns/while/loop_counterö	
sequential/omilqdycns/whileWhile1sequential/omilqdycns/while/loop_counter:output:07sequential/omilqdycns/while/maximum_iterations:output:0#sequential/omilqdycns/time:output:0.sequential/omilqdycns/TensorArrayV2_1:handle:0$sequential/omilqdycns/zeros:output:0&sequential/omilqdycns/zeros_1:output:0.sequential/omilqdycns/strided_slice_1:output:0Msequential/omilqdycns/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_omilqdycns_ddlymsxapn_matmul_readvariableop_resourceAsequential_omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource@sequential_omilqdycns_ddlymsxapn_biasadd_readvariableop_resource8sequential_omilqdycns_ddlymsxapn_readvariableop_resource:sequential_omilqdycns_ddlymsxapn_readvariableop_1_resource:sequential_omilqdycns_ddlymsxapn_readvariableop_2_resource*
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
'sequential_omilqdycns_while_body_220103*3
cond+R)
'sequential_omilqdycns_while_cond_220102*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/omilqdycns/whileá
Fsequential/omilqdycns/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/omilqdycns/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/omilqdycns/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/omilqdycns/while:output:3Osequential/omilqdycns/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/omilqdycns/TensorArrayV2Stack/TensorListStack­
+sequential/omilqdycns/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/omilqdycns/strided_slice_3/stack¨
-sequential/omilqdycns/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/omilqdycns/strided_slice_3/stack_1¨
-sequential/omilqdycns/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/omilqdycns/strided_slice_3/stack_2
%sequential/omilqdycns/strided_slice_3StridedSliceAsequential/omilqdycns/TensorArrayV2Stack/TensorListStack:tensor:04sequential/omilqdycns/strided_slice_3/stack:output:06sequential/omilqdycns/strided_slice_3/stack_1:output:06sequential/omilqdycns/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/omilqdycns/strided_slice_3¥
&sequential/omilqdycns/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/omilqdycns/transpose_1/permý
!sequential/omilqdycns/transpose_1	TransposeAsequential/omilqdycns/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/omilqdycns/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/omilqdycns/transpose_1
sequential/vlxoswgdqw/ShapeShape%sequential/omilqdycns/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/vlxoswgdqw/Shape 
)sequential/vlxoswgdqw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/vlxoswgdqw/strided_slice/stack¤
+sequential/vlxoswgdqw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/vlxoswgdqw/strided_slice/stack_1¤
+sequential/vlxoswgdqw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/vlxoswgdqw/strided_slice/stack_2æ
#sequential/vlxoswgdqw/strided_sliceStridedSlice$sequential/vlxoswgdqw/Shape:output:02sequential/vlxoswgdqw/strided_slice/stack:output:04sequential/vlxoswgdqw/strided_slice/stack_1:output:04sequential/vlxoswgdqw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/vlxoswgdqw/strided_slice
!sequential/vlxoswgdqw/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/vlxoswgdqw/zeros/mul/yÄ
sequential/vlxoswgdqw/zeros/mulMul,sequential/vlxoswgdqw/strided_slice:output:0*sequential/vlxoswgdqw/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/vlxoswgdqw/zeros/mul
"sequential/vlxoswgdqw/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/vlxoswgdqw/zeros/Less/y¿
 sequential/vlxoswgdqw/zeros/LessLess#sequential/vlxoswgdqw/zeros/mul:z:0+sequential/vlxoswgdqw/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/vlxoswgdqw/zeros/Less
$sequential/vlxoswgdqw/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/vlxoswgdqw/zeros/packed/1Û
"sequential/vlxoswgdqw/zeros/packedPack,sequential/vlxoswgdqw/strided_slice:output:0-sequential/vlxoswgdqw/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/vlxoswgdqw/zeros/packed
!sequential/vlxoswgdqw/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/vlxoswgdqw/zeros/ConstÍ
sequential/vlxoswgdqw/zerosFill+sequential/vlxoswgdqw/zeros/packed:output:0*sequential/vlxoswgdqw/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/vlxoswgdqw/zeros
#sequential/vlxoswgdqw/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/vlxoswgdqw/zeros_1/mul/yÊ
!sequential/vlxoswgdqw/zeros_1/mulMul,sequential/vlxoswgdqw/strided_slice:output:0,sequential/vlxoswgdqw/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/vlxoswgdqw/zeros_1/mul
$sequential/vlxoswgdqw/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/vlxoswgdqw/zeros_1/Less/yÇ
"sequential/vlxoswgdqw/zeros_1/LessLess%sequential/vlxoswgdqw/zeros_1/mul:z:0-sequential/vlxoswgdqw/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/vlxoswgdqw/zeros_1/Less
&sequential/vlxoswgdqw/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/vlxoswgdqw/zeros_1/packed/1á
$sequential/vlxoswgdqw/zeros_1/packedPack,sequential/vlxoswgdqw/strided_slice:output:0/sequential/vlxoswgdqw/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/vlxoswgdqw/zeros_1/packed
#sequential/vlxoswgdqw/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/vlxoswgdqw/zeros_1/ConstÕ
sequential/vlxoswgdqw/zeros_1Fill-sequential/vlxoswgdqw/zeros_1/packed:output:0,sequential/vlxoswgdqw/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/vlxoswgdqw/zeros_1¡
$sequential/vlxoswgdqw/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/vlxoswgdqw/transpose/permÛ
sequential/vlxoswgdqw/transpose	Transpose%sequential/omilqdycns/transpose_1:y:0-sequential/vlxoswgdqw/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/vlxoswgdqw/transpose
sequential/vlxoswgdqw/Shape_1Shape#sequential/vlxoswgdqw/transpose:y:0*
T0*
_output_shapes
:2
sequential/vlxoswgdqw/Shape_1¤
+sequential/vlxoswgdqw/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/vlxoswgdqw/strided_slice_1/stack¨
-sequential/vlxoswgdqw/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vlxoswgdqw/strided_slice_1/stack_1¨
-sequential/vlxoswgdqw/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vlxoswgdqw/strided_slice_1/stack_2ò
%sequential/vlxoswgdqw/strided_slice_1StridedSlice&sequential/vlxoswgdqw/Shape_1:output:04sequential/vlxoswgdqw/strided_slice_1/stack:output:06sequential/vlxoswgdqw/strided_slice_1/stack_1:output:06sequential/vlxoswgdqw/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/vlxoswgdqw/strided_slice_1±
1sequential/vlxoswgdqw/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/vlxoswgdqw/TensorArrayV2/element_shape
#sequential/vlxoswgdqw/TensorArrayV2TensorListReserve:sequential/vlxoswgdqw/TensorArrayV2/element_shape:output:0.sequential/vlxoswgdqw/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/vlxoswgdqw/TensorArrayV2ë
Ksequential/vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/vlxoswgdqw/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/vlxoswgdqw/transpose:y:0Tsequential/vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor¤
+sequential/vlxoswgdqw/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/vlxoswgdqw/strided_slice_2/stack¨
-sequential/vlxoswgdqw/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vlxoswgdqw/strided_slice_2/stack_1¨
-sequential/vlxoswgdqw/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vlxoswgdqw/strided_slice_2/stack_2
%sequential/vlxoswgdqw/strided_slice_2StridedSlice#sequential/vlxoswgdqw/transpose:y:04sequential/vlxoswgdqw/strided_slice_2/stack:output:06sequential/vlxoswgdqw/strided_slice_2/stack_1:output:06sequential/vlxoswgdqw/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/vlxoswgdqw/strided_slice_2ñ
6sequential/vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp?sequential_vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOpÿ
'sequential/vlxoswgdqw/vdaevhnmja/MatMulMatMul.sequential/vlxoswgdqw/strided_slice_2:output:0>sequential/vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/vlxoswgdqw/vdaevhnmja/MatMul÷
8sequential/vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOpAsequential_vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOpû
)sequential/vlxoswgdqw/vdaevhnmja/MatMul_1MatMul$sequential/vlxoswgdqw/zeros:output:0@sequential/vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/vlxoswgdqw/vdaevhnmja/MatMul_1ð
$sequential/vlxoswgdqw/vdaevhnmja/addAddV21sequential/vlxoswgdqw/vdaevhnmja/MatMul:product:03sequential/vlxoswgdqw/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/vlxoswgdqw/vdaevhnmja/addð
7sequential/vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp@sequential_vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOpý
(sequential/vlxoswgdqw/vdaevhnmja/BiasAddBiasAdd(sequential/vlxoswgdqw/vdaevhnmja/add:z:0?sequential/vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/vlxoswgdqw/vdaevhnmja/BiasAdd¦
0sequential/vlxoswgdqw/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/vlxoswgdqw/vdaevhnmja/split/split_dimÃ
&sequential/vlxoswgdqw/vdaevhnmja/splitSplit9sequential/vlxoswgdqw/vdaevhnmja/split/split_dim:output:01sequential/vlxoswgdqw/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/vlxoswgdqw/vdaevhnmja/split×
/sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOpReadVariableOp8sequential_vlxoswgdqw_vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOpæ
$sequential/vlxoswgdqw/vdaevhnmja/mulMul7sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp:value:0&sequential/vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/vlxoswgdqw/vdaevhnmja/mulæ
&sequential/vlxoswgdqw/vdaevhnmja/add_1AddV2/sequential/vlxoswgdqw/vdaevhnmja/split:output:0(sequential/vlxoswgdqw/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/add_1½
(sequential/vlxoswgdqw/vdaevhnmja/SigmoidSigmoid*sequential/vlxoswgdqw/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/vlxoswgdqw/vdaevhnmja/SigmoidÝ
1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_1ReadVariableOp:sequential_vlxoswgdqw_vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_1ì
&sequential/vlxoswgdqw/vdaevhnmja/mul_1Mul9sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_1:value:0&sequential/vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/mul_1è
&sequential/vlxoswgdqw/vdaevhnmja/add_2AddV2/sequential/vlxoswgdqw/vdaevhnmja/split:output:1*sequential/vlxoswgdqw/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/add_2Á
*sequential/vlxoswgdqw/vdaevhnmja/Sigmoid_1Sigmoid*sequential/vlxoswgdqw/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/vlxoswgdqw/vdaevhnmja/Sigmoid_1á
&sequential/vlxoswgdqw/vdaevhnmja/mul_2Mul.sequential/vlxoswgdqw/vdaevhnmja/Sigmoid_1:y:0&sequential/vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/mul_2¹
%sequential/vlxoswgdqw/vdaevhnmja/TanhTanh/sequential/vlxoswgdqw/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/vlxoswgdqw/vdaevhnmja/Tanhâ
&sequential/vlxoswgdqw/vdaevhnmja/mul_3Mul,sequential/vlxoswgdqw/vdaevhnmja/Sigmoid:y:0)sequential/vlxoswgdqw/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/mul_3ã
&sequential/vlxoswgdqw/vdaevhnmja/add_3AddV2*sequential/vlxoswgdqw/vdaevhnmja/mul_2:z:0*sequential/vlxoswgdqw/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/add_3Ý
1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_2ReadVariableOp:sequential_vlxoswgdqw_vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_2ð
&sequential/vlxoswgdqw/vdaevhnmja/mul_4Mul9sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_2:value:0*sequential/vlxoswgdqw/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/mul_4è
&sequential/vlxoswgdqw/vdaevhnmja/add_4AddV2/sequential/vlxoswgdqw/vdaevhnmja/split:output:3*sequential/vlxoswgdqw/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/add_4Á
*sequential/vlxoswgdqw/vdaevhnmja/Sigmoid_2Sigmoid*sequential/vlxoswgdqw/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/vlxoswgdqw/vdaevhnmja/Sigmoid_2¸
'sequential/vlxoswgdqw/vdaevhnmja/Tanh_1Tanh*sequential/vlxoswgdqw/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/vlxoswgdqw/vdaevhnmja/Tanh_1æ
&sequential/vlxoswgdqw/vdaevhnmja/mul_5Mul.sequential/vlxoswgdqw/vdaevhnmja/Sigmoid_2:y:0+sequential/vlxoswgdqw/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/vdaevhnmja/mul_5»
3sequential/vlxoswgdqw/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/vlxoswgdqw/TensorArrayV2_1/element_shape
%sequential/vlxoswgdqw/TensorArrayV2_1TensorListReserve<sequential/vlxoswgdqw/TensorArrayV2_1/element_shape:output:0.sequential/vlxoswgdqw/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/vlxoswgdqw/TensorArrayV2_1z
sequential/vlxoswgdqw/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/vlxoswgdqw/time«
.sequential/vlxoswgdqw/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/vlxoswgdqw/while/maximum_iterations
(sequential/vlxoswgdqw/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/vlxoswgdqw/while/loop_counterö	
sequential/vlxoswgdqw/whileWhile1sequential/vlxoswgdqw/while/loop_counter:output:07sequential/vlxoswgdqw/while/maximum_iterations:output:0#sequential/vlxoswgdqw/time:output:0.sequential/vlxoswgdqw/TensorArrayV2_1:handle:0$sequential/vlxoswgdqw/zeros:output:0&sequential/vlxoswgdqw/zeros_1:output:0.sequential/vlxoswgdqw/strided_slice_1:output:0Msequential/vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resourceAsequential_vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource@sequential_vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource8sequential_vlxoswgdqw_vdaevhnmja_readvariableop_resource:sequential_vlxoswgdqw_vdaevhnmja_readvariableop_1_resource:sequential_vlxoswgdqw_vdaevhnmja_readvariableop_2_resource*
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
'sequential_vlxoswgdqw_while_body_220279*3
cond+R)
'sequential_vlxoswgdqw_while_cond_220278*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/vlxoswgdqw/whileá
Fsequential/vlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/vlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/vlxoswgdqw/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/vlxoswgdqw/while:output:3Osequential/vlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/vlxoswgdqw/TensorArrayV2Stack/TensorListStack­
+sequential/vlxoswgdqw/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/vlxoswgdqw/strided_slice_3/stack¨
-sequential/vlxoswgdqw/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/vlxoswgdqw/strided_slice_3/stack_1¨
-sequential/vlxoswgdqw/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vlxoswgdqw/strided_slice_3/stack_2
%sequential/vlxoswgdqw/strided_slice_3StridedSliceAsequential/vlxoswgdqw/TensorArrayV2Stack/TensorListStack:tensor:04sequential/vlxoswgdqw/strided_slice_3/stack:output:06sequential/vlxoswgdqw/strided_slice_3/stack_1:output:06sequential/vlxoswgdqw/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/vlxoswgdqw/strided_slice_3¥
&sequential/vlxoswgdqw/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/vlxoswgdqw/transpose_1/permý
!sequential/vlxoswgdqw/transpose_1	TransposeAsequential/vlxoswgdqw/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/vlxoswgdqw/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/vlxoswgdqw/transpose_1Ï
+sequential/iktogmlrmp/MatMul/ReadVariableOpReadVariableOp4sequential_iktogmlrmp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/iktogmlrmp/MatMul/ReadVariableOpÝ
sequential/iktogmlrmp/MatMulMatMul.sequential/vlxoswgdqw/strided_slice_3:output:03sequential/iktogmlrmp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/iktogmlrmp/MatMulÎ
,sequential/iktogmlrmp/BiasAdd/ReadVariableOpReadVariableOp5sequential_iktogmlrmp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/iktogmlrmp/BiasAdd/ReadVariableOpÙ
sequential/iktogmlrmp/BiasAddBiasAdd&sequential/iktogmlrmp/MatMul:product:04sequential/iktogmlrmp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/iktogmlrmp/BiasAdd 
IdentityIdentity&sequential/iktogmlrmp/BiasAdd:output:09^sequential/iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp@^sequential/iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp-^sequential/iktogmlrmp/BiasAdd/ReadVariableOp,^sequential/iktogmlrmp/MatMul/ReadVariableOp8^sequential/omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp7^sequential/omilqdycns/ddlymsxapn/MatMul/ReadVariableOp9^sequential/omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp0^sequential/omilqdycns/ddlymsxapn/ReadVariableOp2^sequential/omilqdycns/ddlymsxapn/ReadVariableOp_12^sequential/omilqdycns/ddlymsxapn/ReadVariableOp_2^sequential/omilqdycns/while8^sequential/vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp7^sequential/vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp9^sequential/vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp0^sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp2^sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_12^sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_2^sequential/vlxoswgdqw/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2t
8sequential/iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp8sequential/iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,sequential/iktogmlrmp/BiasAdd/ReadVariableOp,sequential/iktogmlrmp/BiasAdd/ReadVariableOp2Z
+sequential/iktogmlrmp/MatMul/ReadVariableOp+sequential/iktogmlrmp/MatMul/ReadVariableOp2r
7sequential/omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp7sequential/omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp2p
6sequential/omilqdycns/ddlymsxapn/MatMul/ReadVariableOp6sequential/omilqdycns/ddlymsxapn/MatMul/ReadVariableOp2t
8sequential/omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp8sequential/omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp2b
/sequential/omilqdycns/ddlymsxapn/ReadVariableOp/sequential/omilqdycns/ddlymsxapn/ReadVariableOp2f
1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_11sequential/omilqdycns/ddlymsxapn/ReadVariableOp_12f
1sequential/omilqdycns/ddlymsxapn/ReadVariableOp_21sequential/omilqdycns/ddlymsxapn/ReadVariableOp_22:
sequential/omilqdycns/whilesequential/omilqdycns/while2r
7sequential/vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp7sequential/vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp2p
6sequential/vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp6sequential/vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp2t
8sequential/vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp8sequential/vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp2b
/sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp/sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp2f
1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_11sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_12f
1sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_21sequential/vlxoswgdqw/vdaevhnmja/ReadVariableOp_22:
sequential/vlxoswgdqw/whilesequential/vlxoswgdqw/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
ahzwxypkrh
Ýh

F__inference_omilqdycns_layer_call_and_return_conditional_losses_224512
inputs_0<
)ddlymsxapn_matmul_readvariableop_resource:	>
+ddlymsxapn_matmul_1_readvariableop_resource:	 9
*ddlymsxapn_biasadd_readvariableop_resource:	0
"ddlymsxapn_readvariableop_resource: 2
$ddlymsxapn_readvariableop_1_resource: 2
$ddlymsxapn_readvariableop_2_resource: 
identity¢!ddlymsxapn/BiasAdd/ReadVariableOp¢ ddlymsxapn/MatMul/ReadVariableOp¢"ddlymsxapn/MatMul_1/ReadVariableOp¢ddlymsxapn/ReadVariableOp¢ddlymsxapn/ReadVariableOp_1¢ddlymsxapn/ReadVariableOp_2¢whileF
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
 ddlymsxapn/MatMul/ReadVariableOpReadVariableOp)ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ddlymsxapn/MatMul/ReadVariableOp§
ddlymsxapn/MatMulMatMulstrided_slice_2:output:0(ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMulµ
"ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp+ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ddlymsxapn/MatMul_1/ReadVariableOp£
ddlymsxapn/MatMul_1MatMulzeros:output:0*ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMul_1
ddlymsxapn/addAddV2ddlymsxapn/MatMul:product:0ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/add®
!ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp*ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ddlymsxapn/BiasAdd/ReadVariableOp¥
ddlymsxapn/BiasAddBiasAddddlymsxapn/add:z:0)ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/BiasAddz
ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ddlymsxapn/split/split_dimë
ddlymsxapn/splitSplit#ddlymsxapn/split/split_dim:output:0ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ddlymsxapn/split
ddlymsxapn/ReadVariableOpReadVariableOp"ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp
ddlymsxapn/mulMul!ddlymsxapn/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul
ddlymsxapn/add_1AddV2ddlymsxapn/split:output:0ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_1{
ddlymsxapn/SigmoidSigmoidddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid
ddlymsxapn/ReadVariableOp_1ReadVariableOp$ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_1
ddlymsxapn/mul_1Mul#ddlymsxapn/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_1
ddlymsxapn/add_2AddV2ddlymsxapn/split:output:1ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_2
ddlymsxapn/Sigmoid_1Sigmoidddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_1
ddlymsxapn/mul_2Mulddlymsxapn/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_2w
ddlymsxapn/TanhTanhddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh
ddlymsxapn/mul_3Mulddlymsxapn/Sigmoid:y:0ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_3
ddlymsxapn/add_3AddV2ddlymsxapn/mul_2:z:0ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_3
ddlymsxapn/ReadVariableOp_2ReadVariableOp$ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_2
ddlymsxapn/mul_4Mul#ddlymsxapn/ReadVariableOp_2:value:0ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_4
ddlymsxapn/add_4AddV2ddlymsxapn/split:output:3ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_4
ddlymsxapn/Sigmoid_2Sigmoidddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_2v
ddlymsxapn/Tanh_1Tanhddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh_1
ddlymsxapn/mul_5Mulddlymsxapn/Sigmoid_2:y:0ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ddlymsxapn_matmul_readvariableop_resource+ddlymsxapn_matmul_1_readvariableop_resource*ddlymsxapn_biasadd_readvariableop_resource"ddlymsxapn_readvariableop_resource$ddlymsxapn_readvariableop_1_resource$ddlymsxapn_readvariableop_2_resource*
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
while_body_224411*
condR
while_cond_224410*Q
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
IdentityIdentitytranspose_1:y:0"^ddlymsxapn/BiasAdd/ReadVariableOp!^ddlymsxapn/MatMul/ReadVariableOp#^ddlymsxapn/MatMul_1/ReadVariableOp^ddlymsxapn/ReadVariableOp^ddlymsxapn/ReadVariableOp_1^ddlymsxapn/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ddlymsxapn/BiasAdd/ReadVariableOp!ddlymsxapn/BiasAdd/ReadVariableOp2D
 ddlymsxapn/MatMul/ReadVariableOp ddlymsxapn/MatMul/ReadVariableOp2H
"ddlymsxapn/MatMul_1/ReadVariableOp"ddlymsxapn/MatMul_1/ReadVariableOp26
ddlymsxapn/ReadVariableOpddlymsxapn/ReadVariableOp2:
ddlymsxapn/ReadVariableOp_1ddlymsxapn/ReadVariableOp_12:
ddlymsxapn/ReadVariableOp_2ddlymsxapn/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ò	
÷
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_225679

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
p
É
omilqdycns_while_body_2233332
.omilqdycns_while_omilqdycns_while_loop_counter8
4omilqdycns_while_omilqdycns_while_maximum_iterations 
omilqdycns_while_placeholder"
omilqdycns_while_placeholder_1"
omilqdycns_while_placeholder_2"
omilqdycns_while_placeholder_31
-omilqdycns_while_omilqdycns_strided_slice_1_0m
iomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0O
<omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0:	Q
>omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 L
=omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0:	C
5omilqdycns_while_ddlymsxapn_readvariableop_resource_0: E
7omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0: E
7omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0: 
omilqdycns_while_identity
omilqdycns_while_identity_1
omilqdycns_while_identity_2
omilqdycns_while_identity_3
omilqdycns_while_identity_4
omilqdycns_while_identity_5/
+omilqdycns_while_omilqdycns_strided_slice_1k
gomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensorM
:omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource:	O
<omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource:	 J
;omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource:	A
3omilqdycns_while_ddlymsxapn_readvariableop_resource: C
5omilqdycns_while_ddlymsxapn_readvariableop_1_resource: C
5omilqdycns_while_ddlymsxapn_readvariableop_2_resource: ¢2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp¢1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp¢3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp¢*omilqdycns/while/ddlymsxapn/ReadVariableOp¢,omilqdycns/while/ddlymsxapn/ReadVariableOp_1¢,omilqdycns/while/ddlymsxapn/ReadVariableOp_2Ù
Bomilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bomilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shape
4omilqdycns/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0omilqdycns_while_placeholderKomilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4omilqdycns/while/TensorArrayV2Read/TensorListGetItemä
1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp<omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOpý
"omilqdycns/while/ddlymsxapn/MatMulMatMul;omilqdycns/while/TensorArrayV2Read/TensorListGetItem:item:09omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"omilqdycns/while/ddlymsxapn/MatMulê
3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp>omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOpæ
$omilqdycns/while/ddlymsxapn/MatMul_1MatMulomilqdycns_while_placeholder_2;omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$omilqdycns/while/ddlymsxapn/MatMul_1Ü
omilqdycns/while/ddlymsxapn/addAddV2,omilqdycns/while/ddlymsxapn/MatMul:product:0.omilqdycns/while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
omilqdycns/while/ddlymsxapn/addã
2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp=omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOpé
#omilqdycns/while/ddlymsxapn/BiasAddBiasAdd#omilqdycns/while/ddlymsxapn/add:z:0:omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#omilqdycns/while/ddlymsxapn/BiasAdd
+omilqdycns/while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+omilqdycns/while/ddlymsxapn/split/split_dim¯
!omilqdycns/while/ddlymsxapn/splitSplit4omilqdycns/while/ddlymsxapn/split/split_dim:output:0,omilqdycns/while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!omilqdycns/while/ddlymsxapn/splitÊ
*omilqdycns/while/ddlymsxapn/ReadVariableOpReadVariableOp5omilqdycns_while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*omilqdycns/while/ddlymsxapn/ReadVariableOpÏ
omilqdycns/while/ddlymsxapn/mulMul2omilqdycns/while/ddlymsxapn/ReadVariableOp:value:0omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
omilqdycns/while/ddlymsxapn/mulÒ
!omilqdycns/while/ddlymsxapn/add_1AddV2*omilqdycns/while/ddlymsxapn/split:output:0#omilqdycns/while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_1®
#omilqdycns/while/ddlymsxapn/SigmoidSigmoid%omilqdycns/while/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#omilqdycns/while/ddlymsxapn/SigmoidÐ
,omilqdycns/while/ddlymsxapn/ReadVariableOp_1ReadVariableOp7omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,omilqdycns/while/ddlymsxapn/ReadVariableOp_1Õ
!omilqdycns/while/ddlymsxapn/mul_1Mul4omilqdycns/while/ddlymsxapn/ReadVariableOp_1:value:0omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_1Ô
!omilqdycns/while/ddlymsxapn/add_2AddV2*omilqdycns/while/ddlymsxapn/split:output:1%omilqdycns/while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_2²
%omilqdycns/while/ddlymsxapn/Sigmoid_1Sigmoid%omilqdycns/while/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%omilqdycns/while/ddlymsxapn/Sigmoid_1Ê
!omilqdycns/while/ddlymsxapn/mul_2Mul)omilqdycns/while/ddlymsxapn/Sigmoid_1:y:0omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_2ª
 omilqdycns/while/ddlymsxapn/TanhTanh*omilqdycns/while/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 omilqdycns/while/ddlymsxapn/TanhÎ
!omilqdycns/while/ddlymsxapn/mul_3Mul'omilqdycns/while/ddlymsxapn/Sigmoid:y:0$omilqdycns/while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_3Ï
!omilqdycns/while/ddlymsxapn/add_3AddV2%omilqdycns/while/ddlymsxapn/mul_2:z:0%omilqdycns/while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_3Ð
,omilqdycns/while/ddlymsxapn/ReadVariableOp_2ReadVariableOp7omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,omilqdycns/while/ddlymsxapn/ReadVariableOp_2Ü
!omilqdycns/while/ddlymsxapn/mul_4Mul4omilqdycns/while/ddlymsxapn/ReadVariableOp_2:value:0%omilqdycns/while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_4Ô
!omilqdycns/while/ddlymsxapn/add_4AddV2*omilqdycns/while/ddlymsxapn/split:output:3%omilqdycns/while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/add_4²
%omilqdycns/while/ddlymsxapn/Sigmoid_2Sigmoid%omilqdycns/while/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%omilqdycns/while/ddlymsxapn/Sigmoid_2©
"omilqdycns/while/ddlymsxapn/Tanh_1Tanh%omilqdycns/while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"omilqdycns/while/ddlymsxapn/Tanh_1Ò
!omilqdycns/while/ddlymsxapn/mul_5Mul)omilqdycns/while/ddlymsxapn/Sigmoid_2:y:0&omilqdycns/while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!omilqdycns/while/ddlymsxapn/mul_5
5omilqdycns/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemomilqdycns_while_placeholder_1omilqdycns_while_placeholder%omilqdycns/while/ddlymsxapn/mul_5:z:0*
_output_shapes
: *
element_dtype027
5omilqdycns/while/TensorArrayV2Write/TensorListSetItemr
omilqdycns/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
omilqdycns/while/add/y
omilqdycns/while/addAddV2omilqdycns_while_placeholderomilqdycns/while/add/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/while/addv
omilqdycns/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
omilqdycns/while/add_1/y­
omilqdycns/while/add_1AddV2.omilqdycns_while_omilqdycns_while_loop_counter!omilqdycns/while/add_1/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/while/add_1©
omilqdycns/while/IdentityIdentityomilqdycns/while/add_1:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/IdentityÇ
omilqdycns/while/Identity_1Identity4omilqdycns_while_omilqdycns_while_maximum_iterations3^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/Identity_1«
omilqdycns/while/Identity_2Identityomilqdycns/while/add:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/Identity_2Ø
omilqdycns/while/Identity_3IdentityEomilqdycns/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
omilqdycns/while/Identity_3É
omilqdycns/while/Identity_4Identity%omilqdycns/while/ddlymsxapn/mul_5:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/while/Identity_4É
omilqdycns/while/Identity_5Identity%omilqdycns/while/ddlymsxapn/add_3:z:03^omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2^omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp4^omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp+^omilqdycns/while/ddlymsxapn/ReadVariableOp-^omilqdycns/while/ddlymsxapn/ReadVariableOp_1-^omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/while/Identity_5"|
;omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource=omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0"~
<omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource>omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0"z
:omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource<omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0"p
5omilqdycns_while_ddlymsxapn_readvariableop_1_resource7omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0"p
5omilqdycns_while_ddlymsxapn_readvariableop_2_resource7omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0"l
3omilqdycns_while_ddlymsxapn_readvariableop_resource5omilqdycns_while_ddlymsxapn_readvariableop_resource_0"?
omilqdycns_while_identity"omilqdycns/while/Identity:output:0"C
omilqdycns_while_identity_1$omilqdycns/while/Identity_1:output:0"C
omilqdycns_while_identity_2$omilqdycns/while/Identity_2:output:0"C
omilqdycns_while_identity_3$omilqdycns/while/Identity_3:output:0"C
omilqdycns_while_identity_4$omilqdycns/while/Identity_4:output:0"C
omilqdycns_while_identity_5$omilqdycns/while/Identity_5:output:0"\
+omilqdycns_while_omilqdycns_strided_slice_1-omilqdycns_while_omilqdycns_strided_slice_1_0"Ô
gomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensoriomilqdycns_while_tensorarrayv2read_tensorlistgetitem_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2f
1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp1omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp2j
3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp3omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp2X
*omilqdycns/while/ddlymsxapn/ReadVariableOp*omilqdycns/while/ddlymsxapn/ReadVariableOp2\
,omilqdycns/while/ddlymsxapn/ReadVariableOp_1,omilqdycns/while/ddlymsxapn/ReadVariableOp_12\
,omilqdycns/while/ddlymsxapn/ReadVariableOp_2,omilqdycns/while/ddlymsxapn/ReadVariableOp_2: 
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
+__inference_omilqdycns_layer_call_fn_224118
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2208362
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
p
É
vlxoswgdqw_while_body_2239132
.vlxoswgdqw_while_vlxoswgdqw_while_loop_counter8
4vlxoswgdqw_while_vlxoswgdqw_while_maximum_iterations 
vlxoswgdqw_while_placeholder"
vlxoswgdqw_while_placeholder_1"
vlxoswgdqw_while_placeholder_2"
vlxoswgdqw_while_placeholder_31
-vlxoswgdqw_while_vlxoswgdqw_strided_slice_1_0m
ivlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0O
<vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0:	 Q
>vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 L
=vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0:	C
5vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0: E
7vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0: E
7vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0: 
vlxoswgdqw_while_identity
vlxoswgdqw_while_identity_1
vlxoswgdqw_while_identity_2
vlxoswgdqw_while_identity_3
vlxoswgdqw_while_identity_4
vlxoswgdqw_while_identity_5/
+vlxoswgdqw_while_vlxoswgdqw_strided_slice_1k
gvlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensorM
:vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource:	 O
<vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource:	 J
;vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource:	A
3vlxoswgdqw_while_vdaevhnmja_readvariableop_resource: C
5vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource: C
5vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource: ¢2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp¢1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp¢3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp¢*vlxoswgdqw/while/vdaevhnmja/ReadVariableOp¢,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1¢,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2Ù
Bvlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bvlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shape
4vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemivlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0vlxoswgdqw_while_placeholderKvlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItemä
1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp<vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOpý
"vlxoswgdqw/while/vdaevhnmja/MatMulMatMul;vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem:item:09vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"vlxoswgdqw/while/vdaevhnmja/MatMulê
3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp>vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOpæ
$vlxoswgdqw/while/vdaevhnmja/MatMul_1MatMulvlxoswgdqw_while_placeholder_2;vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$vlxoswgdqw/while/vdaevhnmja/MatMul_1Ü
vlxoswgdqw/while/vdaevhnmja/addAddV2,vlxoswgdqw/while/vdaevhnmja/MatMul:product:0.vlxoswgdqw/while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
vlxoswgdqw/while/vdaevhnmja/addã
2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp=vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOpé
#vlxoswgdqw/while/vdaevhnmja/BiasAddBiasAdd#vlxoswgdqw/while/vdaevhnmja/add:z:0:vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#vlxoswgdqw/while/vdaevhnmja/BiasAdd
+vlxoswgdqw/while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+vlxoswgdqw/while/vdaevhnmja/split/split_dim¯
!vlxoswgdqw/while/vdaevhnmja/splitSplit4vlxoswgdqw/while/vdaevhnmja/split/split_dim:output:0,vlxoswgdqw/while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!vlxoswgdqw/while/vdaevhnmja/splitÊ
*vlxoswgdqw/while/vdaevhnmja/ReadVariableOpReadVariableOp5vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*vlxoswgdqw/while/vdaevhnmja/ReadVariableOpÏ
vlxoswgdqw/while/vdaevhnmja/mulMul2vlxoswgdqw/while/vdaevhnmja/ReadVariableOp:value:0vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vlxoswgdqw/while/vdaevhnmja/mulÒ
!vlxoswgdqw/while/vdaevhnmja/add_1AddV2*vlxoswgdqw/while/vdaevhnmja/split:output:0#vlxoswgdqw/while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_1®
#vlxoswgdqw/while/vdaevhnmja/SigmoidSigmoid%vlxoswgdqw/while/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#vlxoswgdqw/while/vdaevhnmja/SigmoidÐ
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1ReadVariableOp7vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1Õ
!vlxoswgdqw/while/vdaevhnmja/mul_1Mul4vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1:value:0vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_1Ô
!vlxoswgdqw/while/vdaevhnmja/add_2AddV2*vlxoswgdqw/while/vdaevhnmja/split:output:1%vlxoswgdqw/while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_2²
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_1Sigmoid%vlxoswgdqw/while/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_1Ê
!vlxoswgdqw/while/vdaevhnmja/mul_2Mul)vlxoswgdqw/while/vdaevhnmja/Sigmoid_1:y:0vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_2ª
 vlxoswgdqw/while/vdaevhnmja/TanhTanh*vlxoswgdqw/while/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 vlxoswgdqw/while/vdaevhnmja/TanhÎ
!vlxoswgdqw/while/vdaevhnmja/mul_3Mul'vlxoswgdqw/while/vdaevhnmja/Sigmoid:y:0$vlxoswgdqw/while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_3Ï
!vlxoswgdqw/while/vdaevhnmja/add_3AddV2%vlxoswgdqw/while/vdaevhnmja/mul_2:z:0%vlxoswgdqw/while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_3Ð
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2ReadVariableOp7vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2Ü
!vlxoswgdqw/while/vdaevhnmja/mul_4Mul4vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2:value:0%vlxoswgdqw/while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_4Ô
!vlxoswgdqw/while/vdaevhnmja/add_4AddV2*vlxoswgdqw/while/vdaevhnmja/split:output:3%vlxoswgdqw/while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_4²
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_2Sigmoid%vlxoswgdqw/while/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_2©
"vlxoswgdqw/while/vdaevhnmja/Tanh_1Tanh%vlxoswgdqw/while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"vlxoswgdqw/while/vdaevhnmja/Tanh_1Ò
!vlxoswgdqw/while/vdaevhnmja/mul_5Mul)vlxoswgdqw/while/vdaevhnmja/Sigmoid_2:y:0&vlxoswgdqw/while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_5
5vlxoswgdqw/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemvlxoswgdqw_while_placeholder_1vlxoswgdqw_while_placeholder%vlxoswgdqw/while/vdaevhnmja/mul_5:z:0*
_output_shapes
: *
element_dtype027
5vlxoswgdqw/while/TensorArrayV2Write/TensorListSetItemr
vlxoswgdqw/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
vlxoswgdqw/while/add/y
vlxoswgdqw/while/addAddV2vlxoswgdqw_while_placeholdervlxoswgdqw/while/add/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/while/addv
vlxoswgdqw/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
vlxoswgdqw/while/add_1/y­
vlxoswgdqw/while/add_1AddV2.vlxoswgdqw_while_vlxoswgdqw_while_loop_counter!vlxoswgdqw/while/add_1/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/while/add_1©
vlxoswgdqw/while/IdentityIdentityvlxoswgdqw/while/add_1:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/IdentityÇ
vlxoswgdqw/while/Identity_1Identity4vlxoswgdqw_while_vlxoswgdqw_while_maximum_iterations3^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Identity_1«
vlxoswgdqw/while/Identity_2Identityvlxoswgdqw/while/add:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Identity_2Ø
vlxoswgdqw/while/Identity_3IdentityEvlxoswgdqw/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Identity_3É
vlxoswgdqw/while/Identity_4Identity%vlxoswgdqw/while/vdaevhnmja/mul_5:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/while/Identity_4É
vlxoswgdqw/while/Identity_5Identity%vlxoswgdqw/while/vdaevhnmja/add_3:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/while/Identity_5"?
vlxoswgdqw_while_identity"vlxoswgdqw/while/Identity:output:0"C
vlxoswgdqw_while_identity_1$vlxoswgdqw/while/Identity_1:output:0"C
vlxoswgdqw_while_identity_2$vlxoswgdqw/while/Identity_2:output:0"C
vlxoswgdqw_while_identity_3$vlxoswgdqw/while/Identity_3:output:0"C
vlxoswgdqw_while_identity_4$vlxoswgdqw/while/Identity_4:output:0"C
vlxoswgdqw_while_identity_5$vlxoswgdqw/while/Identity_5:output:0"Ô
gvlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensorivlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0"|
;vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource=vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0"~
<vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource>vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0"z
:vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource<vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0"p
5vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource7vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0"p
5vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource7vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0"l
3vlxoswgdqw_while_vdaevhnmja_readvariableop_resource5vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0"\
+vlxoswgdqw_while_vlxoswgdqw_strided_slice_1-vlxoswgdqw_while_vlxoswgdqw_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2f
1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp2j
3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp2X
*vlxoswgdqw/while/vdaevhnmja/ReadVariableOp*vlxoswgdqw/while/vdaevhnmja/ReadVariableOp2\
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_12\
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2: 
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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_220660

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
ÿ
¿
+__inference_ddlymsxapn_layer_call_fn_225702

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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_2204732
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


vlxoswgdqw_while_cond_2235082
.vlxoswgdqw_while_vlxoswgdqw_while_loop_counter8
4vlxoswgdqw_while_vlxoswgdqw_while_maximum_iterations 
vlxoswgdqw_while_placeholder"
vlxoswgdqw_while_placeholder_1"
vlxoswgdqw_while_placeholder_2"
vlxoswgdqw_while_placeholder_34
0vlxoswgdqw_while_less_vlxoswgdqw_strided_slice_1J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223508___redundant_placeholder0J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223508___redundant_placeholder1J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223508___redundant_placeholder2J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223508___redundant_placeholder3J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223508___redundant_placeholder4J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223508___redundant_placeholder5J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223508___redundant_placeholder6
vlxoswgdqw_while_identity
§
vlxoswgdqw/while/LessLessvlxoswgdqw_while_placeholder0vlxoswgdqw_while_less_vlxoswgdqw_strided_slice_1*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Less~
vlxoswgdqw/while/IdentityIdentityvlxoswgdqw/while/Less:z:0*
T0
*
_output_shapes
: 2
vlxoswgdqw/while/Identity"?
vlxoswgdqw_while_identity"vlxoswgdqw/while/Identity:output:0*(
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
while_body_222727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ddlymsxapn_matmul_readvariableop_resource_0:	F
3while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 A
2while_ddlymsxapn_biasadd_readvariableop_resource_0:	8
*while_ddlymsxapn_readvariableop_resource_0: :
,while_ddlymsxapn_readvariableop_1_resource_0: :
,while_ddlymsxapn_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ddlymsxapn_matmul_readvariableop_resource:	D
1while_ddlymsxapn_matmul_1_readvariableop_resource:	 ?
0while_ddlymsxapn_biasadd_readvariableop_resource:	6
(while_ddlymsxapn_readvariableop_resource: 8
*while_ddlymsxapn_readvariableop_1_resource: 8
*while_ddlymsxapn_readvariableop_2_resource: ¢'while/ddlymsxapn/BiasAdd/ReadVariableOp¢&while/ddlymsxapn/MatMul/ReadVariableOp¢(while/ddlymsxapn/MatMul_1/ReadVariableOp¢while/ddlymsxapn/ReadVariableOp¢!while/ddlymsxapn/ReadVariableOp_1¢!while/ddlymsxapn/ReadVariableOp_2Ã
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
&while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp1while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ddlymsxapn/MatMul/ReadVariableOpÑ
while/ddlymsxapn/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMulÉ
(while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp3while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ddlymsxapn/MatMul_1/ReadVariableOpº
while/ddlymsxapn/MatMul_1MatMulwhile_placeholder_20while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMul_1°
while/ddlymsxapn/addAddV2!while/ddlymsxapn/MatMul:product:0#while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/addÂ
'while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp2while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ddlymsxapn/BiasAdd/ReadVariableOp½
while/ddlymsxapn/BiasAddBiasAddwhile/ddlymsxapn/add:z:0/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/BiasAdd
 while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ddlymsxapn/split/split_dim
while/ddlymsxapn/splitSplit)while/ddlymsxapn/split/split_dim:output:0!while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ddlymsxapn/split©
while/ddlymsxapn/ReadVariableOpReadVariableOp*while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ddlymsxapn/ReadVariableOp£
while/ddlymsxapn/mulMul'while/ddlymsxapn/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul¦
while/ddlymsxapn/add_1AddV2while/ddlymsxapn/split:output:0while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_1
while/ddlymsxapn/SigmoidSigmoidwhile/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid¯
!while/ddlymsxapn/ReadVariableOp_1ReadVariableOp,while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_1©
while/ddlymsxapn/mul_1Mul)while/ddlymsxapn/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_1¨
while/ddlymsxapn/add_2AddV2while/ddlymsxapn/split:output:1while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_2
while/ddlymsxapn/Sigmoid_1Sigmoidwhile/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_1
while/ddlymsxapn/mul_2Mulwhile/ddlymsxapn/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_2
while/ddlymsxapn/TanhTanhwhile/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh¢
while/ddlymsxapn/mul_3Mulwhile/ddlymsxapn/Sigmoid:y:0while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_3£
while/ddlymsxapn/add_3AddV2while/ddlymsxapn/mul_2:z:0while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_3¯
!while/ddlymsxapn/ReadVariableOp_2ReadVariableOp,while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_2°
while/ddlymsxapn/mul_4Mul)while/ddlymsxapn/ReadVariableOp_2:value:0while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_4¨
while/ddlymsxapn/add_4AddV2while/ddlymsxapn/split:output:3while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_4
while/ddlymsxapn/Sigmoid_2Sigmoidwhile/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_2
while/ddlymsxapn/Tanh_1Tanhwhile/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh_1¦
while/ddlymsxapn/mul_5Mulwhile/ddlymsxapn/Sigmoid_2:y:0while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ddlymsxapn/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ddlymsxapn/mul_5:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ddlymsxapn/add_3:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ddlymsxapn_biasadd_readvariableop_resource2while_ddlymsxapn_biasadd_readvariableop_resource_0"h
1while_ddlymsxapn_matmul_1_readvariableop_resource3while_ddlymsxapn_matmul_1_readvariableop_resource_0"d
/while_ddlymsxapn_matmul_readvariableop_resource1while_ddlymsxapn_matmul_readvariableop_resource_0"Z
*while_ddlymsxapn_readvariableop_1_resource,while_ddlymsxapn_readvariableop_1_resource_0"Z
*while_ddlymsxapn_readvariableop_2_resource,while_ddlymsxapn_readvariableop_2_resource_0"V
(while_ddlymsxapn_readvariableop_resource*while_ddlymsxapn_readvariableop_resource_0")
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
'while/ddlymsxapn/BiasAdd/ReadVariableOp'while/ddlymsxapn/BiasAdd/ReadVariableOp2P
&while/ddlymsxapn/MatMul/ReadVariableOp&while/ddlymsxapn/MatMul/ReadVariableOp2T
(while/ddlymsxapn/MatMul_1/ReadVariableOp(while/ddlymsxapn/MatMul_1/ReadVariableOp2B
while/ddlymsxapn/ReadVariableOpwhile/ddlymsxapn/ReadVariableOp2F
!while/ddlymsxapn/ReadVariableOp_1!while/ddlymsxapn/ReadVariableOp_12F
!while/ddlymsxapn/ReadVariableOp_2!while/ddlymsxapn/ReadVariableOp_2: 
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
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_221946

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
Ýh

F__inference_omilqdycns_layer_call_and_return_conditional_losses_224332
inputs_0<
)ddlymsxapn_matmul_readvariableop_resource:	>
+ddlymsxapn_matmul_1_readvariableop_resource:	 9
*ddlymsxapn_biasadd_readvariableop_resource:	0
"ddlymsxapn_readvariableop_resource: 2
$ddlymsxapn_readvariableop_1_resource: 2
$ddlymsxapn_readvariableop_2_resource: 
identity¢!ddlymsxapn/BiasAdd/ReadVariableOp¢ ddlymsxapn/MatMul/ReadVariableOp¢"ddlymsxapn/MatMul_1/ReadVariableOp¢ddlymsxapn/ReadVariableOp¢ddlymsxapn/ReadVariableOp_1¢ddlymsxapn/ReadVariableOp_2¢whileF
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
 ddlymsxapn/MatMul/ReadVariableOpReadVariableOp)ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ddlymsxapn/MatMul/ReadVariableOp§
ddlymsxapn/MatMulMatMulstrided_slice_2:output:0(ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMulµ
"ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp+ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ddlymsxapn/MatMul_1/ReadVariableOp£
ddlymsxapn/MatMul_1MatMulzeros:output:0*ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMul_1
ddlymsxapn/addAddV2ddlymsxapn/MatMul:product:0ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/add®
!ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp*ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ddlymsxapn/BiasAdd/ReadVariableOp¥
ddlymsxapn/BiasAddBiasAddddlymsxapn/add:z:0)ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/BiasAddz
ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ddlymsxapn/split/split_dimë
ddlymsxapn/splitSplit#ddlymsxapn/split/split_dim:output:0ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ddlymsxapn/split
ddlymsxapn/ReadVariableOpReadVariableOp"ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp
ddlymsxapn/mulMul!ddlymsxapn/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul
ddlymsxapn/add_1AddV2ddlymsxapn/split:output:0ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_1{
ddlymsxapn/SigmoidSigmoidddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid
ddlymsxapn/ReadVariableOp_1ReadVariableOp$ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_1
ddlymsxapn/mul_1Mul#ddlymsxapn/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_1
ddlymsxapn/add_2AddV2ddlymsxapn/split:output:1ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_2
ddlymsxapn/Sigmoid_1Sigmoidddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_1
ddlymsxapn/mul_2Mulddlymsxapn/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_2w
ddlymsxapn/TanhTanhddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh
ddlymsxapn/mul_3Mulddlymsxapn/Sigmoid:y:0ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_3
ddlymsxapn/add_3AddV2ddlymsxapn/mul_2:z:0ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_3
ddlymsxapn/ReadVariableOp_2ReadVariableOp$ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_2
ddlymsxapn/mul_4Mul#ddlymsxapn/ReadVariableOp_2:value:0ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_4
ddlymsxapn/add_4AddV2ddlymsxapn/split:output:3ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_4
ddlymsxapn/Sigmoid_2Sigmoidddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_2v
ddlymsxapn/Tanh_1Tanhddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh_1
ddlymsxapn/mul_5Mulddlymsxapn/Sigmoid_2:y:0ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ddlymsxapn_matmul_readvariableop_resource+ddlymsxapn_matmul_1_readvariableop_resource*ddlymsxapn_biasadd_readvariableop_resource"ddlymsxapn_readvariableop_resource$ddlymsxapn_readvariableop_1_resource$ddlymsxapn_readvariableop_2_resource*
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
while_body_224231*
condR
while_cond_224230*Q
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
IdentityIdentitytranspose_1:y:0"^ddlymsxapn/BiasAdd/ReadVariableOp!^ddlymsxapn/MatMul/ReadVariableOp#^ddlymsxapn/MatMul_1/ReadVariableOp^ddlymsxapn/ReadVariableOp^ddlymsxapn/ReadVariableOp_1^ddlymsxapn/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ddlymsxapn/BiasAdd/ReadVariableOp!ddlymsxapn/BiasAdd/ReadVariableOp2D
 ddlymsxapn/MatMul/ReadVariableOp ddlymsxapn/MatMul/ReadVariableOp2H
"ddlymsxapn/MatMul_1/ReadVariableOp"ddlymsxapn/MatMul_1/ReadVariableOp26
ddlymsxapn/ReadVariableOpddlymsxapn/ReadVariableOp2:
ddlymsxapn/ReadVariableOp_1ddlymsxapn/ReadVariableOp_12:
ddlymsxapn/ReadVariableOp_2ddlymsxapn/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

¡	
'sequential_vlxoswgdqw_while_cond_220278H
Dsequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_loop_counterN
Jsequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_maximum_iterations+
'sequential_vlxoswgdqw_while_placeholder-
)sequential_vlxoswgdqw_while_placeholder_1-
)sequential_vlxoswgdqw_while_placeholder_2-
)sequential_vlxoswgdqw_while_placeholder_3J
Fsequential_vlxoswgdqw_while_less_sequential_vlxoswgdqw_strided_slice_1`
\sequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_cond_220278___redundant_placeholder0`
\sequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_cond_220278___redundant_placeholder1`
\sequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_cond_220278___redundant_placeholder2`
\sequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_cond_220278___redundant_placeholder3`
\sequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_cond_220278___redundant_placeholder4`
\sequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_cond_220278___redundant_placeholder5`
\sequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_cond_220278___redundant_placeholder6(
$sequential_vlxoswgdqw_while_identity
Þ
 sequential/vlxoswgdqw/while/LessLess'sequential_vlxoswgdqw_while_placeholderFsequential_vlxoswgdqw_while_less_sequential_vlxoswgdqw_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/vlxoswgdqw/while/Less
$sequential/vlxoswgdqw/while/IdentityIdentity$sequential/vlxoswgdqw/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/vlxoswgdqw/while/Identity"U
$sequential_vlxoswgdqw_while_identity-sequential/vlxoswgdqw/while/Identity:output:0*(
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
omilqdycns_while_cond_2237362
.omilqdycns_while_omilqdycns_while_loop_counter8
4omilqdycns_while_omilqdycns_while_maximum_iterations 
omilqdycns_while_placeholder"
omilqdycns_while_placeholder_1"
omilqdycns_while_placeholder_2"
omilqdycns_while_placeholder_34
0omilqdycns_while_less_omilqdycns_strided_slice_1J
Fomilqdycns_while_omilqdycns_while_cond_223736___redundant_placeholder0J
Fomilqdycns_while_omilqdycns_while_cond_223736___redundant_placeholder1J
Fomilqdycns_while_omilqdycns_while_cond_223736___redundant_placeholder2J
Fomilqdycns_while_omilqdycns_while_cond_223736___redundant_placeholder3J
Fomilqdycns_while_omilqdycns_while_cond_223736___redundant_placeholder4J
Fomilqdycns_while_omilqdycns_while_cond_223736___redundant_placeholder5J
Fomilqdycns_while_omilqdycns_while_cond_223736___redundant_placeholder6
omilqdycns_while_identity
§
omilqdycns/while/LessLessomilqdycns_while_placeholder0omilqdycns_while_less_omilqdycns_strided_slice_1*
T0*
_output_shapes
: 2
omilqdycns/while/Less~
omilqdycns/while/IdentityIdentityomilqdycns/while/Less:z:0*
T0
*
_output_shapes
: 2
omilqdycns/while/Identity"?
omilqdycns_while_identity"omilqdycns/while/Identity:output:0*(
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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_225903

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
h

F__inference_omilqdycns_layer_call_and_return_conditional_losses_224692

inputs<
)ddlymsxapn_matmul_readvariableop_resource:	>
+ddlymsxapn_matmul_1_readvariableop_resource:	 9
*ddlymsxapn_biasadd_readvariableop_resource:	0
"ddlymsxapn_readvariableop_resource: 2
$ddlymsxapn_readvariableop_1_resource: 2
$ddlymsxapn_readvariableop_2_resource: 
identity¢!ddlymsxapn/BiasAdd/ReadVariableOp¢ ddlymsxapn/MatMul/ReadVariableOp¢"ddlymsxapn/MatMul_1/ReadVariableOp¢ddlymsxapn/ReadVariableOp¢ddlymsxapn/ReadVariableOp_1¢ddlymsxapn/ReadVariableOp_2¢whileD
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
 ddlymsxapn/MatMul/ReadVariableOpReadVariableOp)ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ddlymsxapn/MatMul/ReadVariableOp§
ddlymsxapn/MatMulMatMulstrided_slice_2:output:0(ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMulµ
"ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp+ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ddlymsxapn/MatMul_1/ReadVariableOp£
ddlymsxapn/MatMul_1MatMulzeros:output:0*ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMul_1
ddlymsxapn/addAddV2ddlymsxapn/MatMul:product:0ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/add®
!ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp*ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ddlymsxapn/BiasAdd/ReadVariableOp¥
ddlymsxapn/BiasAddBiasAddddlymsxapn/add:z:0)ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/BiasAddz
ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ddlymsxapn/split/split_dimë
ddlymsxapn/splitSplit#ddlymsxapn/split/split_dim:output:0ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ddlymsxapn/split
ddlymsxapn/ReadVariableOpReadVariableOp"ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp
ddlymsxapn/mulMul!ddlymsxapn/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul
ddlymsxapn/add_1AddV2ddlymsxapn/split:output:0ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_1{
ddlymsxapn/SigmoidSigmoidddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid
ddlymsxapn/ReadVariableOp_1ReadVariableOp$ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_1
ddlymsxapn/mul_1Mul#ddlymsxapn/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_1
ddlymsxapn/add_2AddV2ddlymsxapn/split:output:1ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_2
ddlymsxapn/Sigmoid_1Sigmoidddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_1
ddlymsxapn/mul_2Mulddlymsxapn/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_2w
ddlymsxapn/TanhTanhddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh
ddlymsxapn/mul_3Mulddlymsxapn/Sigmoid:y:0ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_3
ddlymsxapn/add_3AddV2ddlymsxapn/mul_2:z:0ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_3
ddlymsxapn/ReadVariableOp_2ReadVariableOp$ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_2
ddlymsxapn/mul_4Mul#ddlymsxapn/ReadVariableOp_2:value:0ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_4
ddlymsxapn/add_4AddV2ddlymsxapn/split:output:3ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_4
ddlymsxapn/Sigmoid_2Sigmoidddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_2v
ddlymsxapn/Tanh_1Tanhddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh_1
ddlymsxapn/mul_5Mulddlymsxapn/Sigmoid_2:y:0ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ddlymsxapn_matmul_readvariableop_resource+ddlymsxapn_matmul_1_readvariableop_resource*ddlymsxapn_biasadd_readvariableop_resource"ddlymsxapn_readvariableop_resource$ddlymsxapn_readvariableop_1_resource$ddlymsxapn_readvariableop_2_resource*
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
while_body_224591*
condR
while_cond_224590*Q
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
IdentityIdentitytranspose_1:y:0"^ddlymsxapn/BiasAdd/ReadVariableOp!^ddlymsxapn/MatMul/ReadVariableOp#^ddlymsxapn/MatMul_1/ReadVariableOp^ddlymsxapn/ReadVariableOp^ddlymsxapn/ReadVariableOp_1^ddlymsxapn/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ddlymsxapn/BiasAdd/ReadVariableOp!ddlymsxapn/BiasAdd/ReadVariableOp2D
 ddlymsxapn/MatMul/ReadVariableOp ddlymsxapn/MatMul/ReadVariableOp2H
"ddlymsxapn/MatMul_1/ReadVariableOp"ddlymsxapn/MatMul_1/ReadVariableOp26
ddlymsxapn/ReadVariableOpddlymsxapn/ReadVariableOp2:
ddlymsxapn/ReadVariableOp_1ddlymsxapn/ReadVariableOp_12:
ddlymsxapn/ReadVariableOp_2ddlymsxapn/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°'
²
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_221231

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
Ñ

+__inference_vlxoswgdqw_layer_call_fn_224923

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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2223392
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
Ç)
Å
while_body_221251
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_vdaevhnmja_221275_0:	 ,
while_vdaevhnmja_221277_0:	 (
while_vdaevhnmja_221279_0:	'
while_vdaevhnmja_221281_0: '
while_vdaevhnmja_221283_0: '
while_vdaevhnmja_221285_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_vdaevhnmja_221275:	 *
while_vdaevhnmja_221277:	 &
while_vdaevhnmja_221279:	%
while_vdaevhnmja_221281: %
while_vdaevhnmja_221283: %
while_vdaevhnmja_221285: ¢(while/vdaevhnmja/StatefulPartitionedCallÃ
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
(while/vdaevhnmja/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_vdaevhnmja_221275_0while_vdaevhnmja_221277_0while_vdaevhnmja_221279_0while_vdaevhnmja_221281_0while_vdaevhnmja_221283_0while_vdaevhnmja_221285_0*
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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_2212312*
(while/vdaevhnmja/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/vdaevhnmja/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/vdaevhnmja/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/vdaevhnmja/StatefulPartitionedCall:output:1)^while/vdaevhnmja/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/vdaevhnmja/StatefulPartitionedCall:output:2)^while/vdaevhnmja/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"4
while_vdaevhnmja_221275while_vdaevhnmja_221275_0"4
while_vdaevhnmja_221277while_vdaevhnmja_221277_0"4
while_vdaevhnmja_221279while_vdaevhnmja_221279_0"4
while_vdaevhnmja_221281while_vdaevhnmja_221281_0"4
while_vdaevhnmja_221283while_vdaevhnmja_221283_0"4
while_vdaevhnmja_221285while_vdaevhnmja_221285_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/vdaevhnmja/StatefulPartitionedCall(while/vdaevhnmja/StatefulPartitionedCall: 
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224872

inputs<
)ddlymsxapn_matmul_readvariableop_resource:	>
+ddlymsxapn_matmul_1_readvariableop_resource:	 9
*ddlymsxapn_biasadd_readvariableop_resource:	0
"ddlymsxapn_readvariableop_resource: 2
$ddlymsxapn_readvariableop_1_resource: 2
$ddlymsxapn_readvariableop_2_resource: 
identity¢!ddlymsxapn/BiasAdd/ReadVariableOp¢ ddlymsxapn/MatMul/ReadVariableOp¢"ddlymsxapn/MatMul_1/ReadVariableOp¢ddlymsxapn/ReadVariableOp¢ddlymsxapn/ReadVariableOp_1¢ddlymsxapn/ReadVariableOp_2¢whileD
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
 ddlymsxapn/MatMul/ReadVariableOpReadVariableOp)ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ddlymsxapn/MatMul/ReadVariableOp§
ddlymsxapn/MatMulMatMulstrided_slice_2:output:0(ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMulµ
"ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp+ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ddlymsxapn/MatMul_1/ReadVariableOp£
ddlymsxapn/MatMul_1MatMulzeros:output:0*ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMul_1
ddlymsxapn/addAddV2ddlymsxapn/MatMul:product:0ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/add®
!ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp*ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ddlymsxapn/BiasAdd/ReadVariableOp¥
ddlymsxapn/BiasAddBiasAddddlymsxapn/add:z:0)ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/BiasAddz
ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ddlymsxapn/split/split_dimë
ddlymsxapn/splitSplit#ddlymsxapn/split/split_dim:output:0ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ddlymsxapn/split
ddlymsxapn/ReadVariableOpReadVariableOp"ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp
ddlymsxapn/mulMul!ddlymsxapn/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul
ddlymsxapn/add_1AddV2ddlymsxapn/split:output:0ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_1{
ddlymsxapn/SigmoidSigmoidddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid
ddlymsxapn/ReadVariableOp_1ReadVariableOp$ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_1
ddlymsxapn/mul_1Mul#ddlymsxapn/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_1
ddlymsxapn/add_2AddV2ddlymsxapn/split:output:1ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_2
ddlymsxapn/Sigmoid_1Sigmoidddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_1
ddlymsxapn/mul_2Mulddlymsxapn/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_2w
ddlymsxapn/TanhTanhddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh
ddlymsxapn/mul_3Mulddlymsxapn/Sigmoid:y:0ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_3
ddlymsxapn/add_3AddV2ddlymsxapn/mul_2:z:0ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_3
ddlymsxapn/ReadVariableOp_2ReadVariableOp$ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_2
ddlymsxapn/mul_4Mul#ddlymsxapn/ReadVariableOp_2:value:0ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_4
ddlymsxapn/add_4AddV2ddlymsxapn/split:output:3ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_4
ddlymsxapn/Sigmoid_2Sigmoidddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_2v
ddlymsxapn/Tanh_1Tanhddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh_1
ddlymsxapn/mul_5Mulddlymsxapn/Sigmoid_2:y:0ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ddlymsxapn_matmul_readvariableop_resource+ddlymsxapn_matmul_1_readvariableop_resource*ddlymsxapn_biasadd_readvariableop_resource"ddlymsxapn_readvariableop_resource$ddlymsxapn_readvariableop_1_resource$ddlymsxapn_readvariableop_2_resource*
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
while_body_224771*
condR
while_cond_224770*Q
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
IdentityIdentitytranspose_1:y:0"^ddlymsxapn/BiasAdd/ReadVariableOp!^ddlymsxapn/MatMul/ReadVariableOp#^ddlymsxapn/MatMul_1/ReadVariableOp^ddlymsxapn/ReadVariableOp^ddlymsxapn/ReadVariableOp_1^ddlymsxapn/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ddlymsxapn/BiasAdd/ReadVariableOp!ddlymsxapn/BiasAdd/ReadVariableOp2D
 ddlymsxapn/MatMul/ReadVariableOp ddlymsxapn/MatMul/ReadVariableOp2H
"ddlymsxapn/MatMul_1/ReadVariableOp"ddlymsxapn/MatMul_1/ReadVariableOp26
ddlymsxapn/ReadVariableOpddlymsxapn/ReadVariableOp2:
ddlymsxapn/ReadVariableOp_1ddlymsxapn/ReadVariableOp_12:
ddlymsxapn/ReadVariableOp_2ddlymsxapn/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_222044
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_222044___redundant_placeholder04
0while_while_cond_222044___redundant_placeholder14
0while_while_cond_222044___redundant_placeholder24
0while_while_cond_222044___redundant_placeholder34
0while_while_cond_222044___redundant_placeholder44
0while_while_cond_222044___redundant_placeholder54
0while_while_cond_222044___redundant_placeholder6
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


+__inference_sequential_layer_call_fn_223212

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
F__inference_sequential_layer_call_and_return_conditional_losses_2229392
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
ßY

while_body_222238
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_vdaevhnmja_matmul_readvariableop_resource_0:	 F
3while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 A
2while_vdaevhnmja_biasadd_readvariableop_resource_0:	8
*while_vdaevhnmja_readvariableop_resource_0: :
,while_vdaevhnmja_readvariableop_1_resource_0: :
,while_vdaevhnmja_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_vdaevhnmja_matmul_readvariableop_resource:	 D
1while_vdaevhnmja_matmul_1_readvariableop_resource:	 ?
0while_vdaevhnmja_biasadd_readvariableop_resource:	6
(while_vdaevhnmja_readvariableop_resource: 8
*while_vdaevhnmja_readvariableop_1_resource: 8
*while_vdaevhnmja_readvariableop_2_resource: ¢'while/vdaevhnmja/BiasAdd/ReadVariableOp¢&while/vdaevhnmja/MatMul/ReadVariableOp¢(while/vdaevhnmja/MatMul_1/ReadVariableOp¢while/vdaevhnmja/ReadVariableOp¢!while/vdaevhnmja/ReadVariableOp_1¢!while/vdaevhnmja/ReadVariableOp_2Ã
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
&while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp1while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/vdaevhnmja/MatMul/ReadVariableOpÑ
while/vdaevhnmja/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMulÉ
(while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp3while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/vdaevhnmja/MatMul_1/ReadVariableOpº
while/vdaevhnmja/MatMul_1MatMulwhile_placeholder_20while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMul_1°
while/vdaevhnmja/addAddV2!while/vdaevhnmja/MatMul:product:0#while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/addÂ
'while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp2while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/vdaevhnmja/BiasAdd/ReadVariableOp½
while/vdaevhnmja/BiasAddBiasAddwhile/vdaevhnmja/add:z:0/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/BiasAdd
 while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/vdaevhnmja/split/split_dim
while/vdaevhnmja/splitSplit)while/vdaevhnmja/split/split_dim:output:0!while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/vdaevhnmja/split©
while/vdaevhnmja/ReadVariableOpReadVariableOp*while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/vdaevhnmja/ReadVariableOp£
while/vdaevhnmja/mulMul'while/vdaevhnmja/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul¦
while/vdaevhnmja/add_1AddV2while/vdaevhnmja/split:output:0while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_1
while/vdaevhnmja/SigmoidSigmoidwhile/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid¯
!while/vdaevhnmja/ReadVariableOp_1ReadVariableOp,while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_1©
while/vdaevhnmja/mul_1Mul)while/vdaevhnmja/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_1¨
while/vdaevhnmja/add_2AddV2while/vdaevhnmja/split:output:1while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_2
while/vdaevhnmja/Sigmoid_1Sigmoidwhile/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_1
while/vdaevhnmja/mul_2Mulwhile/vdaevhnmja/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_2
while/vdaevhnmja/TanhTanhwhile/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh¢
while/vdaevhnmja/mul_3Mulwhile/vdaevhnmja/Sigmoid:y:0while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_3£
while/vdaevhnmja/add_3AddV2while/vdaevhnmja/mul_2:z:0while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_3¯
!while/vdaevhnmja/ReadVariableOp_2ReadVariableOp,while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_2°
while/vdaevhnmja/mul_4Mul)while/vdaevhnmja/ReadVariableOp_2:value:0while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_4¨
while/vdaevhnmja/add_4AddV2while/vdaevhnmja/split:output:3while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_4
while/vdaevhnmja/Sigmoid_2Sigmoidwhile/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_2
while/vdaevhnmja/Tanh_1Tanhwhile/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh_1¦
while/vdaevhnmja/mul_5Mulwhile/vdaevhnmja/Sigmoid_2:y:0while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/vdaevhnmja/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/vdaevhnmja/mul_5:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/vdaevhnmja/add_3:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"f
0while_vdaevhnmja_biasadd_readvariableop_resource2while_vdaevhnmja_biasadd_readvariableop_resource_0"h
1while_vdaevhnmja_matmul_1_readvariableop_resource3while_vdaevhnmja_matmul_1_readvariableop_resource_0"d
/while_vdaevhnmja_matmul_readvariableop_resource1while_vdaevhnmja_matmul_readvariableop_resource_0"Z
*while_vdaevhnmja_readvariableop_1_resource,while_vdaevhnmja_readvariableop_1_resource_0"Z
*while_vdaevhnmja_readvariableop_2_resource,while_vdaevhnmja_readvariableop_2_resource_0"V
(while_vdaevhnmja_readvariableop_resource*while_vdaevhnmja_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/vdaevhnmja/BiasAdd/ReadVariableOp'while/vdaevhnmja/BiasAdd/ReadVariableOp2P
&while/vdaevhnmja/MatMul/ReadVariableOp&while/vdaevhnmja/MatMul/ReadVariableOp2T
(while/vdaevhnmja/MatMul_1/ReadVariableOp(while/vdaevhnmja/MatMul_1/ReadVariableOp2B
while/vdaevhnmja/ReadVariableOpwhile/vdaevhnmja/ReadVariableOp2F
!while/vdaevhnmja/ReadVariableOp_1!while/vdaevhnmja/ReadVariableOp_12F
!while/vdaevhnmja/ReadVariableOp_2!while/vdaevhnmja/ReadVariableOp_2: 
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
while_cond_222512
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_222512___redundant_placeholder04
0while_while_cond_222512___redundant_placeholder14
0while_while_cond_222512___redundant_placeholder24
0while_while_cond_222512___redundant_placeholder34
0while_while_cond_222512___redundant_placeholder44
0while_while_cond_222512___redundant_placeholder54
0while_while_cond_222512___redundant_placeholder6
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
while_cond_220755
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_220755___redundant_placeholder04
0while_while_cond_220755___redundant_placeholder14
0while_while_cond_220755___redundant_placeholder24
0while_while_cond_220755___redundant_placeholder34
0while_while_cond_220755___redundant_placeholder44
0while_while_cond_220755___redundant_placeholder54
0while_while_cond_220755___redundant_placeholder6
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
while_body_225379
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_vdaevhnmja_matmul_readvariableop_resource_0:	 F
3while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 A
2while_vdaevhnmja_biasadd_readvariableop_resource_0:	8
*while_vdaevhnmja_readvariableop_resource_0: :
,while_vdaevhnmja_readvariableop_1_resource_0: :
,while_vdaevhnmja_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_vdaevhnmja_matmul_readvariableop_resource:	 D
1while_vdaevhnmja_matmul_1_readvariableop_resource:	 ?
0while_vdaevhnmja_biasadd_readvariableop_resource:	6
(while_vdaevhnmja_readvariableop_resource: 8
*while_vdaevhnmja_readvariableop_1_resource: 8
*while_vdaevhnmja_readvariableop_2_resource: ¢'while/vdaevhnmja/BiasAdd/ReadVariableOp¢&while/vdaevhnmja/MatMul/ReadVariableOp¢(while/vdaevhnmja/MatMul_1/ReadVariableOp¢while/vdaevhnmja/ReadVariableOp¢!while/vdaevhnmja/ReadVariableOp_1¢!while/vdaevhnmja/ReadVariableOp_2Ã
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
&while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp1while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/vdaevhnmja/MatMul/ReadVariableOpÑ
while/vdaevhnmja/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMulÉ
(while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp3while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/vdaevhnmja/MatMul_1/ReadVariableOpº
while/vdaevhnmja/MatMul_1MatMulwhile_placeholder_20while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMul_1°
while/vdaevhnmja/addAddV2!while/vdaevhnmja/MatMul:product:0#while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/addÂ
'while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp2while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/vdaevhnmja/BiasAdd/ReadVariableOp½
while/vdaevhnmja/BiasAddBiasAddwhile/vdaevhnmja/add:z:0/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/BiasAdd
 while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/vdaevhnmja/split/split_dim
while/vdaevhnmja/splitSplit)while/vdaevhnmja/split/split_dim:output:0!while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/vdaevhnmja/split©
while/vdaevhnmja/ReadVariableOpReadVariableOp*while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/vdaevhnmja/ReadVariableOp£
while/vdaevhnmja/mulMul'while/vdaevhnmja/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul¦
while/vdaevhnmja/add_1AddV2while/vdaevhnmja/split:output:0while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_1
while/vdaevhnmja/SigmoidSigmoidwhile/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid¯
!while/vdaevhnmja/ReadVariableOp_1ReadVariableOp,while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_1©
while/vdaevhnmja/mul_1Mul)while/vdaevhnmja/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_1¨
while/vdaevhnmja/add_2AddV2while/vdaevhnmja/split:output:1while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_2
while/vdaevhnmja/Sigmoid_1Sigmoidwhile/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_1
while/vdaevhnmja/mul_2Mulwhile/vdaevhnmja/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_2
while/vdaevhnmja/TanhTanhwhile/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh¢
while/vdaevhnmja/mul_3Mulwhile/vdaevhnmja/Sigmoid:y:0while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_3£
while/vdaevhnmja/add_3AddV2while/vdaevhnmja/mul_2:z:0while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_3¯
!while/vdaevhnmja/ReadVariableOp_2ReadVariableOp,while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_2°
while/vdaevhnmja/mul_4Mul)while/vdaevhnmja/ReadVariableOp_2:value:0while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_4¨
while/vdaevhnmja/add_4AddV2while/vdaevhnmja/split:output:3while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_4
while/vdaevhnmja/Sigmoid_2Sigmoidwhile/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_2
while/vdaevhnmja/Tanh_1Tanhwhile/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh_1¦
while/vdaevhnmja/mul_5Mulwhile/vdaevhnmja/Sigmoid_2:y:0while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/vdaevhnmja/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/vdaevhnmja/mul_5:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/vdaevhnmja/add_3:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"f
0while_vdaevhnmja_biasadd_readvariableop_resource2while_vdaevhnmja_biasadd_readvariableop_resource_0"h
1while_vdaevhnmja_matmul_1_readvariableop_resource3while_vdaevhnmja_matmul_1_readvariableop_resource_0"d
/while_vdaevhnmja_matmul_readvariableop_resource1while_vdaevhnmja_matmul_readvariableop_resource_0"Z
*while_vdaevhnmja_readvariableop_1_resource,while_vdaevhnmja_readvariableop_1_resource_0"Z
*while_vdaevhnmja_readvariableop_2_resource,while_vdaevhnmja_readvariableop_2_resource_0"V
(while_vdaevhnmja_readvariableop_resource*while_vdaevhnmja_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/vdaevhnmja/BiasAdd/ReadVariableOp'while/vdaevhnmja/BiasAdd/ReadVariableOp2P
&while/vdaevhnmja/MatMul/ReadVariableOp&while/vdaevhnmja/MatMul/ReadVariableOp2T
(while/vdaevhnmja/MatMul_1/ReadVariableOp(while/vdaevhnmja/MatMul_1/ReadVariableOp2B
while/vdaevhnmja/ReadVariableOpwhile/vdaevhnmja/ReadVariableOp2F
!while/vdaevhnmja/ReadVariableOp_1!while/vdaevhnmja/ReadVariableOp_12F
!while/vdaevhnmja/ReadVariableOp_2!while/vdaevhnmja/ReadVariableOp_2: 
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
while_body_222513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_vdaevhnmja_matmul_readvariableop_resource_0:	 F
3while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 A
2while_vdaevhnmja_biasadd_readvariableop_resource_0:	8
*while_vdaevhnmja_readvariableop_resource_0: :
,while_vdaevhnmja_readvariableop_1_resource_0: :
,while_vdaevhnmja_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_vdaevhnmja_matmul_readvariableop_resource:	 D
1while_vdaevhnmja_matmul_1_readvariableop_resource:	 ?
0while_vdaevhnmja_biasadd_readvariableop_resource:	6
(while_vdaevhnmja_readvariableop_resource: 8
*while_vdaevhnmja_readvariableop_1_resource: 8
*while_vdaevhnmja_readvariableop_2_resource: ¢'while/vdaevhnmja/BiasAdd/ReadVariableOp¢&while/vdaevhnmja/MatMul/ReadVariableOp¢(while/vdaevhnmja/MatMul_1/ReadVariableOp¢while/vdaevhnmja/ReadVariableOp¢!while/vdaevhnmja/ReadVariableOp_1¢!while/vdaevhnmja/ReadVariableOp_2Ã
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
&while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp1while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/vdaevhnmja/MatMul/ReadVariableOpÑ
while/vdaevhnmja/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMulÉ
(while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp3while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/vdaevhnmja/MatMul_1/ReadVariableOpº
while/vdaevhnmja/MatMul_1MatMulwhile_placeholder_20while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMul_1°
while/vdaevhnmja/addAddV2!while/vdaevhnmja/MatMul:product:0#while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/addÂ
'while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp2while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/vdaevhnmja/BiasAdd/ReadVariableOp½
while/vdaevhnmja/BiasAddBiasAddwhile/vdaevhnmja/add:z:0/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/BiasAdd
 while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/vdaevhnmja/split/split_dim
while/vdaevhnmja/splitSplit)while/vdaevhnmja/split/split_dim:output:0!while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/vdaevhnmja/split©
while/vdaevhnmja/ReadVariableOpReadVariableOp*while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/vdaevhnmja/ReadVariableOp£
while/vdaevhnmja/mulMul'while/vdaevhnmja/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul¦
while/vdaevhnmja/add_1AddV2while/vdaevhnmja/split:output:0while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_1
while/vdaevhnmja/SigmoidSigmoidwhile/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid¯
!while/vdaevhnmja/ReadVariableOp_1ReadVariableOp,while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_1©
while/vdaevhnmja/mul_1Mul)while/vdaevhnmja/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_1¨
while/vdaevhnmja/add_2AddV2while/vdaevhnmja/split:output:1while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_2
while/vdaevhnmja/Sigmoid_1Sigmoidwhile/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_1
while/vdaevhnmja/mul_2Mulwhile/vdaevhnmja/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_2
while/vdaevhnmja/TanhTanhwhile/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh¢
while/vdaevhnmja/mul_3Mulwhile/vdaevhnmja/Sigmoid:y:0while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_3£
while/vdaevhnmja/add_3AddV2while/vdaevhnmja/mul_2:z:0while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_3¯
!while/vdaevhnmja/ReadVariableOp_2ReadVariableOp,while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_2°
while/vdaevhnmja/mul_4Mul)while/vdaevhnmja/ReadVariableOp_2:value:0while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_4¨
while/vdaevhnmja/add_4AddV2while/vdaevhnmja/split:output:3while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_4
while/vdaevhnmja/Sigmoid_2Sigmoidwhile/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_2
while/vdaevhnmja/Tanh_1Tanhwhile/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh_1¦
while/vdaevhnmja/mul_5Mulwhile/vdaevhnmja/Sigmoid_2:y:0while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/vdaevhnmja/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/vdaevhnmja/mul_5:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/vdaevhnmja/add_3:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"f
0while_vdaevhnmja_biasadd_readvariableop_resource2while_vdaevhnmja_biasadd_readvariableop_resource_0"h
1while_vdaevhnmja_matmul_1_readvariableop_resource3while_vdaevhnmja_matmul_1_readvariableop_resource_0"d
/while_vdaevhnmja_matmul_readvariableop_resource1while_vdaevhnmja_matmul_readvariableop_resource_0"Z
*while_vdaevhnmja_readvariableop_1_resource,while_vdaevhnmja_readvariableop_1_resource_0"Z
*while_vdaevhnmja_readvariableop_2_resource,while_vdaevhnmja_readvariableop_2_resource_0"V
(while_vdaevhnmja_readvariableop_resource*while_vdaevhnmja_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/vdaevhnmja/BiasAdd/ReadVariableOp'while/vdaevhnmja/BiasAdd/ReadVariableOp2P
&while/vdaevhnmja/MatMul/ReadVariableOp&while/vdaevhnmja/MatMul/ReadVariableOp2T
(while/vdaevhnmja/MatMul_1/ReadVariableOp(while/vdaevhnmja/MatMul_1/ReadVariableOp2B
while/vdaevhnmja/ReadVariableOpwhile/vdaevhnmja/ReadVariableOp2F
!while/vdaevhnmja/ReadVariableOp_1!while/vdaevhnmja/ReadVariableOp_12F
!while/vdaevhnmja/ReadVariableOp_2!while/vdaevhnmja/ReadVariableOp_2: 
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
Ç)
Å
while_body_220756
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_ddlymsxapn_220780_0:	,
while_ddlymsxapn_220782_0:	 (
while_ddlymsxapn_220784_0:	'
while_ddlymsxapn_220786_0: '
while_ddlymsxapn_220788_0: '
while_ddlymsxapn_220790_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_ddlymsxapn_220780:	*
while_ddlymsxapn_220782:	 &
while_ddlymsxapn_220784:	%
while_ddlymsxapn_220786: %
while_ddlymsxapn_220788: %
while_ddlymsxapn_220790: ¢(while/ddlymsxapn/StatefulPartitionedCallÃ
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
(while/ddlymsxapn/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_ddlymsxapn_220780_0while_ddlymsxapn_220782_0while_ddlymsxapn_220784_0while_ddlymsxapn_220786_0while_ddlymsxapn_220788_0while_ddlymsxapn_220790_0*
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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_2206602*
(while/ddlymsxapn/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/ddlymsxapn/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/ddlymsxapn/StatefulPartitionedCall:output:1)^while/ddlymsxapn/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/ddlymsxapn/StatefulPartitionedCall:output:2)^while/ddlymsxapn/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_ddlymsxapn_220780while_ddlymsxapn_220780_0"4
while_ddlymsxapn_220782while_ddlymsxapn_220782_0"4
while_ddlymsxapn_220784while_ddlymsxapn_220784_0"4
while_ddlymsxapn_220786while_ddlymsxapn_220786_0"4
while_ddlymsxapn_220788while_ddlymsxapn_220788_0"4
while_ddlymsxapn_220790while_ddlymsxapn_220790_0")
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
(while/ddlymsxapn/StatefulPartitionedCall(while/ddlymsxapn/StatefulPartitionedCall: 
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

b
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_224084

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
ê

$__inference_signature_wrapper_223138

ahzwxypkrh
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
ahzwxypkrhunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_2203862
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
ahzwxypkrh
h

F__inference_omilqdycns_layer_call_and_return_conditional_losses_222146

inputs<
)ddlymsxapn_matmul_readvariableop_resource:	>
+ddlymsxapn_matmul_1_readvariableop_resource:	 9
*ddlymsxapn_biasadd_readvariableop_resource:	0
"ddlymsxapn_readvariableop_resource: 2
$ddlymsxapn_readvariableop_1_resource: 2
$ddlymsxapn_readvariableop_2_resource: 
identity¢!ddlymsxapn/BiasAdd/ReadVariableOp¢ ddlymsxapn/MatMul/ReadVariableOp¢"ddlymsxapn/MatMul_1/ReadVariableOp¢ddlymsxapn/ReadVariableOp¢ddlymsxapn/ReadVariableOp_1¢ddlymsxapn/ReadVariableOp_2¢whileD
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
 ddlymsxapn/MatMul/ReadVariableOpReadVariableOp)ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ddlymsxapn/MatMul/ReadVariableOp§
ddlymsxapn/MatMulMatMulstrided_slice_2:output:0(ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMulµ
"ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp+ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ddlymsxapn/MatMul_1/ReadVariableOp£
ddlymsxapn/MatMul_1MatMulzeros:output:0*ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/MatMul_1
ddlymsxapn/addAddV2ddlymsxapn/MatMul:product:0ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/add®
!ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp*ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ddlymsxapn/BiasAdd/ReadVariableOp¥
ddlymsxapn/BiasAddBiasAddddlymsxapn/add:z:0)ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ddlymsxapn/BiasAddz
ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ddlymsxapn/split/split_dimë
ddlymsxapn/splitSplit#ddlymsxapn/split/split_dim:output:0ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ddlymsxapn/split
ddlymsxapn/ReadVariableOpReadVariableOp"ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp
ddlymsxapn/mulMul!ddlymsxapn/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul
ddlymsxapn/add_1AddV2ddlymsxapn/split:output:0ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_1{
ddlymsxapn/SigmoidSigmoidddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid
ddlymsxapn/ReadVariableOp_1ReadVariableOp$ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_1
ddlymsxapn/mul_1Mul#ddlymsxapn/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_1
ddlymsxapn/add_2AddV2ddlymsxapn/split:output:1ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_2
ddlymsxapn/Sigmoid_1Sigmoidddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_1
ddlymsxapn/mul_2Mulddlymsxapn/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_2w
ddlymsxapn/TanhTanhddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh
ddlymsxapn/mul_3Mulddlymsxapn/Sigmoid:y:0ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_3
ddlymsxapn/add_3AddV2ddlymsxapn/mul_2:z:0ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_3
ddlymsxapn/ReadVariableOp_2ReadVariableOp$ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02
ddlymsxapn/ReadVariableOp_2
ddlymsxapn/mul_4Mul#ddlymsxapn/ReadVariableOp_2:value:0ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_4
ddlymsxapn/add_4AddV2ddlymsxapn/split:output:3ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/add_4
ddlymsxapn/Sigmoid_2Sigmoidddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Sigmoid_2v
ddlymsxapn/Tanh_1Tanhddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/Tanh_1
ddlymsxapn/mul_5Mulddlymsxapn/Sigmoid_2:y:0ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ddlymsxapn/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ddlymsxapn_matmul_readvariableop_resource+ddlymsxapn_matmul_1_readvariableop_resource*ddlymsxapn_biasadd_readvariableop_resource"ddlymsxapn_readvariableop_resource$ddlymsxapn_readvariableop_1_resource$ddlymsxapn_readvariableop_2_resource*
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
while_body_222045*
condR
while_cond_222044*Q
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
IdentityIdentitytranspose_1:y:0"^ddlymsxapn/BiasAdd/ReadVariableOp!^ddlymsxapn/MatMul/ReadVariableOp#^ddlymsxapn/MatMul_1/ReadVariableOp^ddlymsxapn/ReadVariableOp^ddlymsxapn/ReadVariableOp_1^ddlymsxapn/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ddlymsxapn/BiasAdd/ReadVariableOp!ddlymsxapn/BiasAdd/ReadVariableOp2D
 ddlymsxapn/MatMul/ReadVariableOp ddlymsxapn/MatMul/ReadVariableOp2H
"ddlymsxapn/MatMul_1/ReadVariableOp"ddlymsxapn/MatMul_1/ReadVariableOp26
ddlymsxapn/ReadVariableOpddlymsxapn/ReadVariableOp2:
ddlymsxapn/ReadVariableOp_1ddlymsxapn/ReadVariableOp_12:
ddlymsxapn/ReadVariableOp_2ddlymsxapn/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

«
F__inference_sequential_layer_call_and_return_conditional_losses_222370

inputs'
iigfihrkup_221947:
iigfihrkup_221949:$
omilqdycns_222147:	$
omilqdycns_222149:	  
omilqdycns_222151:	
omilqdycns_222153: 
omilqdycns_222155: 
omilqdycns_222157: $
vlxoswgdqw_222340:	 $
vlxoswgdqw_222342:	  
vlxoswgdqw_222344:	
vlxoswgdqw_222346: 
vlxoswgdqw_222348: 
vlxoswgdqw_222350: #
iktogmlrmp_222364: 
iktogmlrmp_222366:
identity¢"iigfihrkup/StatefulPartitionedCall¢"iktogmlrmp/StatefulPartitionedCall¢"omilqdycns/StatefulPartitionedCall¢"vlxoswgdqw/StatefulPartitionedCall©
"iigfihrkup/StatefulPartitionedCallStatefulPartitionedCallinputsiigfihrkup_221947iigfihrkup_221949*
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
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_2219462$
"iigfihrkup/StatefulPartitionedCall
xfbsciqeco/PartitionedCallPartitionedCall+iigfihrkup/StatefulPartitionedCall:output:0*
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
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_2219652
xfbsciqeco/PartitionedCall
"omilqdycns/StatefulPartitionedCallStatefulPartitionedCall#xfbsciqeco/PartitionedCall:output:0omilqdycns_222147omilqdycns_222149omilqdycns_222151omilqdycns_222153omilqdycns_222155omilqdycns_222157*
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2221462$
"omilqdycns/StatefulPartitionedCall
"vlxoswgdqw/StatefulPartitionedCallStatefulPartitionedCall+omilqdycns/StatefulPartitionedCall:output:0vlxoswgdqw_222340vlxoswgdqw_222342vlxoswgdqw_222344vlxoswgdqw_222346vlxoswgdqw_222348vlxoswgdqw_222350*
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2223392$
"vlxoswgdqw/StatefulPartitionedCallÆ
"iktogmlrmp/StatefulPartitionedCallStatefulPartitionedCall+vlxoswgdqw/StatefulPartitionedCall:output:0iktogmlrmp_222364iktogmlrmp_222366*
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
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_2223632$
"iktogmlrmp/StatefulPartitionedCall
IdentityIdentity+iktogmlrmp/StatefulPartitionedCall:output:0#^iigfihrkup/StatefulPartitionedCall#^iktogmlrmp/StatefulPartitionedCall#^omilqdycns/StatefulPartitionedCall#^vlxoswgdqw/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"iigfihrkup/StatefulPartitionedCall"iigfihrkup/StatefulPartitionedCall2H
"iktogmlrmp/StatefulPartitionedCall"iktogmlrmp/StatefulPartitionedCall2H
"omilqdycns/StatefulPartitionedCall"omilqdycns/StatefulPartitionedCall2H
"vlxoswgdqw/StatefulPartitionedCall"vlxoswgdqw/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

«
F__inference_sequential_layer_call_and_return_conditional_losses_222939

inputs'
iigfihrkup_222901:
iigfihrkup_222903:$
omilqdycns_222907:	$
omilqdycns_222909:	  
omilqdycns_222911:	
omilqdycns_222913: 
omilqdycns_222915: 
omilqdycns_222917: $
vlxoswgdqw_222920:	 $
vlxoswgdqw_222922:	  
vlxoswgdqw_222924:	
vlxoswgdqw_222926: 
vlxoswgdqw_222928: 
vlxoswgdqw_222930: #
iktogmlrmp_222933: 
iktogmlrmp_222935:
identity¢"iigfihrkup/StatefulPartitionedCall¢"iktogmlrmp/StatefulPartitionedCall¢"omilqdycns/StatefulPartitionedCall¢"vlxoswgdqw/StatefulPartitionedCall©
"iigfihrkup/StatefulPartitionedCallStatefulPartitionedCallinputsiigfihrkup_222901iigfihrkup_222903*
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
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_2219462$
"iigfihrkup/StatefulPartitionedCall
xfbsciqeco/PartitionedCallPartitionedCall+iigfihrkup/StatefulPartitionedCall:output:0*
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
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_2219652
xfbsciqeco/PartitionedCall
"omilqdycns/StatefulPartitionedCallStatefulPartitionedCall#xfbsciqeco/PartitionedCall:output:0omilqdycns_222907omilqdycns_222909omilqdycns_222911omilqdycns_222913omilqdycns_222915omilqdycns_222917*
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2228282$
"omilqdycns/StatefulPartitionedCall
"vlxoswgdqw/StatefulPartitionedCallStatefulPartitionedCall+omilqdycns/StatefulPartitionedCall:output:0vlxoswgdqw_222920vlxoswgdqw_222922vlxoswgdqw_222924vlxoswgdqw_222926vlxoswgdqw_222928vlxoswgdqw_222930*
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2226142$
"vlxoswgdqw/StatefulPartitionedCallÆ
"iktogmlrmp/StatefulPartitionedCallStatefulPartitionedCall+vlxoswgdqw/StatefulPartitionedCall:output:0iktogmlrmp_222933iktogmlrmp_222935*
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
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_2223632$
"iktogmlrmp/StatefulPartitionedCall
IdentityIdentity+iktogmlrmp/StatefulPartitionedCall:output:0#^iigfihrkup/StatefulPartitionedCall#^iktogmlrmp/StatefulPartitionedCall#^omilqdycns/StatefulPartitionedCall#^vlxoswgdqw/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"iigfihrkup/StatefulPartitionedCall"iigfihrkup/StatefulPartitionedCall2H
"iktogmlrmp/StatefulPartitionedCall"iktogmlrmp/StatefulPartitionedCall2H
"omilqdycns/StatefulPartitionedCall"omilqdycns/StatefulPartitionedCall2H
"vlxoswgdqw/StatefulPartitionedCall"vlxoswgdqw/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


omilqdycns_while_cond_2233322
.omilqdycns_while_omilqdycns_while_loop_counter8
4omilqdycns_while_omilqdycns_while_maximum_iterations 
omilqdycns_while_placeholder"
omilqdycns_while_placeholder_1"
omilqdycns_while_placeholder_2"
omilqdycns_while_placeholder_34
0omilqdycns_while_less_omilqdycns_strided_slice_1J
Fomilqdycns_while_omilqdycns_while_cond_223332___redundant_placeholder0J
Fomilqdycns_while_omilqdycns_while_cond_223332___redundant_placeholder1J
Fomilqdycns_while_omilqdycns_while_cond_223332___redundant_placeholder2J
Fomilqdycns_while_omilqdycns_while_cond_223332___redundant_placeholder3J
Fomilqdycns_while_omilqdycns_while_cond_223332___redundant_placeholder4J
Fomilqdycns_while_omilqdycns_while_cond_223332___redundant_placeholder5J
Fomilqdycns_while_omilqdycns_while_cond_223332___redundant_placeholder6
omilqdycns_while_identity
§
omilqdycns/while/LessLessomilqdycns_while_placeholder0omilqdycns_while_less_omilqdycns_strided_slice_1*
T0*
_output_shapes
: 2
omilqdycns/while/Less~
omilqdycns/while/IdentityIdentityomilqdycns/while/Less:z:0*
T0
*
_output_shapes
: 2
omilqdycns/while/Identity"?
omilqdycns_while_identity"omilqdycns/while/Identity:output:0*(
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
while_cond_225378
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_225378___redundant_placeholder04
0while_while_cond_225378___redundant_placeholder14
0while_while_cond_225378___redundant_placeholder24
0while_while_cond_225378___redundant_placeholder34
0while_while_cond_225378___redundant_placeholder44
0while_while_cond_225378___redundant_placeholder54
0while_while_cond_225378___redundant_placeholder6
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
¨0
»
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_224066

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
Ò	
÷
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_222363

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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_225947

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
ÿ
¿
+__inference_vdaevhnmja_layer_call_fn_225859

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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_2214182
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


å
while_cond_225018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_225018___redundant_placeholder04
0while_while_cond_225018___redundant_placeholder14
0while_while_cond_225018___redundant_placeholder24
0while_while_cond_225018___redundant_placeholder34
0while_while_cond_225018___redundant_placeholder44
0while_while_cond_225018___redundant_placeholder54
0while_while_cond_225018___redundant_placeholder6
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
while_body_224591
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ddlymsxapn_matmul_readvariableop_resource_0:	F
3while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 A
2while_ddlymsxapn_biasadd_readvariableop_resource_0:	8
*while_ddlymsxapn_readvariableop_resource_0: :
,while_ddlymsxapn_readvariableop_1_resource_0: :
,while_ddlymsxapn_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ddlymsxapn_matmul_readvariableop_resource:	D
1while_ddlymsxapn_matmul_1_readvariableop_resource:	 ?
0while_ddlymsxapn_biasadd_readvariableop_resource:	6
(while_ddlymsxapn_readvariableop_resource: 8
*while_ddlymsxapn_readvariableop_1_resource: 8
*while_ddlymsxapn_readvariableop_2_resource: ¢'while/ddlymsxapn/BiasAdd/ReadVariableOp¢&while/ddlymsxapn/MatMul/ReadVariableOp¢(while/ddlymsxapn/MatMul_1/ReadVariableOp¢while/ddlymsxapn/ReadVariableOp¢!while/ddlymsxapn/ReadVariableOp_1¢!while/ddlymsxapn/ReadVariableOp_2Ã
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
&while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp1while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ddlymsxapn/MatMul/ReadVariableOpÑ
while/ddlymsxapn/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMulÉ
(while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp3while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ddlymsxapn/MatMul_1/ReadVariableOpº
while/ddlymsxapn/MatMul_1MatMulwhile_placeholder_20while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMul_1°
while/ddlymsxapn/addAddV2!while/ddlymsxapn/MatMul:product:0#while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/addÂ
'while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp2while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ddlymsxapn/BiasAdd/ReadVariableOp½
while/ddlymsxapn/BiasAddBiasAddwhile/ddlymsxapn/add:z:0/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/BiasAdd
 while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ddlymsxapn/split/split_dim
while/ddlymsxapn/splitSplit)while/ddlymsxapn/split/split_dim:output:0!while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ddlymsxapn/split©
while/ddlymsxapn/ReadVariableOpReadVariableOp*while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ddlymsxapn/ReadVariableOp£
while/ddlymsxapn/mulMul'while/ddlymsxapn/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul¦
while/ddlymsxapn/add_1AddV2while/ddlymsxapn/split:output:0while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_1
while/ddlymsxapn/SigmoidSigmoidwhile/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid¯
!while/ddlymsxapn/ReadVariableOp_1ReadVariableOp,while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_1©
while/ddlymsxapn/mul_1Mul)while/ddlymsxapn/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_1¨
while/ddlymsxapn/add_2AddV2while/ddlymsxapn/split:output:1while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_2
while/ddlymsxapn/Sigmoid_1Sigmoidwhile/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_1
while/ddlymsxapn/mul_2Mulwhile/ddlymsxapn/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_2
while/ddlymsxapn/TanhTanhwhile/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh¢
while/ddlymsxapn/mul_3Mulwhile/ddlymsxapn/Sigmoid:y:0while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_3£
while/ddlymsxapn/add_3AddV2while/ddlymsxapn/mul_2:z:0while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_3¯
!while/ddlymsxapn/ReadVariableOp_2ReadVariableOp,while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_2°
while/ddlymsxapn/mul_4Mul)while/ddlymsxapn/ReadVariableOp_2:value:0while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_4¨
while/ddlymsxapn/add_4AddV2while/ddlymsxapn/split:output:3while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_4
while/ddlymsxapn/Sigmoid_2Sigmoidwhile/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_2
while/ddlymsxapn/Tanh_1Tanhwhile/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh_1¦
while/ddlymsxapn/mul_5Mulwhile/ddlymsxapn/Sigmoid_2:y:0while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ddlymsxapn/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ddlymsxapn/mul_5:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ddlymsxapn/add_3:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ddlymsxapn_biasadd_readvariableop_resource2while_ddlymsxapn_biasadd_readvariableop_resource_0"h
1while_ddlymsxapn_matmul_1_readvariableop_resource3while_ddlymsxapn_matmul_1_readvariableop_resource_0"d
/while_ddlymsxapn_matmul_readvariableop_resource1while_ddlymsxapn_matmul_readvariableop_resource_0"Z
*while_ddlymsxapn_readvariableop_1_resource,while_ddlymsxapn_readvariableop_1_resource_0"Z
*while_ddlymsxapn_readvariableop_2_resource,while_ddlymsxapn_readvariableop_2_resource_0"V
(while_ddlymsxapn_readvariableop_resource*while_ddlymsxapn_readvariableop_resource_0")
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
'while/ddlymsxapn/BiasAdd/ReadVariableOp'while/ddlymsxapn/BiasAdd/ReadVariableOp2P
&while/ddlymsxapn/MatMul/ReadVariableOp&while/ddlymsxapn/MatMul/ReadVariableOp2T
(while/ddlymsxapn/MatMul_1/ReadVariableOp(while/ddlymsxapn/MatMul_1/ReadVariableOp2B
while/ddlymsxapn/ReadVariableOpwhile/ddlymsxapn/ReadVariableOp2F
!while/ddlymsxapn/ReadVariableOp_1!while/ddlymsxapn/ReadVariableOp_12F
!while/ddlymsxapn/ReadVariableOp_2!while/ddlymsxapn/ReadVariableOp_2: 
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_221331

inputs$
vdaevhnmja_221232:	 $
vdaevhnmja_221234:	  
vdaevhnmja_221236:	
vdaevhnmja_221238: 
vdaevhnmja_221240: 
vdaevhnmja_221242: 
identity¢"vdaevhnmja/StatefulPartitionedCall¢whileD
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
"vdaevhnmja/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0vdaevhnmja_221232vdaevhnmja_221234vdaevhnmja_221236vdaevhnmja_221238vdaevhnmja_221240vdaevhnmja_221242*
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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_2212312$
"vdaevhnmja/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0vdaevhnmja_221232vdaevhnmja_221234vdaevhnmja_221236vdaevhnmja_221238vdaevhnmja_221240vdaevhnmja_221242*
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
while_body_221251*
condR
while_cond_221250*Q
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
IdentityIdentitystrided_slice_3:output:0#^vdaevhnmja/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"vdaevhnmja/StatefulPartitionedCall"vdaevhnmja/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


å
while_cond_222726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_222726___redundant_placeholder04
0while_while_cond_222726___redundant_placeholder14
0while_while_cond_222726___redundant_placeholder24
0while_while_cond_222726___redundant_placeholder34
0while_while_cond_222726___redundant_placeholder44
0while_while_cond_222726___redundant_placeholder54
0while_while_cond_222726___redundant_placeholder6
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


+__inference_sequential_layer_call_fn_223011

ahzwxypkrh
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
ahzwxypkrhunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
F__inference_sequential_layer_call_and_return_conditional_losses_2229392
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
ahzwxypkrh
£h

F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225660

inputs<
)vdaevhnmja_matmul_readvariableop_resource:	 >
+vdaevhnmja_matmul_1_readvariableop_resource:	 9
*vdaevhnmja_biasadd_readvariableop_resource:	0
"vdaevhnmja_readvariableop_resource: 2
$vdaevhnmja_readvariableop_1_resource: 2
$vdaevhnmja_readvariableop_2_resource: 
identity¢!vdaevhnmja/BiasAdd/ReadVariableOp¢ vdaevhnmja/MatMul/ReadVariableOp¢"vdaevhnmja/MatMul_1/ReadVariableOp¢vdaevhnmja/ReadVariableOp¢vdaevhnmja/ReadVariableOp_1¢vdaevhnmja/ReadVariableOp_2¢whileD
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
 vdaevhnmja/MatMul/ReadVariableOpReadVariableOp)vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 vdaevhnmja/MatMul/ReadVariableOp§
vdaevhnmja/MatMulMatMulstrided_slice_2:output:0(vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMulµ
"vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp+vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"vdaevhnmja/MatMul_1/ReadVariableOp£
vdaevhnmja/MatMul_1MatMulzeros:output:0*vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMul_1
vdaevhnmja/addAddV2vdaevhnmja/MatMul:product:0vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/add®
!vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp*vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!vdaevhnmja/BiasAdd/ReadVariableOp¥
vdaevhnmja/BiasAddBiasAddvdaevhnmja/add:z:0)vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/BiasAddz
vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vdaevhnmja/split/split_dimë
vdaevhnmja/splitSplit#vdaevhnmja/split/split_dim:output:0vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vdaevhnmja/split
vdaevhnmja/ReadVariableOpReadVariableOp"vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp
vdaevhnmja/mulMul!vdaevhnmja/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul
vdaevhnmja/add_1AddV2vdaevhnmja/split:output:0vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_1{
vdaevhnmja/SigmoidSigmoidvdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid
vdaevhnmja/ReadVariableOp_1ReadVariableOp$vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_1
vdaevhnmja/mul_1Mul#vdaevhnmja/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_1
vdaevhnmja/add_2AddV2vdaevhnmja/split:output:1vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_2
vdaevhnmja/Sigmoid_1Sigmoidvdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_1
vdaevhnmja/mul_2Mulvdaevhnmja/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_2w
vdaevhnmja/TanhTanhvdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh
vdaevhnmja/mul_3Mulvdaevhnmja/Sigmoid:y:0vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_3
vdaevhnmja/add_3AddV2vdaevhnmja/mul_2:z:0vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_3
vdaevhnmja/ReadVariableOp_2ReadVariableOp$vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_2
vdaevhnmja/mul_4Mul#vdaevhnmja/ReadVariableOp_2:value:0vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_4
vdaevhnmja/add_4AddV2vdaevhnmja/split:output:3vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_4
vdaevhnmja/Sigmoid_2Sigmoidvdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_2v
vdaevhnmja/Tanh_1Tanhvdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh_1
vdaevhnmja/mul_5Mulvdaevhnmja/Sigmoid_2:y:0vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)vdaevhnmja_matmul_readvariableop_resource+vdaevhnmja_matmul_1_readvariableop_resource*vdaevhnmja_biasadd_readvariableop_resource"vdaevhnmja_readvariableop_resource$vdaevhnmja_readvariableop_1_resource$vdaevhnmja_readvariableop_2_resource*
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
while_body_225559*
condR
while_cond_225558*Q
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
IdentityIdentitystrided_slice_3:output:0"^vdaevhnmja/BiasAdd/ReadVariableOp!^vdaevhnmja/MatMul/ReadVariableOp#^vdaevhnmja/MatMul_1/ReadVariableOp^vdaevhnmja/ReadVariableOp^vdaevhnmja/ReadVariableOp_1^vdaevhnmja/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!vdaevhnmja/BiasAdd/ReadVariableOp!vdaevhnmja/BiasAdd/ReadVariableOp2D
 vdaevhnmja/MatMul/ReadVariableOp vdaevhnmja/MatMul/ReadVariableOp2H
"vdaevhnmja/MatMul_1/ReadVariableOp"vdaevhnmja/MatMul_1/ReadVariableOp26
vdaevhnmja/ReadVariableOpvdaevhnmja/ReadVariableOp2:
vdaevhnmja/ReadVariableOp_1vdaevhnmja/ReadVariableOp_12:
vdaevhnmja/ReadVariableOp_2vdaevhnmja/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ù

+__inference_omilqdycns_layer_call_fn_224152

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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2228282
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
°'
²
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_220473

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


vlxoswgdqw_while_cond_2239122
.vlxoswgdqw_while_vlxoswgdqw_while_loop_counter8
4vlxoswgdqw_while_vlxoswgdqw_while_maximum_iterations 
vlxoswgdqw_while_placeholder"
vlxoswgdqw_while_placeholder_1"
vlxoswgdqw_while_placeholder_2"
vlxoswgdqw_while_placeholder_34
0vlxoswgdqw_while_less_vlxoswgdqw_strided_slice_1J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223912___redundant_placeholder0J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223912___redundant_placeholder1J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223912___redundant_placeholder2J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223912___redundant_placeholder3J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223912___redundant_placeholder4J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223912___redundant_placeholder5J
Fvlxoswgdqw_while_vlxoswgdqw_while_cond_223912___redundant_placeholder6
vlxoswgdqw_while_identity
§
vlxoswgdqw/while/LessLessvlxoswgdqw_while_placeholder0vlxoswgdqw_while_less_vlxoswgdqw_strided_slice_1*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Less~
vlxoswgdqw/while/IdentityIdentityvlxoswgdqw/while/Less:z:0*
T0
*
_output_shapes
: 2
vlxoswgdqw/while/Identity"?
vlxoswgdqw_while_identity"vlxoswgdqw/while/Identity:output:0*(
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
while_cond_224590
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_224590___redundant_placeholder04
0while_while_cond_224590___redundant_placeholder14
0while_while_cond_224590___redundant_placeholder24
0while_while_cond_224590___redundant_placeholder34
0while_while_cond_224590___redundant_placeholder44
0while_while_cond_224590___redundant_placeholder54
0while_while_cond_224590___redundant_placeholder6
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

¯
F__inference_sequential_layer_call_and_return_conditional_losses_223052

ahzwxypkrh'
iigfihrkup_223014:
iigfihrkup_223016:$
omilqdycns_223020:	$
omilqdycns_223022:	  
omilqdycns_223024:	
omilqdycns_223026: 
omilqdycns_223028: 
omilqdycns_223030: $
vlxoswgdqw_223033:	 $
vlxoswgdqw_223035:	  
vlxoswgdqw_223037:	
vlxoswgdqw_223039: 
vlxoswgdqw_223041: 
vlxoswgdqw_223043: #
iktogmlrmp_223046: 
iktogmlrmp_223048:
identity¢"iigfihrkup/StatefulPartitionedCall¢"iktogmlrmp/StatefulPartitionedCall¢"omilqdycns/StatefulPartitionedCall¢"vlxoswgdqw/StatefulPartitionedCall­
"iigfihrkup/StatefulPartitionedCallStatefulPartitionedCall
ahzwxypkrhiigfihrkup_223014iigfihrkup_223016*
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
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_2219462$
"iigfihrkup/StatefulPartitionedCall
xfbsciqeco/PartitionedCallPartitionedCall+iigfihrkup/StatefulPartitionedCall:output:0*
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
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_2219652
xfbsciqeco/PartitionedCall
"omilqdycns/StatefulPartitionedCallStatefulPartitionedCall#xfbsciqeco/PartitionedCall:output:0omilqdycns_223020omilqdycns_223022omilqdycns_223024omilqdycns_223026omilqdycns_223028omilqdycns_223030*
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2221462$
"omilqdycns/StatefulPartitionedCall
"vlxoswgdqw/StatefulPartitionedCallStatefulPartitionedCall+omilqdycns/StatefulPartitionedCall:output:0vlxoswgdqw_223033vlxoswgdqw_223035vlxoswgdqw_223037vlxoswgdqw_223039vlxoswgdqw_223041vlxoswgdqw_223043*
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2223392$
"vlxoswgdqw/StatefulPartitionedCallÆ
"iktogmlrmp/StatefulPartitionedCallStatefulPartitionedCall+vlxoswgdqw/StatefulPartitionedCall:output:0iktogmlrmp_223046iktogmlrmp_223048*
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
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_2223632$
"iktogmlrmp/StatefulPartitionedCall
IdentityIdentity+iktogmlrmp/StatefulPartitionedCall:output:0#^iigfihrkup/StatefulPartitionedCall#^iktogmlrmp/StatefulPartitionedCall#^omilqdycns/StatefulPartitionedCall#^vlxoswgdqw/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"iigfihrkup/StatefulPartitionedCall"iigfihrkup/StatefulPartitionedCall2H
"iktogmlrmp/StatefulPartitionedCall"iktogmlrmp/StatefulPartitionedCall2H
"omilqdycns/StatefulPartitionedCall"omilqdycns/StatefulPartitionedCall2H
"vlxoswgdqw/StatefulPartitionedCall"vlxoswgdqw/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
ahzwxypkrh


+__inference_sequential_layer_call_fn_222405

ahzwxypkrh
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
ahzwxypkrhunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
F__inference_sequential_layer_call_and_return_conditional_losses_2223702
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
ahzwxypkrh
ÿ
¿
+__inference_ddlymsxapn_layer_call_fn_225725

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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_2206602
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


å
while_cond_225558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_225558___redundant_placeholder04
0while_while_cond_225558___redundant_placeholder14
0while_while_cond_225558___redundant_placeholder24
0while_while_cond_225558___redundant_placeholder34
0while_while_cond_225558___redundant_placeholder44
0while_while_cond_225558___redundant_placeholder54
0while_while_cond_225558___redundant_placeholder6
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

¡	
'sequential_omilqdycns_while_cond_220102H
Dsequential_omilqdycns_while_sequential_omilqdycns_while_loop_counterN
Jsequential_omilqdycns_while_sequential_omilqdycns_while_maximum_iterations+
'sequential_omilqdycns_while_placeholder-
)sequential_omilqdycns_while_placeholder_1-
)sequential_omilqdycns_while_placeholder_2-
)sequential_omilqdycns_while_placeholder_3J
Fsequential_omilqdycns_while_less_sequential_omilqdycns_strided_slice_1`
\sequential_omilqdycns_while_sequential_omilqdycns_while_cond_220102___redundant_placeholder0`
\sequential_omilqdycns_while_sequential_omilqdycns_while_cond_220102___redundant_placeholder1`
\sequential_omilqdycns_while_sequential_omilqdycns_while_cond_220102___redundant_placeholder2`
\sequential_omilqdycns_while_sequential_omilqdycns_while_cond_220102___redundant_placeholder3`
\sequential_omilqdycns_while_sequential_omilqdycns_while_cond_220102___redundant_placeholder4`
\sequential_omilqdycns_while_sequential_omilqdycns_while_cond_220102___redundant_placeholder5`
\sequential_omilqdycns_while_sequential_omilqdycns_while_cond_220102___redundant_placeholder6(
$sequential_omilqdycns_while_identity
Þ
 sequential/omilqdycns/while/LessLess'sequential_omilqdycns_while_placeholderFsequential_omilqdycns_while_less_sequential_omilqdycns_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/omilqdycns/while/Less
$sequential/omilqdycns/while/IdentityIdentity$sequential/omilqdycns/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/omilqdycns/while/Identity"U
$sequential_omilqdycns_while_identity-sequential/omilqdycns/while/Identity:output:0*(
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
F
ã
F__inference_omilqdycns_layer_call_and_return_conditional_losses_220836

inputs$
ddlymsxapn_220737:	$
ddlymsxapn_220739:	  
ddlymsxapn_220741:	
ddlymsxapn_220743: 
ddlymsxapn_220745: 
ddlymsxapn_220747: 
identity¢"ddlymsxapn/StatefulPartitionedCall¢whileD
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
"ddlymsxapn/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0ddlymsxapn_220737ddlymsxapn_220739ddlymsxapn_220741ddlymsxapn_220743ddlymsxapn_220745ddlymsxapn_220747*
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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_2206602$
"ddlymsxapn/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0ddlymsxapn_220737ddlymsxapn_220739ddlymsxapn_220741ddlymsxapn_220743ddlymsxapn_220745ddlymsxapn_220747*
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
while_body_220756*
condR
while_cond_220755*Q
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
IdentityIdentitytranspose_1:y:0#^ddlymsxapn/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"ddlymsxapn/StatefulPartitionedCall"ddlymsxapn/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
F
ã
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_221594

inputs$
vdaevhnmja_221495:	 $
vdaevhnmja_221497:	  
vdaevhnmja_221499:	
vdaevhnmja_221501: 
vdaevhnmja_221503: 
vdaevhnmja_221505: 
identity¢"vdaevhnmja/StatefulPartitionedCall¢whileD
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
"vdaevhnmja/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0vdaevhnmja_221495vdaevhnmja_221497vdaevhnmja_221499vdaevhnmja_221501vdaevhnmja_221503vdaevhnmja_221505*
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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_2214182$
"vdaevhnmja/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0vdaevhnmja_221495vdaevhnmja_221497vdaevhnmja_221499vdaevhnmja_221501vdaevhnmja_221503vdaevhnmja_221505*
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
while_body_221514*
condR
while_cond_221513*Q
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
IdentityIdentitystrided_slice_3:output:0#^vdaevhnmja/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"vdaevhnmja/StatefulPartitionedCall"vdaevhnmja/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


å
while_cond_224770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_224770___redundant_placeholder04
0while_while_cond_224770___redundant_placeholder14
0while_while_cond_224770___redundant_placeholder24
0while_while_cond_224770___redundant_placeholder34
0while_while_cond_224770___redundant_placeholder44
0while_while_cond_224770___redundant_placeholder54
0while_while_cond_224770___redundant_placeholder6
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_222339

inputs<
)vdaevhnmja_matmul_readvariableop_resource:	 >
+vdaevhnmja_matmul_1_readvariableop_resource:	 9
*vdaevhnmja_biasadd_readvariableop_resource:	0
"vdaevhnmja_readvariableop_resource: 2
$vdaevhnmja_readvariableop_1_resource: 2
$vdaevhnmja_readvariableop_2_resource: 
identity¢!vdaevhnmja/BiasAdd/ReadVariableOp¢ vdaevhnmja/MatMul/ReadVariableOp¢"vdaevhnmja/MatMul_1/ReadVariableOp¢vdaevhnmja/ReadVariableOp¢vdaevhnmja/ReadVariableOp_1¢vdaevhnmja/ReadVariableOp_2¢whileD
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
 vdaevhnmja/MatMul/ReadVariableOpReadVariableOp)vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 vdaevhnmja/MatMul/ReadVariableOp§
vdaevhnmja/MatMulMatMulstrided_slice_2:output:0(vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMulµ
"vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp+vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"vdaevhnmja/MatMul_1/ReadVariableOp£
vdaevhnmja/MatMul_1MatMulzeros:output:0*vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMul_1
vdaevhnmja/addAddV2vdaevhnmja/MatMul:product:0vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/add®
!vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp*vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!vdaevhnmja/BiasAdd/ReadVariableOp¥
vdaevhnmja/BiasAddBiasAddvdaevhnmja/add:z:0)vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/BiasAddz
vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vdaevhnmja/split/split_dimë
vdaevhnmja/splitSplit#vdaevhnmja/split/split_dim:output:0vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vdaevhnmja/split
vdaevhnmja/ReadVariableOpReadVariableOp"vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp
vdaevhnmja/mulMul!vdaevhnmja/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul
vdaevhnmja/add_1AddV2vdaevhnmja/split:output:0vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_1{
vdaevhnmja/SigmoidSigmoidvdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid
vdaevhnmja/ReadVariableOp_1ReadVariableOp$vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_1
vdaevhnmja/mul_1Mul#vdaevhnmja/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_1
vdaevhnmja/add_2AddV2vdaevhnmja/split:output:1vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_2
vdaevhnmja/Sigmoid_1Sigmoidvdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_1
vdaevhnmja/mul_2Mulvdaevhnmja/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_2w
vdaevhnmja/TanhTanhvdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh
vdaevhnmja/mul_3Mulvdaevhnmja/Sigmoid:y:0vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_3
vdaevhnmja/add_3AddV2vdaevhnmja/mul_2:z:0vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_3
vdaevhnmja/ReadVariableOp_2ReadVariableOp$vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_2
vdaevhnmja/mul_4Mul#vdaevhnmja/ReadVariableOp_2:value:0vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_4
vdaevhnmja/add_4AddV2vdaevhnmja/split:output:3vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_4
vdaevhnmja/Sigmoid_2Sigmoidvdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_2v
vdaevhnmja/Tanh_1Tanhvdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh_1
vdaevhnmja/mul_5Mulvdaevhnmja/Sigmoid_2:y:0vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)vdaevhnmja_matmul_readvariableop_resource+vdaevhnmja_matmul_1_readvariableop_resource*vdaevhnmja_biasadd_readvariableop_resource"vdaevhnmja_readvariableop_resource$vdaevhnmja_readvariableop_1_resource$vdaevhnmja_readvariableop_2_resource*
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
while_body_222238*
condR
while_cond_222237*Q
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
IdentityIdentitystrided_slice_3:output:0"^vdaevhnmja/BiasAdd/ReadVariableOp!^vdaevhnmja/MatMul/ReadVariableOp#^vdaevhnmja/MatMul_1/ReadVariableOp^vdaevhnmja/ReadVariableOp^vdaevhnmja/ReadVariableOp_1^vdaevhnmja/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!vdaevhnmja/BiasAdd/ReadVariableOp!vdaevhnmja/BiasAdd/ReadVariableOp2D
 vdaevhnmja/MatMul/ReadVariableOp vdaevhnmja/MatMul/ReadVariableOp2H
"vdaevhnmja/MatMul_1/ReadVariableOp"vdaevhnmja/MatMul_1/ReadVariableOp26
vdaevhnmja/ReadVariableOpvdaevhnmja/ReadVariableOp2:
vdaevhnmja/ReadVariableOp_1vdaevhnmja/ReadVariableOp_12:
vdaevhnmja/ReadVariableOp_2vdaevhnmja/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Æ

+__inference_iigfihrkup_layer_call_fn_224029

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
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_2219462
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
Ç)
Å
while_body_220493
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_ddlymsxapn_220517_0:	,
while_ddlymsxapn_220519_0:	 (
while_ddlymsxapn_220521_0:	'
while_ddlymsxapn_220523_0: '
while_ddlymsxapn_220525_0: '
while_ddlymsxapn_220527_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_ddlymsxapn_220517:	*
while_ddlymsxapn_220519:	 &
while_ddlymsxapn_220521:	%
while_ddlymsxapn_220523: %
while_ddlymsxapn_220525: %
while_ddlymsxapn_220527: ¢(while/ddlymsxapn/StatefulPartitionedCallÃ
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
(while/ddlymsxapn/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_ddlymsxapn_220517_0while_ddlymsxapn_220519_0while_ddlymsxapn_220521_0while_ddlymsxapn_220523_0while_ddlymsxapn_220525_0while_ddlymsxapn_220527_0*
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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_2204732*
(while/ddlymsxapn/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/ddlymsxapn/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/ddlymsxapn/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/ddlymsxapn/StatefulPartitionedCall:output:1)^while/ddlymsxapn/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/ddlymsxapn/StatefulPartitionedCall:output:2)^while/ddlymsxapn/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_ddlymsxapn_220517while_ddlymsxapn_220517_0"4
while_ddlymsxapn_220519while_ddlymsxapn_220519_0"4
while_ddlymsxapn_220521while_ddlymsxapn_220521_0"4
while_ddlymsxapn_220523while_ddlymsxapn_220523_0"4
while_ddlymsxapn_220525while_ddlymsxapn_220525_0"4
while_ddlymsxapn_220527while_ddlymsxapn_220527_0")
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
(while/ddlymsxapn/StatefulPartitionedCall(while/ddlymsxapn/StatefulPartitionedCall: 
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
while_cond_221513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_221513___redundant_placeholder04
0while_while_cond_221513___redundant_placeholder14
0while_while_cond_221513___redundant_placeholder24
0while_while_cond_221513___redundant_placeholder34
0while_while_cond_221513___redundant_placeholder44
0while_while_cond_221513___redundant_placeholder54
0while_while_cond_221513___redundant_placeholder6
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
while_cond_224410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_224410___redundant_placeholder04
0while_while_cond_224410___redundant_placeholder14
0while_while_cond_224410___redundant_placeholder24
0while_while_cond_224410___redundant_placeholder34
0while_while_cond_224410___redundant_placeholder44
0while_while_cond_224410___redundant_placeholder54
0while_while_cond_224410___redundant_placeholder6
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

+__inference_omilqdycns_layer_call_fn_224101
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2205732
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
¸'
´
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_225813

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
¢

+__inference_iktogmlrmp_layer_call_fn_225669

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
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_2223632
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
ßY

while_body_224411
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ddlymsxapn_matmul_readvariableop_resource_0:	F
3while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 A
2while_ddlymsxapn_biasadd_readvariableop_resource_0:	8
*while_ddlymsxapn_readvariableop_resource_0: :
,while_ddlymsxapn_readvariableop_1_resource_0: :
,while_ddlymsxapn_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ddlymsxapn_matmul_readvariableop_resource:	D
1while_ddlymsxapn_matmul_1_readvariableop_resource:	 ?
0while_ddlymsxapn_biasadd_readvariableop_resource:	6
(while_ddlymsxapn_readvariableop_resource: 8
*while_ddlymsxapn_readvariableop_1_resource: 8
*while_ddlymsxapn_readvariableop_2_resource: ¢'while/ddlymsxapn/BiasAdd/ReadVariableOp¢&while/ddlymsxapn/MatMul/ReadVariableOp¢(while/ddlymsxapn/MatMul_1/ReadVariableOp¢while/ddlymsxapn/ReadVariableOp¢!while/ddlymsxapn/ReadVariableOp_1¢!while/ddlymsxapn/ReadVariableOp_2Ã
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
&while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp1while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ddlymsxapn/MatMul/ReadVariableOpÑ
while/ddlymsxapn/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMulÉ
(while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp3while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ddlymsxapn/MatMul_1/ReadVariableOpº
while/ddlymsxapn/MatMul_1MatMulwhile_placeholder_20while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMul_1°
while/ddlymsxapn/addAddV2!while/ddlymsxapn/MatMul:product:0#while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/addÂ
'while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp2while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ddlymsxapn/BiasAdd/ReadVariableOp½
while/ddlymsxapn/BiasAddBiasAddwhile/ddlymsxapn/add:z:0/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/BiasAdd
 while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ddlymsxapn/split/split_dim
while/ddlymsxapn/splitSplit)while/ddlymsxapn/split/split_dim:output:0!while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ddlymsxapn/split©
while/ddlymsxapn/ReadVariableOpReadVariableOp*while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ddlymsxapn/ReadVariableOp£
while/ddlymsxapn/mulMul'while/ddlymsxapn/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul¦
while/ddlymsxapn/add_1AddV2while/ddlymsxapn/split:output:0while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_1
while/ddlymsxapn/SigmoidSigmoidwhile/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid¯
!while/ddlymsxapn/ReadVariableOp_1ReadVariableOp,while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_1©
while/ddlymsxapn/mul_1Mul)while/ddlymsxapn/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_1¨
while/ddlymsxapn/add_2AddV2while/ddlymsxapn/split:output:1while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_2
while/ddlymsxapn/Sigmoid_1Sigmoidwhile/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_1
while/ddlymsxapn/mul_2Mulwhile/ddlymsxapn/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_2
while/ddlymsxapn/TanhTanhwhile/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh¢
while/ddlymsxapn/mul_3Mulwhile/ddlymsxapn/Sigmoid:y:0while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_3£
while/ddlymsxapn/add_3AddV2while/ddlymsxapn/mul_2:z:0while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_3¯
!while/ddlymsxapn/ReadVariableOp_2ReadVariableOp,while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_2°
while/ddlymsxapn/mul_4Mul)while/ddlymsxapn/ReadVariableOp_2:value:0while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_4¨
while/ddlymsxapn/add_4AddV2while/ddlymsxapn/split:output:3while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_4
while/ddlymsxapn/Sigmoid_2Sigmoidwhile/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_2
while/ddlymsxapn/Tanh_1Tanhwhile/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh_1¦
while/ddlymsxapn/mul_5Mulwhile/ddlymsxapn/Sigmoid_2:y:0while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ddlymsxapn/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ddlymsxapn/mul_5:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ddlymsxapn/add_3:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ddlymsxapn_biasadd_readvariableop_resource2while_ddlymsxapn_biasadd_readvariableop_resource_0"h
1while_ddlymsxapn_matmul_1_readvariableop_resource3while_ddlymsxapn_matmul_1_readvariableop_resource_0"d
/while_ddlymsxapn_matmul_readvariableop_resource1while_ddlymsxapn_matmul_readvariableop_resource_0"Z
*while_ddlymsxapn_readvariableop_1_resource,while_ddlymsxapn_readvariableop_1_resource_0"Z
*while_ddlymsxapn_readvariableop_2_resource,while_ddlymsxapn_readvariableop_2_resource_0"V
(while_ddlymsxapn_readvariableop_resource*while_ddlymsxapn_readvariableop_resource_0")
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
'while/ddlymsxapn/BiasAdd/ReadVariableOp'while/ddlymsxapn/BiasAdd/ReadVariableOp2P
&while/ddlymsxapn/MatMul/ReadVariableOp&while/ddlymsxapn/MatMul/ReadVariableOp2T
(while/ddlymsxapn/MatMul_1/ReadVariableOp(while/ddlymsxapn/MatMul_1/ReadVariableOp2B
while/ddlymsxapn/ReadVariableOpwhile/ddlymsxapn/ReadVariableOp2F
!while/ddlymsxapn/ReadVariableOp_1!while/ddlymsxapn/ReadVariableOp_12F
!while/ddlymsxapn/ReadVariableOp_2!while/ddlymsxapn/ReadVariableOp_2: 
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
+__inference_vlxoswgdqw_layer_call_fn_224889
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2213312
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
Ù

+__inference_omilqdycns_layer_call_fn_224135

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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2221462
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


å
while_cond_222237
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_222237___redundant_placeholder04
0while_while_cond_222237___redundant_placeholder14
0while_while_cond_222237___redundant_placeholder24
0while_while_cond_222237___redundant_placeholder34
0while_while_cond_222237___redundant_placeholder44
0while_while_cond_222237___redundant_placeholder54
0while_while_cond_222237___redundant_placeholder6
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225480

inputs<
)vdaevhnmja_matmul_readvariableop_resource:	 >
+vdaevhnmja_matmul_1_readvariableop_resource:	 9
*vdaevhnmja_biasadd_readvariableop_resource:	0
"vdaevhnmja_readvariableop_resource: 2
$vdaevhnmja_readvariableop_1_resource: 2
$vdaevhnmja_readvariableop_2_resource: 
identity¢!vdaevhnmja/BiasAdd/ReadVariableOp¢ vdaevhnmja/MatMul/ReadVariableOp¢"vdaevhnmja/MatMul_1/ReadVariableOp¢vdaevhnmja/ReadVariableOp¢vdaevhnmja/ReadVariableOp_1¢vdaevhnmja/ReadVariableOp_2¢whileD
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
 vdaevhnmja/MatMul/ReadVariableOpReadVariableOp)vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 vdaevhnmja/MatMul/ReadVariableOp§
vdaevhnmja/MatMulMatMulstrided_slice_2:output:0(vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMulµ
"vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp+vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"vdaevhnmja/MatMul_1/ReadVariableOp£
vdaevhnmja/MatMul_1MatMulzeros:output:0*vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMul_1
vdaevhnmja/addAddV2vdaevhnmja/MatMul:product:0vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/add®
!vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp*vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!vdaevhnmja/BiasAdd/ReadVariableOp¥
vdaevhnmja/BiasAddBiasAddvdaevhnmja/add:z:0)vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/BiasAddz
vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vdaevhnmja/split/split_dimë
vdaevhnmja/splitSplit#vdaevhnmja/split/split_dim:output:0vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vdaevhnmja/split
vdaevhnmja/ReadVariableOpReadVariableOp"vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp
vdaevhnmja/mulMul!vdaevhnmja/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul
vdaevhnmja/add_1AddV2vdaevhnmja/split:output:0vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_1{
vdaevhnmja/SigmoidSigmoidvdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid
vdaevhnmja/ReadVariableOp_1ReadVariableOp$vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_1
vdaevhnmja/mul_1Mul#vdaevhnmja/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_1
vdaevhnmja/add_2AddV2vdaevhnmja/split:output:1vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_2
vdaevhnmja/Sigmoid_1Sigmoidvdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_1
vdaevhnmja/mul_2Mulvdaevhnmja/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_2w
vdaevhnmja/TanhTanhvdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh
vdaevhnmja/mul_3Mulvdaevhnmja/Sigmoid:y:0vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_3
vdaevhnmja/add_3AddV2vdaevhnmja/mul_2:z:0vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_3
vdaevhnmja/ReadVariableOp_2ReadVariableOp$vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_2
vdaevhnmja/mul_4Mul#vdaevhnmja/ReadVariableOp_2:value:0vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_4
vdaevhnmja/add_4AddV2vdaevhnmja/split:output:3vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_4
vdaevhnmja/Sigmoid_2Sigmoidvdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_2v
vdaevhnmja/Tanh_1Tanhvdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh_1
vdaevhnmja/mul_5Mulvdaevhnmja/Sigmoid_2:y:0vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)vdaevhnmja_matmul_readvariableop_resource+vdaevhnmja_matmul_1_readvariableop_resource*vdaevhnmja_biasadd_readvariableop_resource"vdaevhnmja_readvariableop_resource$vdaevhnmja_readvariableop_1_resource$vdaevhnmja_readvariableop_2_resource*
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
while_body_225379*
condR
while_cond_225378*Q
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
IdentityIdentitystrided_slice_3:output:0"^vdaevhnmja/BiasAdd/ReadVariableOp!^vdaevhnmja/MatMul/ReadVariableOp#^vdaevhnmja/MatMul_1/ReadVariableOp^vdaevhnmja/ReadVariableOp^vdaevhnmja/ReadVariableOp_1^vdaevhnmja/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!vdaevhnmja/BiasAdd/ReadVariableOp!vdaevhnmja/BiasAdd/ReadVariableOp2D
 vdaevhnmja/MatMul/ReadVariableOp vdaevhnmja/MatMul/ReadVariableOp2H
"vdaevhnmja/MatMul_1/ReadVariableOp"vdaevhnmja/MatMul_1/ReadVariableOp26
vdaevhnmja/ReadVariableOpvdaevhnmja/ReadVariableOp2:
vdaevhnmja/ReadVariableOp_1vdaevhnmja/ReadVariableOp_12:
vdaevhnmja/ReadVariableOp_2vdaevhnmja/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ùh

F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225120
inputs_0<
)vdaevhnmja_matmul_readvariableop_resource:	 >
+vdaevhnmja_matmul_1_readvariableop_resource:	 9
*vdaevhnmja_biasadd_readvariableop_resource:	0
"vdaevhnmja_readvariableop_resource: 2
$vdaevhnmja_readvariableop_1_resource: 2
$vdaevhnmja_readvariableop_2_resource: 
identity¢!vdaevhnmja/BiasAdd/ReadVariableOp¢ vdaevhnmja/MatMul/ReadVariableOp¢"vdaevhnmja/MatMul_1/ReadVariableOp¢vdaevhnmja/ReadVariableOp¢vdaevhnmja/ReadVariableOp_1¢vdaevhnmja/ReadVariableOp_2¢whileF
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
 vdaevhnmja/MatMul/ReadVariableOpReadVariableOp)vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 vdaevhnmja/MatMul/ReadVariableOp§
vdaevhnmja/MatMulMatMulstrided_slice_2:output:0(vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMulµ
"vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp+vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"vdaevhnmja/MatMul_1/ReadVariableOp£
vdaevhnmja/MatMul_1MatMulzeros:output:0*vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMul_1
vdaevhnmja/addAddV2vdaevhnmja/MatMul:product:0vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/add®
!vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp*vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!vdaevhnmja/BiasAdd/ReadVariableOp¥
vdaevhnmja/BiasAddBiasAddvdaevhnmja/add:z:0)vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/BiasAddz
vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vdaevhnmja/split/split_dimë
vdaevhnmja/splitSplit#vdaevhnmja/split/split_dim:output:0vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vdaevhnmja/split
vdaevhnmja/ReadVariableOpReadVariableOp"vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp
vdaevhnmja/mulMul!vdaevhnmja/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul
vdaevhnmja/add_1AddV2vdaevhnmja/split:output:0vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_1{
vdaevhnmja/SigmoidSigmoidvdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid
vdaevhnmja/ReadVariableOp_1ReadVariableOp$vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_1
vdaevhnmja/mul_1Mul#vdaevhnmja/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_1
vdaevhnmja/add_2AddV2vdaevhnmja/split:output:1vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_2
vdaevhnmja/Sigmoid_1Sigmoidvdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_1
vdaevhnmja/mul_2Mulvdaevhnmja/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_2w
vdaevhnmja/TanhTanhvdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh
vdaevhnmja/mul_3Mulvdaevhnmja/Sigmoid:y:0vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_3
vdaevhnmja/add_3AddV2vdaevhnmja/mul_2:z:0vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_3
vdaevhnmja/ReadVariableOp_2ReadVariableOp$vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_2
vdaevhnmja/mul_4Mul#vdaevhnmja/ReadVariableOp_2:value:0vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_4
vdaevhnmja/add_4AddV2vdaevhnmja/split:output:3vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_4
vdaevhnmja/Sigmoid_2Sigmoidvdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_2v
vdaevhnmja/Tanh_1Tanhvdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh_1
vdaevhnmja/mul_5Mulvdaevhnmja/Sigmoid_2:y:0vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)vdaevhnmja_matmul_readvariableop_resource+vdaevhnmja_matmul_1_readvariableop_resource*vdaevhnmja_biasadd_readvariableop_resource"vdaevhnmja_readvariableop_resource$vdaevhnmja_readvariableop_1_resource$vdaevhnmja_readvariableop_2_resource*
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
while_body_225019*
condR
while_cond_225018*Q
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
IdentityIdentitystrided_slice_3:output:0"^vdaevhnmja/BiasAdd/ReadVariableOp!^vdaevhnmja/MatMul/ReadVariableOp#^vdaevhnmja/MatMul_1/ReadVariableOp^vdaevhnmja/ReadVariableOp^vdaevhnmja/ReadVariableOp_1^vdaevhnmja/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!vdaevhnmja/BiasAdd/ReadVariableOp!vdaevhnmja/BiasAdd/ReadVariableOp2D
 vdaevhnmja/MatMul/ReadVariableOp vdaevhnmja/MatMul/ReadVariableOp2H
"vdaevhnmja/MatMul_1/ReadVariableOp"vdaevhnmja/MatMul_1/ReadVariableOp26
vdaevhnmja/ReadVariableOpvdaevhnmja/ReadVariableOp2:
vdaevhnmja/ReadVariableOp_1vdaevhnmja/ReadVariableOp_12:
vdaevhnmja/ReadVariableOp_2vdaevhnmja/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Ùh

F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225300
inputs_0<
)vdaevhnmja_matmul_readvariableop_resource:	 >
+vdaevhnmja_matmul_1_readvariableop_resource:	 9
*vdaevhnmja_biasadd_readvariableop_resource:	0
"vdaevhnmja_readvariableop_resource: 2
$vdaevhnmja_readvariableop_1_resource: 2
$vdaevhnmja_readvariableop_2_resource: 
identity¢!vdaevhnmja/BiasAdd/ReadVariableOp¢ vdaevhnmja/MatMul/ReadVariableOp¢"vdaevhnmja/MatMul_1/ReadVariableOp¢vdaevhnmja/ReadVariableOp¢vdaevhnmja/ReadVariableOp_1¢vdaevhnmja/ReadVariableOp_2¢whileF
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
 vdaevhnmja/MatMul/ReadVariableOpReadVariableOp)vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 vdaevhnmja/MatMul/ReadVariableOp§
vdaevhnmja/MatMulMatMulstrided_slice_2:output:0(vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMulµ
"vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp+vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"vdaevhnmja/MatMul_1/ReadVariableOp£
vdaevhnmja/MatMul_1MatMulzeros:output:0*vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/MatMul_1
vdaevhnmja/addAddV2vdaevhnmja/MatMul:product:0vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/add®
!vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp*vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!vdaevhnmja/BiasAdd/ReadVariableOp¥
vdaevhnmja/BiasAddBiasAddvdaevhnmja/add:z:0)vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vdaevhnmja/BiasAddz
vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
vdaevhnmja/split/split_dimë
vdaevhnmja/splitSplit#vdaevhnmja/split/split_dim:output:0vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vdaevhnmja/split
vdaevhnmja/ReadVariableOpReadVariableOp"vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp
vdaevhnmja/mulMul!vdaevhnmja/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul
vdaevhnmja/add_1AddV2vdaevhnmja/split:output:0vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_1{
vdaevhnmja/SigmoidSigmoidvdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid
vdaevhnmja/ReadVariableOp_1ReadVariableOp$vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_1
vdaevhnmja/mul_1Mul#vdaevhnmja/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_1
vdaevhnmja/add_2AddV2vdaevhnmja/split:output:1vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_2
vdaevhnmja/Sigmoid_1Sigmoidvdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_1
vdaevhnmja/mul_2Mulvdaevhnmja/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_2w
vdaevhnmja/TanhTanhvdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh
vdaevhnmja/mul_3Mulvdaevhnmja/Sigmoid:y:0vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_3
vdaevhnmja/add_3AddV2vdaevhnmja/mul_2:z:0vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_3
vdaevhnmja/ReadVariableOp_2ReadVariableOp$vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02
vdaevhnmja/ReadVariableOp_2
vdaevhnmja/mul_4Mul#vdaevhnmja/ReadVariableOp_2:value:0vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_4
vdaevhnmja/add_4AddV2vdaevhnmja/split:output:3vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/add_4
vdaevhnmja/Sigmoid_2Sigmoidvdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Sigmoid_2v
vdaevhnmja/Tanh_1Tanhvdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/Tanh_1
vdaevhnmja/mul_5Mulvdaevhnmja/Sigmoid_2:y:0vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vdaevhnmja/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)vdaevhnmja_matmul_readvariableop_resource+vdaevhnmja_matmul_1_readvariableop_resource*vdaevhnmja_biasadd_readvariableop_resource"vdaevhnmja_readvariableop_resource$vdaevhnmja_readvariableop_1_resource$vdaevhnmja_readvariableop_2_resource*
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
while_body_225199*
condR
while_cond_225198*Q
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
IdentityIdentitystrided_slice_3:output:0"^vdaevhnmja/BiasAdd/ReadVariableOp!^vdaevhnmja/MatMul/ReadVariableOp#^vdaevhnmja/MatMul_1/ReadVariableOp^vdaevhnmja/ReadVariableOp^vdaevhnmja/ReadVariableOp_1^vdaevhnmja/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!vdaevhnmja/BiasAdd/ReadVariableOp!vdaevhnmja/BiasAdd/ReadVariableOp2D
 vdaevhnmja/MatMul/ReadVariableOp vdaevhnmja/MatMul/ReadVariableOp2H
"vdaevhnmja/MatMul_1/ReadVariableOp"vdaevhnmja/MatMul_1/ReadVariableOp26
vdaevhnmja/ReadVariableOpvdaevhnmja/ReadVariableOp2:
vdaevhnmja/ReadVariableOp_1vdaevhnmja/ReadVariableOp_12:
vdaevhnmja/ReadVariableOp_2vdaevhnmja/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0

¯
F__inference_sequential_layer_call_and_return_conditional_losses_223093

ahzwxypkrh'
iigfihrkup_223055:
iigfihrkup_223057:$
omilqdycns_223061:	$
omilqdycns_223063:	  
omilqdycns_223065:	
omilqdycns_223067: 
omilqdycns_223069: 
omilqdycns_223071: $
vlxoswgdqw_223074:	 $
vlxoswgdqw_223076:	  
vlxoswgdqw_223078:	
vlxoswgdqw_223080: 
vlxoswgdqw_223082: 
vlxoswgdqw_223084: #
iktogmlrmp_223087: 
iktogmlrmp_223089:
identity¢"iigfihrkup/StatefulPartitionedCall¢"iktogmlrmp/StatefulPartitionedCall¢"omilqdycns/StatefulPartitionedCall¢"vlxoswgdqw/StatefulPartitionedCall­
"iigfihrkup/StatefulPartitionedCallStatefulPartitionedCall
ahzwxypkrhiigfihrkup_223055iigfihrkup_223057*
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
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_2219462$
"iigfihrkup/StatefulPartitionedCall
xfbsciqeco/PartitionedCallPartitionedCall+iigfihrkup/StatefulPartitionedCall:output:0*
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
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_2219652
xfbsciqeco/PartitionedCall
"omilqdycns/StatefulPartitionedCallStatefulPartitionedCall#xfbsciqeco/PartitionedCall:output:0omilqdycns_223061omilqdycns_223063omilqdycns_223065omilqdycns_223067omilqdycns_223069omilqdycns_223071*
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_2228282$
"omilqdycns/StatefulPartitionedCall
"vlxoswgdqw/StatefulPartitionedCallStatefulPartitionedCall+omilqdycns/StatefulPartitionedCall:output:0vlxoswgdqw_223074vlxoswgdqw_223076vlxoswgdqw_223078vlxoswgdqw_223080vlxoswgdqw_223082vlxoswgdqw_223084*
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_2226142$
"vlxoswgdqw/StatefulPartitionedCallÆ
"iktogmlrmp/StatefulPartitionedCallStatefulPartitionedCall+vlxoswgdqw/StatefulPartitionedCall:output:0iktogmlrmp_223087iktogmlrmp_223089*
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
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_2223632$
"iktogmlrmp/StatefulPartitionedCall
IdentityIdentity+iktogmlrmp/StatefulPartitionedCall:output:0#^iigfihrkup/StatefulPartitionedCall#^iktogmlrmp/StatefulPartitionedCall#^omilqdycns/StatefulPartitionedCall#^vlxoswgdqw/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"iigfihrkup/StatefulPartitionedCall"iigfihrkup/StatefulPartitionedCall2H
"iktogmlrmp/StatefulPartitionedCall"iktogmlrmp/StatefulPartitionedCall2H
"omilqdycns/StatefulPartitionedCall"omilqdycns/StatefulPartitionedCall2H
"vlxoswgdqw/StatefulPartitionedCall"vlxoswgdqw/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
ahzwxypkrh
¯

"__inference__traced_restore_226214
file_prefix8
"assignvariableop_iigfihrkup_kernel:0
"assignvariableop_1_iigfihrkup_bias:6
$assignvariableop_2_iktogmlrmp_kernel: 0
"assignvariableop_3_iktogmlrmp_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_omilqdycns_ddlymsxapn_kernel:	M
:assignvariableop_10_omilqdycns_ddlymsxapn_recurrent_kernel:	 =
.assignvariableop_11_omilqdycns_ddlymsxapn_bias:	S
Eassignvariableop_12_omilqdycns_ddlymsxapn_input_gate_peephole_weights: T
Fassignvariableop_13_omilqdycns_ddlymsxapn_forget_gate_peephole_weights: T
Fassignvariableop_14_omilqdycns_ddlymsxapn_output_gate_peephole_weights: C
0assignvariableop_15_vlxoswgdqw_vdaevhnmja_kernel:	 M
:assignvariableop_16_vlxoswgdqw_vdaevhnmja_recurrent_kernel:	 =
.assignvariableop_17_vlxoswgdqw_vdaevhnmja_bias:	S
Eassignvariableop_18_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weights: T
Fassignvariableop_19_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weights: T
Fassignvariableop_20_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_iigfihrkup_kernel_rms:=
/assignvariableop_24_rmsprop_iigfihrkup_bias_rms:C
1assignvariableop_25_rmsprop_iktogmlrmp_kernel_rms: =
/assignvariableop_26_rmsprop_iktogmlrmp_bias_rms:O
<assignvariableop_27_rmsprop_omilqdycns_ddlymsxapn_kernel_rms:	Y
Fassignvariableop_28_rmsprop_omilqdycns_ddlymsxapn_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_omilqdycns_ddlymsxapn_bias_rms:	_
Qassignvariableop_30_rmsprop_omilqdycns_ddlymsxapn_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_omilqdycns_ddlymsxapn_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_omilqdycns_ddlymsxapn_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_vlxoswgdqw_vdaevhnmja_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_vlxoswgdqw_vdaevhnmja_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_vlxoswgdqw_vdaevhnmja_bias_rms:	_
Qassignvariableop_36_rmsprop_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_iigfihrkup_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_iigfihrkup_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_iktogmlrmp_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_iktogmlrmp_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_omilqdycns_ddlymsxapn_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_omilqdycns_ddlymsxapn_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_omilqdycns_ddlymsxapn_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_omilqdycns_ddlymsxapn_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_omilqdycns_ddlymsxapn_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_omilqdycns_ddlymsxapn_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_vlxoswgdqw_vdaevhnmja_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_vlxoswgdqw_vdaevhnmja_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_vlxoswgdqw_vdaevhnmja_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_iigfihrkup_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_iigfihrkup_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_iktogmlrmp_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_iktogmlrmp_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_omilqdycns_ddlymsxapn_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_omilqdycns_ddlymsxapn_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_omilqdycns_ddlymsxapn_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_omilqdycns_ddlymsxapn_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_omilqdycns_ddlymsxapn_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_omilqdycns_ddlymsxapn_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_vlxoswgdqw_vdaevhnmja_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_vlxoswgdqw_vdaevhnmja_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_vlxoswgdqw_vdaevhnmja_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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
Z
Ê
__inference__traced_save_226087
file_prefix0
,savev2_iigfihrkup_kernel_read_readvariableop.
*savev2_iigfihrkup_bias_read_readvariableop0
,savev2_iktogmlrmp_kernel_read_readvariableop.
*savev2_iktogmlrmp_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_omilqdycns_ddlymsxapn_kernel_read_readvariableopE
Asavev2_omilqdycns_ddlymsxapn_recurrent_kernel_read_readvariableop9
5savev2_omilqdycns_ddlymsxapn_bias_read_readvariableopP
Lsavev2_omilqdycns_ddlymsxapn_input_gate_peephole_weights_read_readvariableopQ
Msavev2_omilqdycns_ddlymsxapn_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_omilqdycns_ddlymsxapn_output_gate_peephole_weights_read_readvariableop;
7savev2_vlxoswgdqw_vdaevhnmja_kernel_read_readvariableopE
Asavev2_vlxoswgdqw_vdaevhnmja_recurrent_kernel_read_readvariableop9
5savev2_vlxoswgdqw_vdaevhnmja_bias_read_readvariableopP
Lsavev2_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weights_read_readvariableopQ
Msavev2_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_iigfihrkup_kernel_rms_read_readvariableop:
6savev2_rmsprop_iigfihrkup_bias_rms_read_readvariableop<
8savev2_rmsprop_iktogmlrmp_kernel_rms_read_readvariableop:
6savev2_rmsprop_iktogmlrmp_bias_rms_read_readvariableopG
Csavev2_rmsprop_omilqdycns_ddlymsxapn_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_omilqdycns_ddlymsxapn_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_omilqdycns_ddlymsxapn_bias_rms_read_readvariableop\
Xsavev2_rmsprop_omilqdycns_ddlymsxapn_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_omilqdycns_ddlymsxapn_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_omilqdycns_ddlymsxapn_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_vlxoswgdqw_vdaevhnmja_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_vlxoswgdqw_vdaevhnmja_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_vlxoswgdqw_vdaevhnmja_bias_rms_read_readvariableop\
Xsavev2_rmsprop_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_iigfihrkup_kernel_read_readvariableop*savev2_iigfihrkup_bias_read_readvariableop,savev2_iktogmlrmp_kernel_read_readvariableop*savev2_iktogmlrmp_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_omilqdycns_ddlymsxapn_kernel_read_readvariableopAsavev2_omilqdycns_ddlymsxapn_recurrent_kernel_read_readvariableop5savev2_omilqdycns_ddlymsxapn_bias_read_readvariableopLsavev2_omilqdycns_ddlymsxapn_input_gate_peephole_weights_read_readvariableopMsavev2_omilqdycns_ddlymsxapn_forget_gate_peephole_weights_read_readvariableopMsavev2_omilqdycns_ddlymsxapn_output_gate_peephole_weights_read_readvariableop7savev2_vlxoswgdqw_vdaevhnmja_kernel_read_readvariableopAsavev2_vlxoswgdqw_vdaevhnmja_recurrent_kernel_read_readvariableop5savev2_vlxoswgdqw_vdaevhnmja_bias_read_readvariableopLsavev2_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weights_read_readvariableopMsavev2_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weights_read_readvariableopMsavev2_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_iigfihrkup_kernel_rms_read_readvariableop6savev2_rmsprop_iigfihrkup_bias_rms_read_readvariableop8savev2_rmsprop_iktogmlrmp_kernel_rms_read_readvariableop6savev2_rmsprop_iktogmlrmp_bias_rms_read_readvariableopCsavev2_rmsprop_omilqdycns_ddlymsxapn_kernel_rms_read_readvariableopMsavev2_rmsprop_omilqdycns_ddlymsxapn_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_omilqdycns_ddlymsxapn_bias_rms_read_readvariableopXsavev2_rmsprop_omilqdycns_ddlymsxapn_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_omilqdycns_ddlymsxapn_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_omilqdycns_ddlymsxapn_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_vlxoswgdqw_vdaevhnmja_kernel_rms_read_readvariableopMsavev2_rmsprop_vlxoswgdqw_vdaevhnmja_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_vlxoswgdqw_vdaevhnmja_bias_rms_read_readvariableopXsavev2_rmsprop_vlxoswgdqw_vdaevhnmja_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_vlxoswgdqw_vdaevhnmja_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_vlxoswgdqw_vdaevhnmja_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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


+__inference_sequential_layer_call_fn_223175

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
F__inference_sequential_layer_call_and_return_conditional_losses_2223702
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
ÙÊ

F__inference_sequential_layer_call_and_return_conditional_losses_224020

inputsL
6iigfihrkup_conv1d_expanddims_1_readvariableop_resource:K
=iigfihrkup_squeeze_batch_dims_biasadd_readvariableop_resource:G
4omilqdycns_ddlymsxapn_matmul_readvariableop_resource:	I
6omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource:	 D
5omilqdycns_ddlymsxapn_biasadd_readvariableop_resource:	;
-omilqdycns_ddlymsxapn_readvariableop_resource: =
/omilqdycns_ddlymsxapn_readvariableop_1_resource: =
/omilqdycns_ddlymsxapn_readvariableop_2_resource: G
4vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource:	 I
6vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource:	 D
5vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource:	;
-vlxoswgdqw_vdaevhnmja_readvariableop_resource: =
/vlxoswgdqw_vdaevhnmja_readvariableop_1_resource: =
/vlxoswgdqw_vdaevhnmja_readvariableop_2_resource: ;
)iktogmlrmp_matmul_readvariableop_resource: 8
*iktogmlrmp_biasadd_readvariableop_resource:
identity¢-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp¢4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!iktogmlrmp/BiasAdd/ReadVariableOp¢ iktogmlrmp/MatMul/ReadVariableOp¢,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp¢+omilqdycns/ddlymsxapn/MatMul/ReadVariableOp¢-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp¢$omilqdycns/ddlymsxapn/ReadVariableOp¢&omilqdycns/ddlymsxapn/ReadVariableOp_1¢&omilqdycns/ddlymsxapn/ReadVariableOp_2¢omilqdycns/while¢,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp¢+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp¢-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp¢$vlxoswgdqw/vdaevhnmja/ReadVariableOp¢&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1¢&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2¢vlxoswgdqw/while
 iigfihrkup/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 iigfihrkup/conv1d/ExpandDims/dim»
iigfihrkup/conv1d/ExpandDims
ExpandDimsinputs)iigfihrkup/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/ExpandDimsÙ
-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6iigfihrkup_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp
"iigfihrkup/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"iigfihrkup/conv1d/ExpandDims_1/dimã
iigfihrkup/conv1d/ExpandDims_1
ExpandDims5iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp:value:0+iigfihrkup/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
iigfihrkup/conv1d/ExpandDims_1
iigfihrkup/conv1d/ShapeShape%iigfihrkup/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
iigfihrkup/conv1d/Shape
%iigfihrkup/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%iigfihrkup/conv1d/strided_slice/stack¥
'iigfihrkup/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'iigfihrkup/conv1d/strided_slice/stack_1
'iigfihrkup/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'iigfihrkup/conv1d/strided_slice/stack_2Ì
iigfihrkup/conv1d/strided_sliceStridedSlice iigfihrkup/conv1d/Shape:output:0.iigfihrkup/conv1d/strided_slice/stack:output:00iigfihrkup/conv1d/strided_slice/stack_1:output:00iigfihrkup/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
iigfihrkup/conv1d/strided_slice
iigfihrkup/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
iigfihrkup/conv1d/Reshape/shapeÌ
iigfihrkup/conv1d/ReshapeReshape%iigfihrkup/conv1d/ExpandDims:output:0(iigfihrkup/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/Reshapeî
iigfihrkup/conv1d/Conv2DConv2D"iigfihrkup/conv1d/Reshape:output:0'iigfihrkup/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
iigfihrkup/conv1d/Conv2D
!iigfihrkup/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!iigfihrkup/conv1d/concat/values_1
iigfihrkup/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/concat/axisì
iigfihrkup/conv1d/concatConcatV2(iigfihrkup/conv1d/strided_slice:output:0*iigfihrkup/conv1d/concat/values_1:output:0&iigfihrkup/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
iigfihrkup/conv1d/concatÉ
iigfihrkup/conv1d/Reshape_1Reshape!iigfihrkup/conv1d/Conv2D:output:0!iigfihrkup/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/Reshape_1Á
iigfihrkup/conv1d/SqueezeSqueeze$iigfihrkup/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/Squeeze
#iigfihrkup/squeeze_batch_dims/ShapeShape"iigfihrkup/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#iigfihrkup/squeeze_batch_dims/Shape°
1iigfihrkup/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1iigfihrkup/squeeze_batch_dims/strided_slice/stack½
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_1´
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_2
+iigfihrkup/squeeze_batch_dims/strided_sliceStridedSlice,iigfihrkup/squeeze_batch_dims/Shape:output:0:iigfihrkup/squeeze_batch_dims/strided_slice/stack:output:0<iigfihrkup/squeeze_batch_dims/strided_slice/stack_1:output:0<iigfihrkup/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+iigfihrkup/squeeze_batch_dims/strided_slice¯
+iigfihrkup/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+iigfihrkup/squeeze_batch_dims/Reshape/shapeé
%iigfihrkup/squeeze_batch_dims/ReshapeReshape"iigfihrkup/conv1d/Squeeze:output:04iigfihrkup/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%iigfihrkup/squeeze_batch_dims/Reshapeæ
4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=iigfihrkup_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%iigfihrkup/squeeze_batch_dims/BiasAddBiasAdd.iigfihrkup/squeeze_batch_dims/Reshape:output:0<iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%iigfihrkup/squeeze_batch_dims/BiasAdd¯
-iigfihrkup/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-iigfihrkup/squeeze_batch_dims/concat/values_1¡
)iigfihrkup/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)iigfihrkup/squeeze_batch_dims/concat/axis¨
$iigfihrkup/squeeze_batch_dims/concatConcatV24iigfihrkup/squeeze_batch_dims/strided_slice:output:06iigfihrkup/squeeze_batch_dims/concat/values_1:output:02iigfihrkup/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$iigfihrkup/squeeze_batch_dims/concatö
'iigfihrkup/squeeze_batch_dims/Reshape_1Reshape.iigfihrkup/squeeze_batch_dims/BiasAdd:output:0-iigfihrkup/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'iigfihrkup/squeeze_batch_dims/Reshape_1
xfbsciqeco/ShapeShape0iigfihrkup/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
xfbsciqeco/Shape
xfbsciqeco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xfbsciqeco/strided_slice/stack
 xfbsciqeco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xfbsciqeco/strided_slice/stack_1
 xfbsciqeco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xfbsciqeco/strided_slice/stack_2¤
xfbsciqeco/strided_sliceStridedSlicexfbsciqeco/Shape:output:0'xfbsciqeco/strided_slice/stack:output:0)xfbsciqeco/strided_slice/stack_1:output:0)xfbsciqeco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xfbsciqeco/strided_slicez
xfbsciqeco/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
xfbsciqeco/Reshape/shape/1z
xfbsciqeco/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
xfbsciqeco/Reshape/shape/2×
xfbsciqeco/Reshape/shapePack!xfbsciqeco/strided_slice:output:0#xfbsciqeco/Reshape/shape/1:output:0#xfbsciqeco/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
xfbsciqeco/Reshape/shape¾
xfbsciqeco/ReshapeReshape0iigfihrkup/squeeze_batch_dims/Reshape_1:output:0!xfbsciqeco/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xfbsciqeco/Reshapeo
omilqdycns/ShapeShapexfbsciqeco/Reshape:output:0*
T0*
_output_shapes
:2
omilqdycns/Shape
omilqdycns/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
omilqdycns/strided_slice/stack
 omilqdycns/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 omilqdycns/strided_slice/stack_1
 omilqdycns/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 omilqdycns/strided_slice/stack_2¤
omilqdycns/strided_sliceStridedSliceomilqdycns/Shape:output:0'omilqdycns/strided_slice/stack:output:0)omilqdycns/strided_slice/stack_1:output:0)omilqdycns/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
omilqdycns/strided_slicer
omilqdycns/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros/mul/y
omilqdycns/zeros/mulMul!omilqdycns/strided_slice:output:0omilqdycns/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros/mulu
omilqdycns/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
omilqdycns/zeros/Less/y
omilqdycns/zeros/LessLessomilqdycns/zeros/mul:z:0 omilqdycns/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros/Lessx
omilqdycns/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros/packed/1¯
omilqdycns/zeros/packedPack!omilqdycns/strided_slice:output:0"omilqdycns/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
omilqdycns/zeros/packedu
omilqdycns/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
omilqdycns/zeros/Const¡
omilqdycns/zerosFill omilqdycns/zeros/packed:output:0omilqdycns/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/zerosv
omilqdycns/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros_1/mul/y
omilqdycns/zeros_1/mulMul!omilqdycns/strided_slice:output:0!omilqdycns/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros_1/muly
omilqdycns/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
omilqdycns/zeros_1/Less/y
omilqdycns/zeros_1/LessLessomilqdycns/zeros_1/mul:z:0"omilqdycns/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros_1/Less|
omilqdycns/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros_1/packed/1µ
omilqdycns/zeros_1/packedPack!omilqdycns/strided_slice:output:0$omilqdycns/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
omilqdycns/zeros_1/packedy
omilqdycns/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
omilqdycns/zeros_1/Const©
omilqdycns/zeros_1Fill"omilqdycns/zeros_1/packed:output:0!omilqdycns/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/zeros_1
omilqdycns/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
omilqdycns/transpose/perm°
omilqdycns/transpose	Transposexfbsciqeco/Reshape:output:0"omilqdycns/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/transposep
omilqdycns/Shape_1Shapeomilqdycns/transpose:y:0*
T0*
_output_shapes
:2
omilqdycns/Shape_1
 omilqdycns/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 omilqdycns/strided_slice_1/stack
"omilqdycns/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_1/stack_1
"omilqdycns/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_1/stack_2°
omilqdycns/strided_slice_1StridedSliceomilqdycns/Shape_1:output:0)omilqdycns/strided_slice_1/stack:output:0+omilqdycns/strided_slice_1/stack_1:output:0+omilqdycns/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
omilqdycns/strided_slice_1
&omilqdycns/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&omilqdycns/TensorArrayV2/element_shapeÞ
omilqdycns/TensorArrayV2TensorListReserve/omilqdycns/TensorArrayV2/element_shape:output:0#omilqdycns/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
omilqdycns/TensorArrayV2Õ
@omilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@omilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2omilqdycns/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoromilqdycns/transpose:y:0Iomilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2omilqdycns/TensorArrayUnstack/TensorListFromTensor
 omilqdycns/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 omilqdycns/strided_slice_2/stack
"omilqdycns/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_2/stack_1
"omilqdycns/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_2/stack_2¾
omilqdycns/strided_slice_2StridedSliceomilqdycns/transpose:y:0)omilqdycns/strided_slice_2/stack:output:0+omilqdycns/strided_slice_2/stack_1:output:0+omilqdycns/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
omilqdycns/strided_slice_2Ð
+omilqdycns/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp4omilqdycns_ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+omilqdycns/ddlymsxapn/MatMul/ReadVariableOpÓ
omilqdycns/ddlymsxapn/MatMulMatMul#omilqdycns/strided_slice_2:output:03omilqdycns/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/ddlymsxapn/MatMulÖ
-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp6omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOpÏ
omilqdycns/ddlymsxapn/MatMul_1MatMulomilqdycns/zeros:output:05omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
omilqdycns/ddlymsxapn/MatMul_1Ä
omilqdycns/ddlymsxapn/addAddV2&omilqdycns/ddlymsxapn/MatMul:product:0(omilqdycns/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/ddlymsxapn/addÏ
,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp5omilqdycns_ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOpÑ
omilqdycns/ddlymsxapn/BiasAddBiasAddomilqdycns/ddlymsxapn/add:z:04omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/ddlymsxapn/BiasAdd
%omilqdycns/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%omilqdycns/ddlymsxapn/split/split_dim
omilqdycns/ddlymsxapn/splitSplit.omilqdycns/ddlymsxapn/split/split_dim:output:0&omilqdycns/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
omilqdycns/ddlymsxapn/split¶
$omilqdycns/ddlymsxapn/ReadVariableOpReadVariableOp-omilqdycns_ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02&
$omilqdycns/ddlymsxapn/ReadVariableOpº
omilqdycns/ddlymsxapn/mulMul,omilqdycns/ddlymsxapn/ReadVariableOp:value:0omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mulº
omilqdycns/ddlymsxapn/add_1AddV2$omilqdycns/ddlymsxapn/split:output:0omilqdycns/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_1
omilqdycns/ddlymsxapn/SigmoidSigmoidomilqdycns/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/Sigmoid¼
&omilqdycns/ddlymsxapn/ReadVariableOp_1ReadVariableOp/omilqdycns_ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&omilqdycns/ddlymsxapn/ReadVariableOp_1À
omilqdycns/ddlymsxapn/mul_1Mul.omilqdycns/ddlymsxapn/ReadVariableOp_1:value:0omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_1¼
omilqdycns/ddlymsxapn/add_2AddV2$omilqdycns/ddlymsxapn/split:output:1omilqdycns/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_2 
omilqdycns/ddlymsxapn/Sigmoid_1Sigmoidomilqdycns/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
omilqdycns/ddlymsxapn/Sigmoid_1µ
omilqdycns/ddlymsxapn/mul_2Mul#omilqdycns/ddlymsxapn/Sigmoid_1:y:0omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_2
omilqdycns/ddlymsxapn/TanhTanh$omilqdycns/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/Tanh¶
omilqdycns/ddlymsxapn/mul_3Mul!omilqdycns/ddlymsxapn/Sigmoid:y:0omilqdycns/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_3·
omilqdycns/ddlymsxapn/add_3AddV2omilqdycns/ddlymsxapn/mul_2:z:0omilqdycns/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_3¼
&omilqdycns/ddlymsxapn/ReadVariableOp_2ReadVariableOp/omilqdycns_ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&omilqdycns/ddlymsxapn/ReadVariableOp_2Ä
omilqdycns/ddlymsxapn/mul_4Mul.omilqdycns/ddlymsxapn/ReadVariableOp_2:value:0omilqdycns/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_4¼
omilqdycns/ddlymsxapn/add_4AddV2$omilqdycns/ddlymsxapn/split:output:3omilqdycns/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_4 
omilqdycns/ddlymsxapn/Sigmoid_2Sigmoidomilqdycns/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
omilqdycns/ddlymsxapn/Sigmoid_2
omilqdycns/ddlymsxapn/Tanh_1Tanhomilqdycns/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/Tanh_1º
omilqdycns/ddlymsxapn/mul_5Mul#omilqdycns/ddlymsxapn/Sigmoid_2:y:0 omilqdycns/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_5¥
(omilqdycns/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(omilqdycns/TensorArrayV2_1/element_shapeä
omilqdycns/TensorArrayV2_1TensorListReserve1omilqdycns/TensorArrayV2_1/element_shape:output:0#omilqdycns/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
omilqdycns/TensorArrayV2_1d
omilqdycns/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/time
#omilqdycns/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#omilqdycns/while/maximum_iterations
omilqdycns/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/while/loop_counter°
omilqdycns/whileWhile&omilqdycns/while/loop_counter:output:0,omilqdycns/while/maximum_iterations:output:0omilqdycns/time:output:0#omilqdycns/TensorArrayV2_1:handle:0omilqdycns/zeros:output:0omilqdycns/zeros_1:output:0#omilqdycns/strided_slice_1:output:0Bomilqdycns/TensorArrayUnstack/TensorListFromTensor:output_handle:04omilqdycns_ddlymsxapn_matmul_readvariableop_resource6omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource5omilqdycns_ddlymsxapn_biasadd_readvariableop_resource-omilqdycns_ddlymsxapn_readvariableop_resource/omilqdycns_ddlymsxapn_readvariableop_1_resource/omilqdycns_ddlymsxapn_readvariableop_2_resource*
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
omilqdycns_while_body_223737*(
cond R
omilqdycns_while_cond_223736*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
omilqdycns/whileË
;omilqdycns/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;omilqdycns/TensorArrayV2Stack/TensorListStack/element_shape
-omilqdycns/TensorArrayV2Stack/TensorListStackTensorListStackomilqdycns/while:output:3Domilqdycns/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-omilqdycns/TensorArrayV2Stack/TensorListStack
 omilqdycns/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 omilqdycns/strided_slice_3/stack
"omilqdycns/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"omilqdycns/strided_slice_3/stack_1
"omilqdycns/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_3/stack_2Ü
omilqdycns/strided_slice_3StridedSlice6omilqdycns/TensorArrayV2Stack/TensorListStack:tensor:0)omilqdycns/strided_slice_3/stack:output:0+omilqdycns/strided_slice_3/stack_1:output:0+omilqdycns/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
omilqdycns/strided_slice_3
omilqdycns/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
omilqdycns/transpose_1/permÑ
omilqdycns/transpose_1	Transpose6omilqdycns/TensorArrayV2Stack/TensorListStack:tensor:0$omilqdycns/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/transpose_1n
vlxoswgdqw/ShapeShapeomilqdycns/transpose_1:y:0*
T0*
_output_shapes
:2
vlxoswgdqw/Shape
vlxoswgdqw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
vlxoswgdqw/strided_slice/stack
 vlxoswgdqw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 vlxoswgdqw/strided_slice/stack_1
 vlxoswgdqw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 vlxoswgdqw/strided_slice/stack_2¤
vlxoswgdqw/strided_sliceStridedSlicevlxoswgdqw/Shape:output:0'vlxoswgdqw/strided_slice/stack:output:0)vlxoswgdqw/strided_slice/stack_1:output:0)vlxoswgdqw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vlxoswgdqw/strided_slicer
vlxoswgdqw/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros/mul/y
vlxoswgdqw/zeros/mulMul!vlxoswgdqw/strided_slice:output:0vlxoswgdqw/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros/mulu
vlxoswgdqw/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vlxoswgdqw/zeros/Less/y
vlxoswgdqw/zeros/LessLessvlxoswgdqw/zeros/mul:z:0 vlxoswgdqw/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros/Lessx
vlxoswgdqw/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros/packed/1¯
vlxoswgdqw/zeros/packedPack!vlxoswgdqw/strided_slice:output:0"vlxoswgdqw/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
vlxoswgdqw/zeros/packedu
vlxoswgdqw/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vlxoswgdqw/zeros/Const¡
vlxoswgdqw/zerosFill vlxoswgdqw/zeros/packed:output:0vlxoswgdqw/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/zerosv
vlxoswgdqw/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros_1/mul/y
vlxoswgdqw/zeros_1/mulMul!vlxoswgdqw/strided_slice:output:0!vlxoswgdqw/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros_1/muly
vlxoswgdqw/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vlxoswgdqw/zeros_1/Less/y
vlxoswgdqw/zeros_1/LessLessvlxoswgdqw/zeros_1/mul:z:0"vlxoswgdqw/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros_1/Less|
vlxoswgdqw/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros_1/packed/1µ
vlxoswgdqw/zeros_1/packedPack!vlxoswgdqw/strided_slice:output:0$vlxoswgdqw/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
vlxoswgdqw/zeros_1/packedy
vlxoswgdqw/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vlxoswgdqw/zeros_1/Const©
vlxoswgdqw/zeros_1Fill"vlxoswgdqw/zeros_1/packed:output:0!vlxoswgdqw/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/zeros_1
vlxoswgdqw/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vlxoswgdqw/transpose/perm¯
vlxoswgdqw/transpose	Transposeomilqdycns/transpose_1:y:0"vlxoswgdqw/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/transposep
vlxoswgdqw/Shape_1Shapevlxoswgdqw/transpose:y:0*
T0*
_output_shapes
:2
vlxoswgdqw/Shape_1
 vlxoswgdqw/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vlxoswgdqw/strided_slice_1/stack
"vlxoswgdqw/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_1/stack_1
"vlxoswgdqw/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_1/stack_2°
vlxoswgdqw/strided_slice_1StridedSlicevlxoswgdqw/Shape_1:output:0)vlxoswgdqw/strided_slice_1/stack:output:0+vlxoswgdqw/strided_slice_1/stack_1:output:0+vlxoswgdqw/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vlxoswgdqw/strided_slice_1
&vlxoswgdqw/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&vlxoswgdqw/TensorArrayV2/element_shapeÞ
vlxoswgdqw/TensorArrayV2TensorListReserve/vlxoswgdqw/TensorArrayV2/element_shape:output:0#vlxoswgdqw/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vlxoswgdqw/TensorArrayV2Õ
@vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2vlxoswgdqw/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorvlxoswgdqw/transpose:y:0Ivlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor
 vlxoswgdqw/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vlxoswgdqw/strided_slice_2/stack
"vlxoswgdqw/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_2/stack_1
"vlxoswgdqw/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_2/stack_2¾
vlxoswgdqw/strided_slice_2StridedSlicevlxoswgdqw/transpose:y:0)vlxoswgdqw/strided_slice_2/stack:output:0+vlxoswgdqw/strided_slice_2/stack_1:output:0+vlxoswgdqw/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
vlxoswgdqw/strided_slice_2Ð
+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp4vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOpÓ
vlxoswgdqw/vdaevhnmja/MatMulMatMul#vlxoswgdqw/strided_slice_2:output:03vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vlxoswgdqw/vdaevhnmja/MatMulÖ
-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp6vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOpÏ
vlxoswgdqw/vdaevhnmja/MatMul_1MatMulvlxoswgdqw/zeros:output:05vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
vlxoswgdqw/vdaevhnmja/MatMul_1Ä
vlxoswgdqw/vdaevhnmja/addAddV2&vlxoswgdqw/vdaevhnmja/MatMul:product:0(vlxoswgdqw/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vlxoswgdqw/vdaevhnmja/addÏ
,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp5vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOpÑ
vlxoswgdqw/vdaevhnmja/BiasAddBiasAddvlxoswgdqw/vdaevhnmja/add:z:04vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vlxoswgdqw/vdaevhnmja/BiasAdd
%vlxoswgdqw/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%vlxoswgdqw/vdaevhnmja/split/split_dim
vlxoswgdqw/vdaevhnmja/splitSplit.vlxoswgdqw/vdaevhnmja/split/split_dim:output:0&vlxoswgdqw/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vlxoswgdqw/vdaevhnmja/split¶
$vlxoswgdqw/vdaevhnmja/ReadVariableOpReadVariableOp-vlxoswgdqw_vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02&
$vlxoswgdqw/vdaevhnmja/ReadVariableOpº
vlxoswgdqw/vdaevhnmja/mulMul,vlxoswgdqw/vdaevhnmja/ReadVariableOp:value:0vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mulº
vlxoswgdqw/vdaevhnmja/add_1AddV2$vlxoswgdqw/vdaevhnmja/split:output:0vlxoswgdqw/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_1
vlxoswgdqw/vdaevhnmja/SigmoidSigmoidvlxoswgdqw/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/Sigmoid¼
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1ReadVariableOp/vlxoswgdqw_vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1À
vlxoswgdqw/vdaevhnmja/mul_1Mul.vlxoswgdqw/vdaevhnmja/ReadVariableOp_1:value:0vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_1¼
vlxoswgdqw/vdaevhnmja/add_2AddV2$vlxoswgdqw/vdaevhnmja/split:output:1vlxoswgdqw/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_2 
vlxoswgdqw/vdaevhnmja/Sigmoid_1Sigmoidvlxoswgdqw/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vlxoswgdqw/vdaevhnmja/Sigmoid_1µ
vlxoswgdqw/vdaevhnmja/mul_2Mul#vlxoswgdqw/vdaevhnmja/Sigmoid_1:y:0vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_2
vlxoswgdqw/vdaevhnmja/TanhTanh$vlxoswgdqw/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/Tanh¶
vlxoswgdqw/vdaevhnmja/mul_3Mul!vlxoswgdqw/vdaevhnmja/Sigmoid:y:0vlxoswgdqw/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_3·
vlxoswgdqw/vdaevhnmja/add_3AddV2vlxoswgdqw/vdaevhnmja/mul_2:z:0vlxoswgdqw/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_3¼
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2ReadVariableOp/vlxoswgdqw_vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2Ä
vlxoswgdqw/vdaevhnmja/mul_4Mul.vlxoswgdqw/vdaevhnmja/ReadVariableOp_2:value:0vlxoswgdqw/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_4¼
vlxoswgdqw/vdaevhnmja/add_4AddV2$vlxoswgdqw/vdaevhnmja/split:output:3vlxoswgdqw/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_4 
vlxoswgdqw/vdaevhnmja/Sigmoid_2Sigmoidvlxoswgdqw/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vlxoswgdqw/vdaevhnmja/Sigmoid_2
vlxoswgdqw/vdaevhnmja/Tanh_1Tanhvlxoswgdqw/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/Tanh_1º
vlxoswgdqw/vdaevhnmja/mul_5Mul#vlxoswgdqw/vdaevhnmja/Sigmoid_2:y:0 vlxoswgdqw/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_5¥
(vlxoswgdqw/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(vlxoswgdqw/TensorArrayV2_1/element_shapeä
vlxoswgdqw/TensorArrayV2_1TensorListReserve1vlxoswgdqw/TensorArrayV2_1/element_shape:output:0#vlxoswgdqw/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vlxoswgdqw/TensorArrayV2_1d
vlxoswgdqw/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/time
#vlxoswgdqw/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#vlxoswgdqw/while/maximum_iterations
vlxoswgdqw/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/while/loop_counter°
vlxoswgdqw/whileWhile&vlxoswgdqw/while/loop_counter:output:0,vlxoswgdqw/while/maximum_iterations:output:0vlxoswgdqw/time:output:0#vlxoswgdqw/TensorArrayV2_1:handle:0vlxoswgdqw/zeros:output:0vlxoswgdqw/zeros_1:output:0#vlxoswgdqw/strided_slice_1:output:0Bvlxoswgdqw/TensorArrayUnstack/TensorListFromTensor:output_handle:04vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource6vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource5vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource-vlxoswgdqw_vdaevhnmja_readvariableop_resource/vlxoswgdqw_vdaevhnmja_readvariableop_1_resource/vlxoswgdqw_vdaevhnmja_readvariableop_2_resource*
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
vlxoswgdqw_while_body_223913*(
cond R
vlxoswgdqw_while_cond_223912*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
vlxoswgdqw/whileË
;vlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;vlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shape
-vlxoswgdqw/TensorArrayV2Stack/TensorListStackTensorListStackvlxoswgdqw/while:output:3Dvlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-vlxoswgdqw/TensorArrayV2Stack/TensorListStack
 vlxoswgdqw/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 vlxoswgdqw/strided_slice_3/stack
"vlxoswgdqw/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"vlxoswgdqw/strided_slice_3/stack_1
"vlxoswgdqw/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_3/stack_2Ü
vlxoswgdqw/strided_slice_3StridedSlice6vlxoswgdqw/TensorArrayV2Stack/TensorListStack:tensor:0)vlxoswgdqw/strided_slice_3/stack:output:0+vlxoswgdqw/strided_slice_3/stack_1:output:0+vlxoswgdqw/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
vlxoswgdqw/strided_slice_3
vlxoswgdqw/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vlxoswgdqw/transpose_1/permÑ
vlxoswgdqw/transpose_1	Transpose6vlxoswgdqw/TensorArrayV2Stack/TensorListStack:tensor:0$vlxoswgdqw/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/transpose_1®
 iktogmlrmp/MatMul/ReadVariableOpReadVariableOp)iktogmlrmp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 iktogmlrmp/MatMul/ReadVariableOp±
iktogmlrmp/MatMulMatMul#vlxoswgdqw/strided_slice_3:output:0(iktogmlrmp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
iktogmlrmp/MatMul­
!iktogmlrmp/BiasAdd/ReadVariableOpReadVariableOp*iktogmlrmp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!iktogmlrmp/BiasAdd/ReadVariableOp­
iktogmlrmp/BiasAddBiasAddiktogmlrmp/MatMul:product:0)iktogmlrmp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
iktogmlrmp/BiasAddÏ
IdentityIdentityiktogmlrmp/BiasAdd:output:0.^iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp5^iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp"^iktogmlrmp/BiasAdd/ReadVariableOp!^iktogmlrmp/MatMul/ReadVariableOp-^omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp,^omilqdycns/ddlymsxapn/MatMul/ReadVariableOp.^omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp%^omilqdycns/ddlymsxapn/ReadVariableOp'^omilqdycns/ddlymsxapn/ReadVariableOp_1'^omilqdycns/ddlymsxapn/ReadVariableOp_2^omilqdycns/while-^vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp,^vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp.^vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp%^vlxoswgdqw/vdaevhnmja/ReadVariableOp'^vlxoswgdqw/vdaevhnmja/ReadVariableOp_1'^vlxoswgdqw/vdaevhnmja/ReadVariableOp_2^vlxoswgdqw/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp2l
4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!iktogmlrmp/BiasAdd/ReadVariableOp!iktogmlrmp/BiasAdd/ReadVariableOp2D
 iktogmlrmp/MatMul/ReadVariableOp iktogmlrmp/MatMul/ReadVariableOp2\
,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp2Z
+omilqdycns/ddlymsxapn/MatMul/ReadVariableOp+omilqdycns/ddlymsxapn/MatMul/ReadVariableOp2^
-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp2L
$omilqdycns/ddlymsxapn/ReadVariableOp$omilqdycns/ddlymsxapn/ReadVariableOp2P
&omilqdycns/ddlymsxapn/ReadVariableOp_1&omilqdycns/ddlymsxapn/ReadVariableOp_12P
&omilqdycns/ddlymsxapn/ReadVariableOp_2&omilqdycns/ddlymsxapn/ReadVariableOp_22$
omilqdycns/whileomilqdycns/while2\
,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp2Z
+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp2^
-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp2L
$vlxoswgdqw/vdaevhnmja/ReadVariableOp$vlxoswgdqw/vdaevhnmja/ReadVariableOp2P
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1&vlxoswgdqw/vdaevhnmja/ReadVariableOp_12P
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2&vlxoswgdqw/vdaevhnmja/ReadVariableOp_22$
vlxoswgdqw/whilevlxoswgdqw/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_224230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_224230___redundant_placeholder04
0while_while_cond_224230___redundant_placeholder14
0while_while_cond_224230___redundant_placeholder24
0while_while_cond_224230___redundant_placeholder34
0while_while_cond_224230___redundant_placeholder44
0while_while_cond_224230___redundant_placeholder54
0while_while_cond_224230___redundant_placeholder6
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
F
ã
F__inference_omilqdycns_layer_call_and_return_conditional_losses_220573

inputs$
ddlymsxapn_220474:	$
ddlymsxapn_220476:	  
ddlymsxapn_220478:	
ddlymsxapn_220480: 
ddlymsxapn_220482: 
ddlymsxapn_220484: 
identity¢"ddlymsxapn/StatefulPartitionedCall¢whileD
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
"ddlymsxapn/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0ddlymsxapn_220474ddlymsxapn_220476ddlymsxapn_220478ddlymsxapn_220480ddlymsxapn_220482ddlymsxapn_220484*
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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_2204732$
"ddlymsxapn/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0ddlymsxapn_220474ddlymsxapn_220476ddlymsxapn_220478ddlymsxapn_220480ddlymsxapn_220482ddlymsxapn_220484*
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
while_body_220493*
condR
while_cond_220492*Q
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
IdentityIdentitytranspose_1:y:0#^ddlymsxapn/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"ddlymsxapn/StatefulPartitionedCall"ddlymsxapn/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
p
É
vlxoswgdqw_while_body_2235092
.vlxoswgdqw_while_vlxoswgdqw_while_loop_counter8
4vlxoswgdqw_while_vlxoswgdqw_while_maximum_iterations 
vlxoswgdqw_while_placeholder"
vlxoswgdqw_while_placeholder_1"
vlxoswgdqw_while_placeholder_2"
vlxoswgdqw_while_placeholder_31
-vlxoswgdqw_while_vlxoswgdqw_strided_slice_1_0m
ivlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0O
<vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0:	 Q
>vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 L
=vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0:	C
5vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0: E
7vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0: E
7vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0: 
vlxoswgdqw_while_identity
vlxoswgdqw_while_identity_1
vlxoswgdqw_while_identity_2
vlxoswgdqw_while_identity_3
vlxoswgdqw_while_identity_4
vlxoswgdqw_while_identity_5/
+vlxoswgdqw_while_vlxoswgdqw_strided_slice_1k
gvlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensorM
:vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource:	 O
<vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource:	 J
;vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource:	A
3vlxoswgdqw_while_vdaevhnmja_readvariableop_resource: C
5vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource: C
5vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource: ¢2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp¢1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp¢3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp¢*vlxoswgdqw/while/vdaevhnmja/ReadVariableOp¢,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1¢,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2Ù
Bvlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bvlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shape
4vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemivlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0vlxoswgdqw_while_placeholderKvlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItemä
1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp<vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOpý
"vlxoswgdqw/while/vdaevhnmja/MatMulMatMul;vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem:item:09vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"vlxoswgdqw/while/vdaevhnmja/MatMulê
3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp>vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOpæ
$vlxoswgdqw/while/vdaevhnmja/MatMul_1MatMulvlxoswgdqw_while_placeholder_2;vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$vlxoswgdqw/while/vdaevhnmja/MatMul_1Ü
vlxoswgdqw/while/vdaevhnmja/addAddV2,vlxoswgdqw/while/vdaevhnmja/MatMul:product:0.vlxoswgdqw/while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
vlxoswgdqw/while/vdaevhnmja/addã
2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp=vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOpé
#vlxoswgdqw/while/vdaevhnmja/BiasAddBiasAdd#vlxoswgdqw/while/vdaevhnmja/add:z:0:vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#vlxoswgdqw/while/vdaevhnmja/BiasAdd
+vlxoswgdqw/while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+vlxoswgdqw/while/vdaevhnmja/split/split_dim¯
!vlxoswgdqw/while/vdaevhnmja/splitSplit4vlxoswgdqw/while/vdaevhnmja/split/split_dim:output:0,vlxoswgdqw/while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!vlxoswgdqw/while/vdaevhnmja/splitÊ
*vlxoswgdqw/while/vdaevhnmja/ReadVariableOpReadVariableOp5vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*vlxoswgdqw/while/vdaevhnmja/ReadVariableOpÏ
vlxoswgdqw/while/vdaevhnmja/mulMul2vlxoswgdqw/while/vdaevhnmja/ReadVariableOp:value:0vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vlxoswgdqw/while/vdaevhnmja/mulÒ
!vlxoswgdqw/while/vdaevhnmja/add_1AddV2*vlxoswgdqw/while/vdaevhnmja/split:output:0#vlxoswgdqw/while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_1®
#vlxoswgdqw/while/vdaevhnmja/SigmoidSigmoid%vlxoswgdqw/while/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#vlxoswgdqw/while/vdaevhnmja/SigmoidÐ
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1ReadVariableOp7vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1Õ
!vlxoswgdqw/while/vdaevhnmja/mul_1Mul4vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1:value:0vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_1Ô
!vlxoswgdqw/while/vdaevhnmja/add_2AddV2*vlxoswgdqw/while/vdaevhnmja/split:output:1%vlxoswgdqw/while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_2²
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_1Sigmoid%vlxoswgdqw/while/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_1Ê
!vlxoswgdqw/while/vdaevhnmja/mul_2Mul)vlxoswgdqw/while/vdaevhnmja/Sigmoid_1:y:0vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_2ª
 vlxoswgdqw/while/vdaevhnmja/TanhTanh*vlxoswgdqw/while/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 vlxoswgdqw/while/vdaevhnmja/TanhÎ
!vlxoswgdqw/while/vdaevhnmja/mul_3Mul'vlxoswgdqw/while/vdaevhnmja/Sigmoid:y:0$vlxoswgdqw/while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_3Ï
!vlxoswgdqw/while/vdaevhnmja/add_3AddV2%vlxoswgdqw/while/vdaevhnmja/mul_2:z:0%vlxoswgdqw/while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_3Ð
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2ReadVariableOp7vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2Ü
!vlxoswgdqw/while/vdaevhnmja/mul_4Mul4vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2:value:0%vlxoswgdqw/while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_4Ô
!vlxoswgdqw/while/vdaevhnmja/add_4AddV2*vlxoswgdqw/while/vdaevhnmja/split:output:3%vlxoswgdqw/while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/add_4²
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_2Sigmoid%vlxoswgdqw/while/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vlxoswgdqw/while/vdaevhnmja/Sigmoid_2©
"vlxoswgdqw/while/vdaevhnmja/Tanh_1Tanh%vlxoswgdqw/while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"vlxoswgdqw/while/vdaevhnmja/Tanh_1Ò
!vlxoswgdqw/while/vdaevhnmja/mul_5Mul)vlxoswgdqw/while/vdaevhnmja/Sigmoid_2:y:0&vlxoswgdqw/while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vlxoswgdqw/while/vdaevhnmja/mul_5
5vlxoswgdqw/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemvlxoswgdqw_while_placeholder_1vlxoswgdqw_while_placeholder%vlxoswgdqw/while/vdaevhnmja/mul_5:z:0*
_output_shapes
: *
element_dtype027
5vlxoswgdqw/while/TensorArrayV2Write/TensorListSetItemr
vlxoswgdqw/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
vlxoswgdqw/while/add/y
vlxoswgdqw/while/addAddV2vlxoswgdqw_while_placeholdervlxoswgdqw/while/add/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/while/addv
vlxoswgdqw/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
vlxoswgdqw/while/add_1/y­
vlxoswgdqw/while/add_1AddV2.vlxoswgdqw_while_vlxoswgdqw_while_loop_counter!vlxoswgdqw/while/add_1/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/while/add_1©
vlxoswgdqw/while/IdentityIdentityvlxoswgdqw/while/add_1:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/IdentityÇ
vlxoswgdqw/while/Identity_1Identity4vlxoswgdqw_while_vlxoswgdqw_while_maximum_iterations3^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Identity_1«
vlxoswgdqw/while/Identity_2Identityvlxoswgdqw/while/add:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Identity_2Ø
vlxoswgdqw/while/Identity_3IdentityEvlxoswgdqw/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
vlxoswgdqw/while/Identity_3É
vlxoswgdqw/while/Identity_4Identity%vlxoswgdqw/while/vdaevhnmja/mul_5:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/while/Identity_4É
vlxoswgdqw/while/Identity_5Identity%vlxoswgdqw/while/vdaevhnmja/add_3:z:03^vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2^vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp4^vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp+^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1-^vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/while/Identity_5"?
vlxoswgdqw_while_identity"vlxoswgdqw/while/Identity:output:0"C
vlxoswgdqw_while_identity_1$vlxoswgdqw/while/Identity_1:output:0"C
vlxoswgdqw_while_identity_2$vlxoswgdqw/while/Identity_2:output:0"C
vlxoswgdqw_while_identity_3$vlxoswgdqw/while/Identity_3:output:0"C
vlxoswgdqw_while_identity_4$vlxoswgdqw/while/Identity_4:output:0"C
vlxoswgdqw_while_identity_5$vlxoswgdqw/while/Identity_5:output:0"Ô
gvlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensorivlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0"|
;vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource=vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0"~
<vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource>vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0"z
:vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource<vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0"p
5vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource7vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0"p
5vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource7vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0"l
3vlxoswgdqw_while_vdaevhnmja_readvariableop_resource5vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0"\
+vlxoswgdqw_while_vlxoswgdqw_strided_slice_1-vlxoswgdqw_while_vlxoswgdqw_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2f
1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp1vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp2j
3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp3vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp2X
*vlxoswgdqw/while/vdaevhnmja/ReadVariableOp*vlxoswgdqw/while/vdaevhnmja/ReadVariableOp2\
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_12\
,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2,vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2: 
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
while_body_224771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ddlymsxapn_matmul_readvariableop_resource_0:	F
3while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 A
2while_ddlymsxapn_biasadd_readvariableop_resource_0:	8
*while_ddlymsxapn_readvariableop_resource_0: :
,while_ddlymsxapn_readvariableop_1_resource_0: :
,while_ddlymsxapn_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ddlymsxapn_matmul_readvariableop_resource:	D
1while_ddlymsxapn_matmul_1_readvariableop_resource:	 ?
0while_ddlymsxapn_biasadd_readvariableop_resource:	6
(while_ddlymsxapn_readvariableop_resource: 8
*while_ddlymsxapn_readvariableop_1_resource: 8
*while_ddlymsxapn_readvariableop_2_resource: ¢'while/ddlymsxapn/BiasAdd/ReadVariableOp¢&while/ddlymsxapn/MatMul/ReadVariableOp¢(while/ddlymsxapn/MatMul_1/ReadVariableOp¢while/ddlymsxapn/ReadVariableOp¢!while/ddlymsxapn/ReadVariableOp_1¢!while/ddlymsxapn/ReadVariableOp_2Ã
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
&while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp1while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ddlymsxapn/MatMul/ReadVariableOpÑ
while/ddlymsxapn/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMulÉ
(while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp3while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ddlymsxapn/MatMul_1/ReadVariableOpº
while/ddlymsxapn/MatMul_1MatMulwhile_placeholder_20while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMul_1°
while/ddlymsxapn/addAddV2!while/ddlymsxapn/MatMul:product:0#while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/addÂ
'while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp2while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ddlymsxapn/BiasAdd/ReadVariableOp½
while/ddlymsxapn/BiasAddBiasAddwhile/ddlymsxapn/add:z:0/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/BiasAdd
 while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ddlymsxapn/split/split_dim
while/ddlymsxapn/splitSplit)while/ddlymsxapn/split/split_dim:output:0!while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ddlymsxapn/split©
while/ddlymsxapn/ReadVariableOpReadVariableOp*while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ddlymsxapn/ReadVariableOp£
while/ddlymsxapn/mulMul'while/ddlymsxapn/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul¦
while/ddlymsxapn/add_1AddV2while/ddlymsxapn/split:output:0while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_1
while/ddlymsxapn/SigmoidSigmoidwhile/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid¯
!while/ddlymsxapn/ReadVariableOp_1ReadVariableOp,while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_1©
while/ddlymsxapn/mul_1Mul)while/ddlymsxapn/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_1¨
while/ddlymsxapn/add_2AddV2while/ddlymsxapn/split:output:1while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_2
while/ddlymsxapn/Sigmoid_1Sigmoidwhile/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_1
while/ddlymsxapn/mul_2Mulwhile/ddlymsxapn/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_2
while/ddlymsxapn/TanhTanhwhile/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh¢
while/ddlymsxapn/mul_3Mulwhile/ddlymsxapn/Sigmoid:y:0while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_3£
while/ddlymsxapn/add_3AddV2while/ddlymsxapn/mul_2:z:0while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_3¯
!while/ddlymsxapn/ReadVariableOp_2ReadVariableOp,while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_2°
while/ddlymsxapn/mul_4Mul)while/ddlymsxapn/ReadVariableOp_2:value:0while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_4¨
while/ddlymsxapn/add_4AddV2while/ddlymsxapn/split:output:3while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_4
while/ddlymsxapn/Sigmoid_2Sigmoidwhile/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_2
while/ddlymsxapn/Tanh_1Tanhwhile/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh_1¦
while/ddlymsxapn/mul_5Mulwhile/ddlymsxapn/Sigmoid_2:y:0while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ddlymsxapn/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ddlymsxapn/mul_5:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ddlymsxapn/add_3:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ddlymsxapn_biasadd_readvariableop_resource2while_ddlymsxapn_biasadd_readvariableop_resource_0"h
1while_ddlymsxapn_matmul_1_readvariableop_resource3while_ddlymsxapn_matmul_1_readvariableop_resource_0"d
/while_ddlymsxapn_matmul_readvariableop_resource1while_ddlymsxapn_matmul_readvariableop_resource_0"Z
*while_ddlymsxapn_readvariableop_1_resource,while_ddlymsxapn_readvariableop_1_resource_0"Z
*while_ddlymsxapn_readvariableop_2_resource,while_ddlymsxapn_readvariableop_2_resource_0"V
(while_ddlymsxapn_readvariableop_resource*while_ddlymsxapn_readvariableop_resource_0")
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
'while/ddlymsxapn/BiasAdd/ReadVariableOp'while/ddlymsxapn/BiasAdd/ReadVariableOp2P
&while/ddlymsxapn/MatMul/ReadVariableOp&while/ddlymsxapn/MatMul/ReadVariableOp2T
(while/ddlymsxapn/MatMul_1/ReadVariableOp(while/ddlymsxapn/MatMul_1/ReadVariableOp2B
while/ddlymsxapn/ReadVariableOpwhile/ddlymsxapn/ReadVariableOp2F
!while/ddlymsxapn/ReadVariableOp_1!while/ddlymsxapn/ReadVariableOp_12F
!while/ddlymsxapn/ReadVariableOp_2!while/ddlymsxapn/ReadVariableOp_2: 
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
ÙÊ

F__inference_sequential_layer_call_and_return_conditional_losses_223616

inputsL
6iigfihrkup_conv1d_expanddims_1_readvariableop_resource:K
=iigfihrkup_squeeze_batch_dims_biasadd_readvariableop_resource:G
4omilqdycns_ddlymsxapn_matmul_readvariableop_resource:	I
6omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource:	 D
5omilqdycns_ddlymsxapn_biasadd_readvariableop_resource:	;
-omilqdycns_ddlymsxapn_readvariableop_resource: =
/omilqdycns_ddlymsxapn_readvariableop_1_resource: =
/omilqdycns_ddlymsxapn_readvariableop_2_resource: G
4vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource:	 I
6vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource:	 D
5vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource:	;
-vlxoswgdqw_vdaevhnmja_readvariableop_resource: =
/vlxoswgdqw_vdaevhnmja_readvariableop_1_resource: =
/vlxoswgdqw_vdaevhnmja_readvariableop_2_resource: ;
)iktogmlrmp_matmul_readvariableop_resource: 8
*iktogmlrmp_biasadd_readvariableop_resource:
identity¢-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp¢4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!iktogmlrmp/BiasAdd/ReadVariableOp¢ iktogmlrmp/MatMul/ReadVariableOp¢,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp¢+omilqdycns/ddlymsxapn/MatMul/ReadVariableOp¢-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp¢$omilqdycns/ddlymsxapn/ReadVariableOp¢&omilqdycns/ddlymsxapn/ReadVariableOp_1¢&omilqdycns/ddlymsxapn/ReadVariableOp_2¢omilqdycns/while¢,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp¢+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp¢-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp¢$vlxoswgdqw/vdaevhnmja/ReadVariableOp¢&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1¢&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2¢vlxoswgdqw/while
 iigfihrkup/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 iigfihrkup/conv1d/ExpandDims/dim»
iigfihrkup/conv1d/ExpandDims
ExpandDimsinputs)iigfihrkup/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/ExpandDimsÙ
-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6iigfihrkup_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp
"iigfihrkup/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"iigfihrkup/conv1d/ExpandDims_1/dimã
iigfihrkup/conv1d/ExpandDims_1
ExpandDims5iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp:value:0+iigfihrkup/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
iigfihrkup/conv1d/ExpandDims_1
iigfihrkup/conv1d/ShapeShape%iigfihrkup/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
iigfihrkup/conv1d/Shape
%iigfihrkup/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%iigfihrkup/conv1d/strided_slice/stack¥
'iigfihrkup/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'iigfihrkup/conv1d/strided_slice/stack_1
'iigfihrkup/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'iigfihrkup/conv1d/strided_slice/stack_2Ì
iigfihrkup/conv1d/strided_sliceStridedSlice iigfihrkup/conv1d/Shape:output:0.iigfihrkup/conv1d/strided_slice/stack:output:00iigfihrkup/conv1d/strided_slice/stack_1:output:00iigfihrkup/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
iigfihrkup/conv1d/strided_slice
iigfihrkup/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
iigfihrkup/conv1d/Reshape/shapeÌ
iigfihrkup/conv1d/ReshapeReshape%iigfihrkup/conv1d/ExpandDims:output:0(iigfihrkup/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/Reshapeî
iigfihrkup/conv1d/Conv2DConv2D"iigfihrkup/conv1d/Reshape:output:0'iigfihrkup/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
iigfihrkup/conv1d/Conv2D
!iigfihrkup/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!iigfihrkup/conv1d/concat/values_1
iigfihrkup/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/concat/axisì
iigfihrkup/conv1d/concatConcatV2(iigfihrkup/conv1d/strided_slice:output:0*iigfihrkup/conv1d/concat/values_1:output:0&iigfihrkup/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
iigfihrkup/conv1d/concatÉ
iigfihrkup/conv1d/Reshape_1Reshape!iigfihrkup/conv1d/Conv2D:output:0!iigfihrkup/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/Reshape_1Á
iigfihrkup/conv1d/SqueezeSqueeze$iigfihrkup/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
iigfihrkup/conv1d/Squeeze
#iigfihrkup/squeeze_batch_dims/ShapeShape"iigfihrkup/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#iigfihrkup/squeeze_batch_dims/Shape°
1iigfihrkup/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1iigfihrkup/squeeze_batch_dims/strided_slice/stack½
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_1´
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3iigfihrkup/squeeze_batch_dims/strided_slice/stack_2
+iigfihrkup/squeeze_batch_dims/strided_sliceStridedSlice,iigfihrkup/squeeze_batch_dims/Shape:output:0:iigfihrkup/squeeze_batch_dims/strided_slice/stack:output:0<iigfihrkup/squeeze_batch_dims/strided_slice/stack_1:output:0<iigfihrkup/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+iigfihrkup/squeeze_batch_dims/strided_slice¯
+iigfihrkup/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+iigfihrkup/squeeze_batch_dims/Reshape/shapeé
%iigfihrkup/squeeze_batch_dims/ReshapeReshape"iigfihrkup/conv1d/Squeeze:output:04iigfihrkup/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%iigfihrkup/squeeze_batch_dims/Reshapeæ
4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=iigfihrkup_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%iigfihrkup/squeeze_batch_dims/BiasAddBiasAdd.iigfihrkup/squeeze_batch_dims/Reshape:output:0<iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%iigfihrkup/squeeze_batch_dims/BiasAdd¯
-iigfihrkup/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-iigfihrkup/squeeze_batch_dims/concat/values_1¡
)iigfihrkup/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)iigfihrkup/squeeze_batch_dims/concat/axis¨
$iigfihrkup/squeeze_batch_dims/concatConcatV24iigfihrkup/squeeze_batch_dims/strided_slice:output:06iigfihrkup/squeeze_batch_dims/concat/values_1:output:02iigfihrkup/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$iigfihrkup/squeeze_batch_dims/concatö
'iigfihrkup/squeeze_batch_dims/Reshape_1Reshape.iigfihrkup/squeeze_batch_dims/BiasAdd:output:0-iigfihrkup/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'iigfihrkup/squeeze_batch_dims/Reshape_1
xfbsciqeco/ShapeShape0iigfihrkup/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
xfbsciqeco/Shape
xfbsciqeco/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xfbsciqeco/strided_slice/stack
 xfbsciqeco/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xfbsciqeco/strided_slice/stack_1
 xfbsciqeco/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xfbsciqeco/strided_slice/stack_2¤
xfbsciqeco/strided_sliceStridedSlicexfbsciqeco/Shape:output:0'xfbsciqeco/strided_slice/stack:output:0)xfbsciqeco/strided_slice/stack_1:output:0)xfbsciqeco/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xfbsciqeco/strided_slicez
xfbsciqeco/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
xfbsciqeco/Reshape/shape/1z
xfbsciqeco/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
xfbsciqeco/Reshape/shape/2×
xfbsciqeco/Reshape/shapePack!xfbsciqeco/strided_slice:output:0#xfbsciqeco/Reshape/shape/1:output:0#xfbsciqeco/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
xfbsciqeco/Reshape/shape¾
xfbsciqeco/ReshapeReshape0iigfihrkup/squeeze_batch_dims/Reshape_1:output:0!xfbsciqeco/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xfbsciqeco/Reshapeo
omilqdycns/ShapeShapexfbsciqeco/Reshape:output:0*
T0*
_output_shapes
:2
omilqdycns/Shape
omilqdycns/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
omilqdycns/strided_slice/stack
 omilqdycns/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 omilqdycns/strided_slice/stack_1
 omilqdycns/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 omilqdycns/strided_slice/stack_2¤
omilqdycns/strided_sliceStridedSliceomilqdycns/Shape:output:0'omilqdycns/strided_slice/stack:output:0)omilqdycns/strided_slice/stack_1:output:0)omilqdycns/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
omilqdycns/strided_slicer
omilqdycns/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros/mul/y
omilqdycns/zeros/mulMul!omilqdycns/strided_slice:output:0omilqdycns/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros/mulu
omilqdycns/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
omilqdycns/zeros/Less/y
omilqdycns/zeros/LessLessomilqdycns/zeros/mul:z:0 omilqdycns/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros/Lessx
omilqdycns/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros/packed/1¯
omilqdycns/zeros/packedPack!omilqdycns/strided_slice:output:0"omilqdycns/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
omilqdycns/zeros/packedu
omilqdycns/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
omilqdycns/zeros/Const¡
omilqdycns/zerosFill omilqdycns/zeros/packed:output:0omilqdycns/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/zerosv
omilqdycns/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros_1/mul/y
omilqdycns/zeros_1/mulMul!omilqdycns/strided_slice:output:0!omilqdycns/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros_1/muly
omilqdycns/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
omilqdycns/zeros_1/Less/y
omilqdycns/zeros_1/LessLessomilqdycns/zeros_1/mul:z:0"omilqdycns/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
omilqdycns/zeros_1/Less|
omilqdycns/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/zeros_1/packed/1µ
omilqdycns/zeros_1/packedPack!omilqdycns/strided_slice:output:0$omilqdycns/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
omilqdycns/zeros_1/packedy
omilqdycns/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
omilqdycns/zeros_1/Const©
omilqdycns/zeros_1Fill"omilqdycns/zeros_1/packed:output:0!omilqdycns/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/zeros_1
omilqdycns/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
omilqdycns/transpose/perm°
omilqdycns/transpose	Transposexfbsciqeco/Reshape:output:0"omilqdycns/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/transposep
omilqdycns/Shape_1Shapeomilqdycns/transpose:y:0*
T0*
_output_shapes
:2
omilqdycns/Shape_1
 omilqdycns/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 omilqdycns/strided_slice_1/stack
"omilqdycns/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_1/stack_1
"omilqdycns/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_1/stack_2°
omilqdycns/strided_slice_1StridedSliceomilqdycns/Shape_1:output:0)omilqdycns/strided_slice_1/stack:output:0+omilqdycns/strided_slice_1/stack_1:output:0+omilqdycns/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
omilqdycns/strided_slice_1
&omilqdycns/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&omilqdycns/TensorArrayV2/element_shapeÞ
omilqdycns/TensorArrayV2TensorListReserve/omilqdycns/TensorArrayV2/element_shape:output:0#omilqdycns/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
omilqdycns/TensorArrayV2Õ
@omilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@omilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2omilqdycns/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoromilqdycns/transpose:y:0Iomilqdycns/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2omilqdycns/TensorArrayUnstack/TensorListFromTensor
 omilqdycns/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 omilqdycns/strided_slice_2/stack
"omilqdycns/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_2/stack_1
"omilqdycns/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_2/stack_2¾
omilqdycns/strided_slice_2StridedSliceomilqdycns/transpose:y:0)omilqdycns/strided_slice_2/stack:output:0+omilqdycns/strided_slice_2/stack_1:output:0+omilqdycns/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
omilqdycns/strided_slice_2Ð
+omilqdycns/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp4omilqdycns_ddlymsxapn_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+omilqdycns/ddlymsxapn/MatMul/ReadVariableOpÓ
omilqdycns/ddlymsxapn/MatMulMatMul#omilqdycns/strided_slice_2:output:03omilqdycns/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/ddlymsxapn/MatMulÖ
-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp6omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOpÏ
omilqdycns/ddlymsxapn/MatMul_1MatMulomilqdycns/zeros:output:05omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
omilqdycns/ddlymsxapn/MatMul_1Ä
omilqdycns/ddlymsxapn/addAddV2&omilqdycns/ddlymsxapn/MatMul:product:0(omilqdycns/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/ddlymsxapn/addÏ
,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp5omilqdycns_ddlymsxapn_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOpÑ
omilqdycns/ddlymsxapn/BiasAddBiasAddomilqdycns/ddlymsxapn/add:z:04omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
omilqdycns/ddlymsxapn/BiasAdd
%omilqdycns/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%omilqdycns/ddlymsxapn/split/split_dim
omilqdycns/ddlymsxapn/splitSplit.omilqdycns/ddlymsxapn/split/split_dim:output:0&omilqdycns/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
omilqdycns/ddlymsxapn/split¶
$omilqdycns/ddlymsxapn/ReadVariableOpReadVariableOp-omilqdycns_ddlymsxapn_readvariableop_resource*
_output_shapes
: *
dtype02&
$omilqdycns/ddlymsxapn/ReadVariableOpº
omilqdycns/ddlymsxapn/mulMul,omilqdycns/ddlymsxapn/ReadVariableOp:value:0omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mulº
omilqdycns/ddlymsxapn/add_1AddV2$omilqdycns/ddlymsxapn/split:output:0omilqdycns/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_1
omilqdycns/ddlymsxapn/SigmoidSigmoidomilqdycns/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/Sigmoid¼
&omilqdycns/ddlymsxapn/ReadVariableOp_1ReadVariableOp/omilqdycns_ddlymsxapn_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&omilqdycns/ddlymsxapn/ReadVariableOp_1À
omilqdycns/ddlymsxapn/mul_1Mul.omilqdycns/ddlymsxapn/ReadVariableOp_1:value:0omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_1¼
omilqdycns/ddlymsxapn/add_2AddV2$omilqdycns/ddlymsxapn/split:output:1omilqdycns/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_2 
omilqdycns/ddlymsxapn/Sigmoid_1Sigmoidomilqdycns/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
omilqdycns/ddlymsxapn/Sigmoid_1µ
omilqdycns/ddlymsxapn/mul_2Mul#omilqdycns/ddlymsxapn/Sigmoid_1:y:0omilqdycns/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_2
omilqdycns/ddlymsxapn/TanhTanh$omilqdycns/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/Tanh¶
omilqdycns/ddlymsxapn/mul_3Mul!omilqdycns/ddlymsxapn/Sigmoid:y:0omilqdycns/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_3·
omilqdycns/ddlymsxapn/add_3AddV2omilqdycns/ddlymsxapn/mul_2:z:0omilqdycns/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_3¼
&omilqdycns/ddlymsxapn/ReadVariableOp_2ReadVariableOp/omilqdycns_ddlymsxapn_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&omilqdycns/ddlymsxapn/ReadVariableOp_2Ä
omilqdycns/ddlymsxapn/mul_4Mul.omilqdycns/ddlymsxapn/ReadVariableOp_2:value:0omilqdycns/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_4¼
omilqdycns/ddlymsxapn/add_4AddV2$omilqdycns/ddlymsxapn/split:output:3omilqdycns/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/add_4 
omilqdycns/ddlymsxapn/Sigmoid_2Sigmoidomilqdycns/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
omilqdycns/ddlymsxapn/Sigmoid_2
omilqdycns/ddlymsxapn/Tanh_1Tanhomilqdycns/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/Tanh_1º
omilqdycns/ddlymsxapn/mul_5Mul#omilqdycns/ddlymsxapn/Sigmoid_2:y:0 omilqdycns/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/ddlymsxapn/mul_5¥
(omilqdycns/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(omilqdycns/TensorArrayV2_1/element_shapeä
omilqdycns/TensorArrayV2_1TensorListReserve1omilqdycns/TensorArrayV2_1/element_shape:output:0#omilqdycns/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
omilqdycns/TensorArrayV2_1d
omilqdycns/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/time
#omilqdycns/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#omilqdycns/while/maximum_iterations
omilqdycns/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
omilqdycns/while/loop_counter°
omilqdycns/whileWhile&omilqdycns/while/loop_counter:output:0,omilqdycns/while/maximum_iterations:output:0omilqdycns/time:output:0#omilqdycns/TensorArrayV2_1:handle:0omilqdycns/zeros:output:0omilqdycns/zeros_1:output:0#omilqdycns/strided_slice_1:output:0Bomilqdycns/TensorArrayUnstack/TensorListFromTensor:output_handle:04omilqdycns_ddlymsxapn_matmul_readvariableop_resource6omilqdycns_ddlymsxapn_matmul_1_readvariableop_resource5omilqdycns_ddlymsxapn_biasadd_readvariableop_resource-omilqdycns_ddlymsxapn_readvariableop_resource/omilqdycns_ddlymsxapn_readvariableop_1_resource/omilqdycns_ddlymsxapn_readvariableop_2_resource*
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
omilqdycns_while_body_223333*(
cond R
omilqdycns_while_cond_223332*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
omilqdycns/whileË
;omilqdycns/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;omilqdycns/TensorArrayV2Stack/TensorListStack/element_shape
-omilqdycns/TensorArrayV2Stack/TensorListStackTensorListStackomilqdycns/while:output:3Domilqdycns/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-omilqdycns/TensorArrayV2Stack/TensorListStack
 omilqdycns/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 omilqdycns/strided_slice_3/stack
"omilqdycns/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"omilqdycns/strided_slice_3/stack_1
"omilqdycns/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"omilqdycns/strided_slice_3/stack_2Ü
omilqdycns/strided_slice_3StridedSlice6omilqdycns/TensorArrayV2Stack/TensorListStack:tensor:0)omilqdycns/strided_slice_3/stack:output:0+omilqdycns/strided_slice_3/stack_1:output:0+omilqdycns/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
omilqdycns/strided_slice_3
omilqdycns/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
omilqdycns/transpose_1/permÑ
omilqdycns/transpose_1	Transpose6omilqdycns/TensorArrayV2Stack/TensorListStack:tensor:0$omilqdycns/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
omilqdycns/transpose_1n
vlxoswgdqw/ShapeShapeomilqdycns/transpose_1:y:0*
T0*
_output_shapes
:2
vlxoswgdqw/Shape
vlxoswgdqw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
vlxoswgdqw/strided_slice/stack
 vlxoswgdqw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 vlxoswgdqw/strided_slice/stack_1
 vlxoswgdqw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 vlxoswgdqw/strided_slice/stack_2¤
vlxoswgdqw/strided_sliceStridedSlicevlxoswgdqw/Shape:output:0'vlxoswgdqw/strided_slice/stack:output:0)vlxoswgdqw/strided_slice/stack_1:output:0)vlxoswgdqw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vlxoswgdqw/strided_slicer
vlxoswgdqw/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros/mul/y
vlxoswgdqw/zeros/mulMul!vlxoswgdqw/strided_slice:output:0vlxoswgdqw/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros/mulu
vlxoswgdqw/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vlxoswgdqw/zeros/Less/y
vlxoswgdqw/zeros/LessLessvlxoswgdqw/zeros/mul:z:0 vlxoswgdqw/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros/Lessx
vlxoswgdqw/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros/packed/1¯
vlxoswgdqw/zeros/packedPack!vlxoswgdqw/strided_slice:output:0"vlxoswgdqw/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
vlxoswgdqw/zeros/packedu
vlxoswgdqw/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vlxoswgdqw/zeros/Const¡
vlxoswgdqw/zerosFill vlxoswgdqw/zeros/packed:output:0vlxoswgdqw/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/zerosv
vlxoswgdqw/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros_1/mul/y
vlxoswgdqw/zeros_1/mulMul!vlxoswgdqw/strided_slice:output:0!vlxoswgdqw/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros_1/muly
vlxoswgdqw/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vlxoswgdqw/zeros_1/Less/y
vlxoswgdqw/zeros_1/LessLessvlxoswgdqw/zeros_1/mul:z:0"vlxoswgdqw/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
vlxoswgdqw/zeros_1/Less|
vlxoswgdqw/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/zeros_1/packed/1µ
vlxoswgdqw/zeros_1/packedPack!vlxoswgdqw/strided_slice:output:0$vlxoswgdqw/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
vlxoswgdqw/zeros_1/packedy
vlxoswgdqw/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vlxoswgdqw/zeros_1/Const©
vlxoswgdqw/zeros_1Fill"vlxoswgdqw/zeros_1/packed:output:0!vlxoswgdqw/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/zeros_1
vlxoswgdqw/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vlxoswgdqw/transpose/perm¯
vlxoswgdqw/transpose	Transposeomilqdycns/transpose_1:y:0"vlxoswgdqw/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/transposep
vlxoswgdqw/Shape_1Shapevlxoswgdqw/transpose:y:0*
T0*
_output_shapes
:2
vlxoswgdqw/Shape_1
 vlxoswgdqw/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vlxoswgdqw/strided_slice_1/stack
"vlxoswgdqw/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_1/stack_1
"vlxoswgdqw/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_1/stack_2°
vlxoswgdqw/strided_slice_1StridedSlicevlxoswgdqw/Shape_1:output:0)vlxoswgdqw/strided_slice_1/stack:output:0+vlxoswgdqw/strided_slice_1/stack_1:output:0+vlxoswgdqw/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vlxoswgdqw/strided_slice_1
&vlxoswgdqw/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&vlxoswgdqw/TensorArrayV2/element_shapeÞ
vlxoswgdqw/TensorArrayV2TensorListReserve/vlxoswgdqw/TensorArrayV2/element_shape:output:0#vlxoswgdqw/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vlxoswgdqw/TensorArrayV2Õ
@vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2vlxoswgdqw/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorvlxoswgdqw/transpose:y:0Ivlxoswgdqw/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2vlxoswgdqw/TensorArrayUnstack/TensorListFromTensor
 vlxoswgdqw/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vlxoswgdqw/strided_slice_2/stack
"vlxoswgdqw/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_2/stack_1
"vlxoswgdqw/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_2/stack_2¾
vlxoswgdqw/strided_slice_2StridedSlicevlxoswgdqw/transpose:y:0)vlxoswgdqw/strided_slice_2/stack:output:0+vlxoswgdqw/strided_slice_2/stack_1:output:0+vlxoswgdqw/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
vlxoswgdqw/strided_slice_2Ð
+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp4vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOpÓ
vlxoswgdqw/vdaevhnmja/MatMulMatMul#vlxoswgdqw/strided_slice_2:output:03vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vlxoswgdqw/vdaevhnmja/MatMulÖ
-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp6vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOpÏ
vlxoswgdqw/vdaevhnmja/MatMul_1MatMulvlxoswgdqw/zeros:output:05vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
vlxoswgdqw/vdaevhnmja/MatMul_1Ä
vlxoswgdqw/vdaevhnmja/addAddV2&vlxoswgdqw/vdaevhnmja/MatMul:product:0(vlxoswgdqw/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vlxoswgdqw/vdaevhnmja/addÏ
,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp5vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOpÑ
vlxoswgdqw/vdaevhnmja/BiasAddBiasAddvlxoswgdqw/vdaevhnmja/add:z:04vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vlxoswgdqw/vdaevhnmja/BiasAdd
%vlxoswgdqw/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%vlxoswgdqw/vdaevhnmja/split/split_dim
vlxoswgdqw/vdaevhnmja/splitSplit.vlxoswgdqw/vdaevhnmja/split/split_dim:output:0&vlxoswgdqw/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vlxoswgdqw/vdaevhnmja/split¶
$vlxoswgdqw/vdaevhnmja/ReadVariableOpReadVariableOp-vlxoswgdqw_vdaevhnmja_readvariableop_resource*
_output_shapes
: *
dtype02&
$vlxoswgdqw/vdaevhnmja/ReadVariableOpº
vlxoswgdqw/vdaevhnmja/mulMul,vlxoswgdqw/vdaevhnmja/ReadVariableOp:value:0vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mulº
vlxoswgdqw/vdaevhnmja/add_1AddV2$vlxoswgdqw/vdaevhnmja/split:output:0vlxoswgdqw/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_1
vlxoswgdqw/vdaevhnmja/SigmoidSigmoidvlxoswgdqw/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/Sigmoid¼
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1ReadVariableOp/vlxoswgdqw_vdaevhnmja_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1À
vlxoswgdqw/vdaevhnmja/mul_1Mul.vlxoswgdqw/vdaevhnmja/ReadVariableOp_1:value:0vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_1¼
vlxoswgdqw/vdaevhnmja/add_2AddV2$vlxoswgdqw/vdaevhnmja/split:output:1vlxoswgdqw/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_2 
vlxoswgdqw/vdaevhnmja/Sigmoid_1Sigmoidvlxoswgdqw/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vlxoswgdqw/vdaevhnmja/Sigmoid_1µ
vlxoswgdqw/vdaevhnmja/mul_2Mul#vlxoswgdqw/vdaevhnmja/Sigmoid_1:y:0vlxoswgdqw/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_2
vlxoswgdqw/vdaevhnmja/TanhTanh$vlxoswgdqw/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/Tanh¶
vlxoswgdqw/vdaevhnmja/mul_3Mul!vlxoswgdqw/vdaevhnmja/Sigmoid:y:0vlxoswgdqw/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_3·
vlxoswgdqw/vdaevhnmja/add_3AddV2vlxoswgdqw/vdaevhnmja/mul_2:z:0vlxoswgdqw/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_3¼
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2ReadVariableOp/vlxoswgdqw_vdaevhnmja_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2Ä
vlxoswgdqw/vdaevhnmja/mul_4Mul.vlxoswgdqw/vdaevhnmja/ReadVariableOp_2:value:0vlxoswgdqw/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_4¼
vlxoswgdqw/vdaevhnmja/add_4AddV2$vlxoswgdqw/vdaevhnmja/split:output:3vlxoswgdqw/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/add_4 
vlxoswgdqw/vdaevhnmja/Sigmoid_2Sigmoidvlxoswgdqw/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vlxoswgdqw/vdaevhnmja/Sigmoid_2
vlxoswgdqw/vdaevhnmja/Tanh_1Tanhvlxoswgdqw/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/Tanh_1º
vlxoswgdqw/vdaevhnmja/mul_5Mul#vlxoswgdqw/vdaevhnmja/Sigmoid_2:y:0 vlxoswgdqw/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/vdaevhnmja/mul_5¥
(vlxoswgdqw/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(vlxoswgdqw/TensorArrayV2_1/element_shapeä
vlxoswgdqw/TensorArrayV2_1TensorListReserve1vlxoswgdqw/TensorArrayV2_1/element_shape:output:0#vlxoswgdqw/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vlxoswgdqw/TensorArrayV2_1d
vlxoswgdqw/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/time
#vlxoswgdqw/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#vlxoswgdqw/while/maximum_iterations
vlxoswgdqw/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
vlxoswgdqw/while/loop_counter°
vlxoswgdqw/whileWhile&vlxoswgdqw/while/loop_counter:output:0,vlxoswgdqw/while/maximum_iterations:output:0vlxoswgdqw/time:output:0#vlxoswgdqw/TensorArrayV2_1:handle:0vlxoswgdqw/zeros:output:0vlxoswgdqw/zeros_1:output:0#vlxoswgdqw/strided_slice_1:output:0Bvlxoswgdqw/TensorArrayUnstack/TensorListFromTensor:output_handle:04vlxoswgdqw_vdaevhnmja_matmul_readvariableop_resource6vlxoswgdqw_vdaevhnmja_matmul_1_readvariableop_resource5vlxoswgdqw_vdaevhnmja_biasadd_readvariableop_resource-vlxoswgdqw_vdaevhnmja_readvariableop_resource/vlxoswgdqw_vdaevhnmja_readvariableop_1_resource/vlxoswgdqw_vdaevhnmja_readvariableop_2_resource*
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
vlxoswgdqw_while_body_223509*(
cond R
vlxoswgdqw_while_cond_223508*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
vlxoswgdqw/whileË
;vlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;vlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shape
-vlxoswgdqw/TensorArrayV2Stack/TensorListStackTensorListStackvlxoswgdqw/while:output:3Dvlxoswgdqw/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-vlxoswgdqw/TensorArrayV2Stack/TensorListStack
 vlxoswgdqw/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 vlxoswgdqw/strided_slice_3/stack
"vlxoswgdqw/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"vlxoswgdqw/strided_slice_3/stack_1
"vlxoswgdqw/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vlxoswgdqw/strided_slice_3/stack_2Ü
vlxoswgdqw/strided_slice_3StridedSlice6vlxoswgdqw/TensorArrayV2Stack/TensorListStack:tensor:0)vlxoswgdqw/strided_slice_3/stack:output:0+vlxoswgdqw/strided_slice_3/stack_1:output:0+vlxoswgdqw/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
vlxoswgdqw/strided_slice_3
vlxoswgdqw/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vlxoswgdqw/transpose_1/permÑ
vlxoswgdqw/transpose_1	Transpose6vlxoswgdqw/TensorArrayV2Stack/TensorListStack:tensor:0$vlxoswgdqw/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vlxoswgdqw/transpose_1®
 iktogmlrmp/MatMul/ReadVariableOpReadVariableOp)iktogmlrmp_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 iktogmlrmp/MatMul/ReadVariableOp±
iktogmlrmp/MatMulMatMul#vlxoswgdqw/strided_slice_3:output:0(iktogmlrmp/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
iktogmlrmp/MatMul­
!iktogmlrmp/BiasAdd/ReadVariableOpReadVariableOp*iktogmlrmp_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!iktogmlrmp/BiasAdd/ReadVariableOp­
iktogmlrmp/BiasAddBiasAddiktogmlrmp/MatMul:product:0)iktogmlrmp/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
iktogmlrmp/BiasAddÏ
IdentityIdentityiktogmlrmp/BiasAdd:output:0.^iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp5^iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp"^iktogmlrmp/BiasAdd/ReadVariableOp!^iktogmlrmp/MatMul/ReadVariableOp-^omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp,^omilqdycns/ddlymsxapn/MatMul/ReadVariableOp.^omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp%^omilqdycns/ddlymsxapn/ReadVariableOp'^omilqdycns/ddlymsxapn/ReadVariableOp_1'^omilqdycns/ddlymsxapn/ReadVariableOp_2^omilqdycns/while-^vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp,^vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp.^vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp%^vlxoswgdqw/vdaevhnmja/ReadVariableOp'^vlxoswgdqw/vdaevhnmja/ReadVariableOp_1'^vlxoswgdqw/vdaevhnmja/ReadVariableOp_2^vlxoswgdqw/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp-iigfihrkup/conv1d/ExpandDims_1/ReadVariableOp2l
4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp4iigfihrkup/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!iktogmlrmp/BiasAdd/ReadVariableOp!iktogmlrmp/BiasAdd/ReadVariableOp2D
 iktogmlrmp/MatMul/ReadVariableOp iktogmlrmp/MatMul/ReadVariableOp2\
,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp,omilqdycns/ddlymsxapn/BiasAdd/ReadVariableOp2Z
+omilqdycns/ddlymsxapn/MatMul/ReadVariableOp+omilqdycns/ddlymsxapn/MatMul/ReadVariableOp2^
-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp-omilqdycns/ddlymsxapn/MatMul_1/ReadVariableOp2L
$omilqdycns/ddlymsxapn/ReadVariableOp$omilqdycns/ddlymsxapn/ReadVariableOp2P
&omilqdycns/ddlymsxapn/ReadVariableOp_1&omilqdycns/ddlymsxapn/ReadVariableOp_12P
&omilqdycns/ddlymsxapn/ReadVariableOp_2&omilqdycns/ddlymsxapn/ReadVariableOp_22$
omilqdycns/whileomilqdycns/while2\
,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp,vlxoswgdqw/vdaevhnmja/BiasAdd/ReadVariableOp2Z
+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp+vlxoswgdqw/vdaevhnmja/MatMul/ReadVariableOp2^
-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp-vlxoswgdqw/vdaevhnmja/MatMul_1/ReadVariableOp2L
$vlxoswgdqw/vdaevhnmja/ReadVariableOp$vlxoswgdqw/vdaevhnmja/ReadVariableOp2P
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_1&vlxoswgdqw/vdaevhnmja/ReadVariableOp_12P
&vlxoswgdqw/vdaevhnmja/ReadVariableOp_2&vlxoswgdqw/vdaevhnmja/ReadVariableOp_22$
vlxoswgdqw/whilevlxoswgdqw/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ßY

while_body_224231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ddlymsxapn_matmul_readvariableop_resource_0:	F
3while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 A
2while_ddlymsxapn_biasadd_readvariableop_resource_0:	8
*while_ddlymsxapn_readvariableop_resource_0: :
,while_ddlymsxapn_readvariableop_1_resource_0: :
,while_ddlymsxapn_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ddlymsxapn_matmul_readvariableop_resource:	D
1while_ddlymsxapn_matmul_1_readvariableop_resource:	 ?
0while_ddlymsxapn_biasadd_readvariableop_resource:	6
(while_ddlymsxapn_readvariableop_resource: 8
*while_ddlymsxapn_readvariableop_1_resource: 8
*while_ddlymsxapn_readvariableop_2_resource: ¢'while/ddlymsxapn/BiasAdd/ReadVariableOp¢&while/ddlymsxapn/MatMul/ReadVariableOp¢(while/ddlymsxapn/MatMul_1/ReadVariableOp¢while/ddlymsxapn/ReadVariableOp¢!while/ddlymsxapn/ReadVariableOp_1¢!while/ddlymsxapn/ReadVariableOp_2Ã
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
&while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOp1while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ddlymsxapn/MatMul/ReadVariableOpÑ
while/ddlymsxapn/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMulÉ
(while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOp3while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ddlymsxapn/MatMul_1/ReadVariableOpº
while/ddlymsxapn/MatMul_1MatMulwhile_placeholder_20while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/MatMul_1°
while/ddlymsxapn/addAddV2!while/ddlymsxapn/MatMul:product:0#while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/addÂ
'while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOp2while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ddlymsxapn/BiasAdd/ReadVariableOp½
while/ddlymsxapn/BiasAddBiasAddwhile/ddlymsxapn/add:z:0/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ddlymsxapn/BiasAdd
 while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ddlymsxapn/split/split_dim
while/ddlymsxapn/splitSplit)while/ddlymsxapn/split/split_dim:output:0!while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ddlymsxapn/split©
while/ddlymsxapn/ReadVariableOpReadVariableOp*while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ddlymsxapn/ReadVariableOp£
while/ddlymsxapn/mulMul'while/ddlymsxapn/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul¦
while/ddlymsxapn/add_1AddV2while/ddlymsxapn/split:output:0while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_1
while/ddlymsxapn/SigmoidSigmoidwhile/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid¯
!while/ddlymsxapn/ReadVariableOp_1ReadVariableOp,while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_1©
while/ddlymsxapn/mul_1Mul)while/ddlymsxapn/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_1¨
while/ddlymsxapn/add_2AddV2while/ddlymsxapn/split:output:1while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_2
while/ddlymsxapn/Sigmoid_1Sigmoidwhile/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_1
while/ddlymsxapn/mul_2Mulwhile/ddlymsxapn/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_2
while/ddlymsxapn/TanhTanhwhile/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh¢
while/ddlymsxapn/mul_3Mulwhile/ddlymsxapn/Sigmoid:y:0while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_3£
while/ddlymsxapn/add_3AddV2while/ddlymsxapn/mul_2:z:0while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_3¯
!while/ddlymsxapn/ReadVariableOp_2ReadVariableOp,while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ddlymsxapn/ReadVariableOp_2°
while/ddlymsxapn/mul_4Mul)while/ddlymsxapn/ReadVariableOp_2:value:0while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_4¨
while/ddlymsxapn/add_4AddV2while/ddlymsxapn/split:output:3while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/add_4
while/ddlymsxapn/Sigmoid_2Sigmoidwhile/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Sigmoid_2
while/ddlymsxapn/Tanh_1Tanhwhile/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/Tanh_1¦
while/ddlymsxapn/mul_5Mulwhile/ddlymsxapn/Sigmoid_2:y:0while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ddlymsxapn/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ddlymsxapn/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ddlymsxapn/mul_5:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ddlymsxapn/add_3:z:0(^while/ddlymsxapn/BiasAdd/ReadVariableOp'^while/ddlymsxapn/MatMul/ReadVariableOp)^while/ddlymsxapn/MatMul_1/ReadVariableOp ^while/ddlymsxapn/ReadVariableOp"^while/ddlymsxapn/ReadVariableOp_1"^while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ddlymsxapn_biasadd_readvariableop_resource2while_ddlymsxapn_biasadd_readvariableop_resource_0"h
1while_ddlymsxapn_matmul_1_readvariableop_resource3while_ddlymsxapn_matmul_1_readvariableop_resource_0"d
/while_ddlymsxapn_matmul_readvariableop_resource1while_ddlymsxapn_matmul_readvariableop_resource_0"Z
*while_ddlymsxapn_readvariableop_1_resource,while_ddlymsxapn_readvariableop_1_resource_0"Z
*while_ddlymsxapn_readvariableop_2_resource,while_ddlymsxapn_readvariableop_2_resource_0"V
(while_ddlymsxapn_readvariableop_resource*while_ddlymsxapn_readvariableop_resource_0")
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
'while/ddlymsxapn/BiasAdd/ReadVariableOp'while/ddlymsxapn/BiasAdd/ReadVariableOp2P
&while/ddlymsxapn/MatMul/ReadVariableOp&while/ddlymsxapn/MatMul/ReadVariableOp2T
(while/ddlymsxapn/MatMul_1/ReadVariableOp(while/ddlymsxapn/MatMul_1/ReadVariableOp2B
while/ddlymsxapn/ReadVariableOpwhile/ddlymsxapn/ReadVariableOp2F
!while/ddlymsxapn/ReadVariableOp_1!while/ddlymsxapn/ReadVariableOp_12F
!while/ddlymsxapn/ReadVariableOp_2!while/ddlymsxapn/ReadVariableOp_2: 
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
while_body_225559
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_vdaevhnmja_matmul_readvariableop_resource_0:	 F
3while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 A
2while_vdaevhnmja_biasadd_readvariableop_resource_0:	8
*while_vdaevhnmja_readvariableop_resource_0: :
,while_vdaevhnmja_readvariableop_1_resource_0: :
,while_vdaevhnmja_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_vdaevhnmja_matmul_readvariableop_resource:	 D
1while_vdaevhnmja_matmul_1_readvariableop_resource:	 ?
0while_vdaevhnmja_biasadd_readvariableop_resource:	6
(while_vdaevhnmja_readvariableop_resource: 8
*while_vdaevhnmja_readvariableop_1_resource: 8
*while_vdaevhnmja_readvariableop_2_resource: ¢'while/vdaevhnmja/BiasAdd/ReadVariableOp¢&while/vdaevhnmja/MatMul/ReadVariableOp¢(while/vdaevhnmja/MatMul_1/ReadVariableOp¢while/vdaevhnmja/ReadVariableOp¢!while/vdaevhnmja/ReadVariableOp_1¢!while/vdaevhnmja/ReadVariableOp_2Ã
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
&while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp1while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/vdaevhnmja/MatMul/ReadVariableOpÑ
while/vdaevhnmja/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMulÉ
(while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp3while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/vdaevhnmja/MatMul_1/ReadVariableOpº
while/vdaevhnmja/MatMul_1MatMulwhile_placeholder_20while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMul_1°
while/vdaevhnmja/addAddV2!while/vdaevhnmja/MatMul:product:0#while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/addÂ
'while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp2while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/vdaevhnmja/BiasAdd/ReadVariableOp½
while/vdaevhnmja/BiasAddBiasAddwhile/vdaevhnmja/add:z:0/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/BiasAdd
 while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/vdaevhnmja/split/split_dim
while/vdaevhnmja/splitSplit)while/vdaevhnmja/split/split_dim:output:0!while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/vdaevhnmja/split©
while/vdaevhnmja/ReadVariableOpReadVariableOp*while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/vdaevhnmja/ReadVariableOp£
while/vdaevhnmja/mulMul'while/vdaevhnmja/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul¦
while/vdaevhnmja/add_1AddV2while/vdaevhnmja/split:output:0while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_1
while/vdaevhnmja/SigmoidSigmoidwhile/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid¯
!while/vdaevhnmja/ReadVariableOp_1ReadVariableOp,while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_1©
while/vdaevhnmja/mul_1Mul)while/vdaevhnmja/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_1¨
while/vdaevhnmja/add_2AddV2while/vdaevhnmja/split:output:1while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_2
while/vdaevhnmja/Sigmoid_1Sigmoidwhile/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_1
while/vdaevhnmja/mul_2Mulwhile/vdaevhnmja/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_2
while/vdaevhnmja/TanhTanhwhile/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh¢
while/vdaevhnmja/mul_3Mulwhile/vdaevhnmja/Sigmoid:y:0while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_3£
while/vdaevhnmja/add_3AddV2while/vdaevhnmja/mul_2:z:0while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_3¯
!while/vdaevhnmja/ReadVariableOp_2ReadVariableOp,while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_2°
while/vdaevhnmja/mul_4Mul)while/vdaevhnmja/ReadVariableOp_2:value:0while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_4¨
while/vdaevhnmja/add_4AddV2while/vdaevhnmja/split:output:3while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_4
while/vdaevhnmja/Sigmoid_2Sigmoidwhile/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_2
while/vdaevhnmja/Tanh_1Tanhwhile/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh_1¦
while/vdaevhnmja/mul_5Mulwhile/vdaevhnmja/Sigmoid_2:y:0while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/vdaevhnmja/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/vdaevhnmja/mul_5:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/vdaevhnmja/add_3:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"f
0while_vdaevhnmja_biasadd_readvariableop_resource2while_vdaevhnmja_biasadd_readvariableop_resource_0"h
1while_vdaevhnmja_matmul_1_readvariableop_resource3while_vdaevhnmja_matmul_1_readvariableop_resource_0"d
/while_vdaevhnmja_matmul_readvariableop_resource1while_vdaevhnmja_matmul_readvariableop_resource_0"Z
*while_vdaevhnmja_readvariableop_1_resource,while_vdaevhnmja_readvariableop_1_resource_0"Z
*while_vdaevhnmja_readvariableop_2_resource,while_vdaevhnmja_readvariableop_2_resource_0"V
(while_vdaevhnmja_readvariableop_resource*while_vdaevhnmja_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/vdaevhnmja/BiasAdd/ReadVariableOp'while/vdaevhnmja/BiasAdd/ReadVariableOp2P
&while/vdaevhnmja/MatMul/ReadVariableOp&while/vdaevhnmja/MatMul/ReadVariableOp2T
(while/vdaevhnmja/MatMul_1/ReadVariableOp(while/vdaevhnmja/MatMul_1/ReadVariableOp2B
while/vdaevhnmja/ReadVariableOpwhile/vdaevhnmja/ReadVariableOp2F
!while/vdaevhnmja/ReadVariableOp_1!while/vdaevhnmja/ReadVariableOp_12F
!while/vdaevhnmja/ReadVariableOp_2!while/vdaevhnmja/ReadVariableOp_2: 
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
while_body_225019
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_vdaevhnmja_matmul_readvariableop_resource_0:	 F
3while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 A
2while_vdaevhnmja_biasadd_readvariableop_resource_0:	8
*while_vdaevhnmja_readvariableop_resource_0: :
,while_vdaevhnmja_readvariableop_1_resource_0: :
,while_vdaevhnmja_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_vdaevhnmja_matmul_readvariableop_resource:	 D
1while_vdaevhnmja_matmul_1_readvariableop_resource:	 ?
0while_vdaevhnmja_biasadd_readvariableop_resource:	6
(while_vdaevhnmja_readvariableop_resource: 8
*while_vdaevhnmja_readvariableop_1_resource: 8
*while_vdaevhnmja_readvariableop_2_resource: ¢'while/vdaevhnmja/BiasAdd/ReadVariableOp¢&while/vdaevhnmja/MatMul/ReadVariableOp¢(while/vdaevhnmja/MatMul_1/ReadVariableOp¢while/vdaevhnmja/ReadVariableOp¢!while/vdaevhnmja/ReadVariableOp_1¢!while/vdaevhnmja/ReadVariableOp_2Ã
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
&while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp1while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/vdaevhnmja/MatMul/ReadVariableOpÑ
while/vdaevhnmja/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMulÉ
(while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp3while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/vdaevhnmja/MatMul_1/ReadVariableOpº
while/vdaevhnmja/MatMul_1MatMulwhile_placeholder_20while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMul_1°
while/vdaevhnmja/addAddV2!while/vdaevhnmja/MatMul:product:0#while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/addÂ
'while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp2while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/vdaevhnmja/BiasAdd/ReadVariableOp½
while/vdaevhnmja/BiasAddBiasAddwhile/vdaevhnmja/add:z:0/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/BiasAdd
 while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/vdaevhnmja/split/split_dim
while/vdaevhnmja/splitSplit)while/vdaevhnmja/split/split_dim:output:0!while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/vdaevhnmja/split©
while/vdaevhnmja/ReadVariableOpReadVariableOp*while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/vdaevhnmja/ReadVariableOp£
while/vdaevhnmja/mulMul'while/vdaevhnmja/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul¦
while/vdaevhnmja/add_1AddV2while/vdaevhnmja/split:output:0while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_1
while/vdaevhnmja/SigmoidSigmoidwhile/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid¯
!while/vdaevhnmja/ReadVariableOp_1ReadVariableOp,while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_1©
while/vdaevhnmja/mul_1Mul)while/vdaevhnmja/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_1¨
while/vdaevhnmja/add_2AddV2while/vdaevhnmja/split:output:1while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_2
while/vdaevhnmja/Sigmoid_1Sigmoidwhile/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_1
while/vdaevhnmja/mul_2Mulwhile/vdaevhnmja/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_2
while/vdaevhnmja/TanhTanhwhile/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh¢
while/vdaevhnmja/mul_3Mulwhile/vdaevhnmja/Sigmoid:y:0while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_3£
while/vdaevhnmja/add_3AddV2while/vdaevhnmja/mul_2:z:0while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_3¯
!while/vdaevhnmja/ReadVariableOp_2ReadVariableOp,while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_2°
while/vdaevhnmja/mul_4Mul)while/vdaevhnmja/ReadVariableOp_2:value:0while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_4¨
while/vdaevhnmja/add_4AddV2while/vdaevhnmja/split:output:3while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_4
while/vdaevhnmja/Sigmoid_2Sigmoidwhile/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_2
while/vdaevhnmja/Tanh_1Tanhwhile/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh_1¦
while/vdaevhnmja/mul_5Mulwhile/vdaevhnmja/Sigmoid_2:y:0while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/vdaevhnmja/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/vdaevhnmja/mul_5:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/vdaevhnmja/add_3:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"f
0while_vdaevhnmja_biasadd_readvariableop_resource2while_vdaevhnmja_biasadd_readvariableop_resource_0"h
1while_vdaevhnmja_matmul_1_readvariableop_resource3while_vdaevhnmja_matmul_1_readvariableop_resource_0"d
/while_vdaevhnmja_matmul_readvariableop_resource1while_vdaevhnmja_matmul_readvariableop_resource_0"Z
*while_vdaevhnmja_readvariableop_1_resource,while_vdaevhnmja_readvariableop_1_resource_0"Z
*while_vdaevhnmja_readvariableop_2_resource,while_vdaevhnmja_readvariableop_2_resource_0"V
(while_vdaevhnmja_readvariableop_resource*while_vdaevhnmja_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/vdaevhnmja/BiasAdd/ReadVariableOp'while/vdaevhnmja/BiasAdd/ReadVariableOp2P
&while/vdaevhnmja/MatMul/ReadVariableOp&while/vdaevhnmja/MatMul/ReadVariableOp2T
(while/vdaevhnmja/MatMul_1/ReadVariableOp(while/vdaevhnmja/MatMul_1/ReadVariableOp2B
while/vdaevhnmja/ReadVariableOpwhile/vdaevhnmja/ReadVariableOp2F
!while/vdaevhnmja/ReadVariableOp_1!while/vdaevhnmja/ReadVariableOp_12F
!while/vdaevhnmja/ReadVariableOp_2!while/vdaevhnmja/ReadVariableOp_2: 
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
while_cond_225198
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_225198___redundant_placeholder04
0while_while_cond_225198___redundant_placeholder14
0while_while_cond_225198___redundant_placeholder24
0while_while_cond_225198___redundant_placeholder34
0while_while_cond_225198___redundant_placeholder44
0while_while_cond_225198___redundant_placeholder54
0while_while_cond_225198___redundant_placeholder6
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
while_cond_221250
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_221250___redundant_placeholder04
0while_while_cond_221250___redundant_placeholder14
0while_while_cond_221250___redundant_placeholder24
0while_while_cond_221250___redundant_placeholder34
0while_while_cond_221250___redundant_placeholder44
0while_while_cond_221250___redundant_placeholder54
0while_while_cond_221250___redundant_placeholder6
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
°'
²
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_221418

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
Û

'sequential_omilqdycns_while_body_220103H
Dsequential_omilqdycns_while_sequential_omilqdycns_while_loop_counterN
Jsequential_omilqdycns_while_sequential_omilqdycns_while_maximum_iterations+
'sequential_omilqdycns_while_placeholder-
)sequential_omilqdycns_while_placeholder_1-
)sequential_omilqdycns_while_placeholder_2-
)sequential_omilqdycns_while_placeholder_3G
Csequential_omilqdycns_while_sequential_omilqdycns_strided_slice_1_0
sequential_omilqdycns_while_tensorarrayv2read_tensorlistgetitem_sequential_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0:	\
Isequential_omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0:	 W
Hsequential_omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0:	N
@sequential_omilqdycns_while_ddlymsxapn_readvariableop_resource_0: P
Bsequential_omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0: P
Bsequential_omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0: (
$sequential_omilqdycns_while_identity*
&sequential_omilqdycns_while_identity_1*
&sequential_omilqdycns_while_identity_2*
&sequential_omilqdycns_while_identity_3*
&sequential_omilqdycns_while_identity_4*
&sequential_omilqdycns_while_identity_5E
Asequential_omilqdycns_while_sequential_omilqdycns_strided_slice_1
}sequential_omilqdycns_while_tensorarrayv2read_tensorlistgetitem_sequential_omilqdycns_tensorarrayunstack_tensorlistfromtensorX
Esequential_omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource:	Z
Gsequential_omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource:	 U
Fsequential_omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource:	L
>sequential_omilqdycns_while_ddlymsxapn_readvariableop_resource: N
@sequential_omilqdycns_while_ddlymsxapn_readvariableop_1_resource: N
@sequential_omilqdycns_while_ddlymsxapn_readvariableop_2_resource: ¢=sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp¢<sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp¢>sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp¢5sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp¢7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_1¢7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2ï
Msequential/omilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/omilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/omilqdycns/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_omilqdycns_while_tensorarrayv2read_tensorlistgetitem_sequential_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0'sequential_omilqdycns_while_placeholderVsequential/omilqdycns/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/omilqdycns/while/TensorArrayV2Read/TensorListGetItem
<sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOpReadVariableOpGsequential_omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp©
-sequential/omilqdycns/while/ddlymsxapn/MatMulMatMulFsequential/omilqdycns/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/omilqdycns/while/ddlymsxapn/MatMul
>sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOpReadVariableOpIsequential_omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp
/sequential/omilqdycns/while/ddlymsxapn/MatMul_1MatMul)sequential_omilqdycns_while_placeholder_2Fsequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/omilqdycns/while/ddlymsxapn/MatMul_1
*sequential/omilqdycns/while/ddlymsxapn/addAddV27sequential/omilqdycns/while/ddlymsxapn/MatMul:product:09sequential/omilqdycns/while/ddlymsxapn/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/omilqdycns/while/ddlymsxapn/add
=sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOpReadVariableOpHsequential_omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp
.sequential/omilqdycns/while/ddlymsxapn/BiasAddBiasAdd.sequential/omilqdycns/while/ddlymsxapn/add:z:0Esequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/omilqdycns/while/ddlymsxapn/BiasAdd²
6sequential/omilqdycns/while/ddlymsxapn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/omilqdycns/while/ddlymsxapn/split/split_dimÛ
,sequential/omilqdycns/while/ddlymsxapn/splitSplit?sequential/omilqdycns/while/ddlymsxapn/split/split_dim:output:07sequential/omilqdycns/while/ddlymsxapn/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/omilqdycns/while/ddlymsxapn/splitë
5sequential/omilqdycns/while/ddlymsxapn/ReadVariableOpReadVariableOp@sequential_omilqdycns_while_ddlymsxapn_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/omilqdycns/while/ddlymsxapn/ReadVariableOpû
*sequential/omilqdycns/while/ddlymsxapn/mulMul=sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp:value:0)sequential_omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/omilqdycns/while/ddlymsxapn/mulþ
,sequential/omilqdycns/while/ddlymsxapn/add_1AddV25sequential/omilqdycns/while/ddlymsxapn/split:output:0.sequential/omilqdycns/while/ddlymsxapn/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/add_1Ï
.sequential/omilqdycns/while/ddlymsxapn/SigmoidSigmoid0sequential/omilqdycns/while/ddlymsxapn/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/omilqdycns/while/ddlymsxapn/Sigmoidñ
7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_1ReadVariableOpBsequential_omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_1
,sequential/omilqdycns/while/ddlymsxapn/mul_1Mul?sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_1:value:0)sequential_omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/mul_1
,sequential/omilqdycns/while/ddlymsxapn/add_2AddV25sequential/omilqdycns/while/ddlymsxapn/split:output:10sequential/omilqdycns/while/ddlymsxapn/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/add_2Ó
0sequential/omilqdycns/while/ddlymsxapn/Sigmoid_1Sigmoid0sequential/omilqdycns/while/ddlymsxapn/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/omilqdycns/while/ddlymsxapn/Sigmoid_1ö
,sequential/omilqdycns/while/ddlymsxapn/mul_2Mul4sequential/omilqdycns/while/ddlymsxapn/Sigmoid_1:y:0)sequential_omilqdycns_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/mul_2Ë
+sequential/omilqdycns/while/ddlymsxapn/TanhTanh5sequential/omilqdycns/while/ddlymsxapn/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/omilqdycns/while/ddlymsxapn/Tanhú
,sequential/omilqdycns/while/ddlymsxapn/mul_3Mul2sequential/omilqdycns/while/ddlymsxapn/Sigmoid:y:0/sequential/omilqdycns/while/ddlymsxapn/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/mul_3û
,sequential/omilqdycns/while/ddlymsxapn/add_3AddV20sequential/omilqdycns/while/ddlymsxapn/mul_2:z:00sequential/omilqdycns/while/ddlymsxapn/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/add_3ñ
7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2ReadVariableOpBsequential_omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2
,sequential/omilqdycns/while/ddlymsxapn/mul_4Mul?sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2:value:00sequential/omilqdycns/while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/mul_4
,sequential/omilqdycns/while/ddlymsxapn/add_4AddV25sequential/omilqdycns/while/ddlymsxapn/split:output:30sequential/omilqdycns/while/ddlymsxapn/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/add_4Ó
0sequential/omilqdycns/while/ddlymsxapn/Sigmoid_2Sigmoid0sequential/omilqdycns/while/ddlymsxapn/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/omilqdycns/while/ddlymsxapn/Sigmoid_2Ê
-sequential/omilqdycns/while/ddlymsxapn/Tanh_1Tanh0sequential/omilqdycns/while/ddlymsxapn/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/omilqdycns/while/ddlymsxapn/Tanh_1þ
,sequential/omilqdycns/while/ddlymsxapn/mul_5Mul4sequential/omilqdycns/while/ddlymsxapn/Sigmoid_2:y:01sequential/omilqdycns/while/ddlymsxapn/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/omilqdycns/while/ddlymsxapn/mul_5Ì
@sequential/omilqdycns/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_omilqdycns_while_placeholder_1'sequential_omilqdycns_while_placeholder0sequential/omilqdycns/while/ddlymsxapn/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/omilqdycns/while/TensorArrayV2Write/TensorListSetItem
!sequential/omilqdycns/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/omilqdycns/while/add/yÁ
sequential/omilqdycns/while/addAddV2'sequential_omilqdycns_while_placeholder*sequential/omilqdycns/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/omilqdycns/while/add
#sequential/omilqdycns/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/omilqdycns/while/add_1/yä
!sequential/omilqdycns/while/add_1AddV2Dsequential_omilqdycns_while_sequential_omilqdycns_while_loop_counter,sequential/omilqdycns/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/omilqdycns/while/add_1
$sequential/omilqdycns/while/IdentityIdentity%sequential/omilqdycns/while/add_1:z:0>^sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp=^sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp?^sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp6^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp8^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_18^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/omilqdycns/while/Identityµ
&sequential/omilqdycns/while/Identity_1IdentityJsequential_omilqdycns_while_sequential_omilqdycns_while_maximum_iterations>^sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp=^sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp?^sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp6^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp8^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_18^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/omilqdycns/while/Identity_1
&sequential/omilqdycns/while/Identity_2Identity#sequential/omilqdycns/while/add:z:0>^sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp=^sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp?^sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp6^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp8^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_18^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/omilqdycns/while/Identity_2»
&sequential/omilqdycns/while/Identity_3IdentityPsequential/omilqdycns/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp=^sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp?^sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp6^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp8^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_18^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/omilqdycns/while/Identity_3¬
&sequential/omilqdycns/while/Identity_4Identity0sequential/omilqdycns/while/ddlymsxapn/mul_5:z:0>^sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp=^sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp?^sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp6^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp8^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_18^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/while/Identity_4¬
&sequential/omilqdycns/while/Identity_5Identity0sequential/omilqdycns/while/ddlymsxapn/add_3:z:0>^sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp=^sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp?^sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp6^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp8^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_18^sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/omilqdycns/while/Identity_5"
Fsequential_omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resourceHsequential_omilqdycns_while_ddlymsxapn_biasadd_readvariableop_resource_0"
Gsequential_omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resourceIsequential_omilqdycns_while_ddlymsxapn_matmul_1_readvariableop_resource_0"
Esequential_omilqdycns_while_ddlymsxapn_matmul_readvariableop_resourceGsequential_omilqdycns_while_ddlymsxapn_matmul_readvariableop_resource_0"
@sequential_omilqdycns_while_ddlymsxapn_readvariableop_1_resourceBsequential_omilqdycns_while_ddlymsxapn_readvariableop_1_resource_0"
@sequential_omilqdycns_while_ddlymsxapn_readvariableop_2_resourceBsequential_omilqdycns_while_ddlymsxapn_readvariableop_2_resource_0"
>sequential_omilqdycns_while_ddlymsxapn_readvariableop_resource@sequential_omilqdycns_while_ddlymsxapn_readvariableop_resource_0"U
$sequential_omilqdycns_while_identity-sequential/omilqdycns/while/Identity:output:0"Y
&sequential_omilqdycns_while_identity_1/sequential/omilqdycns/while/Identity_1:output:0"Y
&sequential_omilqdycns_while_identity_2/sequential/omilqdycns/while/Identity_2:output:0"Y
&sequential_omilqdycns_while_identity_3/sequential/omilqdycns/while/Identity_3:output:0"Y
&sequential_omilqdycns_while_identity_4/sequential/omilqdycns/while/Identity_4:output:0"Y
&sequential_omilqdycns_while_identity_5/sequential/omilqdycns/while/Identity_5:output:0"
Asequential_omilqdycns_while_sequential_omilqdycns_strided_slice_1Csequential_omilqdycns_while_sequential_omilqdycns_strided_slice_1_0"
}sequential_omilqdycns_while_tensorarrayv2read_tensorlistgetitem_sequential_omilqdycns_tensorarrayunstack_tensorlistfromtensorsequential_omilqdycns_while_tensorarrayv2read_tensorlistgetitem_sequential_omilqdycns_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp=sequential/omilqdycns/while/ddlymsxapn/BiasAdd/ReadVariableOp2|
<sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp<sequential/omilqdycns/while/ddlymsxapn/MatMul/ReadVariableOp2
>sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp>sequential/omilqdycns/while/ddlymsxapn/MatMul_1/ReadVariableOp2n
5sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp5sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp2r
7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_17sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_12r
7sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_27sequential/omilqdycns/while/ddlymsxapn/ReadVariableOp_2: 
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
ÿ
¿
+__inference_vdaevhnmja_layer_call_fn_225836

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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_2212312
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
Û
G
+__inference_xfbsciqeco_layer_call_fn_224071

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
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_2219652
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
Û

'sequential_vlxoswgdqw_while_body_220279H
Dsequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_loop_counterN
Jsequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_maximum_iterations+
'sequential_vlxoswgdqw_while_placeholder-
)sequential_vlxoswgdqw_while_placeholder_1-
)sequential_vlxoswgdqw_while_placeholder_2-
)sequential_vlxoswgdqw_while_placeholder_3G
Csequential_vlxoswgdqw_while_sequential_vlxoswgdqw_strided_slice_1_0
sequential_vlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_sequential_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0:	 \
Isequential_vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 W
Hsequential_vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0:	N
@sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0: P
Bsequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0: P
Bsequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0: (
$sequential_vlxoswgdqw_while_identity*
&sequential_vlxoswgdqw_while_identity_1*
&sequential_vlxoswgdqw_while_identity_2*
&sequential_vlxoswgdqw_while_identity_3*
&sequential_vlxoswgdqw_while_identity_4*
&sequential_vlxoswgdqw_while_identity_5E
Asequential_vlxoswgdqw_while_sequential_vlxoswgdqw_strided_slice_1
}sequential_vlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_sequential_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensorX
Esequential_vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource:	 Z
Gsequential_vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource:	 U
Fsequential_vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource:	L
>sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_resource: N
@sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource: N
@sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource: ¢=sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp¢<sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp¢>sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp¢5sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp¢7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1¢7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2ï
Msequential/vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_vlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_sequential_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0'sequential_vlxoswgdqw_while_placeholderVsequential/vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem
<sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOpGsequential_vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp©
-sequential/vlxoswgdqw/while/vdaevhnmja/MatMulMatMulFsequential/vlxoswgdqw/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/vlxoswgdqw/while/vdaevhnmja/MatMul
>sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOpIsequential_vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp
/sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1MatMul)sequential_vlxoswgdqw_while_placeholder_2Fsequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1
*sequential/vlxoswgdqw/while/vdaevhnmja/addAddV27sequential/vlxoswgdqw/while/vdaevhnmja/MatMul:product:09sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/vlxoswgdqw/while/vdaevhnmja/add
=sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOpHsequential_vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp
.sequential/vlxoswgdqw/while/vdaevhnmja/BiasAddBiasAdd.sequential/vlxoswgdqw/while/vdaevhnmja/add:z:0Esequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd²
6sequential/vlxoswgdqw/while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/vlxoswgdqw/while/vdaevhnmja/split/split_dimÛ
,sequential/vlxoswgdqw/while/vdaevhnmja/splitSplit?sequential/vlxoswgdqw/while/vdaevhnmja/split/split_dim:output:07sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/vlxoswgdqw/while/vdaevhnmja/splitë
5sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOpReadVariableOp@sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOpû
*sequential/vlxoswgdqw/while/vdaevhnmja/mulMul=sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp:value:0)sequential_vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/vlxoswgdqw/while/vdaevhnmja/mulþ
,sequential/vlxoswgdqw/while/vdaevhnmja/add_1AddV25sequential/vlxoswgdqw/while/vdaevhnmja/split:output:0.sequential/vlxoswgdqw/while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/add_1Ï
.sequential/vlxoswgdqw/while/vdaevhnmja/SigmoidSigmoid0sequential/vlxoswgdqw/while/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoidñ
7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1ReadVariableOpBsequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_1Mul?sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_1:value:0)sequential_vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_1
,sequential/vlxoswgdqw/while/vdaevhnmja/add_2AddV25sequential/vlxoswgdqw/while/vdaevhnmja/split:output:10sequential/vlxoswgdqw/while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/add_2Ó
0sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoid_1Sigmoid0sequential/vlxoswgdqw/while/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoid_1ö
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_2Mul4sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoid_1:y:0)sequential_vlxoswgdqw_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_2Ë
+sequential/vlxoswgdqw/while/vdaevhnmja/TanhTanh5sequential/vlxoswgdqw/while/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/vlxoswgdqw/while/vdaevhnmja/Tanhú
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_3Mul2sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoid:y:0/sequential/vlxoswgdqw/while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_3û
,sequential/vlxoswgdqw/while/vdaevhnmja/add_3AddV20sequential/vlxoswgdqw/while/vdaevhnmja/mul_2:z:00sequential/vlxoswgdqw/while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/add_3ñ
7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2ReadVariableOpBsequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_4Mul?sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2:value:00sequential/vlxoswgdqw/while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_4
,sequential/vlxoswgdqw/while/vdaevhnmja/add_4AddV25sequential/vlxoswgdqw/while/vdaevhnmja/split:output:30sequential/vlxoswgdqw/while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/add_4Ó
0sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoid_2Sigmoid0sequential/vlxoswgdqw/while/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoid_2Ê
-sequential/vlxoswgdqw/while/vdaevhnmja/Tanh_1Tanh0sequential/vlxoswgdqw/while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/vlxoswgdqw/while/vdaevhnmja/Tanh_1þ
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_5Mul4sequential/vlxoswgdqw/while/vdaevhnmja/Sigmoid_2:y:01sequential/vlxoswgdqw/while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vlxoswgdqw/while/vdaevhnmja/mul_5Ì
@sequential/vlxoswgdqw/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_vlxoswgdqw_while_placeholder_1'sequential_vlxoswgdqw_while_placeholder0sequential/vlxoswgdqw/while/vdaevhnmja/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/vlxoswgdqw/while/TensorArrayV2Write/TensorListSetItem
!sequential/vlxoswgdqw/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/vlxoswgdqw/while/add/yÁ
sequential/vlxoswgdqw/while/addAddV2'sequential_vlxoswgdqw_while_placeholder*sequential/vlxoswgdqw/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/vlxoswgdqw/while/add
#sequential/vlxoswgdqw/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/vlxoswgdqw/while/add_1/yä
!sequential/vlxoswgdqw/while/add_1AddV2Dsequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_loop_counter,sequential/vlxoswgdqw/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/vlxoswgdqw/while/add_1
$sequential/vlxoswgdqw/while/IdentityIdentity%sequential/vlxoswgdqw/while/add_1:z:0>^sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp=^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp?^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp6^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp8^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_18^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/vlxoswgdqw/while/Identityµ
&sequential/vlxoswgdqw/while/Identity_1IdentityJsequential_vlxoswgdqw_while_sequential_vlxoswgdqw_while_maximum_iterations>^sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp=^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp?^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp6^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp8^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_18^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/vlxoswgdqw/while/Identity_1
&sequential/vlxoswgdqw/while/Identity_2Identity#sequential/vlxoswgdqw/while/add:z:0>^sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp=^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp?^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp6^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp8^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_18^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/vlxoswgdqw/while/Identity_2»
&sequential/vlxoswgdqw/while/Identity_3IdentityPsequential/vlxoswgdqw/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp=^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp?^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp6^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp8^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_18^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/vlxoswgdqw/while/Identity_3¬
&sequential/vlxoswgdqw/while/Identity_4Identity0sequential/vlxoswgdqw/while/vdaevhnmja/mul_5:z:0>^sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp=^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp?^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp6^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp8^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_18^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/while/Identity_4¬
&sequential/vlxoswgdqw/while/Identity_5Identity0sequential/vlxoswgdqw/while/vdaevhnmja/add_3:z:0>^sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp=^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp?^sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp6^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp8^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_18^sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vlxoswgdqw/while/Identity_5"U
$sequential_vlxoswgdqw_while_identity-sequential/vlxoswgdqw/while/Identity:output:0"Y
&sequential_vlxoswgdqw_while_identity_1/sequential/vlxoswgdqw/while/Identity_1:output:0"Y
&sequential_vlxoswgdqw_while_identity_2/sequential/vlxoswgdqw/while/Identity_2:output:0"Y
&sequential_vlxoswgdqw_while_identity_3/sequential/vlxoswgdqw/while/Identity_3:output:0"Y
&sequential_vlxoswgdqw_while_identity_4/sequential/vlxoswgdqw/while/Identity_4:output:0"Y
&sequential_vlxoswgdqw_while_identity_5/sequential/vlxoswgdqw/while/Identity_5:output:0"
Asequential_vlxoswgdqw_while_sequential_vlxoswgdqw_strided_slice_1Csequential_vlxoswgdqw_while_sequential_vlxoswgdqw_strided_slice_1_0"
}sequential_vlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_sequential_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensorsequential_vlxoswgdqw_while_tensorarrayv2read_tensorlistgetitem_sequential_vlxoswgdqw_tensorarrayunstack_tensorlistfromtensor_0"
Fsequential_vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resourceHsequential_vlxoswgdqw_while_vdaevhnmja_biasadd_readvariableop_resource_0"
Gsequential_vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resourceIsequential_vlxoswgdqw_while_vdaevhnmja_matmul_1_readvariableop_resource_0"
Esequential_vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resourceGsequential_vlxoswgdqw_while_vdaevhnmja_matmul_readvariableop_resource_0"
@sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resourceBsequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_1_resource_0"
@sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resourceBsequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_2_resource_0"
>sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_resource@sequential_vlxoswgdqw_while_vdaevhnmja_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp=sequential/vlxoswgdqw/while/vdaevhnmja/BiasAdd/ReadVariableOp2|
<sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp<sequential/vlxoswgdqw/while/vdaevhnmja/MatMul/ReadVariableOp2
>sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp>sequential/vlxoswgdqw/while/vdaevhnmja/MatMul_1/ReadVariableOp2n
5sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp5sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp2r
7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_17sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_12r
7sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_27sequential/vlxoswgdqw/while/vdaevhnmja/ReadVariableOp_2: 
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
while_body_225199
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_vdaevhnmja_matmul_readvariableop_resource_0:	 F
3while_vdaevhnmja_matmul_1_readvariableop_resource_0:	 A
2while_vdaevhnmja_biasadd_readvariableop_resource_0:	8
*while_vdaevhnmja_readvariableop_resource_0: :
,while_vdaevhnmja_readvariableop_1_resource_0: :
,while_vdaevhnmja_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_vdaevhnmja_matmul_readvariableop_resource:	 D
1while_vdaevhnmja_matmul_1_readvariableop_resource:	 ?
0while_vdaevhnmja_biasadd_readvariableop_resource:	6
(while_vdaevhnmja_readvariableop_resource: 8
*while_vdaevhnmja_readvariableop_1_resource: 8
*while_vdaevhnmja_readvariableop_2_resource: ¢'while/vdaevhnmja/BiasAdd/ReadVariableOp¢&while/vdaevhnmja/MatMul/ReadVariableOp¢(while/vdaevhnmja/MatMul_1/ReadVariableOp¢while/vdaevhnmja/ReadVariableOp¢!while/vdaevhnmja/ReadVariableOp_1¢!while/vdaevhnmja/ReadVariableOp_2Ã
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
&while/vdaevhnmja/MatMul/ReadVariableOpReadVariableOp1while_vdaevhnmja_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/vdaevhnmja/MatMul/ReadVariableOpÑ
while/vdaevhnmja/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/vdaevhnmja/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMulÉ
(while/vdaevhnmja/MatMul_1/ReadVariableOpReadVariableOp3while_vdaevhnmja_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/vdaevhnmja/MatMul_1/ReadVariableOpº
while/vdaevhnmja/MatMul_1MatMulwhile_placeholder_20while/vdaevhnmja/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/MatMul_1°
while/vdaevhnmja/addAddV2!while/vdaevhnmja/MatMul:product:0#while/vdaevhnmja/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/addÂ
'while/vdaevhnmja/BiasAdd/ReadVariableOpReadVariableOp2while_vdaevhnmja_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/vdaevhnmja/BiasAdd/ReadVariableOp½
while/vdaevhnmja/BiasAddBiasAddwhile/vdaevhnmja/add:z:0/while/vdaevhnmja/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/vdaevhnmja/BiasAdd
 while/vdaevhnmja/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/vdaevhnmja/split/split_dim
while/vdaevhnmja/splitSplit)while/vdaevhnmja/split/split_dim:output:0!while/vdaevhnmja/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/vdaevhnmja/split©
while/vdaevhnmja/ReadVariableOpReadVariableOp*while_vdaevhnmja_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/vdaevhnmja/ReadVariableOp£
while/vdaevhnmja/mulMul'while/vdaevhnmja/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul¦
while/vdaevhnmja/add_1AddV2while/vdaevhnmja/split:output:0while/vdaevhnmja/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_1
while/vdaevhnmja/SigmoidSigmoidwhile/vdaevhnmja/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid¯
!while/vdaevhnmja/ReadVariableOp_1ReadVariableOp,while_vdaevhnmja_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_1©
while/vdaevhnmja/mul_1Mul)while/vdaevhnmja/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_1¨
while/vdaevhnmja/add_2AddV2while/vdaevhnmja/split:output:1while/vdaevhnmja/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_2
while/vdaevhnmja/Sigmoid_1Sigmoidwhile/vdaevhnmja/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_1
while/vdaevhnmja/mul_2Mulwhile/vdaevhnmja/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_2
while/vdaevhnmja/TanhTanhwhile/vdaevhnmja/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh¢
while/vdaevhnmja/mul_3Mulwhile/vdaevhnmja/Sigmoid:y:0while/vdaevhnmja/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_3£
while/vdaevhnmja/add_3AddV2while/vdaevhnmja/mul_2:z:0while/vdaevhnmja/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_3¯
!while/vdaevhnmja/ReadVariableOp_2ReadVariableOp,while_vdaevhnmja_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/vdaevhnmja/ReadVariableOp_2°
while/vdaevhnmja/mul_4Mul)while/vdaevhnmja/ReadVariableOp_2:value:0while/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_4¨
while/vdaevhnmja/add_4AddV2while/vdaevhnmja/split:output:3while/vdaevhnmja/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/add_4
while/vdaevhnmja/Sigmoid_2Sigmoidwhile/vdaevhnmja/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Sigmoid_2
while/vdaevhnmja/Tanh_1Tanhwhile/vdaevhnmja/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/Tanh_1¦
while/vdaevhnmja/mul_5Mulwhile/vdaevhnmja/Sigmoid_2:y:0while/vdaevhnmja/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/vdaevhnmja/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/vdaevhnmja/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/vdaevhnmja/mul_5:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/vdaevhnmja/add_3:z:0(^while/vdaevhnmja/BiasAdd/ReadVariableOp'^while/vdaevhnmja/MatMul/ReadVariableOp)^while/vdaevhnmja/MatMul_1/ReadVariableOp ^while/vdaevhnmja/ReadVariableOp"^while/vdaevhnmja/ReadVariableOp_1"^while/vdaevhnmja/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"f
0while_vdaevhnmja_biasadd_readvariableop_resource2while_vdaevhnmja_biasadd_readvariableop_resource_0"h
1while_vdaevhnmja_matmul_1_readvariableop_resource3while_vdaevhnmja_matmul_1_readvariableop_resource_0"d
/while_vdaevhnmja_matmul_readvariableop_resource1while_vdaevhnmja_matmul_readvariableop_resource_0"Z
*while_vdaevhnmja_readvariableop_1_resource,while_vdaevhnmja_readvariableop_1_resource_0"Z
*while_vdaevhnmja_readvariableop_2_resource,while_vdaevhnmja_readvariableop_2_resource_0"V
(while_vdaevhnmja_readvariableop_resource*while_vdaevhnmja_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/vdaevhnmja/BiasAdd/ReadVariableOp'while/vdaevhnmja/BiasAdd/ReadVariableOp2P
&while/vdaevhnmja/MatMul/ReadVariableOp&while/vdaevhnmja/MatMul/ReadVariableOp2T
(while/vdaevhnmja/MatMul_1/ReadVariableOp(while/vdaevhnmja/MatMul_1/ReadVariableOp2B
while/vdaevhnmja/ReadVariableOpwhile/vdaevhnmja/ReadVariableOp2F
!while/vdaevhnmja/ReadVariableOp_1!while/vdaevhnmja/ReadVariableOp_12F
!while/vdaevhnmja/ReadVariableOp_2!while/vdaevhnmja/ReadVariableOp_2: 
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

ahzwxypkrh;
serving_default_ahzwxypkrh:0ÿÿÿÿÿÿÿÿÿ>

iktogmlrmp0
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
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
_default_save_signature
__call__
+&call_and_return_all_conditional_losses"ÂA
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "ahzwxypkrh"}}, {"class_name": "Conv1D", "config": {"name": "iigfihrkup", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "xfbsciqeco", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "omilqdycns", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ddlymsxapn", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "vlxoswgdqw", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "vdaevhnmja", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "iktogmlrmp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "ahzwxypkrh"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "ahzwxypkrh"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "iigfihrkup", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "xfbsciqeco", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "omilqdycns", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ddlymsxapn", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "vlxoswgdqw", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "vdaevhnmja", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "iktogmlrmp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
Ì

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¥

_tf_keras_layer
{"name": "iigfihrkup", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "iigfihrkup", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ÿ
_tf_keras_layerå{"name": "xfbsciqeco", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "xfbsciqeco", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
­
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_rnn_layerä{"name": "omilqdycns", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "omilqdycns", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ddlymsxapn", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
³
cell

state_spec
	variables
regularization_losses
 trainable_variables
!	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_rnn_layerê{"name": "vlxoswgdqw", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "vlxoswgdqw", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "vdaevhnmja", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
__call__
+&call_and_return_all_conditional_losses"²
_tf_keras_layer{"name": "iktogmlrmp", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "iktogmlrmp", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
9metrics
	variables
:layer_regularization_losses
;layer_metrics

<layers
regularization_losses
=non_trainable_variables
	trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
':%2iigfihrkup/kernel
:2iigfihrkup/bias
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
>metrics
	variables
?layer_regularization_losses
@layer_metrics

Alayers
regularization_losses
Bnon_trainable_variables
trainable_variables
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
Cmetrics
	variables
Dlayer_regularization_losses
Elayer_metrics

Flayers
regularization_losses
Gnon_trainable_variables
trainable_variables
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
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
__call__
+&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"name": "ddlymsxapn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ddlymsxapn", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
Mmetrics
	variables

Nstates
Olayer_regularization_losses
Player_metrics

Qlayers
regularization_losses
Rnon_trainable_variables
trainable_variables
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
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
__call__
+&call_and_return_all_conditional_losses"Ú
_tf_keras_layerÀ{"name": "vdaevhnmja", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "vdaevhnmja", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
Xmetrics
	variables

Ystates
Zlayer_regularization_losses
[layer_metrics

\layers
regularization_losses
]non_trainable_variables
 trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:! 2iktogmlrmp/kernel
:2iktogmlrmp/bias
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
^metrics
$	variables
_layer_regularization_losses
`layer_metrics

alayers
%regularization_losses
bnon_trainable_variables
&trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
/:-	2omilqdycns/ddlymsxapn/kernel
9:7	 2&omilqdycns/ddlymsxapn/recurrent_kernel
):'2omilqdycns/ddlymsxapn/bias
?:= 21omilqdycns/ddlymsxapn/input_gate_peephole_weights
@:> 22omilqdycns/ddlymsxapn/forget_gate_peephole_weights
@:> 22omilqdycns/ddlymsxapn/output_gate_peephole_weights
/:-	 2vlxoswgdqw/vdaevhnmja/kernel
9:7	 2&vlxoswgdqw/vdaevhnmja/recurrent_kernel
):'2vlxoswgdqw/vdaevhnmja/bias
?:= 21vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights
@:> 22vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights
@:> 22vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights
'
c0"
trackable_list_wrapper
 "
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
dmetrics
I	variables
elayer_regularization_losses
flayer_metrics

glayers
Jregularization_losses
hnon_trainable_variables
Ktrainable_variables
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
imetrics
T	variables
jlayer_regularization_losses
klayer_metrics

llayers
Uregularization_losses
mnon_trainable_variables
Vtrainable_variables
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
'
0"
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
1:/2RMSprop/iigfihrkup/kernel/rms
':%2RMSprop/iigfihrkup/bias/rms
-:+ 2RMSprop/iktogmlrmp/kernel/rms
':%2RMSprop/iktogmlrmp/bias/rms
9:7	2(RMSprop/omilqdycns/ddlymsxapn/kernel/rms
C:A	 22RMSprop/omilqdycns/ddlymsxapn/recurrent_kernel/rms
3:12&RMSprop/omilqdycns/ddlymsxapn/bias/rms
I:G 2=RMSprop/omilqdycns/ddlymsxapn/input_gate_peephole_weights/rms
J:H 2>RMSprop/omilqdycns/ddlymsxapn/forget_gate_peephole_weights/rms
J:H 2>RMSprop/omilqdycns/ddlymsxapn/output_gate_peephole_weights/rms
9:7	 2(RMSprop/vlxoswgdqw/vdaevhnmja/kernel/rms
C:A	 22RMSprop/vlxoswgdqw/vdaevhnmja/recurrent_kernel/rms
3:12&RMSprop/vlxoswgdqw/vdaevhnmja/bias/rms
I:G 2=RMSprop/vlxoswgdqw/vdaevhnmja/input_gate_peephole_weights/rms
J:H 2>RMSprop/vlxoswgdqw/vdaevhnmja/forget_gate_peephole_weights/rms
J:H 2>RMSprop/vlxoswgdqw/vdaevhnmja/output_gate_peephole_weights/rms
ê2ç
!__inference__wrapped_model_220386Á
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

ahzwxypkrhÿÿÿÿÿÿÿÿÿ
ú2÷
+__inference_sequential_layer_call_fn_222405
+__inference_sequential_layer_call_fn_223175
+__inference_sequential_layer_call_fn_223212
+__inference_sequential_layer_call_fn_223011À
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
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_223616
F__inference_sequential_layer_call_and_return_conditional_losses_224020
F__inference_sequential_layer_call_and_return_conditional_losses_223052
F__inference_sequential_layer_call_and_return_conditional_losses_223093À
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
Õ2Ò
+__inference_iigfihrkup_layer_call_fn_224029¢
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
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_224066¢
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
+__inference_xfbsciqeco_layer_call_fn_224071¢
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
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_224084¢
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
 2
+__inference_omilqdycns_layer_call_fn_224101
+__inference_omilqdycns_layer_call_fn_224118
+__inference_omilqdycns_layer_call_fn_224135
+__inference_omilqdycns_layer_call_fn_224152æ
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224332
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224512
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224692
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224872æ
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
+__inference_vlxoswgdqw_layer_call_fn_224889
+__inference_vlxoswgdqw_layer_call_fn_224906
+__inference_vlxoswgdqw_layer_call_fn_224923
+__inference_vlxoswgdqw_layer_call_fn_224940æ
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225120
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225300
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225480
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225660æ
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
Õ2Ò
+__inference_iktogmlrmp_layer_call_fn_225669¢
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
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_225679¢
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
$__inference_signature_wrapper_223138
ahzwxypkrh"
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
2
+__inference_ddlymsxapn_layer_call_fn_225702
+__inference_ddlymsxapn_layer_call_fn_225725¾
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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_225769
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_225813¾
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
+__inference_vdaevhnmja_layer_call_fn_225836
+__inference_vdaevhnmja_layer_call_fn_225859¾
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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_225903
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_225947¾
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
!__inference__wrapped_model_220386-./012345678"#;¢8
1¢.
,)

ahzwxypkrhÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

iktogmlrmp$!

iktogmlrmpÿÿÿÿÿÿÿÿÿË
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_225769-./012¢}
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
F__inference_ddlymsxapn_layer_call_and_return_conditional_losses_225813-./012¢}
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
+__inference_ddlymsxapn_layer_call_fn_225702ð-./012¢}
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
+__inference_ddlymsxapn_layer_call_fn_225725ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ ¶
F__inference_iigfihrkup_layer_call_and_return_conditional_losses_224066l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_iigfihrkup_layer_call_fn_224029_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¦
F__inference_iktogmlrmp_layer_call_and_return_conditional_losses_225679\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_iktogmlrmp_layer_call_fn_225669O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÜ
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224332-./012S¢P
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224512-./012S¢P
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224692x-./012C¢@
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
F__inference_omilqdycns_layer_call_and_return_conditional_losses_224872x-./012C¢@
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
+__inference_omilqdycns_layer_call_fn_224101-./012S¢P
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
+__inference_omilqdycns_layer_call_fn_224118-./012S¢P
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
+__inference_omilqdycns_layer_call_fn_224135k-./012C¢@
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
+__inference_omilqdycns_layer_call_fn_224152k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ È
F__inference_sequential_layer_call_and_return_conditional_losses_223052~-./012345678"#C¢@
9¢6
,)

ahzwxypkrhÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
F__inference_sequential_layer_call_and_return_conditional_losses_223093~-./012345678"#C¢@
9¢6
,)

ahzwxypkrhÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_223616z-./012345678"#?¢<
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
F__inference_sequential_layer_call_and_return_conditional_losses_224020z-./012345678"#?¢<
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
+__inference_sequential_layer_call_fn_222405q-./012345678"#C¢@
9¢6
,)

ahzwxypkrhÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_sequential_layer_call_fn_223011q-./012345678"#C¢@
9¢6
,)

ahzwxypkrhÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_223175m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_223212m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
$__inference_signature_wrapper_223138-./012345678"#I¢F
¢ 
?ª<
:

ahzwxypkrh,)

ahzwxypkrhÿÿÿÿÿÿÿÿÿ"7ª4
2

iktogmlrmp$!

iktogmlrmpÿÿÿÿÿÿÿÿÿË
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_225903345678¢}
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
F__inference_vdaevhnmja_layer_call_and_return_conditional_losses_225947345678¢}
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
+__inference_vdaevhnmja_layer_call_fn_225836ð345678¢}
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
+__inference_vdaevhnmja_layer_call_fn_225859ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ï
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225120345678S¢P
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225300345678S¢P
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225480t345678C¢@
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
F__inference_vlxoswgdqw_layer_call_and_return_conditional_losses_225660t345678C¢@
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
+__inference_vlxoswgdqw_layer_call_fn_224889w345678S¢P
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
+__inference_vlxoswgdqw_layer_call_fn_224906w345678S¢P
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
+__inference_vlxoswgdqw_layer_call_fn_224923g345678C¢@
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
+__inference_vlxoswgdqw_layer_call_fn_224940g345678C¢@
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
F__inference_xfbsciqeco_layer_call_and_return_conditional_losses_224084d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_xfbsciqeco_layer_call_fn_224071W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ