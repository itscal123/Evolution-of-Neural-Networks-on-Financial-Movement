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
umdyqemnpr/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameumdyqemnpr/kernel
{
%umdyqemnpr/kernel/Read/ReadVariableOpReadVariableOpumdyqemnpr/kernel*"
_output_shapes
:*
dtype0
v
umdyqemnpr/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameumdyqemnpr/bias
o
#umdyqemnpr/bias/Read/ReadVariableOpReadVariableOpumdyqemnpr/bias*
_output_shapes
:*
dtype0
~
ycxcgxamrr/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameycxcgxamrr/kernel
w
%ycxcgxamrr/kernel/Read/ReadVariableOpReadVariableOpycxcgxamrr/kernel*
_output_shapes

: *
dtype0
v
ycxcgxamrr/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameycxcgxamrr/bias
o
#ycxcgxamrr/bias/Read/ReadVariableOpReadVariableOpycxcgxamrr/bias*
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
subsmtgotc/zdztiqrxwb/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namesubsmtgotc/zdztiqrxwb/kernel

0subsmtgotc/zdztiqrxwb/kernel/Read/ReadVariableOpReadVariableOpsubsmtgotc/zdztiqrxwb/kernel*
_output_shapes
:	*
dtype0
©
&subsmtgotc/zdztiqrxwb/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&subsmtgotc/zdztiqrxwb/recurrent_kernel
¢
:subsmtgotc/zdztiqrxwb/recurrent_kernel/Read/ReadVariableOpReadVariableOp&subsmtgotc/zdztiqrxwb/recurrent_kernel*
_output_shapes
:	 *
dtype0

subsmtgotc/zdztiqrxwb/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesubsmtgotc/zdztiqrxwb/bias

.subsmtgotc/zdztiqrxwb/bias/Read/ReadVariableOpReadVariableOpsubsmtgotc/zdztiqrxwb/bias*
_output_shapes	
:*
dtype0
º
1subsmtgotc/zdztiqrxwb/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31subsmtgotc/zdztiqrxwb/input_gate_peephole_weights
³
Esubsmtgotc/zdztiqrxwb/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1subsmtgotc/zdztiqrxwb/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2subsmtgotc/zdztiqrxwb/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights
µ
Fsubsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2subsmtgotc/zdztiqrxwb/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42subsmtgotc/zdztiqrxwb/output_gate_peephole_weights
µ
Fsubsmtgotc/zdztiqrxwb/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2subsmtgotc/zdztiqrxwb/output_gate_peephole_weights*
_output_shapes
: *
dtype0

quyyatshey/nqcjuhnaut/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namequyyatshey/nqcjuhnaut/kernel

0quyyatshey/nqcjuhnaut/kernel/Read/ReadVariableOpReadVariableOpquyyatshey/nqcjuhnaut/kernel*
_output_shapes
:	 *
dtype0
©
&quyyatshey/nqcjuhnaut/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&quyyatshey/nqcjuhnaut/recurrent_kernel
¢
:quyyatshey/nqcjuhnaut/recurrent_kernel/Read/ReadVariableOpReadVariableOp&quyyatshey/nqcjuhnaut/recurrent_kernel*
_output_shapes
:	 *
dtype0

quyyatshey/nqcjuhnaut/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namequyyatshey/nqcjuhnaut/bias

.quyyatshey/nqcjuhnaut/bias/Read/ReadVariableOpReadVariableOpquyyatshey/nqcjuhnaut/bias*
_output_shapes	
:*
dtype0
º
1quyyatshey/nqcjuhnaut/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31quyyatshey/nqcjuhnaut/input_gate_peephole_weights
³
Equyyatshey/nqcjuhnaut/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1quyyatshey/nqcjuhnaut/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2quyyatshey/nqcjuhnaut/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42quyyatshey/nqcjuhnaut/forget_gate_peephole_weights
µ
Fquyyatshey/nqcjuhnaut/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2quyyatshey/nqcjuhnaut/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2quyyatshey/nqcjuhnaut/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42quyyatshey/nqcjuhnaut/output_gate_peephole_weights
µ
Fquyyatshey/nqcjuhnaut/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2quyyatshey/nqcjuhnaut/output_gate_peephole_weights*
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
RMSprop/umdyqemnpr/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/umdyqemnpr/kernel/rms

1RMSprop/umdyqemnpr/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/umdyqemnpr/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/umdyqemnpr/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/umdyqemnpr/bias/rms

/RMSprop/umdyqemnpr/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/umdyqemnpr/bias/rms*
_output_shapes
:*
dtype0

RMSprop/ycxcgxamrr/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/ycxcgxamrr/kernel/rms

1RMSprop/ycxcgxamrr/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/ycxcgxamrr/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/ycxcgxamrr/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/ycxcgxamrr/bias/rms

/RMSprop/ycxcgxamrr/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/ycxcgxamrr/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/subsmtgotc/zdztiqrxwb/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/subsmtgotc/zdztiqrxwb/kernel/rms
¦
<RMSprop/subsmtgotc/zdztiqrxwb/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/subsmtgotc/zdztiqrxwb/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rms
º
FRMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/subsmtgotc/zdztiqrxwb/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/subsmtgotc/zdztiqrxwb/bias/rms

:RMSprop/subsmtgotc/zdztiqrxwb/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/subsmtgotc/zdztiqrxwb/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rms
Ë
QRMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rms
Í
RRMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rms
Í
RRMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/quyyatshey/nqcjuhnaut/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/quyyatshey/nqcjuhnaut/kernel/rms
¦
<RMSprop/quyyatshey/nqcjuhnaut/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/quyyatshey/nqcjuhnaut/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rms
º
FRMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/quyyatshey/nqcjuhnaut/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/quyyatshey/nqcjuhnaut/bias/rms

:RMSprop/quyyatshey/nqcjuhnaut/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/quyyatshey/nqcjuhnaut/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rms
Ë
QRMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rms
Í
RRMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rms
Í
RRMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rms*
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
VARIABLE_VALUEumdyqemnpr/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEumdyqemnpr/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEycxcgxamrr/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEycxcgxamrr/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEsubsmtgotc/zdztiqrxwb/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&subsmtgotc/zdztiqrxwb/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEsubsmtgotc/zdztiqrxwb/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1subsmtgotc/zdztiqrxwb/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2subsmtgotc/zdztiqrxwb/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEquyyatshey/nqcjuhnaut/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&quyyatshey/nqcjuhnaut/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEquyyatshey/nqcjuhnaut/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1quyyatshey/nqcjuhnaut/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2quyyatshey/nqcjuhnaut/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2quyyatshey/nqcjuhnaut/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUERMSprop/umdyqemnpr/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/umdyqemnpr/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/ycxcgxamrr/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/ycxcgxamrr/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/subsmtgotc/zdztiqrxwb/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/subsmtgotc/zdztiqrxwb/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/quyyatshey/nqcjuhnaut/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/quyyatshey/nqcjuhnaut/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_mkdkkixskmPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_mkdkkixskmumdyqemnpr/kernelumdyqemnpr/biassubsmtgotc/zdztiqrxwb/kernel&subsmtgotc/zdztiqrxwb/recurrent_kernelsubsmtgotc/zdztiqrxwb/bias1subsmtgotc/zdztiqrxwb/input_gate_peephole_weights2subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights2subsmtgotc/zdztiqrxwb/output_gate_peephole_weightsquyyatshey/nqcjuhnaut/kernel&quyyatshey/nqcjuhnaut/recurrent_kernelquyyatshey/nqcjuhnaut/bias1quyyatshey/nqcjuhnaut/input_gate_peephole_weights2quyyatshey/nqcjuhnaut/forget_gate_peephole_weights2quyyatshey/nqcjuhnaut/output_gate_peephole_weightsycxcgxamrr/kernelycxcgxamrr/bias*
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
%__inference_signature_wrapper_1065898
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%umdyqemnpr/kernel/Read/ReadVariableOp#umdyqemnpr/bias/Read/ReadVariableOp%ycxcgxamrr/kernel/Read/ReadVariableOp#ycxcgxamrr/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0subsmtgotc/zdztiqrxwb/kernel/Read/ReadVariableOp:subsmtgotc/zdztiqrxwb/recurrent_kernel/Read/ReadVariableOp.subsmtgotc/zdztiqrxwb/bias/Read/ReadVariableOpEsubsmtgotc/zdztiqrxwb/input_gate_peephole_weights/Read/ReadVariableOpFsubsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/Read/ReadVariableOpFsubsmtgotc/zdztiqrxwb/output_gate_peephole_weights/Read/ReadVariableOp0quyyatshey/nqcjuhnaut/kernel/Read/ReadVariableOp:quyyatshey/nqcjuhnaut/recurrent_kernel/Read/ReadVariableOp.quyyatshey/nqcjuhnaut/bias/Read/ReadVariableOpEquyyatshey/nqcjuhnaut/input_gate_peephole_weights/Read/ReadVariableOpFquyyatshey/nqcjuhnaut/forget_gate_peephole_weights/Read/ReadVariableOpFquyyatshey/nqcjuhnaut/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/umdyqemnpr/kernel/rms/Read/ReadVariableOp/RMSprop/umdyqemnpr/bias/rms/Read/ReadVariableOp1RMSprop/ycxcgxamrr/kernel/rms/Read/ReadVariableOp/RMSprop/ycxcgxamrr/bias/rms/Read/ReadVariableOp<RMSprop/subsmtgotc/zdztiqrxwb/kernel/rms/Read/ReadVariableOpFRMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/subsmtgotc/zdztiqrxwb/bias/rms/Read/ReadVariableOpQRMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/quyyatshey/nqcjuhnaut/kernel/rms/Read/ReadVariableOpFRMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/quyyatshey/nqcjuhnaut/bias/rms/Read/ReadVariableOpQRMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
 __inference__traced_save_1068847
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameumdyqemnpr/kernelumdyqemnpr/biasycxcgxamrr/kernelycxcgxamrr/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhosubsmtgotc/zdztiqrxwb/kernel&subsmtgotc/zdztiqrxwb/recurrent_kernelsubsmtgotc/zdztiqrxwb/bias1subsmtgotc/zdztiqrxwb/input_gate_peephole_weights2subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights2subsmtgotc/zdztiqrxwb/output_gate_peephole_weightsquyyatshey/nqcjuhnaut/kernel&quyyatshey/nqcjuhnaut/recurrent_kernelquyyatshey/nqcjuhnaut/bias1quyyatshey/nqcjuhnaut/input_gate_peephole_weights2quyyatshey/nqcjuhnaut/forget_gate_peephole_weights2quyyatshey/nqcjuhnaut/output_gate_peephole_weightstotalcountRMSprop/umdyqemnpr/kernel/rmsRMSprop/umdyqemnpr/bias/rmsRMSprop/ycxcgxamrr/kernel/rmsRMSprop/ycxcgxamrr/bias/rms(RMSprop/subsmtgotc/zdztiqrxwb/kernel/rms2RMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rms&RMSprop/subsmtgotc/zdztiqrxwb/bias/rms=RMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rms>RMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rms>RMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rms(RMSprop/quyyatshey/nqcjuhnaut/kernel/rms2RMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rms&RMSprop/quyyatshey/nqcjuhnaut/bias/rms=RMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rms>RMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rms>RMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rms*3
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
#__inference__traced_restore_1068974Âà-


,__inference_sequential_layer_call_fn_1065165

mkdkkixskm
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
mkdkkixskmunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_10651302
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
mkdkkixskm
àh

G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067092
inputs_0<
)zdztiqrxwb_matmul_readvariableop_resource:	>
+zdztiqrxwb_matmul_1_readvariableop_resource:	 9
*zdztiqrxwb_biasadd_readvariableop_resource:	0
"zdztiqrxwb_readvariableop_resource: 2
$zdztiqrxwb_readvariableop_1_resource: 2
$zdztiqrxwb_readvariableop_2_resource: 
identity¢while¢!zdztiqrxwb/BiasAdd/ReadVariableOp¢ zdztiqrxwb/MatMul/ReadVariableOp¢"zdztiqrxwb/MatMul_1/ReadVariableOp¢zdztiqrxwb/ReadVariableOp¢zdztiqrxwb/ReadVariableOp_1¢zdztiqrxwb/ReadVariableOp_2F
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
 zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp)zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 zdztiqrxwb/MatMul/ReadVariableOp§
zdztiqrxwb/MatMulMatMulstrided_slice_2:output:0(zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMulµ
"zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp+zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"zdztiqrxwb/MatMul_1/ReadVariableOp£
zdztiqrxwb/MatMul_1MatMulzeros:output:0*zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMul_1
zdztiqrxwb/addAddV2zdztiqrxwb/MatMul:product:0zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/add®
!zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp*zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!zdztiqrxwb/BiasAdd/ReadVariableOp¥
zdztiqrxwb/BiasAddBiasAddzdztiqrxwb/add:z:0)zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/BiasAddz
zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
zdztiqrxwb/split/split_dimë
zdztiqrxwb/splitSplit#zdztiqrxwb/split/split_dim:output:0zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zdztiqrxwb/split
zdztiqrxwb/ReadVariableOpReadVariableOp"zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp
zdztiqrxwb/mulMul!zdztiqrxwb/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul
zdztiqrxwb/add_1AddV2zdztiqrxwb/split:output:0zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_1{
zdztiqrxwb/SigmoidSigmoidzdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid
zdztiqrxwb/ReadVariableOp_1ReadVariableOp$zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_1
zdztiqrxwb/mul_1Mul#zdztiqrxwb/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_1
zdztiqrxwb/add_2AddV2zdztiqrxwb/split:output:1zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_2
zdztiqrxwb/Sigmoid_1Sigmoidzdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_1
zdztiqrxwb/mul_2Mulzdztiqrxwb/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_2w
zdztiqrxwb/TanhTanhzdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh
zdztiqrxwb/mul_3Mulzdztiqrxwb/Sigmoid:y:0zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_3
zdztiqrxwb/add_3AddV2zdztiqrxwb/mul_2:z:0zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_3
zdztiqrxwb/ReadVariableOp_2ReadVariableOp$zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_2
zdztiqrxwb/mul_4Mul#zdztiqrxwb/ReadVariableOp_2:value:0zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_4
zdztiqrxwb/add_4AddV2zdztiqrxwb/split:output:3zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_4
zdztiqrxwb/Sigmoid_2Sigmoidzdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_2v
zdztiqrxwb/Tanh_1Tanhzdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh_1
zdztiqrxwb/mul_5Mulzdztiqrxwb/Sigmoid_2:y:0zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)zdztiqrxwb_matmul_readvariableop_resource+zdztiqrxwb_matmul_1_readvariableop_resource*zdztiqrxwb_biasadd_readvariableop_resource"zdztiqrxwb_readvariableop_resource$zdztiqrxwb_readvariableop_1_resource$zdztiqrxwb_readvariableop_2_resource*
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
while_body_1066991*
condR
while_cond_1066990*Q
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
IdentityIdentitytranspose_1:y:0^while"^zdztiqrxwb/BiasAdd/ReadVariableOp!^zdztiqrxwb/MatMul/ReadVariableOp#^zdztiqrxwb/MatMul_1/ReadVariableOp^zdztiqrxwb/ReadVariableOp^zdztiqrxwb/ReadVariableOp_1^zdztiqrxwb/ReadVariableOp_2*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2F
!zdztiqrxwb/BiasAdd/ReadVariableOp!zdztiqrxwb/BiasAdd/ReadVariableOp2D
 zdztiqrxwb/MatMul/ReadVariableOp zdztiqrxwb/MatMul/ReadVariableOp2H
"zdztiqrxwb/MatMul_1/ReadVariableOp"zdztiqrxwb/MatMul_1/ReadVariableOp26
zdztiqrxwb/ReadVariableOpzdztiqrxwb/ReadVariableOp2:
zdztiqrxwb/ReadVariableOp_1zdztiqrxwb/ReadVariableOp_12:
zdztiqrxwb/ReadVariableOp_2zdztiqrxwb/ReadVariableOp_2:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
p
Ê
quyyatshey_while_body_10662692
.quyyatshey_while_quyyatshey_while_loop_counter8
4quyyatshey_while_quyyatshey_while_maximum_iterations 
quyyatshey_while_placeholder"
quyyatshey_while_placeholder_1"
quyyatshey_while_placeholder_2"
quyyatshey_while_placeholder_31
-quyyatshey_while_quyyatshey_strided_slice_1_0m
iquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0O
<quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0:	 Q
>quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 L
=quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0:	C
5quyyatshey_while_nqcjuhnaut_readvariableop_resource_0: E
7quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0: E
7quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0: 
quyyatshey_while_identity
quyyatshey_while_identity_1
quyyatshey_while_identity_2
quyyatshey_while_identity_3
quyyatshey_while_identity_4
quyyatshey_while_identity_5/
+quyyatshey_while_quyyatshey_strided_slice_1k
gquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensorM
:quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource:	 O
<quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource:	 J
;quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource:	A
3quyyatshey_while_nqcjuhnaut_readvariableop_resource: C
5quyyatshey_while_nqcjuhnaut_readvariableop_1_resource: C
5quyyatshey_while_nqcjuhnaut_readvariableop_2_resource: ¢2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp¢1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp¢3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp¢*quyyatshey/while/nqcjuhnaut/ReadVariableOp¢,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1¢,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2Ù
Bquyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bquyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shape
4quyyatshey/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0quyyatshey_while_placeholderKquyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4quyyatshey/while/TensorArrayV2Read/TensorListGetItemä
1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp<quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOpý
"quyyatshey/while/nqcjuhnaut/MatMulMatMul;quyyatshey/while/TensorArrayV2Read/TensorListGetItem:item:09quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"quyyatshey/while/nqcjuhnaut/MatMulê
3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp>quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOpæ
$quyyatshey/while/nqcjuhnaut/MatMul_1MatMulquyyatshey_while_placeholder_2;quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$quyyatshey/while/nqcjuhnaut/MatMul_1Ü
quyyatshey/while/nqcjuhnaut/addAddV2,quyyatshey/while/nqcjuhnaut/MatMul:product:0.quyyatshey/while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
quyyatshey/while/nqcjuhnaut/addã
2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp=quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOpé
#quyyatshey/while/nqcjuhnaut/BiasAddBiasAdd#quyyatshey/while/nqcjuhnaut/add:z:0:quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#quyyatshey/while/nqcjuhnaut/BiasAdd
+quyyatshey/while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+quyyatshey/while/nqcjuhnaut/split/split_dim¯
!quyyatshey/while/nqcjuhnaut/splitSplit4quyyatshey/while/nqcjuhnaut/split/split_dim:output:0,quyyatshey/while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!quyyatshey/while/nqcjuhnaut/splitÊ
*quyyatshey/while/nqcjuhnaut/ReadVariableOpReadVariableOp5quyyatshey_while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*quyyatshey/while/nqcjuhnaut/ReadVariableOpÏ
quyyatshey/while/nqcjuhnaut/mulMul2quyyatshey/while/nqcjuhnaut/ReadVariableOp:value:0quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
quyyatshey/while/nqcjuhnaut/mulÒ
!quyyatshey/while/nqcjuhnaut/add_1AddV2*quyyatshey/while/nqcjuhnaut/split:output:0#quyyatshey/while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_1®
#quyyatshey/while/nqcjuhnaut/SigmoidSigmoid%quyyatshey/while/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#quyyatshey/while/nqcjuhnaut/SigmoidÐ
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp7quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1Õ
!quyyatshey/while/nqcjuhnaut/mul_1Mul4quyyatshey/while/nqcjuhnaut/ReadVariableOp_1:value:0quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_1Ô
!quyyatshey/while/nqcjuhnaut/add_2AddV2*quyyatshey/while/nqcjuhnaut/split:output:1%quyyatshey/while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_2²
%quyyatshey/while/nqcjuhnaut/Sigmoid_1Sigmoid%quyyatshey/while/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%quyyatshey/while/nqcjuhnaut/Sigmoid_1Ê
!quyyatshey/while/nqcjuhnaut/mul_2Mul)quyyatshey/while/nqcjuhnaut/Sigmoid_1:y:0quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_2ª
 quyyatshey/while/nqcjuhnaut/TanhTanh*quyyatshey/while/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 quyyatshey/while/nqcjuhnaut/TanhÎ
!quyyatshey/while/nqcjuhnaut/mul_3Mul'quyyatshey/while/nqcjuhnaut/Sigmoid:y:0$quyyatshey/while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_3Ï
!quyyatshey/while/nqcjuhnaut/add_3AddV2%quyyatshey/while/nqcjuhnaut/mul_2:z:0%quyyatshey/while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_3Ð
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp7quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2Ü
!quyyatshey/while/nqcjuhnaut/mul_4Mul4quyyatshey/while/nqcjuhnaut/ReadVariableOp_2:value:0%quyyatshey/while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_4Ô
!quyyatshey/while/nqcjuhnaut/add_4AddV2*quyyatshey/while/nqcjuhnaut/split:output:3%quyyatshey/while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_4²
%quyyatshey/while/nqcjuhnaut/Sigmoid_2Sigmoid%quyyatshey/while/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%quyyatshey/while/nqcjuhnaut/Sigmoid_2©
"quyyatshey/while/nqcjuhnaut/Tanh_1Tanh%quyyatshey/while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"quyyatshey/while/nqcjuhnaut/Tanh_1Ò
!quyyatshey/while/nqcjuhnaut/mul_5Mul)quyyatshey/while/nqcjuhnaut/Sigmoid_2:y:0&quyyatshey/while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_5
5quyyatshey/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemquyyatshey_while_placeholder_1quyyatshey_while_placeholder%quyyatshey/while/nqcjuhnaut/mul_5:z:0*
_output_shapes
: *
element_dtype027
5quyyatshey/while/TensorArrayV2Write/TensorListSetItemr
quyyatshey/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
quyyatshey/while/add/y
quyyatshey/while/addAddV2quyyatshey_while_placeholderquyyatshey/while/add/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/while/addv
quyyatshey/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
quyyatshey/while/add_1/y­
quyyatshey/while/add_1AddV2.quyyatshey_while_quyyatshey_while_loop_counter!quyyatshey/while/add_1/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/while/add_1©
quyyatshey/while/IdentityIdentityquyyatshey/while/add_1:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/IdentityÇ
quyyatshey/while/Identity_1Identity4quyyatshey_while_quyyatshey_while_maximum_iterations3^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/Identity_1«
quyyatshey/while/Identity_2Identityquyyatshey/while/add:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/Identity_2Ø
quyyatshey/while/Identity_3IdentityEquyyatshey/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/Identity_3É
quyyatshey/while/Identity_4Identity%quyyatshey/while/nqcjuhnaut/mul_5:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/while/Identity_4É
quyyatshey/while/Identity_5Identity%quyyatshey/while/nqcjuhnaut/add_3:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/while/Identity_5"?
quyyatshey_while_identity"quyyatshey/while/Identity:output:0"C
quyyatshey_while_identity_1$quyyatshey/while/Identity_1:output:0"C
quyyatshey_while_identity_2$quyyatshey/while/Identity_2:output:0"C
quyyatshey_while_identity_3$quyyatshey/while/Identity_3:output:0"C
quyyatshey_while_identity_4$quyyatshey/while/Identity_4:output:0"C
quyyatshey_while_identity_5$quyyatshey/while/Identity_5:output:0"|
;quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource=quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0"~
<quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource>quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0"z
:quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource<quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0"p
5quyyatshey_while_nqcjuhnaut_readvariableop_1_resource7quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0"p
5quyyatshey_while_nqcjuhnaut_readvariableop_2_resource7quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0"l
3quyyatshey_while_nqcjuhnaut_readvariableop_resource5quyyatshey_while_nqcjuhnaut_readvariableop_resource_0"\
+quyyatshey_while_quyyatshey_strided_slice_1-quyyatshey_while_quyyatshey_strided_slice_1_0"Ô
gquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensoriquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2f
1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp2j
3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp2X
*quyyatshey/while/nqcjuhnaut/ReadVariableOp*quyyatshey/while/nqcjuhnaut/ReadVariableOp2\
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1,quyyatshey/while/nqcjuhnaut/ReadVariableOp_12\
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_quyyatshey_layer_call_fn_1067683

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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10650992
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
Û

,__inference_subsmtgotc_layer_call_fn_1066912

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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10655882
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


í
while_cond_1067778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1067778___redundant_placeholder05
1while_while_cond_1067778___redundant_placeholder15
1while_while_cond_1067778___redundant_placeholder25
1while_while_cond_1067778___redundant_placeholder35
1while_while_cond_1067778___redundant_placeholder45
1while_while_cond_1067778___redundant_placeholder55
1while_while_cond_1067778___redundant_placeholder6
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068240

inputs<
)nqcjuhnaut_matmul_readvariableop_resource:	 >
+nqcjuhnaut_matmul_1_readvariableop_resource:	 9
*nqcjuhnaut_biasadd_readvariableop_resource:	0
"nqcjuhnaut_readvariableop_resource: 2
$nqcjuhnaut_readvariableop_1_resource: 2
$nqcjuhnaut_readvariableop_2_resource: 
identity¢!nqcjuhnaut/BiasAdd/ReadVariableOp¢ nqcjuhnaut/MatMul/ReadVariableOp¢"nqcjuhnaut/MatMul_1/ReadVariableOp¢nqcjuhnaut/ReadVariableOp¢nqcjuhnaut/ReadVariableOp_1¢nqcjuhnaut/ReadVariableOp_2¢whileD
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
 nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp)nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 nqcjuhnaut/MatMul/ReadVariableOp§
nqcjuhnaut/MatMulMatMulstrided_slice_2:output:0(nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMulµ
"nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp+nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"nqcjuhnaut/MatMul_1/ReadVariableOp£
nqcjuhnaut/MatMul_1MatMulzeros:output:0*nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMul_1
nqcjuhnaut/addAddV2nqcjuhnaut/MatMul:product:0nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/add®
!nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp*nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!nqcjuhnaut/BiasAdd/ReadVariableOp¥
nqcjuhnaut/BiasAddBiasAddnqcjuhnaut/add:z:0)nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/BiasAddz
nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
nqcjuhnaut/split/split_dimë
nqcjuhnaut/splitSplit#nqcjuhnaut/split/split_dim:output:0nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nqcjuhnaut/split
nqcjuhnaut/ReadVariableOpReadVariableOp"nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp
nqcjuhnaut/mulMul!nqcjuhnaut/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul
nqcjuhnaut/add_1AddV2nqcjuhnaut/split:output:0nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_1{
nqcjuhnaut/SigmoidSigmoidnqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid
nqcjuhnaut/ReadVariableOp_1ReadVariableOp$nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_1
nqcjuhnaut/mul_1Mul#nqcjuhnaut/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_1
nqcjuhnaut/add_2AddV2nqcjuhnaut/split:output:1nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_2
nqcjuhnaut/Sigmoid_1Sigmoidnqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_1
nqcjuhnaut/mul_2Mulnqcjuhnaut/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_2w
nqcjuhnaut/TanhTanhnqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh
nqcjuhnaut/mul_3Mulnqcjuhnaut/Sigmoid:y:0nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_3
nqcjuhnaut/add_3AddV2nqcjuhnaut/mul_2:z:0nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_3
nqcjuhnaut/ReadVariableOp_2ReadVariableOp$nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_2
nqcjuhnaut/mul_4Mul#nqcjuhnaut/ReadVariableOp_2:value:0nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_4
nqcjuhnaut/add_4AddV2nqcjuhnaut/split:output:3nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_4
nqcjuhnaut/Sigmoid_2Sigmoidnqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_2v
nqcjuhnaut/Tanh_1Tanhnqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh_1
nqcjuhnaut/mul_5Mulnqcjuhnaut/Sigmoid_2:y:0nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)nqcjuhnaut_matmul_readvariableop_resource+nqcjuhnaut_matmul_1_readvariableop_resource*nqcjuhnaut_biasadd_readvariableop_resource"nqcjuhnaut_readvariableop_resource$nqcjuhnaut_readvariableop_1_resource$nqcjuhnaut_readvariableop_2_resource*
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
while_body_1068139*
condR
while_cond_1068138*Q
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
IdentityIdentitystrided_slice_3:output:0"^nqcjuhnaut/BiasAdd/ReadVariableOp!^nqcjuhnaut/MatMul/ReadVariableOp#^nqcjuhnaut/MatMul_1/ReadVariableOp^nqcjuhnaut/ReadVariableOp^nqcjuhnaut/ReadVariableOp_1^nqcjuhnaut/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!nqcjuhnaut/BiasAdd/ReadVariableOp!nqcjuhnaut/BiasAdd/ReadVariableOp2D
 nqcjuhnaut/MatMul/ReadVariableOp nqcjuhnaut/MatMul/ReadVariableOp2H
"nqcjuhnaut/MatMul_1/ReadVariableOp"nqcjuhnaut/MatMul_1/ReadVariableOp26
nqcjuhnaut/ReadVariableOpnqcjuhnaut/ReadVariableOp2:
nqcjuhnaut/ReadVariableOp_1nqcjuhnaut/ReadVariableOp_12:
nqcjuhnaut/ReadVariableOp_2nqcjuhnaut/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_1063515
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063515___redundant_placeholder05
1while_while_cond_1063515___redundant_placeholder15
1while_while_cond_1063515___redundant_placeholder25
1while_while_cond_1063515___redundant_placeholder35
1while_while_cond_1063515___redundant_placeholder45
1while_while_cond_1063515___redundant_placeholder55
1while_while_cond_1063515___redundant_placeholder6
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
while_body_1067959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_nqcjuhnaut_matmul_readvariableop_resource_0:	 F
3while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 A
2while_nqcjuhnaut_biasadd_readvariableop_resource_0:	8
*while_nqcjuhnaut_readvariableop_resource_0: :
,while_nqcjuhnaut_readvariableop_1_resource_0: :
,while_nqcjuhnaut_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_nqcjuhnaut_matmul_readvariableop_resource:	 D
1while_nqcjuhnaut_matmul_1_readvariableop_resource:	 ?
0while_nqcjuhnaut_biasadd_readvariableop_resource:	6
(while_nqcjuhnaut_readvariableop_resource: 8
*while_nqcjuhnaut_readvariableop_1_resource: 8
*while_nqcjuhnaut_readvariableop_2_resource: ¢'while/nqcjuhnaut/BiasAdd/ReadVariableOp¢&while/nqcjuhnaut/MatMul/ReadVariableOp¢(while/nqcjuhnaut/MatMul_1/ReadVariableOp¢while/nqcjuhnaut/ReadVariableOp¢!while/nqcjuhnaut/ReadVariableOp_1¢!while/nqcjuhnaut/ReadVariableOp_2Ã
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
&while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp1while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/nqcjuhnaut/MatMul/ReadVariableOpÑ
while/nqcjuhnaut/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMulÉ
(while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp3while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/nqcjuhnaut/MatMul_1/ReadVariableOpº
while/nqcjuhnaut/MatMul_1MatMulwhile_placeholder_20while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMul_1°
while/nqcjuhnaut/addAddV2!while/nqcjuhnaut/MatMul:product:0#while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/addÂ
'while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp2while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/nqcjuhnaut/BiasAdd/ReadVariableOp½
while/nqcjuhnaut/BiasAddBiasAddwhile/nqcjuhnaut/add:z:0/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/BiasAdd
 while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/nqcjuhnaut/split/split_dim
while/nqcjuhnaut/splitSplit)while/nqcjuhnaut/split/split_dim:output:0!while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/nqcjuhnaut/split©
while/nqcjuhnaut/ReadVariableOpReadVariableOp*while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/nqcjuhnaut/ReadVariableOp£
while/nqcjuhnaut/mulMul'while/nqcjuhnaut/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul¦
while/nqcjuhnaut/add_1AddV2while/nqcjuhnaut/split:output:0while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_1
while/nqcjuhnaut/SigmoidSigmoidwhile/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid¯
!while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp,while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_1©
while/nqcjuhnaut/mul_1Mul)while/nqcjuhnaut/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_1¨
while/nqcjuhnaut/add_2AddV2while/nqcjuhnaut/split:output:1while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_2
while/nqcjuhnaut/Sigmoid_1Sigmoidwhile/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_1
while/nqcjuhnaut/mul_2Mulwhile/nqcjuhnaut/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_2
while/nqcjuhnaut/TanhTanhwhile/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh¢
while/nqcjuhnaut/mul_3Mulwhile/nqcjuhnaut/Sigmoid:y:0while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_3£
while/nqcjuhnaut/add_3AddV2while/nqcjuhnaut/mul_2:z:0while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_3¯
!while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp,while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_2°
while/nqcjuhnaut/mul_4Mul)while/nqcjuhnaut/ReadVariableOp_2:value:0while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_4¨
while/nqcjuhnaut/add_4AddV2while/nqcjuhnaut/split:output:3while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_4
while/nqcjuhnaut/Sigmoid_2Sigmoidwhile/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_2
while/nqcjuhnaut/Tanh_1Tanhwhile/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh_1¦
while/nqcjuhnaut/mul_5Mulwhile/nqcjuhnaut/Sigmoid_2:y:0while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/nqcjuhnaut/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/nqcjuhnaut/mul_5:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/nqcjuhnaut/add_3:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
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
0while_nqcjuhnaut_biasadd_readvariableop_resource2while_nqcjuhnaut_biasadd_readvariableop_resource_0"h
1while_nqcjuhnaut_matmul_1_readvariableop_resource3while_nqcjuhnaut_matmul_1_readvariableop_resource_0"d
/while_nqcjuhnaut_matmul_readvariableop_resource1while_nqcjuhnaut_matmul_readvariableop_resource_0"Z
*while_nqcjuhnaut_readvariableop_1_resource,while_nqcjuhnaut_readvariableop_1_resource_0"Z
*while_nqcjuhnaut_readvariableop_2_resource,while_nqcjuhnaut_readvariableop_2_resource_0"V
(while_nqcjuhnaut_readvariableop_resource*while_nqcjuhnaut_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/nqcjuhnaut/BiasAdd/ReadVariableOp'while/nqcjuhnaut/BiasAdd/ReadVariableOp2P
&while/nqcjuhnaut/MatMul/ReadVariableOp&while/nqcjuhnaut/MatMul/ReadVariableOp2T
(while/nqcjuhnaut/MatMul_1/ReadVariableOp(while/nqcjuhnaut/MatMul_1/ReadVariableOp2B
while/nqcjuhnaut/ReadVariableOpwhile/nqcjuhnaut/ReadVariableOp2F
!while/nqcjuhnaut/ReadVariableOp_1!while/nqcjuhnaut/ReadVariableOp_12F
!while/nqcjuhnaut/ReadVariableOp_2!while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1068529

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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1066376

inputsL
6umdyqemnpr_conv1d_expanddims_1_readvariableop_resource:K
=umdyqemnpr_squeeze_batch_dims_biasadd_readvariableop_resource:G
4subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource:	I
6subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource:	 D
5subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource:	;
-subsmtgotc_zdztiqrxwb_readvariableop_resource: =
/subsmtgotc_zdztiqrxwb_readvariableop_1_resource: =
/subsmtgotc_zdztiqrxwb_readvariableop_2_resource: G
4quyyatshey_nqcjuhnaut_matmul_readvariableop_resource:	 I
6quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource:	 D
5quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource:	;
-quyyatshey_nqcjuhnaut_readvariableop_resource: =
/quyyatshey_nqcjuhnaut_readvariableop_1_resource: =
/quyyatshey_nqcjuhnaut_readvariableop_2_resource: ;
)ycxcgxamrr_matmul_readvariableop_resource: 8
*ycxcgxamrr_biasadd_readvariableop_resource:
identity¢,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp¢+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp¢-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp¢$quyyatshey/nqcjuhnaut/ReadVariableOp¢&quyyatshey/nqcjuhnaut/ReadVariableOp_1¢&quyyatshey/nqcjuhnaut/ReadVariableOp_2¢quyyatshey/while¢subsmtgotc/while¢,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp¢+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp¢-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp¢$subsmtgotc/zdztiqrxwb/ReadVariableOp¢&subsmtgotc/zdztiqrxwb/ReadVariableOp_1¢&subsmtgotc/zdztiqrxwb/ReadVariableOp_2¢-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp¢4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!ycxcgxamrr/BiasAdd/ReadVariableOp¢ ycxcgxamrr/MatMul/ReadVariableOp
 umdyqemnpr/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 umdyqemnpr/conv1d/ExpandDims/dim»
umdyqemnpr/conv1d/ExpandDims
ExpandDimsinputs)umdyqemnpr/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/ExpandDimsÙ
-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6umdyqemnpr_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp
"umdyqemnpr/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"umdyqemnpr/conv1d/ExpandDims_1/dimã
umdyqemnpr/conv1d/ExpandDims_1
ExpandDims5umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp:value:0+umdyqemnpr/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
umdyqemnpr/conv1d/ExpandDims_1
umdyqemnpr/conv1d/ShapeShape%umdyqemnpr/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
umdyqemnpr/conv1d/Shape
%umdyqemnpr/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%umdyqemnpr/conv1d/strided_slice/stack¥
'umdyqemnpr/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'umdyqemnpr/conv1d/strided_slice/stack_1
'umdyqemnpr/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'umdyqemnpr/conv1d/strided_slice/stack_2Ì
umdyqemnpr/conv1d/strided_sliceStridedSlice umdyqemnpr/conv1d/Shape:output:0.umdyqemnpr/conv1d/strided_slice/stack:output:00umdyqemnpr/conv1d/strided_slice/stack_1:output:00umdyqemnpr/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
umdyqemnpr/conv1d/strided_slice
umdyqemnpr/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
umdyqemnpr/conv1d/Reshape/shapeÌ
umdyqemnpr/conv1d/ReshapeReshape%umdyqemnpr/conv1d/ExpandDims:output:0(umdyqemnpr/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/Reshapeî
umdyqemnpr/conv1d/Conv2DConv2D"umdyqemnpr/conv1d/Reshape:output:0'umdyqemnpr/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
umdyqemnpr/conv1d/Conv2D
!umdyqemnpr/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!umdyqemnpr/conv1d/concat/values_1
umdyqemnpr/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/concat/axisì
umdyqemnpr/conv1d/concatConcatV2(umdyqemnpr/conv1d/strided_slice:output:0*umdyqemnpr/conv1d/concat/values_1:output:0&umdyqemnpr/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
umdyqemnpr/conv1d/concatÉ
umdyqemnpr/conv1d/Reshape_1Reshape!umdyqemnpr/conv1d/Conv2D:output:0!umdyqemnpr/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/Reshape_1Á
umdyqemnpr/conv1d/SqueezeSqueeze$umdyqemnpr/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/Squeeze
#umdyqemnpr/squeeze_batch_dims/ShapeShape"umdyqemnpr/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#umdyqemnpr/squeeze_batch_dims/Shape°
1umdyqemnpr/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1umdyqemnpr/squeeze_batch_dims/strided_slice/stack½
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1´
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2
+umdyqemnpr/squeeze_batch_dims/strided_sliceStridedSlice,umdyqemnpr/squeeze_batch_dims/Shape:output:0:umdyqemnpr/squeeze_batch_dims/strided_slice/stack:output:0<umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1:output:0<umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+umdyqemnpr/squeeze_batch_dims/strided_slice¯
+umdyqemnpr/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+umdyqemnpr/squeeze_batch_dims/Reshape/shapeé
%umdyqemnpr/squeeze_batch_dims/ReshapeReshape"umdyqemnpr/conv1d/Squeeze:output:04umdyqemnpr/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%umdyqemnpr/squeeze_batch_dims/Reshapeæ
4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=umdyqemnpr_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%umdyqemnpr/squeeze_batch_dims/BiasAddBiasAdd.umdyqemnpr/squeeze_batch_dims/Reshape:output:0<umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%umdyqemnpr/squeeze_batch_dims/BiasAdd¯
-umdyqemnpr/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-umdyqemnpr/squeeze_batch_dims/concat/values_1¡
)umdyqemnpr/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)umdyqemnpr/squeeze_batch_dims/concat/axis¨
$umdyqemnpr/squeeze_batch_dims/concatConcatV24umdyqemnpr/squeeze_batch_dims/strided_slice:output:06umdyqemnpr/squeeze_batch_dims/concat/values_1:output:02umdyqemnpr/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$umdyqemnpr/squeeze_batch_dims/concatö
'umdyqemnpr/squeeze_batch_dims/Reshape_1Reshape.umdyqemnpr/squeeze_batch_dims/BiasAdd:output:0-umdyqemnpr/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'umdyqemnpr/squeeze_batch_dims/Reshape_1
unxqeixodn/ShapeShape0umdyqemnpr/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
unxqeixodn/Shape
unxqeixodn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
unxqeixodn/strided_slice/stack
 unxqeixodn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 unxqeixodn/strided_slice/stack_1
 unxqeixodn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 unxqeixodn/strided_slice/stack_2¤
unxqeixodn/strided_sliceStridedSliceunxqeixodn/Shape:output:0'unxqeixodn/strided_slice/stack:output:0)unxqeixodn/strided_slice/stack_1:output:0)unxqeixodn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
unxqeixodn/strided_slicez
unxqeixodn/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
unxqeixodn/Reshape/shape/1z
unxqeixodn/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
unxqeixodn/Reshape/shape/2×
unxqeixodn/Reshape/shapePack!unxqeixodn/strided_slice:output:0#unxqeixodn/Reshape/shape/1:output:0#unxqeixodn/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
unxqeixodn/Reshape/shape¾
unxqeixodn/ReshapeReshape0umdyqemnpr/squeeze_batch_dims/Reshape_1:output:0!unxqeixodn/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
unxqeixodn/Reshapeo
subsmtgotc/ShapeShapeunxqeixodn/Reshape:output:0*
T0*
_output_shapes
:2
subsmtgotc/Shape
subsmtgotc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
subsmtgotc/strided_slice/stack
 subsmtgotc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 subsmtgotc/strided_slice/stack_1
 subsmtgotc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 subsmtgotc/strided_slice/stack_2¤
subsmtgotc/strided_sliceStridedSlicesubsmtgotc/Shape:output:0'subsmtgotc/strided_slice/stack:output:0)subsmtgotc/strided_slice/stack_1:output:0)subsmtgotc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
subsmtgotc/strided_slicer
subsmtgotc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros/mul/y
subsmtgotc/zeros/mulMul!subsmtgotc/strided_slice:output:0subsmtgotc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros/mulu
subsmtgotc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
subsmtgotc/zeros/Less/y
subsmtgotc/zeros/LessLesssubsmtgotc/zeros/mul:z:0 subsmtgotc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros/Lessx
subsmtgotc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros/packed/1¯
subsmtgotc/zeros/packedPack!subsmtgotc/strided_slice:output:0"subsmtgotc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
subsmtgotc/zeros/packedu
subsmtgotc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
subsmtgotc/zeros/Const¡
subsmtgotc/zerosFill subsmtgotc/zeros/packed:output:0subsmtgotc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zerosv
subsmtgotc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros_1/mul/y
subsmtgotc/zeros_1/mulMul!subsmtgotc/strided_slice:output:0!subsmtgotc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros_1/muly
subsmtgotc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
subsmtgotc/zeros_1/Less/y
subsmtgotc/zeros_1/LessLesssubsmtgotc/zeros_1/mul:z:0"subsmtgotc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros_1/Less|
subsmtgotc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros_1/packed/1µ
subsmtgotc/zeros_1/packedPack!subsmtgotc/strided_slice:output:0$subsmtgotc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
subsmtgotc/zeros_1/packedy
subsmtgotc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
subsmtgotc/zeros_1/Const©
subsmtgotc/zeros_1Fill"subsmtgotc/zeros_1/packed:output:0!subsmtgotc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zeros_1
subsmtgotc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
subsmtgotc/transpose/perm°
subsmtgotc/transpose	Transposeunxqeixodn/Reshape:output:0"subsmtgotc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/transposep
subsmtgotc/Shape_1Shapesubsmtgotc/transpose:y:0*
T0*
_output_shapes
:2
subsmtgotc/Shape_1
 subsmtgotc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 subsmtgotc/strided_slice_1/stack
"subsmtgotc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_1/stack_1
"subsmtgotc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_1/stack_2°
subsmtgotc/strided_slice_1StridedSlicesubsmtgotc/Shape_1:output:0)subsmtgotc/strided_slice_1/stack:output:0+subsmtgotc/strided_slice_1/stack_1:output:0+subsmtgotc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
subsmtgotc/strided_slice_1
&subsmtgotc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&subsmtgotc/TensorArrayV2/element_shapeÞ
subsmtgotc/TensorArrayV2TensorListReserve/subsmtgotc/TensorArrayV2/element_shape:output:0#subsmtgotc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
subsmtgotc/TensorArrayV2Õ
@subsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@subsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2subsmtgotc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsubsmtgotc/transpose:y:0Isubsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2subsmtgotc/TensorArrayUnstack/TensorListFromTensor
 subsmtgotc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 subsmtgotc/strided_slice_2/stack
"subsmtgotc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_2/stack_1
"subsmtgotc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_2/stack_2¾
subsmtgotc/strided_slice_2StridedSlicesubsmtgotc/transpose:y:0)subsmtgotc/strided_slice_2/stack:output:0+subsmtgotc/strided_slice_2/stack_1:output:0+subsmtgotc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
subsmtgotc/strided_slice_2Ð
+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp4subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOpÓ
subsmtgotc/zdztiqrxwb/MatMulMatMul#subsmtgotc/strided_slice_2:output:03subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/zdztiqrxwb/MatMulÖ
-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp6subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOpÏ
subsmtgotc/zdztiqrxwb/MatMul_1MatMulsubsmtgotc/zeros:output:05subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
subsmtgotc/zdztiqrxwb/MatMul_1Ä
subsmtgotc/zdztiqrxwb/addAddV2&subsmtgotc/zdztiqrxwb/MatMul:product:0(subsmtgotc/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/zdztiqrxwb/addÏ
,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp5subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOpÑ
subsmtgotc/zdztiqrxwb/BiasAddBiasAddsubsmtgotc/zdztiqrxwb/add:z:04subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/zdztiqrxwb/BiasAdd
%subsmtgotc/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%subsmtgotc/zdztiqrxwb/split/split_dim
subsmtgotc/zdztiqrxwb/splitSplit.subsmtgotc/zdztiqrxwb/split/split_dim:output:0&subsmtgotc/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
subsmtgotc/zdztiqrxwb/split¶
$subsmtgotc/zdztiqrxwb/ReadVariableOpReadVariableOp-subsmtgotc_zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02&
$subsmtgotc/zdztiqrxwb/ReadVariableOpº
subsmtgotc/zdztiqrxwb/mulMul,subsmtgotc/zdztiqrxwb/ReadVariableOp:value:0subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mulº
subsmtgotc/zdztiqrxwb/add_1AddV2$subsmtgotc/zdztiqrxwb/split:output:0subsmtgotc/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_1
subsmtgotc/zdztiqrxwb/SigmoidSigmoidsubsmtgotc/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/Sigmoid¼
&subsmtgotc/zdztiqrxwb/ReadVariableOp_1ReadVariableOp/subsmtgotc_zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&subsmtgotc/zdztiqrxwb/ReadVariableOp_1À
subsmtgotc/zdztiqrxwb/mul_1Mul.subsmtgotc/zdztiqrxwb/ReadVariableOp_1:value:0subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_1¼
subsmtgotc/zdztiqrxwb/add_2AddV2$subsmtgotc/zdztiqrxwb/split:output:1subsmtgotc/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_2 
subsmtgotc/zdztiqrxwb/Sigmoid_1Sigmoidsubsmtgotc/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
subsmtgotc/zdztiqrxwb/Sigmoid_1µ
subsmtgotc/zdztiqrxwb/mul_2Mul#subsmtgotc/zdztiqrxwb/Sigmoid_1:y:0subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_2
subsmtgotc/zdztiqrxwb/TanhTanh$subsmtgotc/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/Tanh¶
subsmtgotc/zdztiqrxwb/mul_3Mul!subsmtgotc/zdztiqrxwb/Sigmoid:y:0subsmtgotc/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_3·
subsmtgotc/zdztiqrxwb/add_3AddV2subsmtgotc/zdztiqrxwb/mul_2:z:0subsmtgotc/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_3¼
&subsmtgotc/zdztiqrxwb/ReadVariableOp_2ReadVariableOp/subsmtgotc_zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&subsmtgotc/zdztiqrxwb/ReadVariableOp_2Ä
subsmtgotc/zdztiqrxwb/mul_4Mul.subsmtgotc/zdztiqrxwb/ReadVariableOp_2:value:0subsmtgotc/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_4¼
subsmtgotc/zdztiqrxwb/add_4AddV2$subsmtgotc/zdztiqrxwb/split:output:3subsmtgotc/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_4 
subsmtgotc/zdztiqrxwb/Sigmoid_2Sigmoidsubsmtgotc/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
subsmtgotc/zdztiqrxwb/Sigmoid_2
subsmtgotc/zdztiqrxwb/Tanh_1Tanhsubsmtgotc/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/Tanh_1º
subsmtgotc/zdztiqrxwb/mul_5Mul#subsmtgotc/zdztiqrxwb/Sigmoid_2:y:0 subsmtgotc/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_5¥
(subsmtgotc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(subsmtgotc/TensorArrayV2_1/element_shapeä
subsmtgotc/TensorArrayV2_1TensorListReserve1subsmtgotc/TensorArrayV2_1/element_shape:output:0#subsmtgotc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
subsmtgotc/TensorArrayV2_1d
subsmtgotc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/time
#subsmtgotc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#subsmtgotc/while/maximum_iterations
subsmtgotc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/while/loop_counter²
subsmtgotc/whileWhile&subsmtgotc/while/loop_counter:output:0,subsmtgotc/while/maximum_iterations:output:0subsmtgotc/time:output:0#subsmtgotc/TensorArrayV2_1:handle:0subsmtgotc/zeros:output:0subsmtgotc/zeros_1:output:0#subsmtgotc/strided_slice_1:output:0Bsubsmtgotc/TensorArrayUnstack/TensorListFromTensor:output_handle:04subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource6subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource5subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource-subsmtgotc_zdztiqrxwb_readvariableop_resource/subsmtgotc_zdztiqrxwb_readvariableop_1_resource/subsmtgotc_zdztiqrxwb_readvariableop_2_resource*
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
subsmtgotc_while_body_1066093*)
cond!R
subsmtgotc_while_cond_1066092*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
subsmtgotc/whileË
;subsmtgotc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;subsmtgotc/TensorArrayV2Stack/TensorListStack/element_shape
-subsmtgotc/TensorArrayV2Stack/TensorListStackTensorListStacksubsmtgotc/while:output:3Dsubsmtgotc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-subsmtgotc/TensorArrayV2Stack/TensorListStack
 subsmtgotc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 subsmtgotc/strided_slice_3/stack
"subsmtgotc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"subsmtgotc/strided_slice_3/stack_1
"subsmtgotc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_3/stack_2Ü
subsmtgotc/strided_slice_3StridedSlice6subsmtgotc/TensorArrayV2Stack/TensorListStack:tensor:0)subsmtgotc/strided_slice_3/stack:output:0+subsmtgotc/strided_slice_3/stack_1:output:0+subsmtgotc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
subsmtgotc/strided_slice_3
subsmtgotc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
subsmtgotc/transpose_1/permÑ
subsmtgotc/transpose_1	Transpose6subsmtgotc/TensorArrayV2Stack/TensorListStack:tensor:0$subsmtgotc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/transpose_1n
quyyatshey/ShapeShapesubsmtgotc/transpose_1:y:0*
T0*
_output_shapes
:2
quyyatshey/Shape
quyyatshey/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
quyyatshey/strided_slice/stack
 quyyatshey/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 quyyatshey/strided_slice/stack_1
 quyyatshey/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 quyyatshey/strided_slice/stack_2¤
quyyatshey/strided_sliceStridedSlicequyyatshey/Shape:output:0'quyyatshey/strided_slice/stack:output:0)quyyatshey/strided_slice/stack_1:output:0)quyyatshey/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quyyatshey/strided_slicer
quyyatshey/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros/mul/y
quyyatshey/zeros/mulMul!quyyatshey/strided_slice:output:0quyyatshey/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros/mulu
quyyatshey/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
quyyatshey/zeros/Less/y
quyyatshey/zeros/LessLessquyyatshey/zeros/mul:z:0 quyyatshey/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros/Lessx
quyyatshey/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros/packed/1¯
quyyatshey/zeros/packedPack!quyyatshey/strided_slice:output:0"quyyatshey/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
quyyatshey/zeros/packedu
quyyatshey/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
quyyatshey/zeros/Const¡
quyyatshey/zerosFill quyyatshey/zeros/packed:output:0quyyatshey/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/zerosv
quyyatshey/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros_1/mul/y
quyyatshey/zeros_1/mulMul!quyyatshey/strided_slice:output:0!quyyatshey/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros_1/muly
quyyatshey/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
quyyatshey/zeros_1/Less/y
quyyatshey/zeros_1/LessLessquyyatshey/zeros_1/mul:z:0"quyyatshey/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros_1/Less|
quyyatshey/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros_1/packed/1µ
quyyatshey/zeros_1/packedPack!quyyatshey/strided_slice:output:0$quyyatshey/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
quyyatshey/zeros_1/packedy
quyyatshey/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
quyyatshey/zeros_1/Const©
quyyatshey/zeros_1Fill"quyyatshey/zeros_1/packed:output:0!quyyatshey/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/zeros_1
quyyatshey/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
quyyatshey/transpose/perm¯
quyyatshey/transpose	Transposesubsmtgotc/transpose_1:y:0"quyyatshey/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/transposep
quyyatshey/Shape_1Shapequyyatshey/transpose:y:0*
T0*
_output_shapes
:2
quyyatshey/Shape_1
 quyyatshey/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 quyyatshey/strided_slice_1/stack
"quyyatshey/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_1/stack_1
"quyyatshey/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_1/stack_2°
quyyatshey/strided_slice_1StridedSlicequyyatshey/Shape_1:output:0)quyyatshey/strided_slice_1/stack:output:0+quyyatshey/strided_slice_1/stack_1:output:0+quyyatshey/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quyyatshey/strided_slice_1
&quyyatshey/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&quyyatshey/TensorArrayV2/element_shapeÞ
quyyatshey/TensorArrayV2TensorListReserve/quyyatshey/TensorArrayV2/element_shape:output:0#quyyatshey/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
quyyatshey/TensorArrayV2Õ
@quyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@quyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2quyyatshey/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorquyyatshey/transpose:y:0Iquyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2quyyatshey/TensorArrayUnstack/TensorListFromTensor
 quyyatshey/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 quyyatshey/strided_slice_2/stack
"quyyatshey/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_2/stack_1
"quyyatshey/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_2/stack_2¾
quyyatshey/strided_slice_2StridedSlicequyyatshey/transpose:y:0)quyyatshey/strided_slice_2/stack:output:0+quyyatshey/strided_slice_2/stack_1:output:0+quyyatshey/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
quyyatshey/strided_slice_2Ð
+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp4quyyatshey_nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOpÓ
quyyatshey/nqcjuhnaut/MatMulMatMul#quyyatshey/strided_slice_2:output:03quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quyyatshey/nqcjuhnaut/MatMulÖ
-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp6quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOpÏ
quyyatshey/nqcjuhnaut/MatMul_1MatMulquyyatshey/zeros:output:05quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
quyyatshey/nqcjuhnaut/MatMul_1Ä
quyyatshey/nqcjuhnaut/addAddV2&quyyatshey/nqcjuhnaut/MatMul:product:0(quyyatshey/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quyyatshey/nqcjuhnaut/addÏ
,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp5quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOpÑ
quyyatshey/nqcjuhnaut/BiasAddBiasAddquyyatshey/nqcjuhnaut/add:z:04quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quyyatshey/nqcjuhnaut/BiasAdd
%quyyatshey/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%quyyatshey/nqcjuhnaut/split/split_dim
quyyatshey/nqcjuhnaut/splitSplit.quyyatshey/nqcjuhnaut/split/split_dim:output:0&quyyatshey/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
quyyatshey/nqcjuhnaut/split¶
$quyyatshey/nqcjuhnaut/ReadVariableOpReadVariableOp-quyyatshey_nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02&
$quyyatshey/nqcjuhnaut/ReadVariableOpº
quyyatshey/nqcjuhnaut/mulMul,quyyatshey/nqcjuhnaut/ReadVariableOp:value:0quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mulº
quyyatshey/nqcjuhnaut/add_1AddV2$quyyatshey/nqcjuhnaut/split:output:0quyyatshey/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_1
quyyatshey/nqcjuhnaut/SigmoidSigmoidquyyatshey/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/Sigmoid¼
&quyyatshey/nqcjuhnaut/ReadVariableOp_1ReadVariableOp/quyyatshey_nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&quyyatshey/nqcjuhnaut/ReadVariableOp_1À
quyyatshey/nqcjuhnaut/mul_1Mul.quyyatshey/nqcjuhnaut/ReadVariableOp_1:value:0quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_1¼
quyyatshey/nqcjuhnaut/add_2AddV2$quyyatshey/nqcjuhnaut/split:output:1quyyatshey/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_2 
quyyatshey/nqcjuhnaut/Sigmoid_1Sigmoidquyyatshey/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
quyyatshey/nqcjuhnaut/Sigmoid_1µ
quyyatshey/nqcjuhnaut/mul_2Mul#quyyatshey/nqcjuhnaut/Sigmoid_1:y:0quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_2
quyyatshey/nqcjuhnaut/TanhTanh$quyyatshey/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/Tanh¶
quyyatshey/nqcjuhnaut/mul_3Mul!quyyatshey/nqcjuhnaut/Sigmoid:y:0quyyatshey/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_3·
quyyatshey/nqcjuhnaut/add_3AddV2quyyatshey/nqcjuhnaut/mul_2:z:0quyyatshey/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_3¼
&quyyatshey/nqcjuhnaut/ReadVariableOp_2ReadVariableOp/quyyatshey_nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&quyyatshey/nqcjuhnaut/ReadVariableOp_2Ä
quyyatshey/nqcjuhnaut/mul_4Mul.quyyatshey/nqcjuhnaut/ReadVariableOp_2:value:0quyyatshey/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_4¼
quyyatshey/nqcjuhnaut/add_4AddV2$quyyatshey/nqcjuhnaut/split:output:3quyyatshey/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_4 
quyyatshey/nqcjuhnaut/Sigmoid_2Sigmoidquyyatshey/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
quyyatshey/nqcjuhnaut/Sigmoid_2
quyyatshey/nqcjuhnaut/Tanh_1Tanhquyyatshey/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/Tanh_1º
quyyatshey/nqcjuhnaut/mul_5Mul#quyyatshey/nqcjuhnaut/Sigmoid_2:y:0 quyyatshey/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_5¥
(quyyatshey/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(quyyatshey/TensorArrayV2_1/element_shapeä
quyyatshey/TensorArrayV2_1TensorListReserve1quyyatshey/TensorArrayV2_1/element_shape:output:0#quyyatshey/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
quyyatshey/TensorArrayV2_1d
quyyatshey/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/time
#quyyatshey/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#quyyatshey/while/maximum_iterations
quyyatshey/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/while/loop_counter²
quyyatshey/whileWhile&quyyatshey/while/loop_counter:output:0,quyyatshey/while/maximum_iterations:output:0quyyatshey/time:output:0#quyyatshey/TensorArrayV2_1:handle:0quyyatshey/zeros:output:0quyyatshey/zeros_1:output:0#quyyatshey/strided_slice_1:output:0Bquyyatshey/TensorArrayUnstack/TensorListFromTensor:output_handle:04quyyatshey_nqcjuhnaut_matmul_readvariableop_resource6quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource5quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource-quyyatshey_nqcjuhnaut_readvariableop_resource/quyyatshey_nqcjuhnaut_readvariableop_1_resource/quyyatshey_nqcjuhnaut_readvariableop_2_resource*
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
quyyatshey_while_body_1066269*)
cond!R
quyyatshey_while_cond_1066268*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
quyyatshey/whileË
;quyyatshey/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;quyyatshey/TensorArrayV2Stack/TensorListStack/element_shape
-quyyatshey/TensorArrayV2Stack/TensorListStackTensorListStackquyyatshey/while:output:3Dquyyatshey/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-quyyatshey/TensorArrayV2Stack/TensorListStack
 quyyatshey/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 quyyatshey/strided_slice_3/stack
"quyyatshey/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"quyyatshey/strided_slice_3/stack_1
"quyyatshey/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_3/stack_2Ü
quyyatshey/strided_slice_3StridedSlice6quyyatshey/TensorArrayV2Stack/TensorListStack:tensor:0)quyyatshey/strided_slice_3/stack:output:0+quyyatshey/strided_slice_3/stack_1:output:0+quyyatshey/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
quyyatshey/strided_slice_3
quyyatshey/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
quyyatshey/transpose_1/permÑ
quyyatshey/transpose_1	Transpose6quyyatshey/TensorArrayV2Stack/TensorListStack:tensor:0$quyyatshey/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/transpose_1®
 ycxcgxamrr/MatMul/ReadVariableOpReadVariableOp)ycxcgxamrr_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 ycxcgxamrr/MatMul/ReadVariableOp±
ycxcgxamrr/MatMulMatMul#quyyatshey/strided_slice_3:output:0(ycxcgxamrr/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ycxcgxamrr/MatMul­
!ycxcgxamrr/BiasAdd/ReadVariableOpReadVariableOp*ycxcgxamrr_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ycxcgxamrr/BiasAdd/ReadVariableOp­
ycxcgxamrr/BiasAddBiasAddycxcgxamrr/MatMul:product:0)ycxcgxamrr/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ycxcgxamrr/BiasAddÏ
IdentityIdentityycxcgxamrr/BiasAdd:output:0-^quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp,^quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp.^quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp%^quyyatshey/nqcjuhnaut/ReadVariableOp'^quyyatshey/nqcjuhnaut/ReadVariableOp_1'^quyyatshey/nqcjuhnaut/ReadVariableOp_2^quyyatshey/while^subsmtgotc/while-^subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp,^subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp.^subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp%^subsmtgotc/zdztiqrxwb/ReadVariableOp'^subsmtgotc/zdztiqrxwb/ReadVariableOp_1'^subsmtgotc/zdztiqrxwb/ReadVariableOp_2.^umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp5^umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp"^ycxcgxamrr/BiasAdd/ReadVariableOp!^ycxcgxamrr/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp2Z
+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp2^
-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp2L
$quyyatshey/nqcjuhnaut/ReadVariableOp$quyyatshey/nqcjuhnaut/ReadVariableOp2P
&quyyatshey/nqcjuhnaut/ReadVariableOp_1&quyyatshey/nqcjuhnaut/ReadVariableOp_12P
&quyyatshey/nqcjuhnaut/ReadVariableOp_2&quyyatshey/nqcjuhnaut/ReadVariableOp_22$
quyyatshey/whilequyyatshey/while2$
subsmtgotc/whilesubsmtgotc/while2\
,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp2Z
+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp2^
-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp2L
$subsmtgotc/zdztiqrxwb/ReadVariableOp$subsmtgotc/zdztiqrxwb/ReadVariableOp2P
&subsmtgotc/zdztiqrxwb/ReadVariableOp_1&subsmtgotc/zdztiqrxwb/ReadVariableOp_12P
&subsmtgotc/zdztiqrxwb/ReadVariableOp_2&subsmtgotc/zdztiqrxwb/ReadVariableOp_22^
-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp2l
4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!ycxcgxamrr/BiasAdd/ReadVariableOp!ycxcgxamrr/BiasAdd/ReadVariableOp2D
 ycxcgxamrr/MatMul/ReadVariableOp ycxcgxamrr/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹'
µ
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1068707

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


í
while_cond_1064997
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1064997___redundant_placeholder05
1while_while_cond_1064997___redundant_placeholder15
1while_while_cond_1064997___redundant_placeholder25
1while_while_cond_1064997___redundant_placeholder35
1while_while_cond_1064997___redundant_placeholder45
1while_while_cond_1064997___redundant_placeholder55
1while_while_cond_1064997___redundant_placeholder6
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
,__inference_zdztiqrxwb_layer_call_fn_1068462

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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_10632332
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
¤

,__inference_ycxcgxamrr_layer_call_fn_1068429

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
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_10651232
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
ë

,__inference_quyyatshey_layer_call_fn_1067649
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10640912
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
while_cond_1064010
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1064010___redundant_placeholder05
1while_while_cond_1064010___redundant_placeholder15
1while_while_cond_1064010___redundant_placeholder25
1while_while_cond_1064010___redundant_placeholder35
1while_while_cond_1064010___redundant_placeholder45
1while_while_cond_1064010___redundant_placeholder55
1while_while_cond_1064010___redundant_placeholder6
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1065099

inputs<
)nqcjuhnaut_matmul_readvariableop_resource:	 >
+nqcjuhnaut_matmul_1_readvariableop_resource:	 9
*nqcjuhnaut_biasadd_readvariableop_resource:	0
"nqcjuhnaut_readvariableop_resource: 2
$nqcjuhnaut_readvariableop_1_resource: 2
$nqcjuhnaut_readvariableop_2_resource: 
identity¢!nqcjuhnaut/BiasAdd/ReadVariableOp¢ nqcjuhnaut/MatMul/ReadVariableOp¢"nqcjuhnaut/MatMul_1/ReadVariableOp¢nqcjuhnaut/ReadVariableOp¢nqcjuhnaut/ReadVariableOp_1¢nqcjuhnaut/ReadVariableOp_2¢whileD
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
 nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp)nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 nqcjuhnaut/MatMul/ReadVariableOp§
nqcjuhnaut/MatMulMatMulstrided_slice_2:output:0(nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMulµ
"nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp+nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"nqcjuhnaut/MatMul_1/ReadVariableOp£
nqcjuhnaut/MatMul_1MatMulzeros:output:0*nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMul_1
nqcjuhnaut/addAddV2nqcjuhnaut/MatMul:product:0nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/add®
!nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp*nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!nqcjuhnaut/BiasAdd/ReadVariableOp¥
nqcjuhnaut/BiasAddBiasAddnqcjuhnaut/add:z:0)nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/BiasAddz
nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
nqcjuhnaut/split/split_dimë
nqcjuhnaut/splitSplit#nqcjuhnaut/split/split_dim:output:0nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nqcjuhnaut/split
nqcjuhnaut/ReadVariableOpReadVariableOp"nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp
nqcjuhnaut/mulMul!nqcjuhnaut/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul
nqcjuhnaut/add_1AddV2nqcjuhnaut/split:output:0nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_1{
nqcjuhnaut/SigmoidSigmoidnqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid
nqcjuhnaut/ReadVariableOp_1ReadVariableOp$nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_1
nqcjuhnaut/mul_1Mul#nqcjuhnaut/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_1
nqcjuhnaut/add_2AddV2nqcjuhnaut/split:output:1nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_2
nqcjuhnaut/Sigmoid_1Sigmoidnqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_1
nqcjuhnaut/mul_2Mulnqcjuhnaut/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_2w
nqcjuhnaut/TanhTanhnqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh
nqcjuhnaut/mul_3Mulnqcjuhnaut/Sigmoid:y:0nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_3
nqcjuhnaut/add_3AddV2nqcjuhnaut/mul_2:z:0nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_3
nqcjuhnaut/ReadVariableOp_2ReadVariableOp$nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_2
nqcjuhnaut/mul_4Mul#nqcjuhnaut/ReadVariableOp_2:value:0nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_4
nqcjuhnaut/add_4AddV2nqcjuhnaut/split:output:3nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_4
nqcjuhnaut/Sigmoid_2Sigmoidnqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_2v
nqcjuhnaut/Tanh_1Tanhnqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh_1
nqcjuhnaut/mul_5Mulnqcjuhnaut/Sigmoid_2:y:0nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)nqcjuhnaut_matmul_readvariableop_resource+nqcjuhnaut_matmul_1_readvariableop_resource*nqcjuhnaut_biasadd_readvariableop_resource"nqcjuhnaut_readvariableop_resource$nqcjuhnaut_readvariableop_1_resource$nqcjuhnaut_readvariableop_2_resource*
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
while_body_1064998*
condR
while_cond_1064997*Q
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
IdentityIdentitystrided_slice_3:output:0"^nqcjuhnaut/BiasAdd/ReadVariableOp!^nqcjuhnaut/MatMul/ReadVariableOp#^nqcjuhnaut/MatMul_1/ReadVariableOp^nqcjuhnaut/ReadVariableOp^nqcjuhnaut/ReadVariableOp_1^nqcjuhnaut/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!nqcjuhnaut/BiasAdd/ReadVariableOp!nqcjuhnaut/BiasAdd/ReadVariableOp2D
 nqcjuhnaut/MatMul/ReadVariableOp nqcjuhnaut/MatMul/ReadVariableOp2H
"nqcjuhnaut/MatMul_1/ReadVariableOp"nqcjuhnaut/MatMul_1/ReadVariableOp26
nqcjuhnaut/ReadVariableOpnqcjuhnaut/ReadVariableOp2:
nqcjuhnaut/ReadVariableOp_1nqcjuhnaut/ReadVariableOp_12:
nqcjuhnaut/ReadVariableOp_2nqcjuhnaut/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³F
ê
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1063596

inputs%
zdztiqrxwb_1063497:	%
zdztiqrxwb_1063499:	 !
zdztiqrxwb_1063501:	 
zdztiqrxwb_1063503:  
zdztiqrxwb_1063505:  
zdztiqrxwb_1063507: 
identity¢while¢"zdztiqrxwb/StatefulPartitionedCallD
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
"zdztiqrxwb/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0zdztiqrxwb_1063497zdztiqrxwb_1063499zdztiqrxwb_1063501zdztiqrxwb_1063503zdztiqrxwb_1063505zdztiqrxwb_1063507*
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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_10634202$
"zdztiqrxwb/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0zdztiqrxwb_1063497zdztiqrxwb_1063499zdztiqrxwb_1063501zdztiqrxwb_1063503zdztiqrxwb_1063505zdztiqrxwb_1063507*
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
while_body_1063516*
condR
while_cond_1063515*Q
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
IdentityIdentitytranspose_1:y:0^while#^zdztiqrxwb/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2H
"zdztiqrxwb/StatefulPartitionedCall"zdztiqrxwb/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹'
µ
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1068573

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
¹'
µ
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1068663

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
¦h

G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068420

inputs<
)nqcjuhnaut_matmul_readvariableop_resource:	 >
+nqcjuhnaut_matmul_1_readvariableop_resource:	 9
*nqcjuhnaut_biasadd_readvariableop_resource:	0
"nqcjuhnaut_readvariableop_resource: 2
$nqcjuhnaut_readvariableop_1_resource: 2
$nqcjuhnaut_readvariableop_2_resource: 
identity¢!nqcjuhnaut/BiasAdd/ReadVariableOp¢ nqcjuhnaut/MatMul/ReadVariableOp¢"nqcjuhnaut/MatMul_1/ReadVariableOp¢nqcjuhnaut/ReadVariableOp¢nqcjuhnaut/ReadVariableOp_1¢nqcjuhnaut/ReadVariableOp_2¢whileD
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
 nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp)nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 nqcjuhnaut/MatMul/ReadVariableOp§
nqcjuhnaut/MatMulMatMulstrided_slice_2:output:0(nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMulµ
"nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp+nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"nqcjuhnaut/MatMul_1/ReadVariableOp£
nqcjuhnaut/MatMul_1MatMulzeros:output:0*nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMul_1
nqcjuhnaut/addAddV2nqcjuhnaut/MatMul:product:0nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/add®
!nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp*nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!nqcjuhnaut/BiasAdd/ReadVariableOp¥
nqcjuhnaut/BiasAddBiasAddnqcjuhnaut/add:z:0)nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/BiasAddz
nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
nqcjuhnaut/split/split_dimë
nqcjuhnaut/splitSplit#nqcjuhnaut/split/split_dim:output:0nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nqcjuhnaut/split
nqcjuhnaut/ReadVariableOpReadVariableOp"nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp
nqcjuhnaut/mulMul!nqcjuhnaut/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul
nqcjuhnaut/add_1AddV2nqcjuhnaut/split:output:0nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_1{
nqcjuhnaut/SigmoidSigmoidnqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid
nqcjuhnaut/ReadVariableOp_1ReadVariableOp$nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_1
nqcjuhnaut/mul_1Mul#nqcjuhnaut/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_1
nqcjuhnaut/add_2AddV2nqcjuhnaut/split:output:1nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_2
nqcjuhnaut/Sigmoid_1Sigmoidnqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_1
nqcjuhnaut/mul_2Mulnqcjuhnaut/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_2w
nqcjuhnaut/TanhTanhnqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh
nqcjuhnaut/mul_3Mulnqcjuhnaut/Sigmoid:y:0nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_3
nqcjuhnaut/add_3AddV2nqcjuhnaut/mul_2:z:0nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_3
nqcjuhnaut/ReadVariableOp_2ReadVariableOp$nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_2
nqcjuhnaut/mul_4Mul#nqcjuhnaut/ReadVariableOp_2:value:0nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_4
nqcjuhnaut/add_4AddV2nqcjuhnaut/split:output:3nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_4
nqcjuhnaut/Sigmoid_2Sigmoidnqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_2v
nqcjuhnaut/Tanh_1Tanhnqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh_1
nqcjuhnaut/mul_5Mulnqcjuhnaut/Sigmoid_2:y:0nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)nqcjuhnaut_matmul_readvariableop_resource+nqcjuhnaut_matmul_1_readvariableop_resource*nqcjuhnaut_biasadd_readvariableop_resource"nqcjuhnaut_readvariableop_resource$nqcjuhnaut_readvariableop_1_resource$nqcjuhnaut_readvariableop_2_resource*
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
while_body_1068319*
condR
while_cond_1068318*Q
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
IdentityIdentitystrided_slice_3:output:0"^nqcjuhnaut/BiasAdd/ReadVariableOp!^nqcjuhnaut/MatMul/ReadVariableOp#^nqcjuhnaut/MatMul_1/ReadVariableOp^nqcjuhnaut/ReadVariableOp^nqcjuhnaut/ReadVariableOp_1^nqcjuhnaut/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!nqcjuhnaut/BiasAdd/ReadVariableOp!nqcjuhnaut/BiasAdd/ReadVariableOp2D
 nqcjuhnaut/MatMul/ReadVariableOp nqcjuhnaut/MatMul/ReadVariableOp2H
"nqcjuhnaut/MatMul_1/ReadVariableOp"nqcjuhnaut/MatMul_1/ReadVariableOp26
nqcjuhnaut/ReadVariableOpnqcjuhnaut/ReadVariableOp2:
nqcjuhnaut/ReadVariableOp_1nqcjuhnaut/ReadVariableOp_12:
nqcjuhnaut/ReadVariableOp_2nqcjuhnaut/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


quyyatshey_while_cond_10666722
.quyyatshey_while_quyyatshey_while_loop_counter8
4quyyatshey_while_quyyatshey_while_maximum_iterations 
quyyatshey_while_placeholder"
quyyatshey_while_placeholder_1"
quyyatshey_while_placeholder_2"
quyyatshey_while_placeholder_34
0quyyatshey_while_less_quyyatshey_strided_slice_1K
Gquyyatshey_while_quyyatshey_while_cond_1066672___redundant_placeholder0K
Gquyyatshey_while_quyyatshey_while_cond_1066672___redundant_placeholder1K
Gquyyatshey_while_quyyatshey_while_cond_1066672___redundant_placeholder2K
Gquyyatshey_while_quyyatshey_while_cond_1066672___redundant_placeholder3K
Gquyyatshey_while_quyyatshey_while_cond_1066672___redundant_placeholder4K
Gquyyatshey_while_quyyatshey_while_cond_1066672___redundant_placeholder5K
Gquyyatshey_while_quyyatshey_while_cond_1066672___redundant_placeholder6
quyyatshey_while_identity
§
quyyatshey/while/LessLessquyyatshey_while_placeholder0quyyatshey_while_less_quyyatshey_strided_slice_1*
T0*
_output_shapes
: 2
quyyatshey/while/Less~
quyyatshey/while/IdentityIdentityquyyatshey/while/Less:z:0*
T0
*
_output_shapes
: 2
quyyatshey/while/Identity"?
quyyatshey_while_identity"quyyatshey/while/Identity:output:0*(
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
while_body_1064274
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_nqcjuhnaut_1064298_0:	 -
while_nqcjuhnaut_1064300_0:	 )
while_nqcjuhnaut_1064302_0:	(
while_nqcjuhnaut_1064304_0: (
while_nqcjuhnaut_1064306_0: (
while_nqcjuhnaut_1064308_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_nqcjuhnaut_1064298:	 +
while_nqcjuhnaut_1064300:	 '
while_nqcjuhnaut_1064302:	&
while_nqcjuhnaut_1064304: &
while_nqcjuhnaut_1064306: &
while_nqcjuhnaut_1064308: ¢(while/nqcjuhnaut/StatefulPartitionedCallÃ
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
(while/nqcjuhnaut/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_nqcjuhnaut_1064298_0while_nqcjuhnaut_1064300_0while_nqcjuhnaut_1064302_0while_nqcjuhnaut_1064304_0while_nqcjuhnaut_1064306_0while_nqcjuhnaut_1064308_0*
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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_10641782*
(while/nqcjuhnaut/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/nqcjuhnaut/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/nqcjuhnaut/StatefulPartitionedCall:output:1)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/nqcjuhnaut/StatefulPartitionedCall:output:2)^while/nqcjuhnaut/StatefulPartitionedCall*
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
while_nqcjuhnaut_1064298while_nqcjuhnaut_1064298_0"6
while_nqcjuhnaut_1064300while_nqcjuhnaut_1064300_0"6
while_nqcjuhnaut_1064302while_nqcjuhnaut_1064302_0"6
while_nqcjuhnaut_1064304while_nqcjuhnaut_1064304_0"6
while_nqcjuhnaut_1064306while_nqcjuhnaut_1064306_0"6
while_nqcjuhnaut_1064308while_nqcjuhnaut_1064308_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/nqcjuhnaut/StatefulPartitionedCall(while/nqcjuhnaut/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_quyyatshey_layer_call_fn_1067700

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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10653742
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
,__inference_nqcjuhnaut_layer_call_fn_1068596

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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_10639912
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


í
while_cond_1063252
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063252___redundant_placeholder05
1while_while_cond_1063252___redundant_placeholder15
1while_while_cond_1063252___redundant_placeholder25
1while_while_cond_1063252___redundant_placeholder35
1while_while_cond_1063252___redundant_placeholder45
1while_while_cond_1063252___redundant_placeholder55
1while_while_cond_1063252___redundant_placeholder6
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
quyyatshey_while_cond_10662682
.quyyatshey_while_quyyatshey_while_loop_counter8
4quyyatshey_while_quyyatshey_while_maximum_iterations 
quyyatshey_while_placeholder"
quyyatshey_while_placeholder_1"
quyyatshey_while_placeholder_2"
quyyatshey_while_placeholder_34
0quyyatshey_while_less_quyyatshey_strided_slice_1K
Gquyyatshey_while_quyyatshey_while_cond_1066268___redundant_placeholder0K
Gquyyatshey_while_quyyatshey_while_cond_1066268___redundant_placeholder1K
Gquyyatshey_while_quyyatshey_while_cond_1066268___redundant_placeholder2K
Gquyyatshey_while_quyyatshey_while_cond_1066268___redundant_placeholder3K
Gquyyatshey_while_quyyatshey_while_cond_1066268___redundant_placeholder4K
Gquyyatshey_while_quyyatshey_while_cond_1066268___redundant_placeholder5K
Gquyyatshey_while_quyyatshey_while_cond_1066268___redundant_placeholder6
quyyatshey_while_identity
§
quyyatshey/while/LessLessquyyatshey_while_placeholder0quyyatshey_while_less_quyyatshey_strided_slice_1*
T0*
_output_shapes
: 2
quyyatshey/while/Less~
quyyatshey/while/IdentityIdentityquyyatshey/while/Less:z:0*
T0
*
_output_shapes
: 2
quyyatshey/while/Identity"?
quyyatshey_while_identity"quyyatshey/while/Identity:output:0*(
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
Z
Ë
 __inference__traced_save_1068847
file_prefix0
,savev2_umdyqemnpr_kernel_read_readvariableop.
*savev2_umdyqemnpr_bias_read_readvariableop0
,savev2_ycxcgxamrr_kernel_read_readvariableop.
*savev2_ycxcgxamrr_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_subsmtgotc_zdztiqrxwb_kernel_read_readvariableopE
Asavev2_subsmtgotc_zdztiqrxwb_recurrent_kernel_read_readvariableop9
5savev2_subsmtgotc_zdztiqrxwb_bias_read_readvariableopP
Lsavev2_subsmtgotc_zdztiqrxwb_input_gate_peephole_weights_read_readvariableopQ
Msavev2_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_subsmtgotc_zdztiqrxwb_output_gate_peephole_weights_read_readvariableop;
7savev2_quyyatshey_nqcjuhnaut_kernel_read_readvariableopE
Asavev2_quyyatshey_nqcjuhnaut_recurrent_kernel_read_readvariableop9
5savev2_quyyatshey_nqcjuhnaut_bias_read_readvariableopP
Lsavev2_quyyatshey_nqcjuhnaut_input_gate_peephole_weights_read_readvariableopQ
Msavev2_quyyatshey_nqcjuhnaut_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_quyyatshey_nqcjuhnaut_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_umdyqemnpr_kernel_rms_read_readvariableop:
6savev2_rmsprop_umdyqemnpr_bias_rms_read_readvariableop<
8savev2_rmsprop_ycxcgxamrr_kernel_rms_read_readvariableop:
6savev2_rmsprop_ycxcgxamrr_bias_rms_read_readvariableopG
Csavev2_rmsprop_subsmtgotc_zdztiqrxwb_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_subsmtgotc_zdztiqrxwb_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_subsmtgotc_zdztiqrxwb_bias_rms_read_readvariableop\
Xsavev2_rmsprop_subsmtgotc_zdztiqrxwb_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_subsmtgotc_zdztiqrxwb_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_quyyatshey_nqcjuhnaut_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_quyyatshey_nqcjuhnaut_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_quyyatshey_nqcjuhnaut_bias_rms_read_readvariableop\
Xsavev2_rmsprop_quyyatshey_nqcjuhnaut_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_quyyatshey_nqcjuhnaut_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_quyyatshey_nqcjuhnaut_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_umdyqemnpr_kernel_read_readvariableop*savev2_umdyqemnpr_bias_read_readvariableop,savev2_ycxcgxamrr_kernel_read_readvariableop*savev2_ycxcgxamrr_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_subsmtgotc_zdztiqrxwb_kernel_read_readvariableopAsavev2_subsmtgotc_zdztiqrxwb_recurrent_kernel_read_readvariableop5savev2_subsmtgotc_zdztiqrxwb_bias_read_readvariableopLsavev2_subsmtgotc_zdztiqrxwb_input_gate_peephole_weights_read_readvariableopMsavev2_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weights_read_readvariableopMsavev2_subsmtgotc_zdztiqrxwb_output_gate_peephole_weights_read_readvariableop7savev2_quyyatshey_nqcjuhnaut_kernel_read_readvariableopAsavev2_quyyatshey_nqcjuhnaut_recurrent_kernel_read_readvariableop5savev2_quyyatshey_nqcjuhnaut_bias_read_readvariableopLsavev2_quyyatshey_nqcjuhnaut_input_gate_peephole_weights_read_readvariableopMsavev2_quyyatshey_nqcjuhnaut_forget_gate_peephole_weights_read_readvariableopMsavev2_quyyatshey_nqcjuhnaut_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_umdyqemnpr_kernel_rms_read_readvariableop6savev2_rmsprop_umdyqemnpr_bias_rms_read_readvariableop8savev2_rmsprop_ycxcgxamrr_kernel_rms_read_readvariableop6savev2_rmsprop_ycxcgxamrr_bias_rms_read_readvariableopCsavev2_rmsprop_subsmtgotc_zdztiqrxwb_kernel_rms_read_readvariableopMsavev2_rmsprop_subsmtgotc_zdztiqrxwb_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_subsmtgotc_zdztiqrxwb_bias_rms_read_readvariableopXsavev2_rmsprop_subsmtgotc_zdztiqrxwb_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_subsmtgotc_zdztiqrxwb_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_quyyatshey_nqcjuhnaut_kernel_rms_read_readvariableopMsavev2_rmsprop_quyyatshey_nqcjuhnaut_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_quyyatshey_nqcjuhnaut_bias_rms_read_readvariableopXsavev2_rmsprop_quyyatshey_nqcjuhnaut_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_quyyatshey_nqcjuhnaut_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_quyyatshey_nqcjuhnaut_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
while_cond_1064273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1064273___redundant_placeholder05
1while_while_cond_1064273___redundant_placeholder15
1while_while_cond_1064273___redundant_placeholder25
1while_while_cond_1064273___redundant_placeholder35
1while_while_cond_1064273___redundant_placeholder45
1while_while_cond_1064273___redundant_placeholder55
1while_while_cond_1064273___redundant_placeholder6
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067272
inputs_0<
)zdztiqrxwb_matmul_readvariableop_resource:	>
+zdztiqrxwb_matmul_1_readvariableop_resource:	 9
*zdztiqrxwb_biasadd_readvariableop_resource:	0
"zdztiqrxwb_readvariableop_resource: 2
$zdztiqrxwb_readvariableop_1_resource: 2
$zdztiqrxwb_readvariableop_2_resource: 
identity¢while¢!zdztiqrxwb/BiasAdd/ReadVariableOp¢ zdztiqrxwb/MatMul/ReadVariableOp¢"zdztiqrxwb/MatMul_1/ReadVariableOp¢zdztiqrxwb/ReadVariableOp¢zdztiqrxwb/ReadVariableOp_1¢zdztiqrxwb/ReadVariableOp_2F
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
 zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp)zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 zdztiqrxwb/MatMul/ReadVariableOp§
zdztiqrxwb/MatMulMatMulstrided_slice_2:output:0(zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMulµ
"zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp+zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"zdztiqrxwb/MatMul_1/ReadVariableOp£
zdztiqrxwb/MatMul_1MatMulzeros:output:0*zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMul_1
zdztiqrxwb/addAddV2zdztiqrxwb/MatMul:product:0zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/add®
!zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp*zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!zdztiqrxwb/BiasAdd/ReadVariableOp¥
zdztiqrxwb/BiasAddBiasAddzdztiqrxwb/add:z:0)zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/BiasAddz
zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
zdztiqrxwb/split/split_dimë
zdztiqrxwb/splitSplit#zdztiqrxwb/split/split_dim:output:0zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zdztiqrxwb/split
zdztiqrxwb/ReadVariableOpReadVariableOp"zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp
zdztiqrxwb/mulMul!zdztiqrxwb/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul
zdztiqrxwb/add_1AddV2zdztiqrxwb/split:output:0zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_1{
zdztiqrxwb/SigmoidSigmoidzdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid
zdztiqrxwb/ReadVariableOp_1ReadVariableOp$zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_1
zdztiqrxwb/mul_1Mul#zdztiqrxwb/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_1
zdztiqrxwb/add_2AddV2zdztiqrxwb/split:output:1zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_2
zdztiqrxwb/Sigmoid_1Sigmoidzdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_1
zdztiqrxwb/mul_2Mulzdztiqrxwb/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_2w
zdztiqrxwb/TanhTanhzdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh
zdztiqrxwb/mul_3Mulzdztiqrxwb/Sigmoid:y:0zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_3
zdztiqrxwb/add_3AddV2zdztiqrxwb/mul_2:z:0zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_3
zdztiqrxwb/ReadVariableOp_2ReadVariableOp$zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_2
zdztiqrxwb/mul_4Mul#zdztiqrxwb/ReadVariableOp_2:value:0zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_4
zdztiqrxwb/add_4AddV2zdztiqrxwb/split:output:3zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_4
zdztiqrxwb/Sigmoid_2Sigmoidzdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_2v
zdztiqrxwb/Tanh_1Tanhzdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh_1
zdztiqrxwb/mul_5Mulzdztiqrxwb/Sigmoid_2:y:0zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)zdztiqrxwb_matmul_readvariableop_resource+zdztiqrxwb_matmul_1_readvariableop_resource*zdztiqrxwb_biasadd_readvariableop_resource"zdztiqrxwb_readvariableop_resource$zdztiqrxwb_readvariableop_1_resource$zdztiqrxwb_readvariableop_2_resource*
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
while_body_1067171*
condR
while_cond_1067170*Q
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
IdentityIdentitytranspose_1:y:0^while"^zdztiqrxwb/BiasAdd/ReadVariableOp!^zdztiqrxwb/MatMul/ReadVariableOp#^zdztiqrxwb/MatMul_1/ReadVariableOp^zdztiqrxwb/ReadVariableOp^zdztiqrxwb/ReadVariableOp_1^zdztiqrxwb/ReadVariableOp_2*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2F
!zdztiqrxwb/BiasAdd/ReadVariableOp!zdztiqrxwb/BiasAdd/ReadVariableOp2D
 zdztiqrxwb/MatMul/ReadVariableOp zdztiqrxwb/MatMul/ReadVariableOp2H
"zdztiqrxwb/MatMul_1/ReadVariableOp"zdztiqrxwb/MatMul_1/ReadVariableOp26
zdztiqrxwb/ReadVariableOpzdztiqrxwb/ReadVariableOp2:
zdztiqrxwb/ReadVariableOp_1zdztiqrxwb/ReadVariableOp_12:
zdztiqrxwb/ReadVariableOp_2zdztiqrxwb/ReadVariableOp_2:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

À
,__inference_zdztiqrxwb_layer_call_fn_1068485

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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_10634202
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
while_cond_1067170
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1067170___redundant_placeholder05
1while_while_cond_1067170___redundant_placeholder15
1while_while_cond_1067170___redundant_placeholder25
1while_while_cond_1067170___redundant_placeholder35
1while_while_cond_1067170___redundant_placeholder45
1while_while_cond_1067170___redundant_placeholder55
1while_while_cond_1067170___redundant_placeholder6
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1065588

inputs<
)zdztiqrxwb_matmul_readvariableop_resource:	>
+zdztiqrxwb_matmul_1_readvariableop_resource:	 9
*zdztiqrxwb_biasadd_readvariableop_resource:	0
"zdztiqrxwb_readvariableop_resource: 2
$zdztiqrxwb_readvariableop_1_resource: 2
$zdztiqrxwb_readvariableop_2_resource: 
identity¢while¢!zdztiqrxwb/BiasAdd/ReadVariableOp¢ zdztiqrxwb/MatMul/ReadVariableOp¢"zdztiqrxwb/MatMul_1/ReadVariableOp¢zdztiqrxwb/ReadVariableOp¢zdztiqrxwb/ReadVariableOp_1¢zdztiqrxwb/ReadVariableOp_2D
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
 zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp)zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 zdztiqrxwb/MatMul/ReadVariableOp§
zdztiqrxwb/MatMulMatMulstrided_slice_2:output:0(zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMulµ
"zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp+zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"zdztiqrxwb/MatMul_1/ReadVariableOp£
zdztiqrxwb/MatMul_1MatMulzeros:output:0*zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMul_1
zdztiqrxwb/addAddV2zdztiqrxwb/MatMul:product:0zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/add®
!zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp*zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!zdztiqrxwb/BiasAdd/ReadVariableOp¥
zdztiqrxwb/BiasAddBiasAddzdztiqrxwb/add:z:0)zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/BiasAddz
zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
zdztiqrxwb/split/split_dimë
zdztiqrxwb/splitSplit#zdztiqrxwb/split/split_dim:output:0zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zdztiqrxwb/split
zdztiqrxwb/ReadVariableOpReadVariableOp"zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp
zdztiqrxwb/mulMul!zdztiqrxwb/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul
zdztiqrxwb/add_1AddV2zdztiqrxwb/split:output:0zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_1{
zdztiqrxwb/SigmoidSigmoidzdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid
zdztiqrxwb/ReadVariableOp_1ReadVariableOp$zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_1
zdztiqrxwb/mul_1Mul#zdztiqrxwb/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_1
zdztiqrxwb/add_2AddV2zdztiqrxwb/split:output:1zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_2
zdztiqrxwb/Sigmoid_1Sigmoidzdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_1
zdztiqrxwb/mul_2Mulzdztiqrxwb/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_2w
zdztiqrxwb/TanhTanhzdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh
zdztiqrxwb/mul_3Mulzdztiqrxwb/Sigmoid:y:0zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_3
zdztiqrxwb/add_3AddV2zdztiqrxwb/mul_2:z:0zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_3
zdztiqrxwb/ReadVariableOp_2ReadVariableOp$zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_2
zdztiqrxwb/mul_4Mul#zdztiqrxwb/ReadVariableOp_2:value:0zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_4
zdztiqrxwb/add_4AddV2zdztiqrxwb/split:output:3zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_4
zdztiqrxwb/Sigmoid_2Sigmoidzdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_2v
zdztiqrxwb/Tanh_1Tanhzdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh_1
zdztiqrxwb/mul_5Mulzdztiqrxwb/Sigmoid_2:y:0zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)zdztiqrxwb_matmul_readvariableop_resource+zdztiqrxwb_matmul_1_readvariableop_resource*zdztiqrxwb_biasadd_readvariableop_resource"zdztiqrxwb_readvariableop_resource$zdztiqrxwb_readvariableop_1_resource$zdztiqrxwb_readvariableop_2_resource*
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
while_body_1065487*
condR
while_cond_1065486*Q
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
IdentityIdentitytranspose_1:y:0^while"^zdztiqrxwb/BiasAdd/ReadVariableOp!^zdztiqrxwb/MatMul/ReadVariableOp#^zdztiqrxwb/MatMul_1/ReadVariableOp^zdztiqrxwb/ReadVariableOp^zdztiqrxwb/ReadVariableOp_1^zdztiqrxwb/ReadVariableOp_2*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2F
!zdztiqrxwb/BiasAdd/ReadVariableOp!zdztiqrxwb/BiasAdd/ReadVariableOp2D
 zdztiqrxwb/MatMul/ReadVariableOp zdztiqrxwb/MatMul/ReadVariableOp2H
"zdztiqrxwb/MatMul_1/ReadVariableOp"zdztiqrxwb/MatMul_1/ReadVariableOp26
zdztiqrxwb/ReadVariableOpzdztiqrxwb/ReadVariableOp2:
zdztiqrxwb/ReadVariableOp_1zdztiqrxwb/ReadVariableOp_12:
zdztiqrxwb/ReadVariableOp_2zdztiqrxwb/ReadVariableOp_2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
p
Ê
quyyatshey_while_body_10666732
.quyyatshey_while_quyyatshey_while_loop_counter8
4quyyatshey_while_quyyatshey_while_maximum_iterations 
quyyatshey_while_placeholder"
quyyatshey_while_placeholder_1"
quyyatshey_while_placeholder_2"
quyyatshey_while_placeholder_31
-quyyatshey_while_quyyatshey_strided_slice_1_0m
iquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0O
<quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0:	 Q
>quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 L
=quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0:	C
5quyyatshey_while_nqcjuhnaut_readvariableop_resource_0: E
7quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0: E
7quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0: 
quyyatshey_while_identity
quyyatshey_while_identity_1
quyyatshey_while_identity_2
quyyatshey_while_identity_3
quyyatshey_while_identity_4
quyyatshey_while_identity_5/
+quyyatshey_while_quyyatshey_strided_slice_1k
gquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensorM
:quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource:	 O
<quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource:	 J
;quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource:	A
3quyyatshey_while_nqcjuhnaut_readvariableop_resource: C
5quyyatshey_while_nqcjuhnaut_readvariableop_1_resource: C
5quyyatshey_while_nqcjuhnaut_readvariableop_2_resource: ¢2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp¢1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp¢3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp¢*quyyatshey/while/nqcjuhnaut/ReadVariableOp¢,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1¢,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2Ù
Bquyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bquyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shape
4quyyatshey/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0quyyatshey_while_placeholderKquyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4quyyatshey/while/TensorArrayV2Read/TensorListGetItemä
1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp<quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOpý
"quyyatshey/while/nqcjuhnaut/MatMulMatMul;quyyatshey/while/TensorArrayV2Read/TensorListGetItem:item:09quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"quyyatshey/while/nqcjuhnaut/MatMulê
3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp>quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOpæ
$quyyatshey/while/nqcjuhnaut/MatMul_1MatMulquyyatshey_while_placeholder_2;quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$quyyatshey/while/nqcjuhnaut/MatMul_1Ü
quyyatshey/while/nqcjuhnaut/addAddV2,quyyatshey/while/nqcjuhnaut/MatMul:product:0.quyyatshey/while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
quyyatshey/while/nqcjuhnaut/addã
2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp=quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOpé
#quyyatshey/while/nqcjuhnaut/BiasAddBiasAdd#quyyatshey/while/nqcjuhnaut/add:z:0:quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#quyyatshey/while/nqcjuhnaut/BiasAdd
+quyyatshey/while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+quyyatshey/while/nqcjuhnaut/split/split_dim¯
!quyyatshey/while/nqcjuhnaut/splitSplit4quyyatshey/while/nqcjuhnaut/split/split_dim:output:0,quyyatshey/while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!quyyatshey/while/nqcjuhnaut/splitÊ
*quyyatshey/while/nqcjuhnaut/ReadVariableOpReadVariableOp5quyyatshey_while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*quyyatshey/while/nqcjuhnaut/ReadVariableOpÏ
quyyatshey/while/nqcjuhnaut/mulMul2quyyatshey/while/nqcjuhnaut/ReadVariableOp:value:0quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
quyyatshey/while/nqcjuhnaut/mulÒ
!quyyatshey/while/nqcjuhnaut/add_1AddV2*quyyatshey/while/nqcjuhnaut/split:output:0#quyyatshey/while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_1®
#quyyatshey/while/nqcjuhnaut/SigmoidSigmoid%quyyatshey/while/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#quyyatshey/while/nqcjuhnaut/SigmoidÐ
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp7quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1Õ
!quyyatshey/while/nqcjuhnaut/mul_1Mul4quyyatshey/while/nqcjuhnaut/ReadVariableOp_1:value:0quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_1Ô
!quyyatshey/while/nqcjuhnaut/add_2AddV2*quyyatshey/while/nqcjuhnaut/split:output:1%quyyatshey/while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_2²
%quyyatshey/while/nqcjuhnaut/Sigmoid_1Sigmoid%quyyatshey/while/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%quyyatshey/while/nqcjuhnaut/Sigmoid_1Ê
!quyyatshey/while/nqcjuhnaut/mul_2Mul)quyyatshey/while/nqcjuhnaut/Sigmoid_1:y:0quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_2ª
 quyyatshey/while/nqcjuhnaut/TanhTanh*quyyatshey/while/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 quyyatshey/while/nqcjuhnaut/TanhÎ
!quyyatshey/while/nqcjuhnaut/mul_3Mul'quyyatshey/while/nqcjuhnaut/Sigmoid:y:0$quyyatshey/while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_3Ï
!quyyatshey/while/nqcjuhnaut/add_3AddV2%quyyatshey/while/nqcjuhnaut/mul_2:z:0%quyyatshey/while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_3Ð
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp7quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2Ü
!quyyatshey/while/nqcjuhnaut/mul_4Mul4quyyatshey/while/nqcjuhnaut/ReadVariableOp_2:value:0%quyyatshey/while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_4Ô
!quyyatshey/while/nqcjuhnaut/add_4AddV2*quyyatshey/while/nqcjuhnaut/split:output:3%quyyatshey/while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/add_4²
%quyyatshey/while/nqcjuhnaut/Sigmoid_2Sigmoid%quyyatshey/while/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%quyyatshey/while/nqcjuhnaut/Sigmoid_2©
"quyyatshey/while/nqcjuhnaut/Tanh_1Tanh%quyyatshey/while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"quyyatshey/while/nqcjuhnaut/Tanh_1Ò
!quyyatshey/while/nqcjuhnaut/mul_5Mul)quyyatshey/while/nqcjuhnaut/Sigmoid_2:y:0&quyyatshey/while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!quyyatshey/while/nqcjuhnaut/mul_5
5quyyatshey/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemquyyatshey_while_placeholder_1quyyatshey_while_placeholder%quyyatshey/while/nqcjuhnaut/mul_5:z:0*
_output_shapes
: *
element_dtype027
5quyyatshey/while/TensorArrayV2Write/TensorListSetItemr
quyyatshey/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
quyyatshey/while/add/y
quyyatshey/while/addAddV2quyyatshey_while_placeholderquyyatshey/while/add/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/while/addv
quyyatshey/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
quyyatshey/while/add_1/y­
quyyatshey/while/add_1AddV2.quyyatshey_while_quyyatshey_while_loop_counter!quyyatshey/while/add_1/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/while/add_1©
quyyatshey/while/IdentityIdentityquyyatshey/while/add_1:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/IdentityÇ
quyyatshey/while/Identity_1Identity4quyyatshey_while_quyyatshey_while_maximum_iterations3^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/Identity_1«
quyyatshey/while/Identity_2Identityquyyatshey/while/add:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/Identity_2Ø
quyyatshey/while/Identity_3IdentityEquyyatshey/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
quyyatshey/while/Identity_3É
quyyatshey/while/Identity_4Identity%quyyatshey/while/nqcjuhnaut/mul_5:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/while/Identity_4É
quyyatshey/while/Identity_5Identity%quyyatshey/while/nqcjuhnaut/add_3:z:03^quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2^quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp4^quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp+^quyyatshey/while/nqcjuhnaut/ReadVariableOp-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_1-^quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/while/Identity_5"?
quyyatshey_while_identity"quyyatshey/while/Identity:output:0"C
quyyatshey_while_identity_1$quyyatshey/while/Identity_1:output:0"C
quyyatshey_while_identity_2$quyyatshey/while/Identity_2:output:0"C
quyyatshey_while_identity_3$quyyatshey/while/Identity_3:output:0"C
quyyatshey_while_identity_4$quyyatshey/while/Identity_4:output:0"C
quyyatshey_while_identity_5$quyyatshey/while/Identity_5:output:0"|
;quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource=quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0"~
<quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource>quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0"z
:quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource<quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0"p
5quyyatshey_while_nqcjuhnaut_readvariableop_1_resource7quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0"p
5quyyatshey_while_nqcjuhnaut_readvariableop_2_resource7quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0"l
3quyyatshey_while_nqcjuhnaut_readvariableop_resource5quyyatshey_while_nqcjuhnaut_readvariableop_resource_0"\
+quyyatshey_while_quyyatshey_strided_slice_1-quyyatshey_while_quyyatshey_strided_slice_1_0"Ô
gquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensoriquyyatshey_while_tensorarrayv2read_tensorlistgetitem_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2f
1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp1quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp2j
3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp3quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp2X
*quyyatshey/while/nqcjuhnaut/ReadVariableOp*quyyatshey/while/nqcjuhnaut/ReadVariableOp2\
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_1,quyyatshey/while/nqcjuhnaut/ReadVariableOp_12\
,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2,quyyatshey/while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
subsmtgotc_while_body_10660932
.subsmtgotc_while_subsmtgotc_while_loop_counter8
4subsmtgotc_while_subsmtgotc_while_maximum_iterations 
subsmtgotc_while_placeholder"
subsmtgotc_while_placeholder_1"
subsmtgotc_while_placeholder_2"
subsmtgotc_while_placeholder_31
-subsmtgotc_while_subsmtgotc_strided_slice_1_0m
isubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0O
<subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0:	Q
>subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 L
=subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0:	C
5subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0: E
7subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0: E
7subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0: 
subsmtgotc_while_identity
subsmtgotc_while_identity_1
subsmtgotc_while_identity_2
subsmtgotc_while_identity_3
subsmtgotc_while_identity_4
subsmtgotc_while_identity_5/
+subsmtgotc_while_subsmtgotc_strided_slice_1k
gsubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensorM
:subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource:	O
<subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource:	 J
;subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource:	A
3subsmtgotc_while_zdztiqrxwb_readvariableop_resource: C
5subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource: C
5subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource: ¢2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp¢1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp¢3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp¢*subsmtgotc/while/zdztiqrxwb/ReadVariableOp¢,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1¢,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2Ù
Bsubsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bsubsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4subsmtgotc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemisubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0subsmtgotc_while_placeholderKsubsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4subsmtgotc/while/TensorArrayV2Read/TensorListGetItemä
1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp<subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOpý
"subsmtgotc/while/zdztiqrxwb/MatMulMatMul;subsmtgotc/while/TensorArrayV2Read/TensorListGetItem:item:09subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"subsmtgotc/while/zdztiqrxwb/MatMulê
3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp>subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOpæ
$subsmtgotc/while/zdztiqrxwb/MatMul_1MatMulsubsmtgotc_while_placeholder_2;subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$subsmtgotc/while/zdztiqrxwb/MatMul_1Ü
subsmtgotc/while/zdztiqrxwb/addAddV2,subsmtgotc/while/zdztiqrxwb/MatMul:product:0.subsmtgotc/while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
subsmtgotc/while/zdztiqrxwb/addã
2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp=subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOpé
#subsmtgotc/while/zdztiqrxwb/BiasAddBiasAdd#subsmtgotc/while/zdztiqrxwb/add:z:0:subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#subsmtgotc/while/zdztiqrxwb/BiasAdd
+subsmtgotc/while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+subsmtgotc/while/zdztiqrxwb/split/split_dim¯
!subsmtgotc/while/zdztiqrxwb/splitSplit4subsmtgotc/while/zdztiqrxwb/split/split_dim:output:0,subsmtgotc/while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!subsmtgotc/while/zdztiqrxwb/splitÊ
*subsmtgotc/while/zdztiqrxwb/ReadVariableOpReadVariableOp5subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*subsmtgotc/while/zdztiqrxwb/ReadVariableOpÏ
subsmtgotc/while/zdztiqrxwb/mulMul2subsmtgotc/while/zdztiqrxwb/ReadVariableOp:value:0subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
subsmtgotc/while/zdztiqrxwb/mulÒ
!subsmtgotc/while/zdztiqrxwb/add_1AddV2*subsmtgotc/while/zdztiqrxwb/split:output:0#subsmtgotc/while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_1®
#subsmtgotc/while/zdztiqrxwb/SigmoidSigmoid%subsmtgotc/while/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#subsmtgotc/while/zdztiqrxwb/SigmoidÐ
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp7subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1Õ
!subsmtgotc/while/zdztiqrxwb/mul_1Mul4subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1:value:0subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_1Ô
!subsmtgotc/while/zdztiqrxwb/add_2AddV2*subsmtgotc/while/zdztiqrxwb/split:output:1%subsmtgotc/while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_2²
%subsmtgotc/while/zdztiqrxwb/Sigmoid_1Sigmoid%subsmtgotc/while/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%subsmtgotc/while/zdztiqrxwb/Sigmoid_1Ê
!subsmtgotc/while/zdztiqrxwb/mul_2Mul)subsmtgotc/while/zdztiqrxwb/Sigmoid_1:y:0subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_2ª
 subsmtgotc/while/zdztiqrxwb/TanhTanh*subsmtgotc/while/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 subsmtgotc/while/zdztiqrxwb/TanhÎ
!subsmtgotc/while/zdztiqrxwb/mul_3Mul'subsmtgotc/while/zdztiqrxwb/Sigmoid:y:0$subsmtgotc/while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_3Ï
!subsmtgotc/while/zdztiqrxwb/add_3AddV2%subsmtgotc/while/zdztiqrxwb/mul_2:z:0%subsmtgotc/while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_3Ð
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp7subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2Ü
!subsmtgotc/while/zdztiqrxwb/mul_4Mul4subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2:value:0%subsmtgotc/while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_4Ô
!subsmtgotc/while/zdztiqrxwb/add_4AddV2*subsmtgotc/while/zdztiqrxwb/split:output:3%subsmtgotc/while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_4²
%subsmtgotc/while/zdztiqrxwb/Sigmoid_2Sigmoid%subsmtgotc/while/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%subsmtgotc/while/zdztiqrxwb/Sigmoid_2©
"subsmtgotc/while/zdztiqrxwb/Tanh_1Tanh%subsmtgotc/while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"subsmtgotc/while/zdztiqrxwb/Tanh_1Ò
!subsmtgotc/while/zdztiqrxwb/mul_5Mul)subsmtgotc/while/zdztiqrxwb/Sigmoid_2:y:0&subsmtgotc/while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_5
5subsmtgotc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsubsmtgotc_while_placeholder_1subsmtgotc_while_placeholder%subsmtgotc/while/zdztiqrxwb/mul_5:z:0*
_output_shapes
: *
element_dtype027
5subsmtgotc/while/TensorArrayV2Write/TensorListSetItemr
subsmtgotc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
subsmtgotc/while/add/y
subsmtgotc/while/addAddV2subsmtgotc_while_placeholdersubsmtgotc/while/add/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/while/addv
subsmtgotc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
subsmtgotc/while/add_1/y­
subsmtgotc/while/add_1AddV2.subsmtgotc_while_subsmtgotc_while_loop_counter!subsmtgotc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/while/add_1©
subsmtgotc/while/IdentityIdentitysubsmtgotc/while/add_1:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/IdentityÇ
subsmtgotc/while/Identity_1Identity4subsmtgotc_while_subsmtgotc_while_maximum_iterations3^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/Identity_1«
subsmtgotc/while/Identity_2Identitysubsmtgotc/while/add:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/Identity_2Ø
subsmtgotc/while/Identity_3IdentityEsubsmtgotc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/Identity_3É
subsmtgotc/while/Identity_4Identity%subsmtgotc/while/zdztiqrxwb/mul_5:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/while/Identity_4É
subsmtgotc/while/Identity_5Identity%subsmtgotc/while/zdztiqrxwb/add_3:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/while/Identity_5"?
subsmtgotc_while_identity"subsmtgotc/while/Identity:output:0"C
subsmtgotc_while_identity_1$subsmtgotc/while/Identity_1:output:0"C
subsmtgotc_while_identity_2$subsmtgotc/while/Identity_2:output:0"C
subsmtgotc_while_identity_3$subsmtgotc/while/Identity_3:output:0"C
subsmtgotc_while_identity_4$subsmtgotc/while/Identity_4:output:0"C
subsmtgotc_while_identity_5$subsmtgotc/while/Identity_5:output:0"\
+subsmtgotc_while_subsmtgotc_strided_slice_1-subsmtgotc_while_subsmtgotc_strided_slice_1_0"Ô
gsubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensorisubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0"|
;subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource=subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0"~
<subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource>subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0"z
:subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource<subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0"p
5subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource7subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0"p
5subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource7subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0"l
3subsmtgotc_while_zdztiqrxwb_readvariableop_resource5subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2f
1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp2j
3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp2X
*subsmtgotc/while/zdztiqrxwb/ReadVariableOp*subsmtgotc/while/zdztiqrxwb/ReadVariableOp2\
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_12\
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1067880
inputs_0<
)nqcjuhnaut_matmul_readvariableop_resource:	 >
+nqcjuhnaut_matmul_1_readvariableop_resource:	 9
*nqcjuhnaut_biasadd_readvariableop_resource:	0
"nqcjuhnaut_readvariableop_resource: 2
$nqcjuhnaut_readvariableop_1_resource: 2
$nqcjuhnaut_readvariableop_2_resource: 
identity¢!nqcjuhnaut/BiasAdd/ReadVariableOp¢ nqcjuhnaut/MatMul/ReadVariableOp¢"nqcjuhnaut/MatMul_1/ReadVariableOp¢nqcjuhnaut/ReadVariableOp¢nqcjuhnaut/ReadVariableOp_1¢nqcjuhnaut/ReadVariableOp_2¢whileF
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
 nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp)nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 nqcjuhnaut/MatMul/ReadVariableOp§
nqcjuhnaut/MatMulMatMulstrided_slice_2:output:0(nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMulµ
"nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp+nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"nqcjuhnaut/MatMul_1/ReadVariableOp£
nqcjuhnaut/MatMul_1MatMulzeros:output:0*nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMul_1
nqcjuhnaut/addAddV2nqcjuhnaut/MatMul:product:0nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/add®
!nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp*nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!nqcjuhnaut/BiasAdd/ReadVariableOp¥
nqcjuhnaut/BiasAddBiasAddnqcjuhnaut/add:z:0)nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/BiasAddz
nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
nqcjuhnaut/split/split_dimë
nqcjuhnaut/splitSplit#nqcjuhnaut/split/split_dim:output:0nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nqcjuhnaut/split
nqcjuhnaut/ReadVariableOpReadVariableOp"nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp
nqcjuhnaut/mulMul!nqcjuhnaut/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul
nqcjuhnaut/add_1AddV2nqcjuhnaut/split:output:0nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_1{
nqcjuhnaut/SigmoidSigmoidnqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid
nqcjuhnaut/ReadVariableOp_1ReadVariableOp$nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_1
nqcjuhnaut/mul_1Mul#nqcjuhnaut/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_1
nqcjuhnaut/add_2AddV2nqcjuhnaut/split:output:1nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_2
nqcjuhnaut/Sigmoid_1Sigmoidnqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_1
nqcjuhnaut/mul_2Mulnqcjuhnaut/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_2w
nqcjuhnaut/TanhTanhnqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh
nqcjuhnaut/mul_3Mulnqcjuhnaut/Sigmoid:y:0nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_3
nqcjuhnaut/add_3AddV2nqcjuhnaut/mul_2:z:0nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_3
nqcjuhnaut/ReadVariableOp_2ReadVariableOp$nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_2
nqcjuhnaut/mul_4Mul#nqcjuhnaut/ReadVariableOp_2:value:0nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_4
nqcjuhnaut/add_4AddV2nqcjuhnaut/split:output:3nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_4
nqcjuhnaut/Sigmoid_2Sigmoidnqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_2v
nqcjuhnaut/Tanh_1Tanhnqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh_1
nqcjuhnaut/mul_5Mulnqcjuhnaut/Sigmoid_2:y:0nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)nqcjuhnaut_matmul_readvariableop_resource+nqcjuhnaut_matmul_1_readvariableop_resource*nqcjuhnaut_biasadd_readvariableop_resource"nqcjuhnaut_readvariableop_resource$nqcjuhnaut_readvariableop_1_resource$nqcjuhnaut_readvariableop_2_resource*
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
while_body_1067779*
condR
while_cond_1067778*Q
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
IdentityIdentitystrided_slice_3:output:0"^nqcjuhnaut/BiasAdd/ReadVariableOp!^nqcjuhnaut/MatMul/ReadVariableOp#^nqcjuhnaut/MatMul_1/ReadVariableOp^nqcjuhnaut/ReadVariableOp^nqcjuhnaut/ReadVariableOp_1^nqcjuhnaut/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!nqcjuhnaut/BiasAdd/ReadVariableOp!nqcjuhnaut/BiasAdd/ReadVariableOp2D
 nqcjuhnaut/MatMul/ReadVariableOp nqcjuhnaut/MatMul/ReadVariableOp2H
"nqcjuhnaut/MatMul_1/ReadVariableOp"nqcjuhnaut/MatMul_1/ReadVariableOp26
nqcjuhnaut/ReadVariableOpnqcjuhnaut/ReadVariableOp2:
nqcjuhnaut/ReadVariableOp_1nqcjuhnaut/ReadVariableOp_12:
nqcjuhnaut/ReadVariableOp_2nqcjuhnaut/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
¯F
ê
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1064091

inputs%
nqcjuhnaut_1063992:	 %
nqcjuhnaut_1063994:	 !
nqcjuhnaut_1063996:	 
nqcjuhnaut_1063998:  
nqcjuhnaut_1064000:  
nqcjuhnaut_1064002: 
identity¢"nqcjuhnaut/StatefulPartitionedCall¢whileD
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
"nqcjuhnaut/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0nqcjuhnaut_1063992nqcjuhnaut_1063994nqcjuhnaut_1063996nqcjuhnaut_1063998nqcjuhnaut_1064000nqcjuhnaut_1064002*
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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_10639912$
"nqcjuhnaut/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0nqcjuhnaut_1063992nqcjuhnaut_1063994nqcjuhnaut_1063996nqcjuhnaut_1063998nqcjuhnaut_1064000nqcjuhnaut_1064002*
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
while_body_1064011*
condR
while_cond_1064010*Q
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
IdentityIdentitystrided_slice_3:output:0#^nqcjuhnaut/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"nqcjuhnaut/StatefulPartitionedCall"nqcjuhnaut/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
àY

while_body_1068139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_nqcjuhnaut_matmul_readvariableop_resource_0:	 F
3while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 A
2while_nqcjuhnaut_biasadd_readvariableop_resource_0:	8
*while_nqcjuhnaut_readvariableop_resource_0: :
,while_nqcjuhnaut_readvariableop_1_resource_0: :
,while_nqcjuhnaut_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_nqcjuhnaut_matmul_readvariableop_resource:	 D
1while_nqcjuhnaut_matmul_1_readvariableop_resource:	 ?
0while_nqcjuhnaut_biasadd_readvariableop_resource:	6
(while_nqcjuhnaut_readvariableop_resource: 8
*while_nqcjuhnaut_readvariableop_1_resource: 8
*while_nqcjuhnaut_readvariableop_2_resource: ¢'while/nqcjuhnaut/BiasAdd/ReadVariableOp¢&while/nqcjuhnaut/MatMul/ReadVariableOp¢(while/nqcjuhnaut/MatMul_1/ReadVariableOp¢while/nqcjuhnaut/ReadVariableOp¢!while/nqcjuhnaut/ReadVariableOp_1¢!while/nqcjuhnaut/ReadVariableOp_2Ã
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
&while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp1while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/nqcjuhnaut/MatMul/ReadVariableOpÑ
while/nqcjuhnaut/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMulÉ
(while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp3while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/nqcjuhnaut/MatMul_1/ReadVariableOpº
while/nqcjuhnaut/MatMul_1MatMulwhile_placeholder_20while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMul_1°
while/nqcjuhnaut/addAddV2!while/nqcjuhnaut/MatMul:product:0#while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/addÂ
'while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp2while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/nqcjuhnaut/BiasAdd/ReadVariableOp½
while/nqcjuhnaut/BiasAddBiasAddwhile/nqcjuhnaut/add:z:0/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/BiasAdd
 while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/nqcjuhnaut/split/split_dim
while/nqcjuhnaut/splitSplit)while/nqcjuhnaut/split/split_dim:output:0!while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/nqcjuhnaut/split©
while/nqcjuhnaut/ReadVariableOpReadVariableOp*while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/nqcjuhnaut/ReadVariableOp£
while/nqcjuhnaut/mulMul'while/nqcjuhnaut/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul¦
while/nqcjuhnaut/add_1AddV2while/nqcjuhnaut/split:output:0while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_1
while/nqcjuhnaut/SigmoidSigmoidwhile/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid¯
!while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp,while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_1©
while/nqcjuhnaut/mul_1Mul)while/nqcjuhnaut/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_1¨
while/nqcjuhnaut/add_2AddV2while/nqcjuhnaut/split:output:1while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_2
while/nqcjuhnaut/Sigmoid_1Sigmoidwhile/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_1
while/nqcjuhnaut/mul_2Mulwhile/nqcjuhnaut/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_2
while/nqcjuhnaut/TanhTanhwhile/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh¢
while/nqcjuhnaut/mul_3Mulwhile/nqcjuhnaut/Sigmoid:y:0while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_3£
while/nqcjuhnaut/add_3AddV2while/nqcjuhnaut/mul_2:z:0while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_3¯
!while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp,while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_2°
while/nqcjuhnaut/mul_4Mul)while/nqcjuhnaut/ReadVariableOp_2:value:0while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_4¨
while/nqcjuhnaut/add_4AddV2while/nqcjuhnaut/split:output:3while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_4
while/nqcjuhnaut/Sigmoid_2Sigmoidwhile/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_2
while/nqcjuhnaut/Tanh_1Tanhwhile/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh_1¦
while/nqcjuhnaut/mul_5Mulwhile/nqcjuhnaut/Sigmoid_2:y:0while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/nqcjuhnaut/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/nqcjuhnaut/mul_5:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/nqcjuhnaut/add_3:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
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
0while_nqcjuhnaut_biasadd_readvariableop_resource2while_nqcjuhnaut_biasadd_readvariableop_resource_0"h
1while_nqcjuhnaut_matmul_1_readvariableop_resource3while_nqcjuhnaut_matmul_1_readvariableop_resource_0"d
/while_nqcjuhnaut_matmul_readvariableop_resource1while_nqcjuhnaut_matmul_readvariableop_resource_0"Z
*while_nqcjuhnaut_readvariableop_1_resource,while_nqcjuhnaut_readvariableop_1_resource_0"Z
*while_nqcjuhnaut_readvariableop_2_resource,while_nqcjuhnaut_readvariableop_2_resource_0"V
(while_nqcjuhnaut_readvariableop_resource*while_nqcjuhnaut_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/nqcjuhnaut/BiasAdd/ReadVariableOp'while/nqcjuhnaut/BiasAdd/ReadVariableOp2P
&while/nqcjuhnaut/MatMul/ReadVariableOp&while/nqcjuhnaut/MatMul/ReadVariableOp2T
(while/nqcjuhnaut/MatMul_1/ReadVariableOp(while/nqcjuhnaut/MatMul_1/ReadVariableOp2B
while/nqcjuhnaut/ReadVariableOpwhile/nqcjuhnaut/ReadVariableOp2F
!while/nqcjuhnaut/ReadVariableOp_1!while/nqcjuhnaut/ReadVariableOp_12F
!while/nqcjuhnaut/ReadVariableOp_2!while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1067350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1067350___redundant_placeholder05
1while_while_cond_1067350___redundant_placeholder15
1while_while_cond_1067350___redundant_placeholder25
1while_while_cond_1067350___redundant_placeholder35
1while_while_cond_1067350___redundant_placeholder45
1while_while_cond_1067350___redundant_placeholder55
1while_while_cond_1067350___redundant_placeholder6
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
Üh

G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068060
inputs_0<
)nqcjuhnaut_matmul_readvariableop_resource:	 >
+nqcjuhnaut_matmul_1_readvariableop_resource:	 9
*nqcjuhnaut_biasadd_readvariableop_resource:	0
"nqcjuhnaut_readvariableop_resource: 2
$nqcjuhnaut_readvariableop_1_resource: 2
$nqcjuhnaut_readvariableop_2_resource: 
identity¢!nqcjuhnaut/BiasAdd/ReadVariableOp¢ nqcjuhnaut/MatMul/ReadVariableOp¢"nqcjuhnaut/MatMul_1/ReadVariableOp¢nqcjuhnaut/ReadVariableOp¢nqcjuhnaut/ReadVariableOp_1¢nqcjuhnaut/ReadVariableOp_2¢whileF
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
 nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp)nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 nqcjuhnaut/MatMul/ReadVariableOp§
nqcjuhnaut/MatMulMatMulstrided_slice_2:output:0(nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMulµ
"nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp+nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"nqcjuhnaut/MatMul_1/ReadVariableOp£
nqcjuhnaut/MatMul_1MatMulzeros:output:0*nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMul_1
nqcjuhnaut/addAddV2nqcjuhnaut/MatMul:product:0nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/add®
!nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp*nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!nqcjuhnaut/BiasAdd/ReadVariableOp¥
nqcjuhnaut/BiasAddBiasAddnqcjuhnaut/add:z:0)nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/BiasAddz
nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
nqcjuhnaut/split/split_dimë
nqcjuhnaut/splitSplit#nqcjuhnaut/split/split_dim:output:0nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nqcjuhnaut/split
nqcjuhnaut/ReadVariableOpReadVariableOp"nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp
nqcjuhnaut/mulMul!nqcjuhnaut/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul
nqcjuhnaut/add_1AddV2nqcjuhnaut/split:output:0nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_1{
nqcjuhnaut/SigmoidSigmoidnqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid
nqcjuhnaut/ReadVariableOp_1ReadVariableOp$nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_1
nqcjuhnaut/mul_1Mul#nqcjuhnaut/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_1
nqcjuhnaut/add_2AddV2nqcjuhnaut/split:output:1nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_2
nqcjuhnaut/Sigmoid_1Sigmoidnqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_1
nqcjuhnaut/mul_2Mulnqcjuhnaut/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_2w
nqcjuhnaut/TanhTanhnqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh
nqcjuhnaut/mul_3Mulnqcjuhnaut/Sigmoid:y:0nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_3
nqcjuhnaut/add_3AddV2nqcjuhnaut/mul_2:z:0nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_3
nqcjuhnaut/ReadVariableOp_2ReadVariableOp$nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_2
nqcjuhnaut/mul_4Mul#nqcjuhnaut/ReadVariableOp_2:value:0nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_4
nqcjuhnaut/add_4AddV2nqcjuhnaut/split:output:3nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_4
nqcjuhnaut/Sigmoid_2Sigmoidnqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_2v
nqcjuhnaut/Tanh_1Tanhnqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh_1
nqcjuhnaut/mul_5Mulnqcjuhnaut/Sigmoid_2:y:0nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)nqcjuhnaut_matmul_readvariableop_resource+nqcjuhnaut_matmul_1_readvariableop_resource*nqcjuhnaut_biasadd_readvariableop_resource"nqcjuhnaut_readvariableop_resource$nqcjuhnaut_readvariableop_1_resource$nqcjuhnaut_readvariableop_2_resource*
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
while_body_1067959*
condR
while_cond_1067958*Q
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
IdentityIdentitystrided_slice_3:output:0"^nqcjuhnaut/BiasAdd/ReadVariableOp!^nqcjuhnaut/MatMul/ReadVariableOp#^nqcjuhnaut/MatMul_1/ReadVariableOp^nqcjuhnaut/ReadVariableOp^nqcjuhnaut/ReadVariableOp_1^nqcjuhnaut/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!nqcjuhnaut/BiasAdd/ReadVariableOp!nqcjuhnaut/BiasAdd/ReadVariableOp2D
 nqcjuhnaut/MatMul/ReadVariableOp nqcjuhnaut/MatMul/ReadVariableOp2H
"nqcjuhnaut/MatMul_1/ReadVariableOp"nqcjuhnaut/MatMul_1/ReadVariableOp26
nqcjuhnaut/ReadVariableOpnqcjuhnaut/ReadVariableOp2:
nqcjuhnaut/ReadVariableOp_1nqcjuhnaut/ReadVariableOp_12:
nqcjuhnaut/ReadVariableOp_2nqcjuhnaut/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
¸
¼
G__inference_sequential_layer_call_and_return_conditional_losses_1065699

inputs(
umdyqemnpr_1065661: 
umdyqemnpr_1065663:%
subsmtgotc_1065667:	%
subsmtgotc_1065669:	 !
subsmtgotc_1065671:	 
subsmtgotc_1065673:  
subsmtgotc_1065675:  
subsmtgotc_1065677: %
quyyatshey_1065680:	 %
quyyatshey_1065682:	 !
quyyatshey_1065684:	 
quyyatshey_1065686:  
quyyatshey_1065688:  
quyyatshey_1065690: $
ycxcgxamrr_1065693:  
ycxcgxamrr_1065695:
identity¢"quyyatshey/StatefulPartitionedCall¢"subsmtgotc/StatefulPartitionedCall¢"umdyqemnpr/StatefulPartitionedCall¢"ycxcgxamrr/StatefulPartitionedCall¬
"umdyqemnpr/StatefulPartitionedCallStatefulPartitionedCallinputsumdyqemnpr_1065661umdyqemnpr_1065663*
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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_10647062$
"umdyqemnpr/StatefulPartitionedCall
unxqeixodn/PartitionedCallPartitionedCall+umdyqemnpr/StatefulPartitionedCall:output:0*
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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_10647252
unxqeixodn/PartitionedCall
"subsmtgotc/StatefulPartitionedCallStatefulPartitionedCall#unxqeixodn/PartitionedCall:output:0subsmtgotc_1065667subsmtgotc_1065669subsmtgotc_1065671subsmtgotc_1065673subsmtgotc_1065675subsmtgotc_1065677*
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10655882$
"subsmtgotc/StatefulPartitionedCall¡
"quyyatshey/StatefulPartitionedCallStatefulPartitionedCall+subsmtgotc/StatefulPartitionedCall:output:0quyyatshey_1065680quyyatshey_1065682quyyatshey_1065684quyyatshey_1065686quyyatshey_1065688quyyatshey_1065690*
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10653742$
"quyyatshey/StatefulPartitionedCallÉ
"ycxcgxamrr/StatefulPartitionedCallStatefulPartitionedCall+quyyatshey/StatefulPartitionedCall:output:0ycxcgxamrr_1065693ycxcgxamrr_1065695*
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
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_10651232$
"ycxcgxamrr/StatefulPartitionedCall
IdentityIdentity+ycxcgxamrr/StatefulPartitionedCall:output:0#^quyyatshey/StatefulPartitionedCall#^subsmtgotc/StatefulPartitionedCall#^umdyqemnpr/StatefulPartitionedCall#^ycxcgxamrr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"quyyatshey/StatefulPartitionedCall"quyyatshey/StatefulPartitionedCall2H
"subsmtgotc/StatefulPartitionedCall"subsmtgotc/StatefulPartitionedCall2H
"umdyqemnpr/StatefulPartitionedCall"umdyqemnpr/StatefulPartitionedCall2H
"ycxcgxamrr/StatefulPartitionedCall"ycxcgxamrr/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç)
Ò
while_body_1064011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_nqcjuhnaut_1064035_0:	 -
while_nqcjuhnaut_1064037_0:	 )
while_nqcjuhnaut_1064039_0:	(
while_nqcjuhnaut_1064041_0: (
while_nqcjuhnaut_1064043_0: (
while_nqcjuhnaut_1064045_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_nqcjuhnaut_1064035:	 +
while_nqcjuhnaut_1064037:	 '
while_nqcjuhnaut_1064039:	&
while_nqcjuhnaut_1064041: &
while_nqcjuhnaut_1064043: &
while_nqcjuhnaut_1064045: ¢(while/nqcjuhnaut/StatefulPartitionedCallÃ
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
(while/nqcjuhnaut/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_nqcjuhnaut_1064035_0while_nqcjuhnaut_1064037_0while_nqcjuhnaut_1064039_0while_nqcjuhnaut_1064041_0while_nqcjuhnaut_1064043_0while_nqcjuhnaut_1064045_0*
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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_10639912*
(while/nqcjuhnaut/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/nqcjuhnaut/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/nqcjuhnaut/StatefulPartitionedCall:output:1)^while/nqcjuhnaut/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/nqcjuhnaut/StatefulPartitionedCall:output:2)^while/nqcjuhnaut/StatefulPartitionedCall*
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
while_nqcjuhnaut_1064035while_nqcjuhnaut_1064035_0"6
while_nqcjuhnaut_1064037while_nqcjuhnaut_1064037_0"6
while_nqcjuhnaut_1064039while_nqcjuhnaut_1064039_0"6
while_nqcjuhnaut_1064041while_nqcjuhnaut_1064041_0"6
while_nqcjuhnaut_1064043while_nqcjuhnaut_1064043_0"6
while_nqcjuhnaut_1064045while_nqcjuhnaut_1064045_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/nqcjuhnaut/StatefulPartitionedCall(while/nqcjuhnaut/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1066990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1066990___redundant_placeholder05
1while_while_cond_1066990___redundant_placeholder15
1while_while_cond_1066990___redundant_placeholder25
1while_while_cond_1066990___redundant_placeholder35
1while_while_cond_1066990___redundant_placeholder45
1while_while_cond_1066990___redundant_placeholder55
1while_while_cond_1066990___redundant_placeholder6
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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_1066844

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
àY

while_body_1068319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_nqcjuhnaut_matmul_readvariableop_resource_0:	 F
3while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 A
2while_nqcjuhnaut_biasadd_readvariableop_resource_0:	8
*while_nqcjuhnaut_readvariableop_resource_0: :
,while_nqcjuhnaut_readvariableop_1_resource_0: :
,while_nqcjuhnaut_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_nqcjuhnaut_matmul_readvariableop_resource:	 D
1while_nqcjuhnaut_matmul_1_readvariableop_resource:	 ?
0while_nqcjuhnaut_biasadd_readvariableop_resource:	6
(while_nqcjuhnaut_readvariableop_resource: 8
*while_nqcjuhnaut_readvariableop_1_resource: 8
*while_nqcjuhnaut_readvariableop_2_resource: ¢'while/nqcjuhnaut/BiasAdd/ReadVariableOp¢&while/nqcjuhnaut/MatMul/ReadVariableOp¢(while/nqcjuhnaut/MatMul_1/ReadVariableOp¢while/nqcjuhnaut/ReadVariableOp¢!while/nqcjuhnaut/ReadVariableOp_1¢!while/nqcjuhnaut/ReadVariableOp_2Ã
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
&while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp1while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/nqcjuhnaut/MatMul/ReadVariableOpÑ
while/nqcjuhnaut/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMulÉ
(while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp3while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/nqcjuhnaut/MatMul_1/ReadVariableOpº
while/nqcjuhnaut/MatMul_1MatMulwhile_placeholder_20while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMul_1°
while/nqcjuhnaut/addAddV2!while/nqcjuhnaut/MatMul:product:0#while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/addÂ
'while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp2while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/nqcjuhnaut/BiasAdd/ReadVariableOp½
while/nqcjuhnaut/BiasAddBiasAddwhile/nqcjuhnaut/add:z:0/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/BiasAdd
 while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/nqcjuhnaut/split/split_dim
while/nqcjuhnaut/splitSplit)while/nqcjuhnaut/split/split_dim:output:0!while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/nqcjuhnaut/split©
while/nqcjuhnaut/ReadVariableOpReadVariableOp*while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/nqcjuhnaut/ReadVariableOp£
while/nqcjuhnaut/mulMul'while/nqcjuhnaut/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul¦
while/nqcjuhnaut/add_1AddV2while/nqcjuhnaut/split:output:0while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_1
while/nqcjuhnaut/SigmoidSigmoidwhile/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid¯
!while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp,while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_1©
while/nqcjuhnaut/mul_1Mul)while/nqcjuhnaut/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_1¨
while/nqcjuhnaut/add_2AddV2while/nqcjuhnaut/split:output:1while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_2
while/nqcjuhnaut/Sigmoid_1Sigmoidwhile/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_1
while/nqcjuhnaut/mul_2Mulwhile/nqcjuhnaut/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_2
while/nqcjuhnaut/TanhTanhwhile/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh¢
while/nqcjuhnaut/mul_3Mulwhile/nqcjuhnaut/Sigmoid:y:0while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_3£
while/nqcjuhnaut/add_3AddV2while/nqcjuhnaut/mul_2:z:0while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_3¯
!while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp,while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_2°
while/nqcjuhnaut/mul_4Mul)while/nqcjuhnaut/ReadVariableOp_2:value:0while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_4¨
while/nqcjuhnaut/add_4AddV2while/nqcjuhnaut/split:output:3while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_4
while/nqcjuhnaut/Sigmoid_2Sigmoidwhile/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_2
while/nqcjuhnaut/Tanh_1Tanhwhile/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh_1¦
while/nqcjuhnaut/mul_5Mulwhile/nqcjuhnaut/Sigmoid_2:y:0while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/nqcjuhnaut/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/nqcjuhnaut/mul_5:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/nqcjuhnaut/add_3:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
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
0while_nqcjuhnaut_biasadd_readvariableop_resource2while_nqcjuhnaut_biasadd_readvariableop_resource_0"h
1while_nqcjuhnaut_matmul_1_readvariableop_resource3while_nqcjuhnaut_matmul_1_readvariableop_resource_0"d
/while_nqcjuhnaut_matmul_readvariableop_resource1while_nqcjuhnaut_matmul_readvariableop_resource_0"Z
*while_nqcjuhnaut_readvariableop_1_resource,while_nqcjuhnaut_readvariableop_1_resource_0"Z
*while_nqcjuhnaut_readvariableop_2_resource,while_nqcjuhnaut_readvariableop_2_resource_0"V
(while_nqcjuhnaut_readvariableop_resource*while_nqcjuhnaut_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/nqcjuhnaut/BiasAdd/ReadVariableOp'while/nqcjuhnaut/BiasAdd/ReadVariableOp2P
&while/nqcjuhnaut/MatMul/ReadVariableOp&while/nqcjuhnaut/MatMul/ReadVariableOp2T
(while/nqcjuhnaut/MatMul_1/ReadVariableOp(while/nqcjuhnaut/MatMul_1/ReadVariableOp2B
while/nqcjuhnaut/ReadVariableOpwhile/nqcjuhnaut/ReadVariableOp2F
!while/nqcjuhnaut/ReadVariableOp_1!while/nqcjuhnaut/ReadVariableOp_12F
!while/nqcjuhnaut/ReadVariableOp_2!while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_nqcjuhnaut_layer_call_fn_1068619

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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_10641782
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
	

,__inference_subsmtgotc_layer_call_fn_1066878
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10635962
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
(sequential_quyyatshey_while_body_1063039H
Dsequential_quyyatshey_while_sequential_quyyatshey_while_loop_counterN
Jsequential_quyyatshey_while_sequential_quyyatshey_while_maximum_iterations+
'sequential_quyyatshey_while_placeholder-
)sequential_quyyatshey_while_placeholder_1-
)sequential_quyyatshey_while_placeholder_2-
)sequential_quyyatshey_while_placeholder_3G
Csequential_quyyatshey_while_sequential_quyyatshey_strided_slice_1_0
sequential_quyyatshey_while_tensorarrayv2read_tensorlistgetitem_sequential_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0:	 \
Isequential_quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 W
Hsequential_quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0:	N
@sequential_quyyatshey_while_nqcjuhnaut_readvariableop_resource_0: P
Bsequential_quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0: P
Bsequential_quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0: (
$sequential_quyyatshey_while_identity*
&sequential_quyyatshey_while_identity_1*
&sequential_quyyatshey_while_identity_2*
&sequential_quyyatshey_while_identity_3*
&sequential_quyyatshey_while_identity_4*
&sequential_quyyatshey_while_identity_5E
Asequential_quyyatshey_while_sequential_quyyatshey_strided_slice_1
}sequential_quyyatshey_while_tensorarrayv2read_tensorlistgetitem_sequential_quyyatshey_tensorarrayunstack_tensorlistfromtensorX
Esequential_quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource:	 Z
Gsequential_quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource:	 U
Fsequential_quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource:	L
>sequential_quyyatshey_while_nqcjuhnaut_readvariableop_resource: N
@sequential_quyyatshey_while_nqcjuhnaut_readvariableop_1_resource: N
@sequential_quyyatshey_while_nqcjuhnaut_readvariableop_2_resource: ¢=sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp¢<sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp¢>sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp¢5sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp¢7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_1¢7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2ï
Msequential/quyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/quyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/quyyatshey/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_quyyatshey_while_tensorarrayv2read_tensorlistgetitem_sequential_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0'sequential_quyyatshey_while_placeholderVsequential/quyyatshey/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/quyyatshey/while/TensorArrayV2Read/TensorListGetItem
<sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOpGsequential_quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp©
-sequential/quyyatshey/while/nqcjuhnaut/MatMulMatMulFsequential/quyyatshey/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/quyyatshey/while/nqcjuhnaut/MatMul
>sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOpIsequential_quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp
/sequential/quyyatshey/while/nqcjuhnaut/MatMul_1MatMul)sequential_quyyatshey_while_placeholder_2Fsequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/quyyatshey/while/nqcjuhnaut/MatMul_1
*sequential/quyyatshey/while/nqcjuhnaut/addAddV27sequential/quyyatshey/while/nqcjuhnaut/MatMul:product:09sequential/quyyatshey/while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/quyyatshey/while/nqcjuhnaut/add
=sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOpHsequential_quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp
.sequential/quyyatshey/while/nqcjuhnaut/BiasAddBiasAdd.sequential/quyyatshey/while/nqcjuhnaut/add:z:0Esequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/quyyatshey/while/nqcjuhnaut/BiasAdd²
6sequential/quyyatshey/while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/quyyatshey/while/nqcjuhnaut/split/split_dimÛ
,sequential/quyyatshey/while/nqcjuhnaut/splitSplit?sequential/quyyatshey/while/nqcjuhnaut/split/split_dim:output:07sequential/quyyatshey/while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/quyyatshey/while/nqcjuhnaut/splitë
5sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOpReadVariableOp@sequential_quyyatshey_while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOpû
*sequential/quyyatshey/while/nqcjuhnaut/mulMul=sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp:value:0)sequential_quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/quyyatshey/while/nqcjuhnaut/mulþ
,sequential/quyyatshey/while/nqcjuhnaut/add_1AddV25sequential/quyyatshey/while/nqcjuhnaut/split:output:0.sequential/quyyatshey/while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/add_1Ï
.sequential/quyyatshey/while/nqcjuhnaut/SigmoidSigmoid0sequential/quyyatshey/while/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/quyyatshey/while/nqcjuhnaut/Sigmoidñ
7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_1ReadVariableOpBsequential_quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_1
,sequential/quyyatshey/while/nqcjuhnaut/mul_1Mul?sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_1:value:0)sequential_quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/mul_1
,sequential/quyyatshey/while/nqcjuhnaut/add_2AddV25sequential/quyyatshey/while/nqcjuhnaut/split:output:10sequential/quyyatshey/while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/add_2Ó
0sequential/quyyatshey/while/nqcjuhnaut/Sigmoid_1Sigmoid0sequential/quyyatshey/while/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/quyyatshey/while/nqcjuhnaut/Sigmoid_1ö
,sequential/quyyatshey/while/nqcjuhnaut/mul_2Mul4sequential/quyyatshey/while/nqcjuhnaut/Sigmoid_1:y:0)sequential_quyyatshey_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/mul_2Ë
+sequential/quyyatshey/while/nqcjuhnaut/TanhTanh5sequential/quyyatshey/while/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/quyyatshey/while/nqcjuhnaut/Tanhú
,sequential/quyyatshey/while/nqcjuhnaut/mul_3Mul2sequential/quyyatshey/while/nqcjuhnaut/Sigmoid:y:0/sequential/quyyatshey/while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/mul_3û
,sequential/quyyatshey/while/nqcjuhnaut/add_3AddV20sequential/quyyatshey/while/nqcjuhnaut/mul_2:z:00sequential/quyyatshey/while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/add_3ñ
7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2ReadVariableOpBsequential_quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2
,sequential/quyyatshey/while/nqcjuhnaut/mul_4Mul?sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2:value:00sequential/quyyatshey/while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/mul_4
,sequential/quyyatshey/while/nqcjuhnaut/add_4AddV25sequential/quyyatshey/while/nqcjuhnaut/split:output:30sequential/quyyatshey/while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/add_4Ó
0sequential/quyyatshey/while/nqcjuhnaut/Sigmoid_2Sigmoid0sequential/quyyatshey/while/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/quyyatshey/while/nqcjuhnaut/Sigmoid_2Ê
-sequential/quyyatshey/while/nqcjuhnaut/Tanh_1Tanh0sequential/quyyatshey/while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/quyyatshey/while/nqcjuhnaut/Tanh_1þ
,sequential/quyyatshey/while/nqcjuhnaut/mul_5Mul4sequential/quyyatshey/while/nqcjuhnaut/Sigmoid_2:y:01sequential/quyyatshey/while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/quyyatshey/while/nqcjuhnaut/mul_5Ì
@sequential/quyyatshey/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_quyyatshey_while_placeholder_1'sequential_quyyatshey_while_placeholder0sequential/quyyatshey/while/nqcjuhnaut/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/quyyatshey/while/TensorArrayV2Write/TensorListSetItem
!sequential/quyyatshey/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/quyyatshey/while/add/yÁ
sequential/quyyatshey/while/addAddV2'sequential_quyyatshey_while_placeholder*sequential/quyyatshey/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/quyyatshey/while/add
#sequential/quyyatshey/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/quyyatshey/while/add_1/yä
!sequential/quyyatshey/while/add_1AddV2Dsequential_quyyatshey_while_sequential_quyyatshey_while_loop_counter,sequential/quyyatshey/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/quyyatshey/while/add_1
$sequential/quyyatshey/while/IdentityIdentity%sequential/quyyatshey/while/add_1:z:0>^sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp=^sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp?^sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp6^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp8^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_18^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/quyyatshey/while/Identityµ
&sequential/quyyatshey/while/Identity_1IdentityJsequential_quyyatshey_while_sequential_quyyatshey_while_maximum_iterations>^sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp=^sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp?^sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp6^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp8^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_18^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/quyyatshey/while/Identity_1
&sequential/quyyatshey/while/Identity_2Identity#sequential/quyyatshey/while/add:z:0>^sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp=^sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp?^sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp6^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp8^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_18^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/quyyatshey/while/Identity_2»
&sequential/quyyatshey/while/Identity_3IdentityPsequential/quyyatshey/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp=^sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp?^sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp6^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp8^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_18^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/quyyatshey/while/Identity_3¬
&sequential/quyyatshey/while/Identity_4Identity0sequential/quyyatshey/while/nqcjuhnaut/mul_5:z:0>^sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp=^sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp?^sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp6^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp8^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_18^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/while/Identity_4¬
&sequential/quyyatshey/while/Identity_5Identity0sequential/quyyatshey/while/nqcjuhnaut/add_3:z:0>^sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp=^sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp?^sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp6^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp8^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_18^sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/while/Identity_5"U
$sequential_quyyatshey_while_identity-sequential/quyyatshey/while/Identity:output:0"Y
&sequential_quyyatshey_while_identity_1/sequential/quyyatshey/while/Identity_1:output:0"Y
&sequential_quyyatshey_while_identity_2/sequential/quyyatshey/while/Identity_2:output:0"Y
&sequential_quyyatshey_while_identity_3/sequential/quyyatshey/while/Identity_3:output:0"Y
&sequential_quyyatshey_while_identity_4/sequential/quyyatshey/while/Identity_4:output:0"Y
&sequential_quyyatshey_while_identity_5/sequential/quyyatshey/while/Identity_5:output:0"
Fsequential_quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resourceHsequential_quyyatshey_while_nqcjuhnaut_biasadd_readvariableop_resource_0"
Gsequential_quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resourceIsequential_quyyatshey_while_nqcjuhnaut_matmul_1_readvariableop_resource_0"
Esequential_quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resourceGsequential_quyyatshey_while_nqcjuhnaut_matmul_readvariableop_resource_0"
@sequential_quyyatshey_while_nqcjuhnaut_readvariableop_1_resourceBsequential_quyyatshey_while_nqcjuhnaut_readvariableop_1_resource_0"
@sequential_quyyatshey_while_nqcjuhnaut_readvariableop_2_resourceBsequential_quyyatshey_while_nqcjuhnaut_readvariableop_2_resource_0"
>sequential_quyyatshey_while_nqcjuhnaut_readvariableop_resource@sequential_quyyatshey_while_nqcjuhnaut_readvariableop_resource_0"
Asequential_quyyatshey_while_sequential_quyyatshey_strided_slice_1Csequential_quyyatshey_while_sequential_quyyatshey_strided_slice_1_0"
}sequential_quyyatshey_while_tensorarrayv2read_tensorlistgetitem_sequential_quyyatshey_tensorarrayunstack_tensorlistfromtensorsequential_quyyatshey_while_tensorarrayv2read_tensorlistgetitem_sequential_quyyatshey_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp=sequential/quyyatshey/while/nqcjuhnaut/BiasAdd/ReadVariableOp2|
<sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp<sequential/quyyatshey/while/nqcjuhnaut/MatMul/ReadVariableOp2
>sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp>sequential/quyyatshey/while/nqcjuhnaut/MatMul_1/ReadVariableOp2n
5sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp5sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp2r
7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_17sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_12r
7sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_27sequential/quyyatshey/while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_1066826

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
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_1065853

mkdkkixskm(
umdyqemnpr_1065815: 
umdyqemnpr_1065817:%
subsmtgotc_1065821:	%
subsmtgotc_1065823:	 !
subsmtgotc_1065825:	 
subsmtgotc_1065827:  
subsmtgotc_1065829:  
subsmtgotc_1065831: %
quyyatshey_1065834:	 %
quyyatshey_1065836:	 !
quyyatshey_1065838:	 
quyyatshey_1065840:  
quyyatshey_1065842:  
quyyatshey_1065844: $
ycxcgxamrr_1065847:  
ycxcgxamrr_1065849:
identity¢"quyyatshey/StatefulPartitionedCall¢"subsmtgotc/StatefulPartitionedCall¢"umdyqemnpr/StatefulPartitionedCall¢"ycxcgxamrr/StatefulPartitionedCall°
"umdyqemnpr/StatefulPartitionedCallStatefulPartitionedCall
mkdkkixskmumdyqemnpr_1065815umdyqemnpr_1065817*
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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_10647062$
"umdyqemnpr/StatefulPartitionedCall
unxqeixodn/PartitionedCallPartitionedCall+umdyqemnpr/StatefulPartitionedCall:output:0*
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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_10647252
unxqeixodn/PartitionedCall
"subsmtgotc/StatefulPartitionedCallStatefulPartitionedCall#unxqeixodn/PartitionedCall:output:0subsmtgotc_1065821subsmtgotc_1065823subsmtgotc_1065825subsmtgotc_1065827subsmtgotc_1065829subsmtgotc_1065831*
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10655882$
"subsmtgotc/StatefulPartitionedCall¡
"quyyatshey/StatefulPartitionedCallStatefulPartitionedCall+subsmtgotc/StatefulPartitionedCall:output:0quyyatshey_1065834quyyatshey_1065836quyyatshey_1065838quyyatshey_1065840quyyatshey_1065842quyyatshey_1065844*
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10653742$
"quyyatshey/StatefulPartitionedCallÉ
"ycxcgxamrr/StatefulPartitionedCallStatefulPartitionedCall+quyyatshey/StatefulPartitionedCall:output:0ycxcgxamrr_1065847ycxcgxamrr_1065849*
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
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_10651232$
"ycxcgxamrr/StatefulPartitionedCall
IdentityIdentity+ycxcgxamrr/StatefulPartitionedCall:output:0#^quyyatshey/StatefulPartitionedCall#^subsmtgotc/StatefulPartitionedCall#^umdyqemnpr/StatefulPartitionedCall#^ycxcgxamrr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"quyyatshey/StatefulPartitionedCall"quyyatshey/StatefulPartitionedCall2H
"subsmtgotc/StatefulPartitionedCall"subsmtgotc/StatefulPartitionedCall2H
"umdyqemnpr/StatefulPartitionedCall"umdyqemnpr/StatefulPartitionedCall2H
"ycxcgxamrr/StatefulPartitionedCall"ycxcgxamrr/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
mkdkkixskm
©0
¼
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_1064706

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


,__inference_sequential_layer_call_fn_1065935

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
G__inference_sequential_layer_call_and_return_conditional_losses_10651302
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
àY

while_body_1064998
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_nqcjuhnaut_matmul_readvariableop_resource_0:	 F
3while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 A
2while_nqcjuhnaut_biasadd_readvariableop_resource_0:	8
*while_nqcjuhnaut_readvariableop_resource_0: :
,while_nqcjuhnaut_readvariableop_1_resource_0: :
,while_nqcjuhnaut_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_nqcjuhnaut_matmul_readvariableop_resource:	 D
1while_nqcjuhnaut_matmul_1_readvariableop_resource:	 ?
0while_nqcjuhnaut_biasadd_readvariableop_resource:	6
(while_nqcjuhnaut_readvariableop_resource: 8
*while_nqcjuhnaut_readvariableop_1_resource: 8
*while_nqcjuhnaut_readvariableop_2_resource: ¢'while/nqcjuhnaut/BiasAdd/ReadVariableOp¢&while/nqcjuhnaut/MatMul/ReadVariableOp¢(while/nqcjuhnaut/MatMul_1/ReadVariableOp¢while/nqcjuhnaut/ReadVariableOp¢!while/nqcjuhnaut/ReadVariableOp_1¢!while/nqcjuhnaut/ReadVariableOp_2Ã
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
&while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp1while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/nqcjuhnaut/MatMul/ReadVariableOpÑ
while/nqcjuhnaut/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMulÉ
(while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp3while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/nqcjuhnaut/MatMul_1/ReadVariableOpº
while/nqcjuhnaut/MatMul_1MatMulwhile_placeholder_20while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMul_1°
while/nqcjuhnaut/addAddV2!while/nqcjuhnaut/MatMul:product:0#while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/addÂ
'while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp2while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/nqcjuhnaut/BiasAdd/ReadVariableOp½
while/nqcjuhnaut/BiasAddBiasAddwhile/nqcjuhnaut/add:z:0/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/BiasAdd
 while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/nqcjuhnaut/split/split_dim
while/nqcjuhnaut/splitSplit)while/nqcjuhnaut/split/split_dim:output:0!while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/nqcjuhnaut/split©
while/nqcjuhnaut/ReadVariableOpReadVariableOp*while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/nqcjuhnaut/ReadVariableOp£
while/nqcjuhnaut/mulMul'while/nqcjuhnaut/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul¦
while/nqcjuhnaut/add_1AddV2while/nqcjuhnaut/split:output:0while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_1
while/nqcjuhnaut/SigmoidSigmoidwhile/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid¯
!while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp,while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_1©
while/nqcjuhnaut/mul_1Mul)while/nqcjuhnaut/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_1¨
while/nqcjuhnaut/add_2AddV2while/nqcjuhnaut/split:output:1while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_2
while/nqcjuhnaut/Sigmoid_1Sigmoidwhile/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_1
while/nqcjuhnaut/mul_2Mulwhile/nqcjuhnaut/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_2
while/nqcjuhnaut/TanhTanhwhile/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh¢
while/nqcjuhnaut/mul_3Mulwhile/nqcjuhnaut/Sigmoid:y:0while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_3£
while/nqcjuhnaut/add_3AddV2while/nqcjuhnaut/mul_2:z:0while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_3¯
!while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp,while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_2°
while/nqcjuhnaut/mul_4Mul)while/nqcjuhnaut/ReadVariableOp_2:value:0while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_4¨
while/nqcjuhnaut/add_4AddV2while/nqcjuhnaut/split:output:3while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_4
while/nqcjuhnaut/Sigmoid_2Sigmoidwhile/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_2
while/nqcjuhnaut/Tanh_1Tanhwhile/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh_1¦
while/nqcjuhnaut/mul_5Mulwhile/nqcjuhnaut/Sigmoid_2:y:0while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/nqcjuhnaut/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/nqcjuhnaut/mul_5:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/nqcjuhnaut/add_3:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
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
0while_nqcjuhnaut_biasadd_readvariableop_resource2while_nqcjuhnaut_biasadd_readvariableop_resource_0"h
1while_nqcjuhnaut_matmul_1_readvariableop_resource3while_nqcjuhnaut_matmul_1_readvariableop_resource_0"d
/while_nqcjuhnaut_matmul_readvariableop_resource1while_nqcjuhnaut_matmul_readvariableop_resource_0"Z
*while_nqcjuhnaut_readvariableop_1_resource,while_nqcjuhnaut_readvariableop_1_resource_0"Z
*while_nqcjuhnaut_readvariableop_2_resource,while_nqcjuhnaut_readvariableop_2_resource_0"V
(while_nqcjuhnaut_readvariableop_resource*while_nqcjuhnaut_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/nqcjuhnaut/BiasAdd/ReadVariableOp'while/nqcjuhnaut/BiasAdd/ReadVariableOp2P
&while/nqcjuhnaut/MatMul/ReadVariableOp&while/nqcjuhnaut/MatMul/ReadVariableOp2T
(while/nqcjuhnaut/MatMul_1/ReadVariableOp(while/nqcjuhnaut/MatMul_1/ReadVariableOp2B
while/nqcjuhnaut/ReadVariableOpwhile/nqcjuhnaut/ReadVariableOp2F
!while/nqcjuhnaut/ReadVariableOp_1!while/nqcjuhnaut/ReadVariableOp_12F
!while/nqcjuhnaut/ReadVariableOp_2!while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1067958
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1067958___redundant_placeholder05
1while_while_cond_1067958___redundant_placeholder15
1while_while_cond_1067958___redundant_placeholder25
1while_while_cond_1067958___redundant_placeholder35
1while_while_cond_1067958___redundant_placeholder45
1while_while_cond_1067958___redundant_placeholder55
1while_while_cond_1067958___redundant_placeholder6
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
%__inference_signature_wrapper_1065898

mkdkkixskm
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
mkdkkixskmunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_10631462
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
mkdkkixskm


,__inference_sequential_layer_call_fn_1065972

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
G__inference_sequential_layer_call_and_return_conditional_losses_10656992
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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_1064725

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
±'
³
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1063233

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
while_body_1067171
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_zdztiqrxwb_matmul_readvariableop_resource_0:	F
3while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 A
2while_zdztiqrxwb_biasadd_readvariableop_resource_0:	8
*while_zdztiqrxwb_readvariableop_resource_0: :
,while_zdztiqrxwb_readvariableop_1_resource_0: :
,while_zdztiqrxwb_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_zdztiqrxwb_matmul_readvariableop_resource:	D
1while_zdztiqrxwb_matmul_1_readvariableop_resource:	 ?
0while_zdztiqrxwb_biasadd_readvariableop_resource:	6
(while_zdztiqrxwb_readvariableop_resource: 8
*while_zdztiqrxwb_readvariableop_1_resource: 8
*while_zdztiqrxwb_readvariableop_2_resource: ¢'while/zdztiqrxwb/BiasAdd/ReadVariableOp¢&while/zdztiqrxwb/MatMul/ReadVariableOp¢(while/zdztiqrxwb/MatMul_1/ReadVariableOp¢while/zdztiqrxwb/ReadVariableOp¢!while/zdztiqrxwb/ReadVariableOp_1¢!while/zdztiqrxwb/ReadVariableOp_2Ã
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
&while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp1while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/zdztiqrxwb/MatMul/ReadVariableOpÑ
while/zdztiqrxwb/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMulÉ
(while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp3while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/zdztiqrxwb/MatMul_1/ReadVariableOpº
while/zdztiqrxwb/MatMul_1MatMulwhile_placeholder_20while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMul_1°
while/zdztiqrxwb/addAddV2!while/zdztiqrxwb/MatMul:product:0#while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/addÂ
'while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp2while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/zdztiqrxwb/BiasAdd/ReadVariableOp½
while/zdztiqrxwb/BiasAddBiasAddwhile/zdztiqrxwb/add:z:0/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/BiasAdd
 while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/zdztiqrxwb/split/split_dim
while/zdztiqrxwb/splitSplit)while/zdztiqrxwb/split/split_dim:output:0!while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/zdztiqrxwb/split©
while/zdztiqrxwb/ReadVariableOpReadVariableOp*while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/zdztiqrxwb/ReadVariableOp£
while/zdztiqrxwb/mulMul'while/zdztiqrxwb/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul¦
while/zdztiqrxwb/add_1AddV2while/zdztiqrxwb/split:output:0while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_1
while/zdztiqrxwb/SigmoidSigmoidwhile/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid¯
!while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp,while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_1©
while/zdztiqrxwb/mul_1Mul)while/zdztiqrxwb/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_1¨
while/zdztiqrxwb/add_2AddV2while/zdztiqrxwb/split:output:1while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_2
while/zdztiqrxwb/Sigmoid_1Sigmoidwhile/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_1
while/zdztiqrxwb/mul_2Mulwhile/zdztiqrxwb/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_2
while/zdztiqrxwb/TanhTanhwhile/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh¢
while/zdztiqrxwb/mul_3Mulwhile/zdztiqrxwb/Sigmoid:y:0while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_3£
while/zdztiqrxwb/add_3AddV2while/zdztiqrxwb/mul_2:z:0while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_3¯
!while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp,while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_2°
while/zdztiqrxwb/mul_4Mul)while/zdztiqrxwb/ReadVariableOp_2:value:0while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_4¨
while/zdztiqrxwb/add_4AddV2while/zdztiqrxwb/split:output:3while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_4
while/zdztiqrxwb/Sigmoid_2Sigmoidwhile/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_2
while/zdztiqrxwb/Tanh_1Tanhwhile/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh_1¦
while/zdztiqrxwb/mul_5Mulwhile/zdztiqrxwb/Sigmoid_2:y:0while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/zdztiqrxwb/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/zdztiqrxwb/mul_5:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/zdztiqrxwb/add_3:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
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
0while_zdztiqrxwb_biasadd_readvariableop_resource2while_zdztiqrxwb_biasadd_readvariableop_resource_0"h
1while_zdztiqrxwb_matmul_1_readvariableop_resource3while_zdztiqrxwb_matmul_1_readvariableop_resource_0"d
/while_zdztiqrxwb_matmul_readvariableop_resource1while_zdztiqrxwb_matmul_readvariableop_resource_0"Z
*while_zdztiqrxwb_readvariableop_1_resource,while_zdztiqrxwb_readvariableop_1_resource_0"Z
*while_zdztiqrxwb_readvariableop_2_resource,while_zdztiqrxwb_readvariableop_2_resource_0"V
(while_zdztiqrxwb_readvariableop_resource*while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/zdztiqrxwb/BiasAdd/ReadVariableOp'while/zdztiqrxwb/BiasAdd/ReadVariableOp2P
&while/zdztiqrxwb/MatMul/ReadVariableOp&while/zdztiqrxwb/MatMul/ReadVariableOp2T
(while/zdztiqrxwb/MatMul_1/ReadVariableOp(while/zdztiqrxwb/MatMul_1/ReadVariableOp2B
while/zdztiqrxwb/ReadVariableOpwhile/zdztiqrxwb/ReadVariableOp2F
!while/zdztiqrxwb/ReadVariableOp_1!while/zdztiqrxwb/ReadVariableOp_12F
!while/zdztiqrxwb/ReadVariableOp_2!while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_unxqeixodn_layer_call_fn_1066831

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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_10647252
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
	

,__inference_subsmtgotc_layer_call_fn_1066861
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10633332
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
ç)
Ò
while_body_1063516
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_zdztiqrxwb_1063540_0:	-
while_zdztiqrxwb_1063542_0:	 )
while_zdztiqrxwb_1063544_0:	(
while_zdztiqrxwb_1063546_0: (
while_zdztiqrxwb_1063548_0: (
while_zdztiqrxwb_1063550_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_zdztiqrxwb_1063540:	+
while_zdztiqrxwb_1063542:	 '
while_zdztiqrxwb_1063544:	&
while_zdztiqrxwb_1063546: &
while_zdztiqrxwb_1063548: &
while_zdztiqrxwb_1063550: ¢(while/zdztiqrxwb/StatefulPartitionedCallÃ
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
(while/zdztiqrxwb/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_zdztiqrxwb_1063540_0while_zdztiqrxwb_1063542_0while_zdztiqrxwb_1063544_0while_zdztiqrxwb_1063546_0while_zdztiqrxwb_1063548_0while_zdztiqrxwb_1063550_0*
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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_10634202*
(while/zdztiqrxwb/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/zdztiqrxwb/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/zdztiqrxwb/StatefulPartitionedCall:output:1)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/zdztiqrxwb/StatefulPartitionedCall:output:2)^while/zdztiqrxwb/StatefulPartitionedCall*
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
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"6
while_zdztiqrxwb_1063540while_zdztiqrxwb_1063540_0"6
while_zdztiqrxwb_1063542while_zdztiqrxwb_1063542_0"6
while_zdztiqrxwb_1063544while_zdztiqrxwb_1063544_0"6
while_zdztiqrxwb_1063546while_zdztiqrxwb_1063546_0"6
while_zdztiqrxwb_1063548while_zdztiqrxwb_1063548_0"6
while_zdztiqrxwb_1063550while_zdztiqrxwb_1063550_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/zdztiqrxwb/StatefulPartitionedCall(while/zdztiqrxwb/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067452

inputs<
)zdztiqrxwb_matmul_readvariableop_resource:	>
+zdztiqrxwb_matmul_1_readvariableop_resource:	 9
*zdztiqrxwb_biasadd_readvariableop_resource:	0
"zdztiqrxwb_readvariableop_resource: 2
$zdztiqrxwb_readvariableop_1_resource: 2
$zdztiqrxwb_readvariableop_2_resource: 
identity¢while¢!zdztiqrxwb/BiasAdd/ReadVariableOp¢ zdztiqrxwb/MatMul/ReadVariableOp¢"zdztiqrxwb/MatMul_1/ReadVariableOp¢zdztiqrxwb/ReadVariableOp¢zdztiqrxwb/ReadVariableOp_1¢zdztiqrxwb/ReadVariableOp_2D
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
 zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp)zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 zdztiqrxwb/MatMul/ReadVariableOp§
zdztiqrxwb/MatMulMatMulstrided_slice_2:output:0(zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMulµ
"zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp+zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"zdztiqrxwb/MatMul_1/ReadVariableOp£
zdztiqrxwb/MatMul_1MatMulzeros:output:0*zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMul_1
zdztiqrxwb/addAddV2zdztiqrxwb/MatMul:product:0zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/add®
!zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp*zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!zdztiqrxwb/BiasAdd/ReadVariableOp¥
zdztiqrxwb/BiasAddBiasAddzdztiqrxwb/add:z:0)zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/BiasAddz
zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
zdztiqrxwb/split/split_dimë
zdztiqrxwb/splitSplit#zdztiqrxwb/split/split_dim:output:0zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zdztiqrxwb/split
zdztiqrxwb/ReadVariableOpReadVariableOp"zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp
zdztiqrxwb/mulMul!zdztiqrxwb/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul
zdztiqrxwb/add_1AddV2zdztiqrxwb/split:output:0zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_1{
zdztiqrxwb/SigmoidSigmoidzdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid
zdztiqrxwb/ReadVariableOp_1ReadVariableOp$zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_1
zdztiqrxwb/mul_1Mul#zdztiqrxwb/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_1
zdztiqrxwb/add_2AddV2zdztiqrxwb/split:output:1zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_2
zdztiqrxwb/Sigmoid_1Sigmoidzdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_1
zdztiqrxwb/mul_2Mulzdztiqrxwb/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_2w
zdztiqrxwb/TanhTanhzdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh
zdztiqrxwb/mul_3Mulzdztiqrxwb/Sigmoid:y:0zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_3
zdztiqrxwb/add_3AddV2zdztiqrxwb/mul_2:z:0zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_3
zdztiqrxwb/ReadVariableOp_2ReadVariableOp$zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_2
zdztiqrxwb/mul_4Mul#zdztiqrxwb/ReadVariableOp_2:value:0zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_4
zdztiqrxwb/add_4AddV2zdztiqrxwb/split:output:3zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_4
zdztiqrxwb/Sigmoid_2Sigmoidzdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_2v
zdztiqrxwb/Tanh_1Tanhzdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh_1
zdztiqrxwb/mul_5Mulzdztiqrxwb/Sigmoid_2:y:0zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)zdztiqrxwb_matmul_readvariableop_resource+zdztiqrxwb_matmul_1_readvariableop_resource*zdztiqrxwb_biasadd_readvariableop_resource"zdztiqrxwb_readvariableop_resource$zdztiqrxwb_readvariableop_1_resource$zdztiqrxwb_readvariableop_2_resource*
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
while_body_1067351*
condR
while_cond_1067350*Q
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
IdentityIdentitytranspose_1:y:0^while"^zdztiqrxwb/BiasAdd/ReadVariableOp!^zdztiqrxwb/MatMul/ReadVariableOp#^zdztiqrxwb/MatMul_1/ReadVariableOp^zdztiqrxwb/ReadVariableOp^zdztiqrxwb/ReadVariableOp_1^zdztiqrxwb/ReadVariableOp_2*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2F
!zdztiqrxwb/BiasAdd/ReadVariableOp!zdztiqrxwb/BiasAdd/ReadVariableOp2D
 zdztiqrxwb/MatMul/ReadVariableOp zdztiqrxwb/MatMul/ReadVariableOp2H
"zdztiqrxwb/MatMul_1/ReadVariableOp"zdztiqrxwb/MatMul_1/ReadVariableOp26
zdztiqrxwb/ReadVariableOpzdztiqrxwb/ReadVariableOp2:
zdztiqrxwb/ReadVariableOp_1zdztiqrxwb/ReadVariableOp_12:
zdztiqrxwb/ReadVariableOp_2zdztiqrxwb/ReadVariableOp_2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó	
ø
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_1068439

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
while_cond_1068318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1068318___redundant_placeholder05
1while_while_cond_1068318___redundant_placeholder15
1while_while_cond_1068318___redundant_placeholder25
1while_while_cond_1068318___redundant_placeholder35
1while_while_cond_1068318___redundant_placeholder45
1while_while_cond_1068318___redundant_placeholder55
1while_while_cond_1068318___redundant_placeholder6
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
while_body_1067531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_zdztiqrxwb_matmul_readvariableop_resource_0:	F
3while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 A
2while_zdztiqrxwb_biasadd_readvariableop_resource_0:	8
*while_zdztiqrxwb_readvariableop_resource_0: :
,while_zdztiqrxwb_readvariableop_1_resource_0: :
,while_zdztiqrxwb_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_zdztiqrxwb_matmul_readvariableop_resource:	D
1while_zdztiqrxwb_matmul_1_readvariableop_resource:	 ?
0while_zdztiqrxwb_biasadd_readvariableop_resource:	6
(while_zdztiqrxwb_readvariableop_resource: 8
*while_zdztiqrxwb_readvariableop_1_resource: 8
*while_zdztiqrxwb_readvariableop_2_resource: ¢'while/zdztiqrxwb/BiasAdd/ReadVariableOp¢&while/zdztiqrxwb/MatMul/ReadVariableOp¢(while/zdztiqrxwb/MatMul_1/ReadVariableOp¢while/zdztiqrxwb/ReadVariableOp¢!while/zdztiqrxwb/ReadVariableOp_1¢!while/zdztiqrxwb/ReadVariableOp_2Ã
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
&while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp1while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/zdztiqrxwb/MatMul/ReadVariableOpÑ
while/zdztiqrxwb/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMulÉ
(while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp3while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/zdztiqrxwb/MatMul_1/ReadVariableOpº
while/zdztiqrxwb/MatMul_1MatMulwhile_placeholder_20while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMul_1°
while/zdztiqrxwb/addAddV2!while/zdztiqrxwb/MatMul:product:0#while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/addÂ
'while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp2while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/zdztiqrxwb/BiasAdd/ReadVariableOp½
while/zdztiqrxwb/BiasAddBiasAddwhile/zdztiqrxwb/add:z:0/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/BiasAdd
 while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/zdztiqrxwb/split/split_dim
while/zdztiqrxwb/splitSplit)while/zdztiqrxwb/split/split_dim:output:0!while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/zdztiqrxwb/split©
while/zdztiqrxwb/ReadVariableOpReadVariableOp*while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/zdztiqrxwb/ReadVariableOp£
while/zdztiqrxwb/mulMul'while/zdztiqrxwb/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul¦
while/zdztiqrxwb/add_1AddV2while/zdztiqrxwb/split:output:0while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_1
while/zdztiqrxwb/SigmoidSigmoidwhile/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid¯
!while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp,while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_1©
while/zdztiqrxwb/mul_1Mul)while/zdztiqrxwb/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_1¨
while/zdztiqrxwb/add_2AddV2while/zdztiqrxwb/split:output:1while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_2
while/zdztiqrxwb/Sigmoid_1Sigmoidwhile/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_1
while/zdztiqrxwb/mul_2Mulwhile/zdztiqrxwb/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_2
while/zdztiqrxwb/TanhTanhwhile/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh¢
while/zdztiqrxwb/mul_3Mulwhile/zdztiqrxwb/Sigmoid:y:0while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_3£
while/zdztiqrxwb/add_3AddV2while/zdztiqrxwb/mul_2:z:0while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_3¯
!while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp,while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_2°
while/zdztiqrxwb/mul_4Mul)while/zdztiqrxwb/ReadVariableOp_2:value:0while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_4¨
while/zdztiqrxwb/add_4AddV2while/zdztiqrxwb/split:output:3while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_4
while/zdztiqrxwb/Sigmoid_2Sigmoidwhile/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_2
while/zdztiqrxwb/Tanh_1Tanhwhile/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh_1¦
while/zdztiqrxwb/mul_5Mulwhile/zdztiqrxwb/Sigmoid_2:y:0while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/zdztiqrxwb/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/zdztiqrxwb/mul_5:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/zdztiqrxwb/add_3:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
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
0while_zdztiqrxwb_biasadd_readvariableop_resource2while_zdztiqrxwb_biasadd_readvariableop_resource_0"h
1while_zdztiqrxwb_matmul_1_readvariableop_resource3while_zdztiqrxwb_matmul_1_readvariableop_resource_0"d
/while_zdztiqrxwb_matmul_readvariableop_resource1while_zdztiqrxwb_matmul_readvariableop_resource_0"Z
*while_zdztiqrxwb_readvariableop_1_resource,while_zdztiqrxwb_readvariableop_1_resource_0"Z
*while_zdztiqrxwb_readvariableop_2_resource,while_zdztiqrxwb_readvariableop_2_resource_0"V
(while_zdztiqrxwb_readvariableop_resource*while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/zdztiqrxwb/BiasAdd/ReadVariableOp'while/zdztiqrxwb/BiasAdd/ReadVariableOp2P
&while/zdztiqrxwb/MatMul/ReadVariableOp&while/zdztiqrxwb/MatMul/ReadVariableOp2T
(while/zdztiqrxwb/MatMul_1/ReadVariableOp(while/zdztiqrxwb/MatMul_1/ReadVariableOp2B
while/zdztiqrxwb/ReadVariableOpwhile/zdztiqrxwb/ReadVariableOp2F
!while/zdztiqrxwb/ReadVariableOp_1!while/zdztiqrxwb/ReadVariableOp_12F
!while/zdztiqrxwb/ReadVariableOp_2!while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_sequential_layer_call_and_return_conditional_losses_1065130

inputs(
umdyqemnpr_1064707: 
umdyqemnpr_1064709:%
subsmtgotc_1064907:	%
subsmtgotc_1064909:	 !
subsmtgotc_1064911:	 
subsmtgotc_1064913:  
subsmtgotc_1064915:  
subsmtgotc_1064917: %
quyyatshey_1065100:	 %
quyyatshey_1065102:	 !
quyyatshey_1065104:	 
quyyatshey_1065106:  
quyyatshey_1065108:  
quyyatshey_1065110: $
ycxcgxamrr_1065124:  
ycxcgxamrr_1065126:
identity¢"quyyatshey/StatefulPartitionedCall¢"subsmtgotc/StatefulPartitionedCall¢"umdyqemnpr/StatefulPartitionedCall¢"ycxcgxamrr/StatefulPartitionedCall¬
"umdyqemnpr/StatefulPartitionedCallStatefulPartitionedCallinputsumdyqemnpr_1064707umdyqemnpr_1064709*
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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_10647062$
"umdyqemnpr/StatefulPartitionedCall
unxqeixodn/PartitionedCallPartitionedCall+umdyqemnpr/StatefulPartitionedCall:output:0*
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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_10647252
unxqeixodn/PartitionedCall
"subsmtgotc/StatefulPartitionedCallStatefulPartitionedCall#unxqeixodn/PartitionedCall:output:0subsmtgotc_1064907subsmtgotc_1064909subsmtgotc_1064911subsmtgotc_1064913subsmtgotc_1064915subsmtgotc_1064917*
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10649062$
"subsmtgotc/StatefulPartitionedCall¡
"quyyatshey/StatefulPartitionedCallStatefulPartitionedCall+subsmtgotc/StatefulPartitionedCall:output:0quyyatshey_1065100quyyatshey_1065102quyyatshey_1065104quyyatshey_1065106quyyatshey_1065108quyyatshey_1065110*
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10650992$
"quyyatshey/StatefulPartitionedCallÉ
"ycxcgxamrr/StatefulPartitionedCallStatefulPartitionedCall+quyyatshey/StatefulPartitionedCall:output:0ycxcgxamrr_1065124ycxcgxamrr_1065126*
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
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_10651232$
"ycxcgxamrr/StatefulPartitionedCall
IdentityIdentity+ycxcgxamrr/StatefulPartitionedCall:output:0#^quyyatshey/StatefulPartitionedCall#^subsmtgotc/StatefulPartitionedCall#^umdyqemnpr/StatefulPartitionedCall#^ycxcgxamrr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"quyyatshey/StatefulPartitionedCall"quyyatshey/StatefulPartitionedCall2H
"subsmtgotc/StatefulPartitionedCall"subsmtgotc/StatefulPartitionedCall2H
"umdyqemnpr/StatefulPartitionedCall"umdyqemnpr/StatefulPartitionedCall2H
"ycxcgxamrr/StatefulPartitionedCall"ycxcgxamrr/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àY

while_body_1065487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_zdztiqrxwb_matmul_readvariableop_resource_0:	F
3while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 A
2while_zdztiqrxwb_biasadd_readvariableop_resource_0:	8
*while_zdztiqrxwb_readvariableop_resource_0: :
,while_zdztiqrxwb_readvariableop_1_resource_0: :
,while_zdztiqrxwb_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_zdztiqrxwb_matmul_readvariableop_resource:	D
1while_zdztiqrxwb_matmul_1_readvariableop_resource:	 ?
0while_zdztiqrxwb_biasadd_readvariableop_resource:	6
(while_zdztiqrxwb_readvariableop_resource: 8
*while_zdztiqrxwb_readvariableop_1_resource: 8
*while_zdztiqrxwb_readvariableop_2_resource: ¢'while/zdztiqrxwb/BiasAdd/ReadVariableOp¢&while/zdztiqrxwb/MatMul/ReadVariableOp¢(while/zdztiqrxwb/MatMul_1/ReadVariableOp¢while/zdztiqrxwb/ReadVariableOp¢!while/zdztiqrxwb/ReadVariableOp_1¢!while/zdztiqrxwb/ReadVariableOp_2Ã
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
&while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp1while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/zdztiqrxwb/MatMul/ReadVariableOpÑ
while/zdztiqrxwb/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMulÉ
(while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp3while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/zdztiqrxwb/MatMul_1/ReadVariableOpº
while/zdztiqrxwb/MatMul_1MatMulwhile_placeholder_20while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMul_1°
while/zdztiqrxwb/addAddV2!while/zdztiqrxwb/MatMul:product:0#while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/addÂ
'while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp2while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/zdztiqrxwb/BiasAdd/ReadVariableOp½
while/zdztiqrxwb/BiasAddBiasAddwhile/zdztiqrxwb/add:z:0/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/BiasAdd
 while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/zdztiqrxwb/split/split_dim
while/zdztiqrxwb/splitSplit)while/zdztiqrxwb/split/split_dim:output:0!while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/zdztiqrxwb/split©
while/zdztiqrxwb/ReadVariableOpReadVariableOp*while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/zdztiqrxwb/ReadVariableOp£
while/zdztiqrxwb/mulMul'while/zdztiqrxwb/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul¦
while/zdztiqrxwb/add_1AddV2while/zdztiqrxwb/split:output:0while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_1
while/zdztiqrxwb/SigmoidSigmoidwhile/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid¯
!while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp,while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_1©
while/zdztiqrxwb/mul_1Mul)while/zdztiqrxwb/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_1¨
while/zdztiqrxwb/add_2AddV2while/zdztiqrxwb/split:output:1while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_2
while/zdztiqrxwb/Sigmoid_1Sigmoidwhile/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_1
while/zdztiqrxwb/mul_2Mulwhile/zdztiqrxwb/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_2
while/zdztiqrxwb/TanhTanhwhile/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh¢
while/zdztiqrxwb/mul_3Mulwhile/zdztiqrxwb/Sigmoid:y:0while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_3£
while/zdztiqrxwb/add_3AddV2while/zdztiqrxwb/mul_2:z:0while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_3¯
!while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp,while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_2°
while/zdztiqrxwb/mul_4Mul)while/zdztiqrxwb/ReadVariableOp_2:value:0while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_4¨
while/zdztiqrxwb/add_4AddV2while/zdztiqrxwb/split:output:3while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_4
while/zdztiqrxwb/Sigmoid_2Sigmoidwhile/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_2
while/zdztiqrxwb/Tanh_1Tanhwhile/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh_1¦
while/zdztiqrxwb/mul_5Mulwhile/zdztiqrxwb/Sigmoid_2:y:0while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/zdztiqrxwb/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/zdztiqrxwb/mul_5:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/zdztiqrxwb/add_3:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
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
0while_zdztiqrxwb_biasadd_readvariableop_resource2while_zdztiqrxwb_biasadd_readvariableop_resource_0"h
1while_zdztiqrxwb_matmul_1_readvariableop_resource3while_zdztiqrxwb_matmul_1_readvariableop_resource_0"d
/while_zdztiqrxwb_matmul_readvariableop_resource1while_zdztiqrxwb_matmul_readvariableop_resource_0"Z
*while_zdztiqrxwb_readvariableop_1_resource,while_zdztiqrxwb_readvariableop_1_resource_0"Z
*while_zdztiqrxwb_readvariableop_2_resource,while_zdztiqrxwb_readvariableop_2_resource_0"V
(while_zdztiqrxwb_readvariableop_resource*while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/zdztiqrxwb/BiasAdd/ReadVariableOp'while/zdztiqrxwb/BiasAdd/ReadVariableOp2P
&while/zdztiqrxwb/MatMul/ReadVariableOp&while/zdztiqrxwb/MatMul/ReadVariableOp2T
(while/zdztiqrxwb/MatMul_1/ReadVariableOp(while/zdztiqrxwb/MatMul_1/ReadVariableOp2B
while/zdztiqrxwb/ReadVariableOpwhile/zdztiqrxwb/ReadVariableOp2F
!while/zdztiqrxwb/ReadVariableOp_1!while/zdztiqrxwb/ReadVariableOp_12F
!while/zdztiqrxwb/ReadVariableOp_2!while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
"__inference__wrapped_model_1063146

mkdkkixskmW
Asequential_umdyqemnpr_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_umdyqemnpr_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource:	T
Asequential_subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource:	 O
@sequential_subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource:	F
8sequential_subsmtgotc_zdztiqrxwb_readvariableop_resource: H
:sequential_subsmtgotc_zdztiqrxwb_readvariableop_1_resource: H
:sequential_subsmtgotc_zdztiqrxwb_readvariableop_2_resource: R
?sequential_quyyatshey_nqcjuhnaut_matmul_readvariableop_resource:	 T
Asequential_quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource:	 O
@sequential_quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource:	F
8sequential_quyyatshey_nqcjuhnaut_readvariableop_resource: H
:sequential_quyyatshey_nqcjuhnaut_readvariableop_1_resource: H
:sequential_quyyatshey_nqcjuhnaut_readvariableop_2_resource: F
4sequential_ycxcgxamrr_matmul_readvariableop_resource: C
5sequential_ycxcgxamrr_biasadd_readvariableop_resource:
identity¢7sequential/quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp¢6sequential/quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp¢8sequential/quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp¢/sequential/quyyatshey/nqcjuhnaut/ReadVariableOp¢1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_1¢1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_2¢sequential/quyyatshey/while¢sequential/subsmtgotc/while¢7sequential/subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp¢6sequential/subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp¢8sequential/subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp¢/sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp¢1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_1¢1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_2¢8sequential/umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,sequential/ycxcgxamrr/BiasAdd/ReadVariableOp¢+sequential/ycxcgxamrr/MatMul/ReadVariableOp¥
+sequential/umdyqemnpr/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/umdyqemnpr/conv1d/ExpandDims/dimà
'sequential/umdyqemnpr/conv1d/ExpandDims
ExpandDims
mkdkkixskm4sequential/umdyqemnpr/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/umdyqemnpr/conv1d/ExpandDimsú
8sequential/umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_umdyqemnpr_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/umdyqemnpr/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/umdyqemnpr/conv1d/ExpandDims_1/dim
)sequential/umdyqemnpr/conv1d/ExpandDims_1
ExpandDims@sequential/umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/umdyqemnpr/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/umdyqemnpr/conv1d/ExpandDims_1¨
"sequential/umdyqemnpr/conv1d/ShapeShape0sequential/umdyqemnpr/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/umdyqemnpr/conv1d/Shape®
0sequential/umdyqemnpr/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/umdyqemnpr/conv1d/strided_slice/stack»
2sequential/umdyqemnpr/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/umdyqemnpr/conv1d/strided_slice/stack_1²
2sequential/umdyqemnpr/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/umdyqemnpr/conv1d/strided_slice/stack_2
*sequential/umdyqemnpr/conv1d/strided_sliceStridedSlice+sequential/umdyqemnpr/conv1d/Shape:output:09sequential/umdyqemnpr/conv1d/strided_slice/stack:output:0;sequential/umdyqemnpr/conv1d/strided_slice/stack_1:output:0;sequential/umdyqemnpr/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/umdyqemnpr/conv1d/strided_slice±
*sequential/umdyqemnpr/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/umdyqemnpr/conv1d/Reshape/shapeø
$sequential/umdyqemnpr/conv1d/ReshapeReshape0sequential/umdyqemnpr/conv1d/ExpandDims:output:03sequential/umdyqemnpr/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/umdyqemnpr/conv1d/Reshape
#sequential/umdyqemnpr/conv1d/Conv2DConv2D-sequential/umdyqemnpr/conv1d/Reshape:output:02sequential/umdyqemnpr/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/umdyqemnpr/conv1d/Conv2D±
,sequential/umdyqemnpr/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/umdyqemnpr/conv1d/concat/values_1
(sequential/umdyqemnpr/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/umdyqemnpr/conv1d/concat/axis£
#sequential/umdyqemnpr/conv1d/concatConcatV23sequential/umdyqemnpr/conv1d/strided_slice:output:05sequential/umdyqemnpr/conv1d/concat/values_1:output:01sequential/umdyqemnpr/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/umdyqemnpr/conv1d/concatõ
&sequential/umdyqemnpr/conv1d/Reshape_1Reshape,sequential/umdyqemnpr/conv1d/Conv2D:output:0,sequential/umdyqemnpr/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/umdyqemnpr/conv1d/Reshape_1â
$sequential/umdyqemnpr/conv1d/SqueezeSqueeze/sequential/umdyqemnpr/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/umdyqemnpr/conv1d/Squeeze½
.sequential/umdyqemnpr/squeeze_batch_dims/ShapeShape-sequential/umdyqemnpr/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/umdyqemnpr/squeeze_batch_dims/ShapeÆ
<sequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stackÓ
>sequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/umdyqemnpr/squeeze_batch_dims/strided_sliceStridedSlice7sequential/umdyqemnpr/squeeze_batch_dims/Shape:output:0Esequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/umdyqemnpr/squeeze_batch_dims/strided_sliceÅ
6sequential/umdyqemnpr/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/umdyqemnpr/squeeze_batch_dims/Reshape/shape
0sequential/umdyqemnpr/squeeze_batch_dims/ReshapeReshape-sequential/umdyqemnpr/conv1d/Squeeze:output:0?sequential/umdyqemnpr/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/umdyqemnpr/squeeze_batch_dims/Reshape
?sequential/umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_umdyqemnpr_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/umdyqemnpr/squeeze_batch_dims/BiasAddBiasAdd9sequential/umdyqemnpr/squeeze_batch_dims/Reshape:output:0Gsequential/umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/umdyqemnpr/squeeze_batch_dims/BiasAddÅ
8sequential/umdyqemnpr/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/umdyqemnpr/squeeze_batch_dims/concat/values_1·
4sequential/umdyqemnpr/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/umdyqemnpr/squeeze_batch_dims/concat/axisß
/sequential/umdyqemnpr/squeeze_batch_dims/concatConcatV2?sequential/umdyqemnpr/squeeze_batch_dims/strided_slice:output:0Asequential/umdyqemnpr/squeeze_batch_dims/concat/values_1:output:0=sequential/umdyqemnpr/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/umdyqemnpr/squeeze_batch_dims/concat¢
2sequential/umdyqemnpr/squeeze_batch_dims/Reshape_1Reshape9sequential/umdyqemnpr/squeeze_batch_dims/BiasAdd:output:08sequential/umdyqemnpr/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/umdyqemnpr/squeeze_batch_dims/Reshape_1¥
sequential/unxqeixodn/ShapeShape;sequential/umdyqemnpr/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/unxqeixodn/Shape 
)sequential/unxqeixodn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/unxqeixodn/strided_slice/stack¤
+sequential/unxqeixodn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/unxqeixodn/strided_slice/stack_1¤
+sequential/unxqeixodn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/unxqeixodn/strided_slice/stack_2æ
#sequential/unxqeixodn/strided_sliceStridedSlice$sequential/unxqeixodn/Shape:output:02sequential/unxqeixodn/strided_slice/stack:output:04sequential/unxqeixodn/strided_slice/stack_1:output:04sequential/unxqeixodn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/unxqeixodn/strided_slice
%sequential/unxqeixodn/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/unxqeixodn/Reshape/shape/1
%sequential/unxqeixodn/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/unxqeixodn/Reshape/shape/2
#sequential/unxqeixodn/Reshape/shapePack,sequential/unxqeixodn/strided_slice:output:0.sequential/unxqeixodn/Reshape/shape/1:output:0.sequential/unxqeixodn/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/unxqeixodn/Reshape/shapeê
sequential/unxqeixodn/ReshapeReshape;sequential/umdyqemnpr/squeeze_batch_dims/Reshape_1:output:0,sequential/unxqeixodn/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/unxqeixodn/Reshape
sequential/subsmtgotc/ShapeShape&sequential/unxqeixodn/Reshape:output:0*
T0*
_output_shapes
:2
sequential/subsmtgotc/Shape 
)sequential/subsmtgotc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/subsmtgotc/strided_slice/stack¤
+sequential/subsmtgotc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/subsmtgotc/strided_slice/stack_1¤
+sequential/subsmtgotc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/subsmtgotc/strided_slice/stack_2æ
#sequential/subsmtgotc/strided_sliceStridedSlice$sequential/subsmtgotc/Shape:output:02sequential/subsmtgotc/strided_slice/stack:output:04sequential/subsmtgotc/strided_slice/stack_1:output:04sequential/subsmtgotc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/subsmtgotc/strided_slice
!sequential/subsmtgotc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/subsmtgotc/zeros/mul/yÄ
sequential/subsmtgotc/zeros/mulMul,sequential/subsmtgotc/strided_slice:output:0*sequential/subsmtgotc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/subsmtgotc/zeros/mul
"sequential/subsmtgotc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/subsmtgotc/zeros/Less/y¿
 sequential/subsmtgotc/zeros/LessLess#sequential/subsmtgotc/zeros/mul:z:0+sequential/subsmtgotc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/subsmtgotc/zeros/Less
$sequential/subsmtgotc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/subsmtgotc/zeros/packed/1Û
"sequential/subsmtgotc/zeros/packedPack,sequential/subsmtgotc/strided_slice:output:0-sequential/subsmtgotc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/subsmtgotc/zeros/packed
!sequential/subsmtgotc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/subsmtgotc/zeros/ConstÍ
sequential/subsmtgotc/zerosFill+sequential/subsmtgotc/zeros/packed:output:0*sequential/subsmtgotc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/subsmtgotc/zeros
#sequential/subsmtgotc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/subsmtgotc/zeros_1/mul/yÊ
!sequential/subsmtgotc/zeros_1/mulMul,sequential/subsmtgotc/strided_slice:output:0,sequential/subsmtgotc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/subsmtgotc/zeros_1/mul
$sequential/subsmtgotc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/subsmtgotc/zeros_1/Less/yÇ
"sequential/subsmtgotc/zeros_1/LessLess%sequential/subsmtgotc/zeros_1/mul:z:0-sequential/subsmtgotc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/subsmtgotc/zeros_1/Less
&sequential/subsmtgotc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/subsmtgotc/zeros_1/packed/1á
$sequential/subsmtgotc/zeros_1/packedPack,sequential/subsmtgotc/strided_slice:output:0/sequential/subsmtgotc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/subsmtgotc/zeros_1/packed
#sequential/subsmtgotc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/subsmtgotc/zeros_1/ConstÕ
sequential/subsmtgotc/zeros_1Fill-sequential/subsmtgotc/zeros_1/packed:output:0,sequential/subsmtgotc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/subsmtgotc/zeros_1¡
$sequential/subsmtgotc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/subsmtgotc/transpose/permÜ
sequential/subsmtgotc/transpose	Transpose&sequential/unxqeixodn/Reshape:output:0-sequential/subsmtgotc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/subsmtgotc/transpose
sequential/subsmtgotc/Shape_1Shape#sequential/subsmtgotc/transpose:y:0*
T0*
_output_shapes
:2
sequential/subsmtgotc/Shape_1¤
+sequential/subsmtgotc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/subsmtgotc/strided_slice_1/stack¨
-sequential/subsmtgotc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/subsmtgotc/strided_slice_1/stack_1¨
-sequential/subsmtgotc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/subsmtgotc/strided_slice_1/stack_2ò
%sequential/subsmtgotc/strided_slice_1StridedSlice&sequential/subsmtgotc/Shape_1:output:04sequential/subsmtgotc/strided_slice_1/stack:output:06sequential/subsmtgotc/strided_slice_1/stack_1:output:06sequential/subsmtgotc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/subsmtgotc/strided_slice_1±
1sequential/subsmtgotc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/subsmtgotc/TensorArrayV2/element_shape
#sequential/subsmtgotc/TensorArrayV2TensorListReserve:sequential/subsmtgotc/TensorArrayV2/element_shape:output:0.sequential/subsmtgotc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/subsmtgotc/TensorArrayV2ë
Ksequential/subsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/subsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/subsmtgotc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/subsmtgotc/transpose:y:0Tsequential/subsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/subsmtgotc/TensorArrayUnstack/TensorListFromTensor¤
+sequential/subsmtgotc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/subsmtgotc/strided_slice_2/stack¨
-sequential/subsmtgotc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/subsmtgotc/strided_slice_2/stack_1¨
-sequential/subsmtgotc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/subsmtgotc/strided_slice_2/stack_2
%sequential/subsmtgotc/strided_slice_2StridedSlice#sequential/subsmtgotc/transpose:y:04sequential/subsmtgotc/strided_slice_2/stack:output:06sequential/subsmtgotc/strided_slice_2/stack_1:output:06sequential/subsmtgotc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/subsmtgotc/strided_slice_2ñ
6sequential/subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp?sequential_subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOpÿ
'sequential/subsmtgotc/zdztiqrxwb/MatMulMatMul.sequential/subsmtgotc/strided_slice_2:output:0>sequential/subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/subsmtgotc/zdztiqrxwb/MatMul÷
8sequential/subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOpAsequential_subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOpû
)sequential/subsmtgotc/zdztiqrxwb/MatMul_1MatMul$sequential/subsmtgotc/zeros:output:0@sequential/subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/subsmtgotc/zdztiqrxwb/MatMul_1ð
$sequential/subsmtgotc/zdztiqrxwb/addAddV21sequential/subsmtgotc/zdztiqrxwb/MatMul:product:03sequential/subsmtgotc/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/subsmtgotc/zdztiqrxwb/addð
7sequential/subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp@sequential_subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOpý
(sequential/subsmtgotc/zdztiqrxwb/BiasAddBiasAdd(sequential/subsmtgotc/zdztiqrxwb/add:z:0?sequential/subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/subsmtgotc/zdztiqrxwb/BiasAdd¦
0sequential/subsmtgotc/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/subsmtgotc/zdztiqrxwb/split/split_dimÃ
&sequential/subsmtgotc/zdztiqrxwb/splitSplit9sequential/subsmtgotc/zdztiqrxwb/split/split_dim:output:01sequential/subsmtgotc/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/subsmtgotc/zdztiqrxwb/split×
/sequential/subsmtgotc/zdztiqrxwb/ReadVariableOpReadVariableOp8sequential_subsmtgotc_zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/subsmtgotc/zdztiqrxwb/ReadVariableOpæ
$sequential/subsmtgotc/zdztiqrxwb/mulMul7sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp:value:0&sequential/subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/subsmtgotc/zdztiqrxwb/mulæ
&sequential/subsmtgotc/zdztiqrxwb/add_1AddV2/sequential/subsmtgotc/zdztiqrxwb/split:output:0(sequential/subsmtgotc/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/add_1½
(sequential/subsmtgotc/zdztiqrxwb/SigmoidSigmoid*sequential/subsmtgotc/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/subsmtgotc/zdztiqrxwb/SigmoidÝ
1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_1ReadVariableOp:sequential_subsmtgotc_zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_1ì
&sequential/subsmtgotc/zdztiqrxwb/mul_1Mul9sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_1:value:0&sequential/subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/mul_1è
&sequential/subsmtgotc/zdztiqrxwb/add_2AddV2/sequential/subsmtgotc/zdztiqrxwb/split:output:1*sequential/subsmtgotc/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/add_2Á
*sequential/subsmtgotc/zdztiqrxwb/Sigmoid_1Sigmoid*sequential/subsmtgotc/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/subsmtgotc/zdztiqrxwb/Sigmoid_1á
&sequential/subsmtgotc/zdztiqrxwb/mul_2Mul.sequential/subsmtgotc/zdztiqrxwb/Sigmoid_1:y:0&sequential/subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/mul_2¹
%sequential/subsmtgotc/zdztiqrxwb/TanhTanh/sequential/subsmtgotc/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/subsmtgotc/zdztiqrxwb/Tanhâ
&sequential/subsmtgotc/zdztiqrxwb/mul_3Mul,sequential/subsmtgotc/zdztiqrxwb/Sigmoid:y:0)sequential/subsmtgotc/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/mul_3ã
&sequential/subsmtgotc/zdztiqrxwb/add_3AddV2*sequential/subsmtgotc/zdztiqrxwb/mul_2:z:0*sequential/subsmtgotc/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/add_3Ý
1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_2ReadVariableOp:sequential_subsmtgotc_zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_2ð
&sequential/subsmtgotc/zdztiqrxwb/mul_4Mul9sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_2:value:0*sequential/subsmtgotc/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/mul_4è
&sequential/subsmtgotc/zdztiqrxwb/add_4AddV2/sequential/subsmtgotc/zdztiqrxwb/split:output:3*sequential/subsmtgotc/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/add_4Á
*sequential/subsmtgotc/zdztiqrxwb/Sigmoid_2Sigmoid*sequential/subsmtgotc/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/subsmtgotc/zdztiqrxwb/Sigmoid_2¸
'sequential/subsmtgotc/zdztiqrxwb/Tanh_1Tanh*sequential/subsmtgotc/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/subsmtgotc/zdztiqrxwb/Tanh_1æ
&sequential/subsmtgotc/zdztiqrxwb/mul_5Mul.sequential/subsmtgotc/zdztiqrxwb/Sigmoid_2:y:0+sequential/subsmtgotc/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/zdztiqrxwb/mul_5»
3sequential/subsmtgotc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/subsmtgotc/TensorArrayV2_1/element_shape
%sequential/subsmtgotc/TensorArrayV2_1TensorListReserve<sequential/subsmtgotc/TensorArrayV2_1/element_shape:output:0.sequential/subsmtgotc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/subsmtgotc/TensorArrayV2_1z
sequential/subsmtgotc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/subsmtgotc/time«
.sequential/subsmtgotc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/subsmtgotc/while/maximum_iterations
(sequential/subsmtgotc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/subsmtgotc/while/loop_counterø	
sequential/subsmtgotc/whileWhile1sequential/subsmtgotc/while/loop_counter:output:07sequential/subsmtgotc/while/maximum_iterations:output:0#sequential/subsmtgotc/time:output:0.sequential/subsmtgotc/TensorArrayV2_1:handle:0$sequential/subsmtgotc/zeros:output:0&sequential/subsmtgotc/zeros_1:output:0.sequential/subsmtgotc/strided_slice_1:output:0Msequential/subsmtgotc/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_subsmtgotc_zdztiqrxwb_matmul_readvariableop_resourceAsequential_subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource@sequential_subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource8sequential_subsmtgotc_zdztiqrxwb_readvariableop_resource:sequential_subsmtgotc_zdztiqrxwb_readvariableop_1_resource:sequential_subsmtgotc_zdztiqrxwb_readvariableop_2_resource*
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
(sequential_subsmtgotc_while_body_1062863*4
cond,R*
(sequential_subsmtgotc_while_cond_1062862*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/subsmtgotc/whileá
Fsequential/subsmtgotc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/subsmtgotc/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/subsmtgotc/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/subsmtgotc/while:output:3Osequential/subsmtgotc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/subsmtgotc/TensorArrayV2Stack/TensorListStack­
+sequential/subsmtgotc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/subsmtgotc/strided_slice_3/stack¨
-sequential/subsmtgotc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/subsmtgotc/strided_slice_3/stack_1¨
-sequential/subsmtgotc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/subsmtgotc/strided_slice_3/stack_2
%sequential/subsmtgotc/strided_slice_3StridedSliceAsequential/subsmtgotc/TensorArrayV2Stack/TensorListStack:tensor:04sequential/subsmtgotc/strided_slice_3/stack:output:06sequential/subsmtgotc/strided_slice_3/stack_1:output:06sequential/subsmtgotc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/subsmtgotc/strided_slice_3¥
&sequential/subsmtgotc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/subsmtgotc/transpose_1/permý
!sequential/subsmtgotc/transpose_1	TransposeAsequential/subsmtgotc/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/subsmtgotc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/subsmtgotc/transpose_1
sequential/quyyatshey/ShapeShape%sequential/subsmtgotc/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/quyyatshey/Shape 
)sequential/quyyatshey/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/quyyatshey/strided_slice/stack¤
+sequential/quyyatshey/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/quyyatshey/strided_slice/stack_1¤
+sequential/quyyatshey/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/quyyatshey/strided_slice/stack_2æ
#sequential/quyyatshey/strided_sliceStridedSlice$sequential/quyyatshey/Shape:output:02sequential/quyyatshey/strided_slice/stack:output:04sequential/quyyatshey/strided_slice/stack_1:output:04sequential/quyyatshey/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/quyyatshey/strided_slice
!sequential/quyyatshey/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/quyyatshey/zeros/mul/yÄ
sequential/quyyatshey/zeros/mulMul,sequential/quyyatshey/strided_slice:output:0*sequential/quyyatshey/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/quyyatshey/zeros/mul
"sequential/quyyatshey/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/quyyatshey/zeros/Less/y¿
 sequential/quyyatshey/zeros/LessLess#sequential/quyyatshey/zeros/mul:z:0+sequential/quyyatshey/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/quyyatshey/zeros/Less
$sequential/quyyatshey/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/quyyatshey/zeros/packed/1Û
"sequential/quyyatshey/zeros/packedPack,sequential/quyyatshey/strided_slice:output:0-sequential/quyyatshey/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/quyyatshey/zeros/packed
!sequential/quyyatshey/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/quyyatshey/zeros/ConstÍ
sequential/quyyatshey/zerosFill+sequential/quyyatshey/zeros/packed:output:0*sequential/quyyatshey/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/quyyatshey/zeros
#sequential/quyyatshey/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/quyyatshey/zeros_1/mul/yÊ
!sequential/quyyatshey/zeros_1/mulMul,sequential/quyyatshey/strided_slice:output:0,sequential/quyyatshey/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/quyyatshey/zeros_1/mul
$sequential/quyyatshey/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/quyyatshey/zeros_1/Less/yÇ
"sequential/quyyatshey/zeros_1/LessLess%sequential/quyyatshey/zeros_1/mul:z:0-sequential/quyyatshey/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/quyyatshey/zeros_1/Less
&sequential/quyyatshey/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/quyyatshey/zeros_1/packed/1á
$sequential/quyyatshey/zeros_1/packedPack,sequential/quyyatshey/strided_slice:output:0/sequential/quyyatshey/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/quyyatshey/zeros_1/packed
#sequential/quyyatshey/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/quyyatshey/zeros_1/ConstÕ
sequential/quyyatshey/zeros_1Fill-sequential/quyyatshey/zeros_1/packed:output:0,sequential/quyyatshey/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/quyyatshey/zeros_1¡
$sequential/quyyatshey/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/quyyatshey/transpose/permÛ
sequential/quyyatshey/transpose	Transpose%sequential/subsmtgotc/transpose_1:y:0-sequential/quyyatshey/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/quyyatshey/transpose
sequential/quyyatshey/Shape_1Shape#sequential/quyyatshey/transpose:y:0*
T0*
_output_shapes
:2
sequential/quyyatshey/Shape_1¤
+sequential/quyyatshey/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/quyyatshey/strided_slice_1/stack¨
-sequential/quyyatshey/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/quyyatshey/strided_slice_1/stack_1¨
-sequential/quyyatshey/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/quyyatshey/strided_slice_1/stack_2ò
%sequential/quyyatshey/strided_slice_1StridedSlice&sequential/quyyatshey/Shape_1:output:04sequential/quyyatshey/strided_slice_1/stack:output:06sequential/quyyatshey/strided_slice_1/stack_1:output:06sequential/quyyatshey/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/quyyatshey/strided_slice_1±
1sequential/quyyatshey/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/quyyatshey/TensorArrayV2/element_shape
#sequential/quyyatshey/TensorArrayV2TensorListReserve:sequential/quyyatshey/TensorArrayV2/element_shape:output:0.sequential/quyyatshey/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/quyyatshey/TensorArrayV2ë
Ksequential/quyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/quyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/quyyatshey/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/quyyatshey/transpose:y:0Tsequential/quyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/quyyatshey/TensorArrayUnstack/TensorListFromTensor¤
+sequential/quyyatshey/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/quyyatshey/strided_slice_2/stack¨
-sequential/quyyatshey/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/quyyatshey/strided_slice_2/stack_1¨
-sequential/quyyatshey/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/quyyatshey/strided_slice_2/stack_2
%sequential/quyyatshey/strided_slice_2StridedSlice#sequential/quyyatshey/transpose:y:04sequential/quyyatshey/strided_slice_2/stack:output:06sequential/quyyatshey/strided_slice_2/stack_1:output:06sequential/quyyatshey/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/quyyatshey/strided_slice_2ñ
6sequential/quyyatshey/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp?sequential_quyyatshey_nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/quyyatshey/nqcjuhnaut/MatMul/ReadVariableOpÿ
'sequential/quyyatshey/nqcjuhnaut/MatMulMatMul.sequential/quyyatshey/strided_slice_2:output:0>sequential/quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/quyyatshey/nqcjuhnaut/MatMul÷
8sequential/quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOpAsequential_quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOpû
)sequential/quyyatshey/nqcjuhnaut/MatMul_1MatMul$sequential/quyyatshey/zeros:output:0@sequential/quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/quyyatshey/nqcjuhnaut/MatMul_1ð
$sequential/quyyatshey/nqcjuhnaut/addAddV21sequential/quyyatshey/nqcjuhnaut/MatMul:product:03sequential/quyyatshey/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/quyyatshey/nqcjuhnaut/addð
7sequential/quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp@sequential_quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOpý
(sequential/quyyatshey/nqcjuhnaut/BiasAddBiasAdd(sequential/quyyatshey/nqcjuhnaut/add:z:0?sequential/quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/quyyatshey/nqcjuhnaut/BiasAdd¦
0sequential/quyyatshey/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/quyyatshey/nqcjuhnaut/split/split_dimÃ
&sequential/quyyatshey/nqcjuhnaut/splitSplit9sequential/quyyatshey/nqcjuhnaut/split/split_dim:output:01sequential/quyyatshey/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/quyyatshey/nqcjuhnaut/split×
/sequential/quyyatshey/nqcjuhnaut/ReadVariableOpReadVariableOp8sequential_quyyatshey_nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/quyyatshey/nqcjuhnaut/ReadVariableOpæ
$sequential/quyyatshey/nqcjuhnaut/mulMul7sequential/quyyatshey/nqcjuhnaut/ReadVariableOp:value:0&sequential/quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/quyyatshey/nqcjuhnaut/mulæ
&sequential/quyyatshey/nqcjuhnaut/add_1AddV2/sequential/quyyatshey/nqcjuhnaut/split:output:0(sequential/quyyatshey/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/add_1½
(sequential/quyyatshey/nqcjuhnaut/SigmoidSigmoid*sequential/quyyatshey/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/quyyatshey/nqcjuhnaut/SigmoidÝ
1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_1ReadVariableOp:sequential_quyyatshey_nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_1ì
&sequential/quyyatshey/nqcjuhnaut/mul_1Mul9sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_1:value:0&sequential/quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/mul_1è
&sequential/quyyatshey/nqcjuhnaut/add_2AddV2/sequential/quyyatshey/nqcjuhnaut/split:output:1*sequential/quyyatshey/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/add_2Á
*sequential/quyyatshey/nqcjuhnaut/Sigmoid_1Sigmoid*sequential/quyyatshey/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/quyyatshey/nqcjuhnaut/Sigmoid_1á
&sequential/quyyatshey/nqcjuhnaut/mul_2Mul.sequential/quyyatshey/nqcjuhnaut/Sigmoid_1:y:0&sequential/quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/mul_2¹
%sequential/quyyatshey/nqcjuhnaut/TanhTanh/sequential/quyyatshey/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/quyyatshey/nqcjuhnaut/Tanhâ
&sequential/quyyatshey/nqcjuhnaut/mul_3Mul,sequential/quyyatshey/nqcjuhnaut/Sigmoid:y:0)sequential/quyyatshey/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/mul_3ã
&sequential/quyyatshey/nqcjuhnaut/add_3AddV2*sequential/quyyatshey/nqcjuhnaut/mul_2:z:0*sequential/quyyatshey/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/add_3Ý
1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_2ReadVariableOp:sequential_quyyatshey_nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_2ð
&sequential/quyyatshey/nqcjuhnaut/mul_4Mul9sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_2:value:0*sequential/quyyatshey/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/mul_4è
&sequential/quyyatshey/nqcjuhnaut/add_4AddV2/sequential/quyyatshey/nqcjuhnaut/split:output:3*sequential/quyyatshey/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/add_4Á
*sequential/quyyatshey/nqcjuhnaut/Sigmoid_2Sigmoid*sequential/quyyatshey/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/quyyatshey/nqcjuhnaut/Sigmoid_2¸
'sequential/quyyatshey/nqcjuhnaut/Tanh_1Tanh*sequential/quyyatshey/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/quyyatshey/nqcjuhnaut/Tanh_1æ
&sequential/quyyatshey/nqcjuhnaut/mul_5Mul.sequential/quyyatshey/nqcjuhnaut/Sigmoid_2:y:0+sequential/quyyatshey/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/quyyatshey/nqcjuhnaut/mul_5»
3sequential/quyyatshey/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/quyyatshey/TensorArrayV2_1/element_shape
%sequential/quyyatshey/TensorArrayV2_1TensorListReserve<sequential/quyyatshey/TensorArrayV2_1/element_shape:output:0.sequential/quyyatshey/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/quyyatshey/TensorArrayV2_1z
sequential/quyyatshey/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/quyyatshey/time«
.sequential/quyyatshey/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/quyyatshey/while/maximum_iterations
(sequential/quyyatshey/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/quyyatshey/while/loop_counterø	
sequential/quyyatshey/whileWhile1sequential/quyyatshey/while/loop_counter:output:07sequential/quyyatshey/while/maximum_iterations:output:0#sequential/quyyatshey/time:output:0.sequential/quyyatshey/TensorArrayV2_1:handle:0$sequential/quyyatshey/zeros:output:0&sequential/quyyatshey/zeros_1:output:0.sequential/quyyatshey/strided_slice_1:output:0Msequential/quyyatshey/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_quyyatshey_nqcjuhnaut_matmul_readvariableop_resourceAsequential_quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource@sequential_quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource8sequential_quyyatshey_nqcjuhnaut_readvariableop_resource:sequential_quyyatshey_nqcjuhnaut_readvariableop_1_resource:sequential_quyyatshey_nqcjuhnaut_readvariableop_2_resource*
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
(sequential_quyyatshey_while_body_1063039*4
cond,R*
(sequential_quyyatshey_while_cond_1063038*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/quyyatshey/whileá
Fsequential/quyyatshey/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/quyyatshey/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/quyyatshey/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/quyyatshey/while:output:3Osequential/quyyatshey/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/quyyatshey/TensorArrayV2Stack/TensorListStack­
+sequential/quyyatshey/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/quyyatshey/strided_slice_3/stack¨
-sequential/quyyatshey/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/quyyatshey/strided_slice_3/stack_1¨
-sequential/quyyatshey/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/quyyatshey/strided_slice_3/stack_2
%sequential/quyyatshey/strided_slice_3StridedSliceAsequential/quyyatshey/TensorArrayV2Stack/TensorListStack:tensor:04sequential/quyyatshey/strided_slice_3/stack:output:06sequential/quyyatshey/strided_slice_3/stack_1:output:06sequential/quyyatshey/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/quyyatshey/strided_slice_3¥
&sequential/quyyatshey/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/quyyatshey/transpose_1/permý
!sequential/quyyatshey/transpose_1	TransposeAsequential/quyyatshey/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/quyyatshey/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/quyyatshey/transpose_1Ï
+sequential/ycxcgxamrr/MatMul/ReadVariableOpReadVariableOp4sequential_ycxcgxamrr_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/ycxcgxamrr/MatMul/ReadVariableOpÝ
sequential/ycxcgxamrr/MatMulMatMul.sequential/quyyatshey/strided_slice_3:output:03sequential/ycxcgxamrr/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/ycxcgxamrr/MatMulÎ
,sequential/ycxcgxamrr/BiasAdd/ReadVariableOpReadVariableOp5sequential_ycxcgxamrr_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/ycxcgxamrr/BiasAdd/ReadVariableOpÙ
sequential/ycxcgxamrr/BiasAddBiasAdd&sequential/ycxcgxamrr/MatMul:product:04sequential/ycxcgxamrr/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/ycxcgxamrr/BiasAdd 
IdentityIdentity&sequential/ycxcgxamrr/BiasAdd:output:08^sequential/quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp7^sequential/quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp9^sequential/quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp0^sequential/quyyatshey/nqcjuhnaut/ReadVariableOp2^sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_12^sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_2^sequential/quyyatshey/while^sequential/subsmtgotc/while8^sequential/subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp7^sequential/subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp9^sequential/subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp0^sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp2^sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_12^sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_29^sequential/umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp@^sequential/umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp-^sequential/ycxcgxamrr/BiasAdd/ReadVariableOp,^sequential/ycxcgxamrr/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2r
7sequential/quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp7sequential/quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp2p
6sequential/quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp6sequential/quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp2t
8sequential/quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp8sequential/quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp2b
/sequential/quyyatshey/nqcjuhnaut/ReadVariableOp/sequential/quyyatshey/nqcjuhnaut/ReadVariableOp2f
1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_11sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_12f
1sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_21sequential/quyyatshey/nqcjuhnaut/ReadVariableOp_22:
sequential/quyyatshey/whilesequential/quyyatshey/while2:
sequential/subsmtgotc/whilesequential/subsmtgotc/while2r
7sequential/subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp7sequential/subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp2p
6sequential/subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp6sequential/subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp2t
8sequential/subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp8sequential/subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp2b
/sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp/sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp2f
1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_11sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_12f
1sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_21sequential/subsmtgotc/zdztiqrxwb/ReadVariableOp_22t
8sequential/umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp8sequential/umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,sequential/ycxcgxamrr/BiasAdd/ReadVariableOp,sequential/ycxcgxamrr/BiasAdd/ReadVariableOp2Z
+sequential/ycxcgxamrr/MatMul/ReadVariableOp+sequential/ycxcgxamrr/MatMul/ReadVariableOp:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
mkdkkixskm
àY

while_body_1066991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_zdztiqrxwb_matmul_readvariableop_resource_0:	F
3while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 A
2while_zdztiqrxwb_biasadd_readvariableop_resource_0:	8
*while_zdztiqrxwb_readvariableop_resource_0: :
,while_zdztiqrxwb_readvariableop_1_resource_0: :
,while_zdztiqrxwb_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_zdztiqrxwb_matmul_readvariableop_resource:	D
1while_zdztiqrxwb_matmul_1_readvariableop_resource:	 ?
0while_zdztiqrxwb_biasadd_readvariableop_resource:	6
(while_zdztiqrxwb_readvariableop_resource: 8
*while_zdztiqrxwb_readvariableop_1_resource: 8
*while_zdztiqrxwb_readvariableop_2_resource: ¢'while/zdztiqrxwb/BiasAdd/ReadVariableOp¢&while/zdztiqrxwb/MatMul/ReadVariableOp¢(while/zdztiqrxwb/MatMul_1/ReadVariableOp¢while/zdztiqrxwb/ReadVariableOp¢!while/zdztiqrxwb/ReadVariableOp_1¢!while/zdztiqrxwb/ReadVariableOp_2Ã
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
&while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp1while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/zdztiqrxwb/MatMul/ReadVariableOpÑ
while/zdztiqrxwb/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMulÉ
(while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp3while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/zdztiqrxwb/MatMul_1/ReadVariableOpº
while/zdztiqrxwb/MatMul_1MatMulwhile_placeholder_20while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMul_1°
while/zdztiqrxwb/addAddV2!while/zdztiqrxwb/MatMul:product:0#while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/addÂ
'while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp2while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/zdztiqrxwb/BiasAdd/ReadVariableOp½
while/zdztiqrxwb/BiasAddBiasAddwhile/zdztiqrxwb/add:z:0/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/BiasAdd
 while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/zdztiqrxwb/split/split_dim
while/zdztiqrxwb/splitSplit)while/zdztiqrxwb/split/split_dim:output:0!while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/zdztiqrxwb/split©
while/zdztiqrxwb/ReadVariableOpReadVariableOp*while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/zdztiqrxwb/ReadVariableOp£
while/zdztiqrxwb/mulMul'while/zdztiqrxwb/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul¦
while/zdztiqrxwb/add_1AddV2while/zdztiqrxwb/split:output:0while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_1
while/zdztiqrxwb/SigmoidSigmoidwhile/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid¯
!while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp,while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_1©
while/zdztiqrxwb/mul_1Mul)while/zdztiqrxwb/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_1¨
while/zdztiqrxwb/add_2AddV2while/zdztiqrxwb/split:output:1while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_2
while/zdztiqrxwb/Sigmoid_1Sigmoidwhile/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_1
while/zdztiqrxwb/mul_2Mulwhile/zdztiqrxwb/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_2
while/zdztiqrxwb/TanhTanhwhile/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh¢
while/zdztiqrxwb/mul_3Mulwhile/zdztiqrxwb/Sigmoid:y:0while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_3£
while/zdztiqrxwb/add_3AddV2while/zdztiqrxwb/mul_2:z:0while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_3¯
!while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp,while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_2°
while/zdztiqrxwb/mul_4Mul)while/zdztiqrxwb/ReadVariableOp_2:value:0while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_4¨
while/zdztiqrxwb/add_4AddV2while/zdztiqrxwb/split:output:3while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_4
while/zdztiqrxwb/Sigmoid_2Sigmoidwhile/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_2
while/zdztiqrxwb/Tanh_1Tanhwhile/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh_1¦
while/zdztiqrxwb/mul_5Mulwhile/zdztiqrxwb/Sigmoid_2:y:0while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/zdztiqrxwb/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/zdztiqrxwb/mul_5:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/zdztiqrxwb/add_3:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
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
0while_zdztiqrxwb_biasadd_readvariableop_resource2while_zdztiqrxwb_biasadd_readvariableop_resource_0"h
1while_zdztiqrxwb_matmul_1_readvariableop_resource3while_zdztiqrxwb_matmul_1_readvariableop_resource_0"d
/while_zdztiqrxwb_matmul_readvariableop_resource1while_zdztiqrxwb_matmul_readvariableop_resource_0"Z
*while_zdztiqrxwb_readvariableop_1_resource,while_zdztiqrxwb_readvariableop_1_resource_0"Z
*while_zdztiqrxwb_readvariableop_2_resource,while_zdztiqrxwb_readvariableop_2_resource_0"V
(while_zdztiqrxwb_readvariableop_resource*while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/zdztiqrxwb/BiasAdd/ReadVariableOp'while/zdztiqrxwb/BiasAdd/ReadVariableOp2P
&while/zdztiqrxwb/MatMul/ReadVariableOp&while/zdztiqrxwb/MatMul/ReadVariableOp2T
(while/zdztiqrxwb/MatMul_1/ReadVariableOp(while/zdztiqrxwb/MatMul_1/ReadVariableOp2B
while/zdztiqrxwb/ReadVariableOpwhile/zdztiqrxwb/ReadVariableOp2F
!while/zdztiqrxwb/ReadVariableOp_1!while/zdztiqrxwb/ReadVariableOp_12F
!while/zdztiqrxwb/ReadVariableOp_2!while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1065374

inputs<
)nqcjuhnaut_matmul_readvariableop_resource:	 >
+nqcjuhnaut_matmul_1_readvariableop_resource:	 9
*nqcjuhnaut_biasadd_readvariableop_resource:	0
"nqcjuhnaut_readvariableop_resource: 2
$nqcjuhnaut_readvariableop_1_resource: 2
$nqcjuhnaut_readvariableop_2_resource: 
identity¢!nqcjuhnaut/BiasAdd/ReadVariableOp¢ nqcjuhnaut/MatMul/ReadVariableOp¢"nqcjuhnaut/MatMul_1/ReadVariableOp¢nqcjuhnaut/ReadVariableOp¢nqcjuhnaut/ReadVariableOp_1¢nqcjuhnaut/ReadVariableOp_2¢whileD
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
 nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp)nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 nqcjuhnaut/MatMul/ReadVariableOp§
nqcjuhnaut/MatMulMatMulstrided_slice_2:output:0(nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMulµ
"nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp+nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"nqcjuhnaut/MatMul_1/ReadVariableOp£
nqcjuhnaut/MatMul_1MatMulzeros:output:0*nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/MatMul_1
nqcjuhnaut/addAddV2nqcjuhnaut/MatMul:product:0nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/add®
!nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp*nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!nqcjuhnaut/BiasAdd/ReadVariableOp¥
nqcjuhnaut/BiasAddBiasAddnqcjuhnaut/add:z:0)nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nqcjuhnaut/BiasAddz
nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
nqcjuhnaut/split/split_dimë
nqcjuhnaut/splitSplit#nqcjuhnaut/split/split_dim:output:0nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nqcjuhnaut/split
nqcjuhnaut/ReadVariableOpReadVariableOp"nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp
nqcjuhnaut/mulMul!nqcjuhnaut/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul
nqcjuhnaut/add_1AddV2nqcjuhnaut/split:output:0nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_1{
nqcjuhnaut/SigmoidSigmoidnqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid
nqcjuhnaut/ReadVariableOp_1ReadVariableOp$nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_1
nqcjuhnaut/mul_1Mul#nqcjuhnaut/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_1
nqcjuhnaut/add_2AddV2nqcjuhnaut/split:output:1nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_2
nqcjuhnaut/Sigmoid_1Sigmoidnqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_1
nqcjuhnaut/mul_2Mulnqcjuhnaut/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_2w
nqcjuhnaut/TanhTanhnqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh
nqcjuhnaut/mul_3Mulnqcjuhnaut/Sigmoid:y:0nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_3
nqcjuhnaut/add_3AddV2nqcjuhnaut/mul_2:z:0nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_3
nqcjuhnaut/ReadVariableOp_2ReadVariableOp$nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02
nqcjuhnaut/ReadVariableOp_2
nqcjuhnaut/mul_4Mul#nqcjuhnaut/ReadVariableOp_2:value:0nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_4
nqcjuhnaut/add_4AddV2nqcjuhnaut/split:output:3nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/add_4
nqcjuhnaut/Sigmoid_2Sigmoidnqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Sigmoid_2v
nqcjuhnaut/Tanh_1Tanhnqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/Tanh_1
nqcjuhnaut/mul_5Mulnqcjuhnaut/Sigmoid_2:y:0nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nqcjuhnaut/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)nqcjuhnaut_matmul_readvariableop_resource+nqcjuhnaut_matmul_1_readvariableop_resource*nqcjuhnaut_biasadd_readvariableop_resource"nqcjuhnaut_readvariableop_resource$nqcjuhnaut_readvariableop_1_resource$nqcjuhnaut_readvariableop_2_resource*
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
while_body_1065273*
condR
while_cond_1065272*Q
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
IdentityIdentitystrided_slice_3:output:0"^nqcjuhnaut/BiasAdd/ReadVariableOp!^nqcjuhnaut/MatMul/ReadVariableOp#^nqcjuhnaut/MatMul_1/ReadVariableOp^nqcjuhnaut/ReadVariableOp^nqcjuhnaut/ReadVariableOp_1^nqcjuhnaut/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!nqcjuhnaut/BiasAdd/ReadVariableOp!nqcjuhnaut/BiasAdd/ReadVariableOp2D
 nqcjuhnaut/MatMul/ReadVariableOp nqcjuhnaut/MatMul/ReadVariableOp2H
"nqcjuhnaut/MatMul_1/ReadVariableOp"nqcjuhnaut/MatMul_1/ReadVariableOp26
nqcjuhnaut/ReadVariableOpnqcjuhnaut/ReadVariableOp2:
nqcjuhnaut/ReadVariableOp_1nqcjuhnaut/ReadVariableOp_12:
nqcjuhnaut/ReadVariableOp_2nqcjuhnaut/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±'
³
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1064178

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
,__inference_sequential_layer_call_fn_1065771

mkdkkixskm
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
mkdkkixskmunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_10656992
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
mkdkkixskm


subsmtgotc_while_cond_10664962
.subsmtgotc_while_subsmtgotc_while_loop_counter8
4subsmtgotc_while_subsmtgotc_while_maximum_iterations 
subsmtgotc_while_placeholder"
subsmtgotc_while_placeholder_1"
subsmtgotc_while_placeholder_2"
subsmtgotc_while_placeholder_34
0subsmtgotc_while_less_subsmtgotc_strided_slice_1K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066496___redundant_placeholder0K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066496___redundant_placeholder1K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066496___redundant_placeholder2K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066496___redundant_placeholder3K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066496___redundant_placeholder4K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066496___redundant_placeholder5K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066496___redundant_placeholder6
subsmtgotc_while_identity
§
subsmtgotc/while/LessLesssubsmtgotc_while_placeholder0subsmtgotc_while_less_subsmtgotc_strided_slice_1*
T0*
_output_shapes
: 2
subsmtgotc/while/Less~
subsmtgotc/while/IdentityIdentitysubsmtgotc/while/Less:z:0*
T0
*
_output_shapes
: 2
subsmtgotc/while/Identity"?
subsmtgotc_while_identity"subsmtgotc/while/Identity:output:0*(
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
(sequential_subsmtgotc_while_cond_1062862H
Dsequential_subsmtgotc_while_sequential_subsmtgotc_while_loop_counterN
Jsequential_subsmtgotc_while_sequential_subsmtgotc_while_maximum_iterations+
'sequential_subsmtgotc_while_placeholder-
)sequential_subsmtgotc_while_placeholder_1-
)sequential_subsmtgotc_while_placeholder_2-
)sequential_subsmtgotc_while_placeholder_3J
Fsequential_subsmtgotc_while_less_sequential_subsmtgotc_strided_slice_1a
]sequential_subsmtgotc_while_sequential_subsmtgotc_while_cond_1062862___redundant_placeholder0a
]sequential_subsmtgotc_while_sequential_subsmtgotc_while_cond_1062862___redundant_placeholder1a
]sequential_subsmtgotc_while_sequential_subsmtgotc_while_cond_1062862___redundant_placeholder2a
]sequential_subsmtgotc_while_sequential_subsmtgotc_while_cond_1062862___redundant_placeholder3a
]sequential_subsmtgotc_while_sequential_subsmtgotc_while_cond_1062862___redundant_placeholder4a
]sequential_subsmtgotc_while_sequential_subsmtgotc_while_cond_1062862___redundant_placeholder5a
]sequential_subsmtgotc_while_sequential_subsmtgotc_while_cond_1062862___redundant_placeholder6(
$sequential_subsmtgotc_while_identity
Þ
 sequential/subsmtgotc/while/LessLess'sequential_subsmtgotc_while_placeholderFsequential_subsmtgotc_while_less_sequential_subsmtgotc_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/subsmtgotc/while/Less
$sequential/subsmtgotc/while/IdentityIdentity$sequential/subsmtgotc/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/subsmtgotc/while/Identity"U
$sequential_subsmtgotc_while_identity-sequential/subsmtgotc/while/Identity:output:0*(
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
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_1065812

mkdkkixskm(
umdyqemnpr_1065774: 
umdyqemnpr_1065776:%
subsmtgotc_1065780:	%
subsmtgotc_1065782:	 !
subsmtgotc_1065784:	 
subsmtgotc_1065786:  
subsmtgotc_1065788:  
subsmtgotc_1065790: %
quyyatshey_1065793:	 %
quyyatshey_1065795:	 !
quyyatshey_1065797:	 
quyyatshey_1065799:  
quyyatshey_1065801:  
quyyatshey_1065803: $
ycxcgxamrr_1065806:  
ycxcgxamrr_1065808:
identity¢"quyyatshey/StatefulPartitionedCall¢"subsmtgotc/StatefulPartitionedCall¢"umdyqemnpr/StatefulPartitionedCall¢"ycxcgxamrr/StatefulPartitionedCall°
"umdyqemnpr/StatefulPartitionedCallStatefulPartitionedCall
mkdkkixskmumdyqemnpr_1065774umdyqemnpr_1065776*
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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_10647062$
"umdyqemnpr/StatefulPartitionedCall
unxqeixodn/PartitionedCallPartitionedCall+umdyqemnpr/StatefulPartitionedCall:output:0*
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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_10647252
unxqeixodn/PartitionedCall
"subsmtgotc/StatefulPartitionedCallStatefulPartitionedCall#unxqeixodn/PartitionedCall:output:0subsmtgotc_1065780subsmtgotc_1065782subsmtgotc_1065784subsmtgotc_1065786subsmtgotc_1065788subsmtgotc_1065790*
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10649062$
"subsmtgotc/StatefulPartitionedCall¡
"quyyatshey/StatefulPartitionedCallStatefulPartitionedCall+subsmtgotc/StatefulPartitionedCall:output:0quyyatshey_1065793quyyatshey_1065795quyyatshey_1065797quyyatshey_1065799quyyatshey_1065801quyyatshey_1065803*
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10650992$
"quyyatshey/StatefulPartitionedCallÉ
"ycxcgxamrr/StatefulPartitionedCallStatefulPartitionedCall+quyyatshey/StatefulPartitionedCall:output:0ycxcgxamrr_1065806ycxcgxamrr_1065808*
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
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_10651232$
"ycxcgxamrr/StatefulPartitionedCall
IdentityIdentity+ycxcgxamrr/StatefulPartitionedCall:output:0#^quyyatshey/StatefulPartitionedCall#^subsmtgotc/StatefulPartitionedCall#^umdyqemnpr/StatefulPartitionedCall#^ycxcgxamrr/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"quyyatshey/StatefulPartitionedCall"quyyatshey/StatefulPartitionedCall2H
"subsmtgotc/StatefulPartitionedCall"subsmtgotc/StatefulPartitionedCall2H
"umdyqemnpr/StatefulPartitionedCall"umdyqemnpr/StatefulPartitionedCall2H
"ycxcgxamrr/StatefulPartitionedCall"ycxcgxamrr/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
mkdkkixskm
±'
³
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1063420

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
Ü

(sequential_subsmtgotc_while_body_1062863H
Dsequential_subsmtgotc_while_sequential_subsmtgotc_while_loop_counterN
Jsequential_subsmtgotc_while_sequential_subsmtgotc_while_maximum_iterations+
'sequential_subsmtgotc_while_placeholder-
)sequential_subsmtgotc_while_placeholder_1-
)sequential_subsmtgotc_while_placeholder_2-
)sequential_subsmtgotc_while_placeholder_3G
Csequential_subsmtgotc_while_sequential_subsmtgotc_strided_slice_1_0
sequential_subsmtgotc_while_tensorarrayv2read_tensorlistgetitem_sequential_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0:	\
Isequential_subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 W
Hsequential_subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0:	N
@sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0: P
Bsequential_subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0: P
Bsequential_subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0: (
$sequential_subsmtgotc_while_identity*
&sequential_subsmtgotc_while_identity_1*
&sequential_subsmtgotc_while_identity_2*
&sequential_subsmtgotc_while_identity_3*
&sequential_subsmtgotc_while_identity_4*
&sequential_subsmtgotc_while_identity_5E
Asequential_subsmtgotc_while_sequential_subsmtgotc_strided_slice_1
}sequential_subsmtgotc_while_tensorarrayv2read_tensorlistgetitem_sequential_subsmtgotc_tensorarrayunstack_tensorlistfromtensorX
Esequential_subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource:	Z
Gsequential_subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource:	 U
Fsequential_subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource:	L
>sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_resource: N
@sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource: N
@sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource: ¢=sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp¢<sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp¢>sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp¢5sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp¢7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1¢7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2ï
Msequential/subsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/subsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/subsmtgotc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_subsmtgotc_while_tensorarrayv2read_tensorlistgetitem_sequential_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0'sequential_subsmtgotc_while_placeholderVsequential/subsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/subsmtgotc/while/TensorArrayV2Read/TensorListGetItem
<sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOpGsequential_subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp©
-sequential/subsmtgotc/while/zdztiqrxwb/MatMulMatMulFsequential/subsmtgotc/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/subsmtgotc/while/zdztiqrxwb/MatMul
>sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOpIsequential_subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp
/sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1MatMul)sequential_subsmtgotc_while_placeholder_2Fsequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1
*sequential/subsmtgotc/while/zdztiqrxwb/addAddV27sequential/subsmtgotc/while/zdztiqrxwb/MatMul:product:09sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/subsmtgotc/while/zdztiqrxwb/add
=sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOpHsequential_subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp
.sequential/subsmtgotc/while/zdztiqrxwb/BiasAddBiasAdd.sequential/subsmtgotc/while/zdztiqrxwb/add:z:0Esequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd²
6sequential/subsmtgotc/while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/subsmtgotc/while/zdztiqrxwb/split/split_dimÛ
,sequential/subsmtgotc/while/zdztiqrxwb/splitSplit?sequential/subsmtgotc/while/zdztiqrxwb/split/split_dim:output:07sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/subsmtgotc/while/zdztiqrxwb/splitë
5sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOpReadVariableOp@sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOpû
*sequential/subsmtgotc/while/zdztiqrxwb/mulMul=sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp:value:0)sequential_subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/subsmtgotc/while/zdztiqrxwb/mulþ
,sequential/subsmtgotc/while/zdztiqrxwb/add_1AddV25sequential/subsmtgotc/while/zdztiqrxwb/split:output:0.sequential/subsmtgotc/while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/add_1Ï
.sequential/subsmtgotc/while/zdztiqrxwb/SigmoidSigmoid0sequential/subsmtgotc/while/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/subsmtgotc/while/zdztiqrxwb/Sigmoidñ
7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1ReadVariableOpBsequential_subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1
,sequential/subsmtgotc/while/zdztiqrxwb/mul_1Mul?sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1:value:0)sequential_subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/mul_1
,sequential/subsmtgotc/while/zdztiqrxwb/add_2AddV25sequential/subsmtgotc/while/zdztiqrxwb/split:output:10sequential/subsmtgotc/while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/add_2Ó
0sequential/subsmtgotc/while/zdztiqrxwb/Sigmoid_1Sigmoid0sequential/subsmtgotc/while/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/subsmtgotc/while/zdztiqrxwb/Sigmoid_1ö
,sequential/subsmtgotc/while/zdztiqrxwb/mul_2Mul4sequential/subsmtgotc/while/zdztiqrxwb/Sigmoid_1:y:0)sequential_subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/mul_2Ë
+sequential/subsmtgotc/while/zdztiqrxwb/TanhTanh5sequential/subsmtgotc/while/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/subsmtgotc/while/zdztiqrxwb/Tanhú
,sequential/subsmtgotc/while/zdztiqrxwb/mul_3Mul2sequential/subsmtgotc/while/zdztiqrxwb/Sigmoid:y:0/sequential/subsmtgotc/while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/mul_3û
,sequential/subsmtgotc/while/zdztiqrxwb/add_3AddV20sequential/subsmtgotc/while/zdztiqrxwb/mul_2:z:00sequential/subsmtgotc/while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/add_3ñ
7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2ReadVariableOpBsequential_subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2
,sequential/subsmtgotc/while/zdztiqrxwb/mul_4Mul?sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2:value:00sequential/subsmtgotc/while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/mul_4
,sequential/subsmtgotc/while/zdztiqrxwb/add_4AddV25sequential/subsmtgotc/while/zdztiqrxwb/split:output:30sequential/subsmtgotc/while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/add_4Ó
0sequential/subsmtgotc/while/zdztiqrxwb/Sigmoid_2Sigmoid0sequential/subsmtgotc/while/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/subsmtgotc/while/zdztiqrxwb/Sigmoid_2Ê
-sequential/subsmtgotc/while/zdztiqrxwb/Tanh_1Tanh0sequential/subsmtgotc/while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/subsmtgotc/while/zdztiqrxwb/Tanh_1þ
,sequential/subsmtgotc/while/zdztiqrxwb/mul_5Mul4sequential/subsmtgotc/while/zdztiqrxwb/Sigmoid_2:y:01sequential/subsmtgotc/while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/subsmtgotc/while/zdztiqrxwb/mul_5Ì
@sequential/subsmtgotc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_subsmtgotc_while_placeholder_1'sequential_subsmtgotc_while_placeholder0sequential/subsmtgotc/while/zdztiqrxwb/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/subsmtgotc/while/TensorArrayV2Write/TensorListSetItem
!sequential/subsmtgotc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/subsmtgotc/while/add/yÁ
sequential/subsmtgotc/while/addAddV2'sequential_subsmtgotc_while_placeholder*sequential/subsmtgotc/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/subsmtgotc/while/add
#sequential/subsmtgotc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/subsmtgotc/while/add_1/yä
!sequential/subsmtgotc/while/add_1AddV2Dsequential_subsmtgotc_while_sequential_subsmtgotc_while_loop_counter,sequential/subsmtgotc/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/subsmtgotc/while/add_1
$sequential/subsmtgotc/while/IdentityIdentity%sequential/subsmtgotc/while/add_1:z:0>^sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp=^sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp?^sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp6^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp8^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_18^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/subsmtgotc/while/Identityµ
&sequential/subsmtgotc/while/Identity_1IdentityJsequential_subsmtgotc_while_sequential_subsmtgotc_while_maximum_iterations>^sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp=^sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp?^sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp6^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp8^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_18^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/subsmtgotc/while/Identity_1
&sequential/subsmtgotc/while/Identity_2Identity#sequential/subsmtgotc/while/add:z:0>^sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp=^sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp?^sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp6^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp8^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_18^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/subsmtgotc/while/Identity_2»
&sequential/subsmtgotc/while/Identity_3IdentityPsequential/subsmtgotc/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp=^sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp?^sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp6^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp8^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_18^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/subsmtgotc/while/Identity_3¬
&sequential/subsmtgotc/while/Identity_4Identity0sequential/subsmtgotc/while/zdztiqrxwb/mul_5:z:0>^sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp=^sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp?^sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp6^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp8^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_18^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/while/Identity_4¬
&sequential/subsmtgotc/while/Identity_5Identity0sequential/subsmtgotc/while/zdztiqrxwb/add_3:z:0>^sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp=^sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp?^sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp6^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp8^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_18^sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/subsmtgotc/while/Identity_5"U
$sequential_subsmtgotc_while_identity-sequential/subsmtgotc/while/Identity:output:0"Y
&sequential_subsmtgotc_while_identity_1/sequential/subsmtgotc/while/Identity_1:output:0"Y
&sequential_subsmtgotc_while_identity_2/sequential/subsmtgotc/while/Identity_2:output:0"Y
&sequential_subsmtgotc_while_identity_3/sequential/subsmtgotc/while/Identity_3:output:0"Y
&sequential_subsmtgotc_while_identity_4/sequential/subsmtgotc/while/Identity_4:output:0"Y
&sequential_subsmtgotc_while_identity_5/sequential/subsmtgotc/while/Identity_5:output:0"
Asequential_subsmtgotc_while_sequential_subsmtgotc_strided_slice_1Csequential_subsmtgotc_while_sequential_subsmtgotc_strided_slice_1_0"
}sequential_subsmtgotc_while_tensorarrayv2read_tensorlistgetitem_sequential_subsmtgotc_tensorarrayunstack_tensorlistfromtensorsequential_subsmtgotc_while_tensorarrayv2read_tensorlistgetitem_sequential_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0"
Fsequential_subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resourceHsequential_subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0"
Gsequential_subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resourceIsequential_subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0"
Esequential_subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resourceGsequential_subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0"
@sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_1_resourceBsequential_subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0"
@sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_2_resourceBsequential_subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0"
>sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_resource@sequential_subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp=sequential/subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2|
<sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp<sequential/subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp2
>sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp>sequential/subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp2n
5sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp5sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp2r
7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_17sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_12r
7sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_27sequential/subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1064906

inputs<
)zdztiqrxwb_matmul_readvariableop_resource:	>
+zdztiqrxwb_matmul_1_readvariableop_resource:	 9
*zdztiqrxwb_biasadd_readvariableop_resource:	0
"zdztiqrxwb_readvariableop_resource: 2
$zdztiqrxwb_readvariableop_1_resource: 2
$zdztiqrxwb_readvariableop_2_resource: 
identity¢while¢!zdztiqrxwb/BiasAdd/ReadVariableOp¢ zdztiqrxwb/MatMul/ReadVariableOp¢"zdztiqrxwb/MatMul_1/ReadVariableOp¢zdztiqrxwb/ReadVariableOp¢zdztiqrxwb/ReadVariableOp_1¢zdztiqrxwb/ReadVariableOp_2D
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
 zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp)zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 zdztiqrxwb/MatMul/ReadVariableOp§
zdztiqrxwb/MatMulMatMulstrided_slice_2:output:0(zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMulµ
"zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp+zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"zdztiqrxwb/MatMul_1/ReadVariableOp£
zdztiqrxwb/MatMul_1MatMulzeros:output:0*zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMul_1
zdztiqrxwb/addAddV2zdztiqrxwb/MatMul:product:0zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/add®
!zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp*zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!zdztiqrxwb/BiasAdd/ReadVariableOp¥
zdztiqrxwb/BiasAddBiasAddzdztiqrxwb/add:z:0)zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/BiasAddz
zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
zdztiqrxwb/split/split_dimë
zdztiqrxwb/splitSplit#zdztiqrxwb/split/split_dim:output:0zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zdztiqrxwb/split
zdztiqrxwb/ReadVariableOpReadVariableOp"zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp
zdztiqrxwb/mulMul!zdztiqrxwb/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul
zdztiqrxwb/add_1AddV2zdztiqrxwb/split:output:0zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_1{
zdztiqrxwb/SigmoidSigmoidzdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid
zdztiqrxwb/ReadVariableOp_1ReadVariableOp$zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_1
zdztiqrxwb/mul_1Mul#zdztiqrxwb/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_1
zdztiqrxwb/add_2AddV2zdztiqrxwb/split:output:1zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_2
zdztiqrxwb/Sigmoid_1Sigmoidzdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_1
zdztiqrxwb/mul_2Mulzdztiqrxwb/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_2w
zdztiqrxwb/TanhTanhzdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh
zdztiqrxwb/mul_3Mulzdztiqrxwb/Sigmoid:y:0zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_3
zdztiqrxwb/add_3AddV2zdztiqrxwb/mul_2:z:0zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_3
zdztiqrxwb/ReadVariableOp_2ReadVariableOp$zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_2
zdztiqrxwb/mul_4Mul#zdztiqrxwb/ReadVariableOp_2:value:0zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_4
zdztiqrxwb/add_4AddV2zdztiqrxwb/split:output:3zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_4
zdztiqrxwb/Sigmoid_2Sigmoidzdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_2v
zdztiqrxwb/Tanh_1Tanhzdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh_1
zdztiqrxwb/mul_5Mulzdztiqrxwb/Sigmoid_2:y:0zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)zdztiqrxwb_matmul_readvariableop_resource+zdztiqrxwb_matmul_1_readvariableop_resource*zdztiqrxwb_biasadd_readvariableop_resource"zdztiqrxwb_readvariableop_resource$zdztiqrxwb_readvariableop_1_resource$zdztiqrxwb_readvariableop_2_resource*
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
while_body_1064805*
condR
while_cond_1064804*Q
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
IdentityIdentitytranspose_1:y:0^while"^zdztiqrxwb/BiasAdd/ReadVariableOp!^zdztiqrxwb/MatMul/ReadVariableOp#^zdztiqrxwb/MatMul_1/ReadVariableOp^zdztiqrxwb/ReadVariableOp^zdztiqrxwb/ReadVariableOp_1^zdztiqrxwb/ReadVariableOp_2*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2F
!zdztiqrxwb/BiasAdd/ReadVariableOp!zdztiqrxwb/BiasAdd/ReadVariableOp2D
 zdztiqrxwb/MatMul/ReadVariableOp zdztiqrxwb/MatMul/ReadVariableOp2H
"zdztiqrxwb/MatMul_1/ReadVariableOp"zdztiqrxwb/MatMul_1/ReadVariableOp26
zdztiqrxwb/ReadVariableOpzdztiqrxwb/ReadVariableOp2:
zdztiqrxwb/ReadVariableOp_1zdztiqrxwb/ReadVariableOp_12:
zdztiqrxwb/ReadVariableOp_2zdztiqrxwb/ReadVariableOp_2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç)
Ò
while_body_1063253
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_zdztiqrxwb_1063277_0:	-
while_zdztiqrxwb_1063279_0:	 )
while_zdztiqrxwb_1063281_0:	(
while_zdztiqrxwb_1063283_0: (
while_zdztiqrxwb_1063285_0: (
while_zdztiqrxwb_1063287_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_zdztiqrxwb_1063277:	+
while_zdztiqrxwb_1063279:	 '
while_zdztiqrxwb_1063281:	&
while_zdztiqrxwb_1063283: &
while_zdztiqrxwb_1063285: &
while_zdztiqrxwb_1063287: ¢(while/zdztiqrxwb/StatefulPartitionedCallÃ
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
(while/zdztiqrxwb/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_zdztiqrxwb_1063277_0while_zdztiqrxwb_1063279_0while_zdztiqrxwb_1063281_0while_zdztiqrxwb_1063283_0while_zdztiqrxwb_1063285_0while_zdztiqrxwb_1063287_0*
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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_10632332*
(while/zdztiqrxwb/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/zdztiqrxwb/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/zdztiqrxwb/StatefulPartitionedCall:output:1)^while/zdztiqrxwb/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/zdztiqrxwb/StatefulPartitionedCall:output:2)^while/zdztiqrxwb/StatefulPartitionedCall*
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
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"6
while_zdztiqrxwb_1063277while_zdztiqrxwb_1063277_0"6
while_zdztiqrxwb_1063279while_zdztiqrxwb_1063279_0"6
while_zdztiqrxwb_1063281while_zdztiqrxwb_1063281_0"6
while_zdztiqrxwb_1063283while_zdztiqrxwb_1063283_0"6
while_zdztiqrxwb_1063285while_zdztiqrxwb_1063285_0"6
while_zdztiqrxwb_1063287while_zdztiqrxwb_1063287_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/zdztiqrxwb/StatefulPartitionedCall(while/zdztiqrxwb/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1065486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1065486___redundant_placeholder05
1while_while_cond_1065486___redundant_placeholder15
1while_while_cond_1065486___redundant_placeholder25
1while_while_cond_1065486___redundant_placeholder35
1while_while_cond_1065486___redundant_placeholder45
1while_while_cond_1065486___redundant_placeholder55
1while_while_cond_1065486___redundant_placeholder6
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
while_body_1065273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_nqcjuhnaut_matmul_readvariableop_resource_0:	 F
3while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 A
2while_nqcjuhnaut_biasadd_readvariableop_resource_0:	8
*while_nqcjuhnaut_readvariableop_resource_0: :
,while_nqcjuhnaut_readvariableop_1_resource_0: :
,while_nqcjuhnaut_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_nqcjuhnaut_matmul_readvariableop_resource:	 D
1while_nqcjuhnaut_matmul_1_readvariableop_resource:	 ?
0while_nqcjuhnaut_biasadd_readvariableop_resource:	6
(while_nqcjuhnaut_readvariableop_resource: 8
*while_nqcjuhnaut_readvariableop_1_resource: 8
*while_nqcjuhnaut_readvariableop_2_resource: ¢'while/nqcjuhnaut/BiasAdd/ReadVariableOp¢&while/nqcjuhnaut/MatMul/ReadVariableOp¢(while/nqcjuhnaut/MatMul_1/ReadVariableOp¢while/nqcjuhnaut/ReadVariableOp¢!while/nqcjuhnaut/ReadVariableOp_1¢!while/nqcjuhnaut/ReadVariableOp_2Ã
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
&while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp1while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/nqcjuhnaut/MatMul/ReadVariableOpÑ
while/nqcjuhnaut/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMulÉ
(while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp3while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/nqcjuhnaut/MatMul_1/ReadVariableOpº
while/nqcjuhnaut/MatMul_1MatMulwhile_placeholder_20while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMul_1°
while/nqcjuhnaut/addAddV2!while/nqcjuhnaut/MatMul:product:0#while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/addÂ
'while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp2while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/nqcjuhnaut/BiasAdd/ReadVariableOp½
while/nqcjuhnaut/BiasAddBiasAddwhile/nqcjuhnaut/add:z:0/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/BiasAdd
 while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/nqcjuhnaut/split/split_dim
while/nqcjuhnaut/splitSplit)while/nqcjuhnaut/split/split_dim:output:0!while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/nqcjuhnaut/split©
while/nqcjuhnaut/ReadVariableOpReadVariableOp*while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/nqcjuhnaut/ReadVariableOp£
while/nqcjuhnaut/mulMul'while/nqcjuhnaut/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul¦
while/nqcjuhnaut/add_1AddV2while/nqcjuhnaut/split:output:0while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_1
while/nqcjuhnaut/SigmoidSigmoidwhile/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid¯
!while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp,while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_1©
while/nqcjuhnaut/mul_1Mul)while/nqcjuhnaut/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_1¨
while/nqcjuhnaut/add_2AddV2while/nqcjuhnaut/split:output:1while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_2
while/nqcjuhnaut/Sigmoid_1Sigmoidwhile/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_1
while/nqcjuhnaut/mul_2Mulwhile/nqcjuhnaut/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_2
while/nqcjuhnaut/TanhTanhwhile/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh¢
while/nqcjuhnaut/mul_3Mulwhile/nqcjuhnaut/Sigmoid:y:0while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_3£
while/nqcjuhnaut/add_3AddV2while/nqcjuhnaut/mul_2:z:0while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_3¯
!while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp,while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_2°
while/nqcjuhnaut/mul_4Mul)while/nqcjuhnaut/ReadVariableOp_2:value:0while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_4¨
while/nqcjuhnaut/add_4AddV2while/nqcjuhnaut/split:output:3while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_4
while/nqcjuhnaut/Sigmoid_2Sigmoidwhile/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_2
while/nqcjuhnaut/Tanh_1Tanhwhile/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh_1¦
while/nqcjuhnaut/mul_5Mulwhile/nqcjuhnaut/Sigmoid_2:y:0while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/nqcjuhnaut/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/nqcjuhnaut/mul_5:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/nqcjuhnaut/add_3:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
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
0while_nqcjuhnaut_biasadd_readvariableop_resource2while_nqcjuhnaut_biasadd_readvariableop_resource_0"h
1while_nqcjuhnaut_matmul_1_readvariableop_resource3while_nqcjuhnaut_matmul_1_readvariableop_resource_0"d
/while_nqcjuhnaut_matmul_readvariableop_resource1while_nqcjuhnaut_matmul_readvariableop_resource_0"Z
*while_nqcjuhnaut_readvariableop_1_resource,while_nqcjuhnaut_readvariableop_1_resource_0"Z
*while_nqcjuhnaut_readvariableop_2_resource,while_nqcjuhnaut_readvariableop_2_resource_0"V
(while_nqcjuhnaut_readvariableop_resource*while_nqcjuhnaut_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/nqcjuhnaut/BiasAdd/ReadVariableOp'while/nqcjuhnaut/BiasAdd/ReadVariableOp2P
&while/nqcjuhnaut/MatMul/ReadVariableOp&while/nqcjuhnaut/MatMul/ReadVariableOp2T
(while/nqcjuhnaut/MatMul_1/ReadVariableOp(while/nqcjuhnaut/MatMul_1/ReadVariableOp2B
while/nqcjuhnaut/ReadVariableOpwhile/nqcjuhnaut/ReadVariableOp2F
!while/nqcjuhnaut/ReadVariableOp_1!while/nqcjuhnaut/ReadVariableOp_12F
!while/nqcjuhnaut/ReadVariableOp_2!while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1063333

inputs%
zdztiqrxwb_1063234:	%
zdztiqrxwb_1063236:	 !
zdztiqrxwb_1063238:	 
zdztiqrxwb_1063240:  
zdztiqrxwb_1063242:  
zdztiqrxwb_1063244: 
identity¢while¢"zdztiqrxwb/StatefulPartitionedCallD
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
"zdztiqrxwb/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0zdztiqrxwb_1063234zdztiqrxwb_1063236zdztiqrxwb_1063238zdztiqrxwb_1063240zdztiqrxwb_1063242zdztiqrxwb_1063244*
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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_10632332$
"zdztiqrxwb/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0zdztiqrxwb_1063234zdztiqrxwb_1063236zdztiqrxwb_1063238zdztiqrxwb_1063240zdztiqrxwb_1063242zdztiqrxwb_1063244*
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
while_body_1063253*
condR
while_cond_1063252*Q
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
IdentityIdentitytranspose_1:y:0^while#^zdztiqrxwb/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2H
"zdztiqrxwb/StatefulPartitionedCall"zdztiqrxwb/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

#__inference__traced_restore_1068974
file_prefix8
"assignvariableop_umdyqemnpr_kernel:0
"assignvariableop_1_umdyqemnpr_bias:6
$assignvariableop_2_ycxcgxamrr_kernel: 0
"assignvariableop_3_ycxcgxamrr_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_subsmtgotc_zdztiqrxwb_kernel:	M
:assignvariableop_10_subsmtgotc_zdztiqrxwb_recurrent_kernel:	 =
.assignvariableop_11_subsmtgotc_zdztiqrxwb_bias:	S
Eassignvariableop_12_subsmtgotc_zdztiqrxwb_input_gate_peephole_weights: T
Fassignvariableop_13_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weights: T
Fassignvariableop_14_subsmtgotc_zdztiqrxwb_output_gate_peephole_weights: C
0assignvariableop_15_quyyatshey_nqcjuhnaut_kernel:	 M
:assignvariableop_16_quyyatshey_nqcjuhnaut_recurrent_kernel:	 =
.assignvariableop_17_quyyatshey_nqcjuhnaut_bias:	S
Eassignvariableop_18_quyyatshey_nqcjuhnaut_input_gate_peephole_weights: T
Fassignvariableop_19_quyyatshey_nqcjuhnaut_forget_gate_peephole_weights: T
Fassignvariableop_20_quyyatshey_nqcjuhnaut_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_umdyqemnpr_kernel_rms:=
/assignvariableop_24_rmsprop_umdyqemnpr_bias_rms:C
1assignvariableop_25_rmsprop_ycxcgxamrr_kernel_rms: =
/assignvariableop_26_rmsprop_ycxcgxamrr_bias_rms:O
<assignvariableop_27_rmsprop_subsmtgotc_zdztiqrxwb_kernel_rms:	Y
Fassignvariableop_28_rmsprop_subsmtgotc_zdztiqrxwb_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_subsmtgotc_zdztiqrxwb_bias_rms:	_
Qassignvariableop_30_rmsprop_subsmtgotc_zdztiqrxwb_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_subsmtgotc_zdztiqrxwb_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_quyyatshey_nqcjuhnaut_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_quyyatshey_nqcjuhnaut_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_quyyatshey_nqcjuhnaut_bias_rms:	_
Qassignvariableop_36_rmsprop_quyyatshey_nqcjuhnaut_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_quyyatshey_nqcjuhnaut_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_quyyatshey_nqcjuhnaut_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_umdyqemnpr_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_umdyqemnpr_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_ycxcgxamrr_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_ycxcgxamrr_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_subsmtgotc_zdztiqrxwb_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_subsmtgotc_zdztiqrxwb_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_subsmtgotc_zdztiqrxwb_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_subsmtgotc_zdztiqrxwb_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_subsmtgotc_zdztiqrxwb_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_quyyatshey_nqcjuhnaut_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_quyyatshey_nqcjuhnaut_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_quyyatshey_nqcjuhnaut_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_quyyatshey_nqcjuhnaut_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_quyyatshey_nqcjuhnaut_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_quyyatshey_nqcjuhnaut_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_umdyqemnpr_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_umdyqemnpr_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_ycxcgxamrr_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_ycxcgxamrr_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_subsmtgotc_zdztiqrxwb_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_subsmtgotc_zdztiqrxwb_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_subsmtgotc_zdztiqrxwb_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_subsmtgotc_zdztiqrxwb_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_subsmtgotc_zdztiqrxwb_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_subsmtgotc_zdztiqrxwb_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_quyyatshey_nqcjuhnaut_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_quyyatshey_nqcjuhnaut_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_quyyatshey_nqcjuhnaut_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_quyyatshey_nqcjuhnaut_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_quyyatshey_nqcjuhnaut_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_quyyatshey_nqcjuhnaut_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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


í
while_cond_1065272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1065272___redundant_placeholder05
1while_while_cond_1065272___redundant_placeholder15
1while_while_cond_1065272___redundant_placeholder25
1while_while_cond_1065272___redundant_placeholder35
1while_while_cond_1065272___redundant_placeholder45
1while_while_cond_1065272___redundant_placeholder55
1while_while_cond_1065272___redundant_placeholder6
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067632

inputs<
)zdztiqrxwb_matmul_readvariableop_resource:	>
+zdztiqrxwb_matmul_1_readvariableop_resource:	 9
*zdztiqrxwb_biasadd_readvariableop_resource:	0
"zdztiqrxwb_readvariableop_resource: 2
$zdztiqrxwb_readvariableop_1_resource: 2
$zdztiqrxwb_readvariableop_2_resource: 
identity¢while¢!zdztiqrxwb/BiasAdd/ReadVariableOp¢ zdztiqrxwb/MatMul/ReadVariableOp¢"zdztiqrxwb/MatMul_1/ReadVariableOp¢zdztiqrxwb/ReadVariableOp¢zdztiqrxwb/ReadVariableOp_1¢zdztiqrxwb/ReadVariableOp_2D
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
 zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp)zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 zdztiqrxwb/MatMul/ReadVariableOp§
zdztiqrxwb/MatMulMatMulstrided_slice_2:output:0(zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMulµ
"zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp+zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"zdztiqrxwb/MatMul_1/ReadVariableOp£
zdztiqrxwb/MatMul_1MatMulzeros:output:0*zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/MatMul_1
zdztiqrxwb/addAddV2zdztiqrxwb/MatMul:product:0zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/add®
!zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp*zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!zdztiqrxwb/BiasAdd/ReadVariableOp¥
zdztiqrxwb/BiasAddBiasAddzdztiqrxwb/add:z:0)zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zdztiqrxwb/BiasAddz
zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
zdztiqrxwb/split/split_dimë
zdztiqrxwb/splitSplit#zdztiqrxwb/split/split_dim:output:0zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zdztiqrxwb/split
zdztiqrxwb/ReadVariableOpReadVariableOp"zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp
zdztiqrxwb/mulMul!zdztiqrxwb/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul
zdztiqrxwb/add_1AddV2zdztiqrxwb/split:output:0zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_1{
zdztiqrxwb/SigmoidSigmoidzdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid
zdztiqrxwb/ReadVariableOp_1ReadVariableOp$zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_1
zdztiqrxwb/mul_1Mul#zdztiqrxwb/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_1
zdztiqrxwb/add_2AddV2zdztiqrxwb/split:output:1zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_2
zdztiqrxwb/Sigmoid_1Sigmoidzdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_1
zdztiqrxwb/mul_2Mulzdztiqrxwb/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_2w
zdztiqrxwb/TanhTanhzdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh
zdztiqrxwb/mul_3Mulzdztiqrxwb/Sigmoid:y:0zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_3
zdztiqrxwb/add_3AddV2zdztiqrxwb/mul_2:z:0zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_3
zdztiqrxwb/ReadVariableOp_2ReadVariableOp$zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02
zdztiqrxwb/ReadVariableOp_2
zdztiqrxwb/mul_4Mul#zdztiqrxwb/ReadVariableOp_2:value:0zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_4
zdztiqrxwb/add_4AddV2zdztiqrxwb/split:output:3zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/add_4
zdztiqrxwb/Sigmoid_2Sigmoidzdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Sigmoid_2v
zdztiqrxwb/Tanh_1Tanhzdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/Tanh_1
zdztiqrxwb/mul_5Mulzdztiqrxwb/Sigmoid_2:y:0zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zdztiqrxwb/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)zdztiqrxwb_matmul_readvariableop_resource+zdztiqrxwb_matmul_1_readvariableop_resource*zdztiqrxwb_biasadd_readvariableop_resource"zdztiqrxwb_readvariableop_resource$zdztiqrxwb_readvariableop_1_resource$zdztiqrxwb_readvariableop_2_resource*
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
while_body_1067531*
condR
while_cond_1067530*Q
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
IdentityIdentitytranspose_1:y:0^while"^zdztiqrxwb/BiasAdd/ReadVariableOp!^zdztiqrxwb/MatMul/ReadVariableOp#^zdztiqrxwb/MatMul_1/ReadVariableOp^zdztiqrxwb/ReadVariableOp^zdztiqrxwb/ReadVariableOp_1^zdztiqrxwb/ReadVariableOp_2*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2
whilewhile2F
!zdztiqrxwb/BiasAdd/ReadVariableOp!zdztiqrxwb/BiasAdd/ReadVariableOp2D
 zdztiqrxwb/MatMul/ReadVariableOp zdztiqrxwb/MatMul/ReadVariableOp2H
"zdztiqrxwb/MatMul_1/ReadVariableOp"zdztiqrxwb/MatMul_1/ReadVariableOp26
zdztiqrxwb/ReadVariableOpzdztiqrxwb/ReadVariableOp2:
zdztiqrxwb/ReadVariableOp_1zdztiqrxwb/ReadVariableOp_12:
zdztiqrxwb/ReadVariableOp_2zdztiqrxwb/ReadVariableOp_2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_1068138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1068138___redundant_placeholder05
1while_while_cond_1068138___redundant_placeholder15
1while_while_cond_1068138___redundant_placeholder25
1while_while_cond_1068138___redundant_placeholder35
1while_while_cond_1068138___redundant_placeholder45
1while_while_cond_1068138___redundant_placeholder55
1while_while_cond_1068138___redundant_placeholder6
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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1063991

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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1066780

inputsL
6umdyqemnpr_conv1d_expanddims_1_readvariableop_resource:K
=umdyqemnpr_squeeze_batch_dims_biasadd_readvariableop_resource:G
4subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource:	I
6subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource:	 D
5subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource:	;
-subsmtgotc_zdztiqrxwb_readvariableop_resource: =
/subsmtgotc_zdztiqrxwb_readvariableop_1_resource: =
/subsmtgotc_zdztiqrxwb_readvariableop_2_resource: G
4quyyatshey_nqcjuhnaut_matmul_readvariableop_resource:	 I
6quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource:	 D
5quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource:	;
-quyyatshey_nqcjuhnaut_readvariableop_resource: =
/quyyatshey_nqcjuhnaut_readvariableop_1_resource: =
/quyyatshey_nqcjuhnaut_readvariableop_2_resource: ;
)ycxcgxamrr_matmul_readvariableop_resource: 8
*ycxcgxamrr_biasadd_readvariableop_resource:
identity¢,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp¢+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp¢-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp¢$quyyatshey/nqcjuhnaut/ReadVariableOp¢&quyyatshey/nqcjuhnaut/ReadVariableOp_1¢&quyyatshey/nqcjuhnaut/ReadVariableOp_2¢quyyatshey/while¢subsmtgotc/while¢,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp¢+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp¢-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp¢$subsmtgotc/zdztiqrxwb/ReadVariableOp¢&subsmtgotc/zdztiqrxwb/ReadVariableOp_1¢&subsmtgotc/zdztiqrxwb/ReadVariableOp_2¢-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp¢4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp¢!ycxcgxamrr/BiasAdd/ReadVariableOp¢ ycxcgxamrr/MatMul/ReadVariableOp
 umdyqemnpr/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 umdyqemnpr/conv1d/ExpandDims/dim»
umdyqemnpr/conv1d/ExpandDims
ExpandDimsinputs)umdyqemnpr/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/ExpandDimsÙ
-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6umdyqemnpr_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp
"umdyqemnpr/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"umdyqemnpr/conv1d/ExpandDims_1/dimã
umdyqemnpr/conv1d/ExpandDims_1
ExpandDims5umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp:value:0+umdyqemnpr/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
umdyqemnpr/conv1d/ExpandDims_1
umdyqemnpr/conv1d/ShapeShape%umdyqemnpr/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
umdyqemnpr/conv1d/Shape
%umdyqemnpr/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%umdyqemnpr/conv1d/strided_slice/stack¥
'umdyqemnpr/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'umdyqemnpr/conv1d/strided_slice/stack_1
'umdyqemnpr/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'umdyqemnpr/conv1d/strided_slice/stack_2Ì
umdyqemnpr/conv1d/strided_sliceStridedSlice umdyqemnpr/conv1d/Shape:output:0.umdyqemnpr/conv1d/strided_slice/stack:output:00umdyqemnpr/conv1d/strided_slice/stack_1:output:00umdyqemnpr/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
umdyqemnpr/conv1d/strided_slice
umdyqemnpr/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
umdyqemnpr/conv1d/Reshape/shapeÌ
umdyqemnpr/conv1d/ReshapeReshape%umdyqemnpr/conv1d/ExpandDims:output:0(umdyqemnpr/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/Reshapeî
umdyqemnpr/conv1d/Conv2DConv2D"umdyqemnpr/conv1d/Reshape:output:0'umdyqemnpr/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
umdyqemnpr/conv1d/Conv2D
!umdyqemnpr/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!umdyqemnpr/conv1d/concat/values_1
umdyqemnpr/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/concat/axisì
umdyqemnpr/conv1d/concatConcatV2(umdyqemnpr/conv1d/strided_slice:output:0*umdyqemnpr/conv1d/concat/values_1:output:0&umdyqemnpr/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
umdyqemnpr/conv1d/concatÉ
umdyqemnpr/conv1d/Reshape_1Reshape!umdyqemnpr/conv1d/Conv2D:output:0!umdyqemnpr/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/Reshape_1Á
umdyqemnpr/conv1d/SqueezeSqueeze$umdyqemnpr/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
umdyqemnpr/conv1d/Squeeze
#umdyqemnpr/squeeze_batch_dims/ShapeShape"umdyqemnpr/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#umdyqemnpr/squeeze_batch_dims/Shape°
1umdyqemnpr/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1umdyqemnpr/squeeze_batch_dims/strided_slice/stack½
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1´
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2
+umdyqemnpr/squeeze_batch_dims/strided_sliceStridedSlice,umdyqemnpr/squeeze_batch_dims/Shape:output:0:umdyqemnpr/squeeze_batch_dims/strided_slice/stack:output:0<umdyqemnpr/squeeze_batch_dims/strided_slice/stack_1:output:0<umdyqemnpr/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+umdyqemnpr/squeeze_batch_dims/strided_slice¯
+umdyqemnpr/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+umdyqemnpr/squeeze_batch_dims/Reshape/shapeé
%umdyqemnpr/squeeze_batch_dims/ReshapeReshape"umdyqemnpr/conv1d/Squeeze:output:04umdyqemnpr/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%umdyqemnpr/squeeze_batch_dims/Reshapeæ
4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=umdyqemnpr_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%umdyqemnpr/squeeze_batch_dims/BiasAddBiasAdd.umdyqemnpr/squeeze_batch_dims/Reshape:output:0<umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%umdyqemnpr/squeeze_batch_dims/BiasAdd¯
-umdyqemnpr/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-umdyqemnpr/squeeze_batch_dims/concat/values_1¡
)umdyqemnpr/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)umdyqemnpr/squeeze_batch_dims/concat/axis¨
$umdyqemnpr/squeeze_batch_dims/concatConcatV24umdyqemnpr/squeeze_batch_dims/strided_slice:output:06umdyqemnpr/squeeze_batch_dims/concat/values_1:output:02umdyqemnpr/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$umdyqemnpr/squeeze_batch_dims/concatö
'umdyqemnpr/squeeze_batch_dims/Reshape_1Reshape.umdyqemnpr/squeeze_batch_dims/BiasAdd:output:0-umdyqemnpr/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'umdyqemnpr/squeeze_batch_dims/Reshape_1
unxqeixodn/ShapeShape0umdyqemnpr/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
unxqeixodn/Shape
unxqeixodn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
unxqeixodn/strided_slice/stack
 unxqeixodn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 unxqeixodn/strided_slice/stack_1
 unxqeixodn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 unxqeixodn/strided_slice/stack_2¤
unxqeixodn/strided_sliceStridedSliceunxqeixodn/Shape:output:0'unxqeixodn/strided_slice/stack:output:0)unxqeixodn/strided_slice/stack_1:output:0)unxqeixodn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
unxqeixodn/strided_slicez
unxqeixodn/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
unxqeixodn/Reshape/shape/1z
unxqeixodn/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
unxqeixodn/Reshape/shape/2×
unxqeixodn/Reshape/shapePack!unxqeixodn/strided_slice:output:0#unxqeixodn/Reshape/shape/1:output:0#unxqeixodn/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
unxqeixodn/Reshape/shape¾
unxqeixodn/ReshapeReshape0umdyqemnpr/squeeze_batch_dims/Reshape_1:output:0!unxqeixodn/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
unxqeixodn/Reshapeo
subsmtgotc/ShapeShapeunxqeixodn/Reshape:output:0*
T0*
_output_shapes
:2
subsmtgotc/Shape
subsmtgotc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
subsmtgotc/strided_slice/stack
 subsmtgotc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 subsmtgotc/strided_slice/stack_1
 subsmtgotc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 subsmtgotc/strided_slice/stack_2¤
subsmtgotc/strided_sliceStridedSlicesubsmtgotc/Shape:output:0'subsmtgotc/strided_slice/stack:output:0)subsmtgotc/strided_slice/stack_1:output:0)subsmtgotc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
subsmtgotc/strided_slicer
subsmtgotc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros/mul/y
subsmtgotc/zeros/mulMul!subsmtgotc/strided_slice:output:0subsmtgotc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros/mulu
subsmtgotc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
subsmtgotc/zeros/Less/y
subsmtgotc/zeros/LessLesssubsmtgotc/zeros/mul:z:0 subsmtgotc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros/Lessx
subsmtgotc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros/packed/1¯
subsmtgotc/zeros/packedPack!subsmtgotc/strided_slice:output:0"subsmtgotc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
subsmtgotc/zeros/packedu
subsmtgotc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
subsmtgotc/zeros/Const¡
subsmtgotc/zerosFill subsmtgotc/zeros/packed:output:0subsmtgotc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zerosv
subsmtgotc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros_1/mul/y
subsmtgotc/zeros_1/mulMul!subsmtgotc/strided_slice:output:0!subsmtgotc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros_1/muly
subsmtgotc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
subsmtgotc/zeros_1/Less/y
subsmtgotc/zeros_1/LessLesssubsmtgotc/zeros_1/mul:z:0"subsmtgotc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/zeros_1/Less|
subsmtgotc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/zeros_1/packed/1µ
subsmtgotc/zeros_1/packedPack!subsmtgotc/strided_slice:output:0$subsmtgotc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
subsmtgotc/zeros_1/packedy
subsmtgotc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
subsmtgotc/zeros_1/Const©
subsmtgotc/zeros_1Fill"subsmtgotc/zeros_1/packed:output:0!subsmtgotc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zeros_1
subsmtgotc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
subsmtgotc/transpose/perm°
subsmtgotc/transpose	Transposeunxqeixodn/Reshape:output:0"subsmtgotc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/transposep
subsmtgotc/Shape_1Shapesubsmtgotc/transpose:y:0*
T0*
_output_shapes
:2
subsmtgotc/Shape_1
 subsmtgotc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 subsmtgotc/strided_slice_1/stack
"subsmtgotc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_1/stack_1
"subsmtgotc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_1/stack_2°
subsmtgotc/strided_slice_1StridedSlicesubsmtgotc/Shape_1:output:0)subsmtgotc/strided_slice_1/stack:output:0+subsmtgotc/strided_slice_1/stack_1:output:0+subsmtgotc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
subsmtgotc/strided_slice_1
&subsmtgotc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&subsmtgotc/TensorArrayV2/element_shapeÞ
subsmtgotc/TensorArrayV2TensorListReserve/subsmtgotc/TensorArrayV2/element_shape:output:0#subsmtgotc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
subsmtgotc/TensorArrayV2Õ
@subsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@subsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2subsmtgotc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsubsmtgotc/transpose:y:0Isubsmtgotc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2subsmtgotc/TensorArrayUnstack/TensorListFromTensor
 subsmtgotc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 subsmtgotc/strided_slice_2/stack
"subsmtgotc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_2/stack_1
"subsmtgotc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_2/stack_2¾
subsmtgotc/strided_slice_2StridedSlicesubsmtgotc/transpose:y:0)subsmtgotc/strided_slice_2/stack:output:0+subsmtgotc/strided_slice_2/stack_1:output:0+subsmtgotc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
subsmtgotc/strided_slice_2Ð
+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp4subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOpÓ
subsmtgotc/zdztiqrxwb/MatMulMatMul#subsmtgotc/strided_slice_2:output:03subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/zdztiqrxwb/MatMulÖ
-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp6subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOpÏ
subsmtgotc/zdztiqrxwb/MatMul_1MatMulsubsmtgotc/zeros:output:05subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
subsmtgotc/zdztiqrxwb/MatMul_1Ä
subsmtgotc/zdztiqrxwb/addAddV2&subsmtgotc/zdztiqrxwb/MatMul:product:0(subsmtgotc/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/zdztiqrxwb/addÏ
,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp5subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOpÑ
subsmtgotc/zdztiqrxwb/BiasAddBiasAddsubsmtgotc/zdztiqrxwb/add:z:04subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subsmtgotc/zdztiqrxwb/BiasAdd
%subsmtgotc/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%subsmtgotc/zdztiqrxwb/split/split_dim
subsmtgotc/zdztiqrxwb/splitSplit.subsmtgotc/zdztiqrxwb/split/split_dim:output:0&subsmtgotc/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
subsmtgotc/zdztiqrxwb/split¶
$subsmtgotc/zdztiqrxwb/ReadVariableOpReadVariableOp-subsmtgotc_zdztiqrxwb_readvariableop_resource*
_output_shapes
: *
dtype02&
$subsmtgotc/zdztiqrxwb/ReadVariableOpº
subsmtgotc/zdztiqrxwb/mulMul,subsmtgotc/zdztiqrxwb/ReadVariableOp:value:0subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mulº
subsmtgotc/zdztiqrxwb/add_1AddV2$subsmtgotc/zdztiqrxwb/split:output:0subsmtgotc/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_1
subsmtgotc/zdztiqrxwb/SigmoidSigmoidsubsmtgotc/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/Sigmoid¼
&subsmtgotc/zdztiqrxwb/ReadVariableOp_1ReadVariableOp/subsmtgotc_zdztiqrxwb_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&subsmtgotc/zdztiqrxwb/ReadVariableOp_1À
subsmtgotc/zdztiqrxwb/mul_1Mul.subsmtgotc/zdztiqrxwb/ReadVariableOp_1:value:0subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_1¼
subsmtgotc/zdztiqrxwb/add_2AddV2$subsmtgotc/zdztiqrxwb/split:output:1subsmtgotc/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_2 
subsmtgotc/zdztiqrxwb/Sigmoid_1Sigmoidsubsmtgotc/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
subsmtgotc/zdztiqrxwb/Sigmoid_1µ
subsmtgotc/zdztiqrxwb/mul_2Mul#subsmtgotc/zdztiqrxwb/Sigmoid_1:y:0subsmtgotc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_2
subsmtgotc/zdztiqrxwb/TanhTanh$subsmtgotc/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/Tanh¶
subsmtgotc/zdztiqrxwb/mul_3Mul!subsmtgotc/zdztiqrxwb/Sigmoid:y:0subsmtgotc/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_3·
subsmtgotc/zdztiqrxwb/add_3AddV2subsmtgotc/zdztiqrxwb/mul_2:z:0subsmtgotc/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_3¼
&subsmtgotc/zdztiqrxwb/ReadVariableOp_2ReadVariableOp/subsmtgotc_zdztiqrxwb_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&subsmtgotc/zdztiqrxwb/ReadVariableOp_2Ä
subsmtgotc/zdztiqrxwb/mul_4Mul.subsmtgotc/zdztiqrxwb/ReadVariableOp_2:value:0subsmtgotc/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_4¼
subsmtgotc/zdztiqrxwb/add_4AddV2$subsmtgotc/zdztiqrxwb/split:output:3subsmtgotc/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/add_4 
subsmtgotc/zdztiqrxwb/Sigmoid_2Sigmoidsubsmtgotc/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
subsmtgotc/zdztiqrxwb/Sigmoid_2
subsmtgotc/zdztiqrxwb/Tanh_1Tanhsubsmtgotc/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/Tanh_1º
subsmtgotc/zdztiqrxwb/mul_5Mul#subsmtgotc/zdztiqrxwb/Sigmoid_2:y:0 subsmtgotc/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/zdztiqrxwb/mul_5¥
(subsmtgotc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(subsmtgotc/TensorArrayV2_1/element_shapeä
subsmtgotc/TensorArrayV2_1TensorListReserve1subsmtgotc/TensorArrayV2_1/element_shape:output:0#subsmtgotc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
subsmtgotc/TensorArrayV2_1d
subsmtgotc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/time
#subsmtgotc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#subsmtgotc/while/maximum_iterations
subsmtgotc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
subsmtgotc/while/loop_counter²
subsmtgotc/whileWhile&subsmtgotc/while/loop_counter:output:0,subsmtgotc/while/maximum_iterations:output:0subsmtgotc/time:output:0#subsmtgotc/TensorArrayV2_1:handle:0subsmtgotc/zeros:output:0subsmtgotc/zeros_1:output:0#subsmtgotc/strided_slice_1:output:0Bsubsmtgotc/TensorArrayUnstack/TensorListFromTensor:output_handle:04subsmtgotc_zdztiqrxwb_matmul_readvariableop_resource6subsmtgotc_zdztiqrxwb_matmul_1_readvariableop_resource5subsmtgotc_zdztiqrxwb_biasadd_readvariableop_resource-subsmtgotc_zdztiqrxwb_readvariableop_resource/subsmtgotc_zdztiqrxwb_readvariableop_1_resource/subsmtgotc_zdztiqrxwb_readvariableop_2_resource*
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
subsmtgotc_while_body_1066497*)
cond!R
subsmtgotc_while_cond_1066496*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
subsmtgotc/whileË
;subsmtgotc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;subsmtgotc/TensorArrayV2Stack/TensorListStack/element_shape
-subsmtgotc/TensorArrayV2Stack/TensorListStackTensorListStacksubsmtgotc/while:output:3Dsubsmtgotc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-subsmtgotc/TensorArrayV2Stack/TensorListStack
 subsmtgotc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 subsmtgotc/strided_slice_3/stack
"subsmtgotc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"subsmtgotc/strided_slice_3/stack_1
"subsmtgotc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"subsmtgotc/strided_slice_3/stack_2Ü
subsmtgotc/strided_slice_3StridedSlice6subsmtgotc/TensorArrayV2Stack/TensorListStack:tensor:0)subsmtgotc/strided_slice_3/stack:output:0+subsmtgotc/strided_slice_3/stack_1:output:0+subsmtgotc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
subsmtgotc/strided_slice_3
subsmtgotc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
subsmtgotc/transpose_1/permÑ
subsmtgotc/transpose_1	Transpose6subsmtgotc/TensorArrayV2Stack/TensorListStack:tensor:0$subsmtgotc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/transpose_1n
quyyatshey/ShapeShapesubsmtgotc/transpose_1:y:0*
T0*
_output_shapes
:2
quyyatshey/Shape
quyyatshey/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
quyyatshey/strided_slice/stack
 quyyatshey/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 quyyatshey/strided_slice/stack_1
 quyyatshey/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 quyyatshey/strided_slice/stack_2¤
quyyatshey/strided_sliceStridedSlicequyyatshey/Shape:output:0'quyyatshey/strided_slice/stack:output:0)quyyatshey/strided_slice/stack_1:output:0)quyyatshey/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quyyatshey/strided_slicer
quyyatshey/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros/mul/y
quyyatshey/zeros/mulMul!quyyatshey/strided_slice:output:0quyyatshey/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros/mulu
quyyatshey/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
quyyatshey/zeros/Less/y
quyyatshey/zeros/LessLessquyyatshey/zeros/mul:z:0 quyyatshey/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros/Lessx
quyyatshey/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros/packed/1¯
quyyatshey/zeros/packedPack!quyyatshey/strided_slice:output:0"quyyatshey/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
quyyatshey/zeros/packedu
quyyatshey/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
quyyatshey/zeros/Const¡
quyyatshey/zerosFill quyyatshey/zeros/packed:output:0quyyatshey/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/zerosv
quyyatshey/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros_1/mul/y
quyyatshey/zeros_1/mulMul!quyyatshey/strided_slice:output:0!quyyatshey/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros_1/muly
quyyatshey/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
quyyatshey/zeros_1/Less/y
quyyatshey/zeros_1/LessLessquyyatshey/zeros_1/mul:z:0"quyyatshey/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
quyyatshey/zeros_1/Less|
quyyatshey/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/zeros_1/packed/1µ
quyyatshey/zeros_1/packedPack!quyyatshey/strided_slice:output:0$quyyatshey/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
quyyatshey/zeros_1/packedy
quyyatshey/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
quyyatshey/zeros_1/Const©
quyyatshey/zeros_1Fill"quyyatshey/zeros_1/packed:output:0!quyyatshey/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/zeros_1
quyyatshey/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
quyyatshey/transpose/perm¯
quyyatshey/transpose	Transposesubsmtgotc/transpose_1:y:0"quyyatshey/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/transposep
quyyatshey/Shape_1Shapequyyatshey/transpose:y:0*
T0*
_output_shapes
:2
quyyatshey/Shape_1
 quyyatshey/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 quyyatshey/strided_slice_1/stack
"quyyatshey/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_1/stack_1
"quyyatshey/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_1/stack_2°
quyyatshey/strided_slice_1StridedSlicequyyatshey/Shape_1:output:0)quyyatshey/strided_slice_1/stack:output:0+quyyatshey/strided_slice_1/stack_1:output:0+quyyatshey/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
quyyatshey/strided_slice_1
&quyyatshey/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&quyyatshey/TensorArrayV2/element_shapeÞ
quyyatshey/TensorArrayV2TensorListReserve/quyyatshey/TensorArrayV2/element_shape:output:0#quyyatshey/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
quyyatshey/TensorArrayV2Õ
@quyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@quyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2quyyatshey/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorquyyatshey/transpose:y:0Iquyyatshey/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2quyyatshey/TensorArrayUnstack/TensorListFromTensor
 quyyatshey/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 quyyatshey/strided_slice_2/stack
"quyyatshey/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_2/stack_1
"quyyatshey/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_2/stack_2¾
quyyatshey/strided_slice_2StridedSlicequyyatshey/transpose:y:0)quyyatshey/strided_slice_2/stack:output:0+quyyatshey/strided_slice_2/stack_1:output:0+quyyatshey/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
quyyatshey/strided_slice_2Ð
+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp4quyyatshey_nqcjuhnaut_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOpÓ
quyyatshey/nqcjuhnaut/MatMulMatMul#quyyatshey/strided_slice_2:output:03quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quyyatshey/nqcjuhnaut/MatMulÖ
-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp6quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOpÏ
quyyatshey/nqcjuhnaut/MatMul_1MatMulquyyatshey/zeros:output:05quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
quyyatshey/nqcjuhnaut/MatMul_1Ä
quyyatshey/nqcjuhnaut/addAddV2&quyyatshey/nqcjuhnaut/MatMul:product:0(quyyatshey/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quyyatshey/nqcjuhnaut/addÏ
,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp5quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOpÑ
quyyatshey/nqcjuhnaut/BiasAddBiasAddquyyatshey/nqcjuhnaut/add:z:04quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
quyyatshey/nqcjuhnaut/BiasAdd
%quyyatshey/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%quyyatshey/nqcjuhnaut/split/split_dim
quyyatshey/nqcjuhnaut/splitSplit.quyyatshey/nqcjuhnaut/split/split_dim:output:0&quyyatshey/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
quyyatshey/nqcjuhnaut/split¶
$quyyatshey/nqcjuhnaut/ReadVariableOpReadVariableOp-quyyatshey_nqcjuhnaut_readvariableop_resource*
_output_shapes
: *
dtype02&
$quyyatshey/nqcjuhnaut/ReadVariableOpº
quyyatshey/nqcjuhnaut/mulMul,quyyatshey/nqcjuhnaut/ReadVariableOp:value:0quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mulº
quyyatshey/nqcjuhnaut/add_1AddV2$quyyatshey/nqcjuhnaut/split:output:0quyyatshey/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_1
quyyatshey/nqcjuhnaut/SigmoidSigmoidquyyatshey/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/Sigmoid¼
&quyyatshey/nqcjuhnaut/ReadVariableOp_1ReadVariableOp/quyyatshey_nqcjuhnaut_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&quyyatshey/nqcjuhnaut/ReadVariableOp_1À
quyyatshey/nqcjuhnaut/mul_1Mul.quyyatshey/nqcjuhnaut/ReadVariableOp_1:value:0quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_1¼
quyyatshey/nqcjuhnaut/add_2AddV2$quyyatshey/nqcjuhnaut/split:output:1quyyatshey/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_2 
quyyatshey/nqcjuhnaut/Sigmoid_1Sigmoidquyyatshey/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
quyyatshey/nqcjuhnaut/Sigmoid_1µ
quyyatshey/nqcjuhnaut/mul_2Mul#quyyatshey/nqcjuhnaut/Sigmoid_1:y:0quyyatshey/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_2
quyyatshey/nqcjuhnaut/TanhTanh$quyyatshey/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/Tanh¶
quyyatshey/nqcjuhnaut/mul_3Mul!quyyatshey/nqcjuhnaut/Sigmoid:y:0quyyatshey/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_3·
quyyatshey/nqcjuhnaut/add_3AddV2quyyatshey/nqcjuhnaut/mul_2:z:0quyyatshey/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_3¼
&quyyatshey/nqcjuhnaut/ReadVariableOp_2ReadVariableOp/quyyatshey_nqcjuhnaut_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&quyyatshey/nqcjuhnaut/ReadVariableOp_2Ä
quyyatshey/nqcjuhnaut/mul_4Mul.quyyatshey/nqcjuhnaut/ReadVariableOp_2:value:0quyyatshey/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_4¼
quyyatshey/nqcjuhnaut/add_4AddV2$quyyatshey/nqcjuhnaut/split:output:3quyyatshey/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/add_4 
quyyatshey/nqcjuhnaut/Sigmoid_2Sigmoidquyyatshey/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
quyyatshey/nqcjuhnaut/Sigmoid_2
quyyatshey/nqcjuhnaut/Tanh_1Tanhquyyatshey/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/Tanh_1º
quyyatshey/nqcjuhnaut/mul_5Mul#quyyatshey/nqcjuhnaut/Sigmoid_2:y:0 quyyatshey/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/nqcjuhnaut/mul_5¥
(quyyatshey/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(quyyatshey/TensorArrayV2_1/element_shapeä
quyyatshey/TensorArrayV2_1TensorListReserve1quyyatshey/TensorArrayV2_1/element_shape:output:0#quyyatshey/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
quyyatshey/TensorArrayV2_1d
quyyatshey/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/time
#quyyatshey/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#quyyatshey/while/maximum_iterations
quyyatshey/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
quyyatshey/while/loop_counter²
quyyatshey/whileWhile&quyyatshey/while/loop_counter:output:0,quyyatshey/while/maximum_iterations:output:0quyyatshey/time:output:0#quyyatshey/TensorArrayV2_1:handle:0quyyatshey/zeros:output:0quyyatshey/zeros_1:output:0#quyyatshey/strided_slice_1:output:0Bquyyatshey/TensorArrayUnstack/TensorListFromTensor:output_handle:04quyyatshey_nqcjuhnaut_matmul_readvariableop_resource6quyyatshey_nqcjuhnaut_matmul_1_readvariableop_resource5quyyatshey_nqcjuhnaut_biasadd_readvariableop_resource-quyyatshey_nqcjuhnaut_readvariableop_resource/quyyatshey_nqcjuhnaut_readvariableop_1_resource/quyyatshey_nqcjuhnaut_readvariableop_2_resource*
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
quyyatshey_while_body_1066673*)
cond!R
quyyatshey_while_cond_1066672*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
quyyatshey/whileË
;quyyatshey/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;quyyatshey/TensorArrayV2Stack/TensorListStack/element_shape
-quyyatshey/TensorArrayV2Stack/TensorListStackTensorListStackquyyatshey/while:output:3Dquyyatshey/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-quyyatshey/TensorArrayV2Stack/TensorListStack
 quyyatshey/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 quyyatshey/strided_slice_3/stack
"quyyatshey/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"quyyatshey/strided_slice_3/stack_1
"quyyatshey/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"quyyatshey/strided_slice_3/stack_2Ü
quyyatshey/strided_slice_3StridedSlice6quyyatshey/TensorArrayV2Stack/TensorListStack:tensor:0)quyyatshey/strided_slice_3/stack:output:0+quyyatshey/strided_slice_3/stack_1:output:0+quyyatshey/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
quyyatshey/strided_slice_3
quyyatshey/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
quyyatshey/transpose_1/permÑ
quyyatshey/transpose_1	Transpose6quyyatshey/TensorArrayV2Stack/TensorListStack:tensor:0$quyyatshey/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
quyyatshey/transpose_1®
 ycxcgxamrr/MatMul/ReadVariableOpReadVariableOp)ycxcgxamrr_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 ycxcgxamrr/MatMul/ReadVariableOp±
ycxcgxamrr/MatMulMatMul#quyyatshey/strided_slice_3:output:0(ycxcgxamrr/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ycxcgxamrr/MatMul­
!ycxcgxamrr/BiasAdd/ReadVariableOpReadVariableOp*ycxcgxamrr_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ycxcgxamrr/BiasAdd/ReadVariableOp­
ycxcgxamrr/BiasAddBiasAddycxcgxamrr/MatMul:product:0)ycxcgxamrr/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ycxcgxamrr/BiasAddÏ
IdentityIdentityycxcgxamrr/BiasAdd:output:0-^quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp,^quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp.^quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp%^quyyatshey/nqcjuhnaut/ReadVariableOp'^quyyatshey/nqcjuhnaut/ReadVariableOp_1'^quyyatshey/nqcjuhnaut/ReadVariableOp_2^quyyatshey/while^subsmtgotc/while-^subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp,^subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp.^subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp%^subsmtgotc/zdztiqrxwb/ReadVariableOp'^subsmtgotc/zdztiqrxwb/ReadVariableOp_1'^subsmtgotc/zdztiqrxwb/ReadVariableOp_2.^umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp5^umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp"^ycxcgxamrr/BiasAdd/ReadVariableOp!^ycxcgxamrr/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp,quyyatshey/nqcjuhnaut/BiasAdd/ReadVariableOp2Z
+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp+quyyatshey/nqcjuhnaut/MatMul/ReadVariableOp2^
-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp-quyyatshey/nqcjuhnaut/MatMul_1/ReadVariableOp2L
$quyyatshey/nqcjuhnaut/ReadVariableOp$quyyatshey/nqcjuhnaut/ReadVariableOp2P
&quyyatshey/nqcjuhnaut/ReadVariableOp_1&quyyatshey/nqcjuhnaut/ReadVariableOp_12P
&quyyatshey/nqcjuhnaut/ReadVariableOp_2&quyyatshey/nqcjuhnaut/ReadVariableOp_22$
quyyatshey/whilequyyatshey/while2$
subsmtgotc/whilesubsmtgotc/while2\
,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp,subsmtgotc/zdztiqrxwb/BiasAdd/ReadVariableOp2Z
+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp+subsmtgotc/zdztiqrxwb/MatMul/ReadVariableOp2^
-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp-subsmtgotc/zdztiqrxwb/MatMul_1/ReadVariableOp2L
$subsmtgotc/zdztiqrxwb/ReadVariableOp$subsmtgotc/zdztiqrxwb/ReadVariableOp2P
&subsmtgotc/zdztiqrxwb/ReadVariableOp_1&subsmtgotc/zdztiqrxwb/ReadVariableOp_12P
&subsmtgotc/zdztiqrxwb/ReadVariableOp_2&subsmtgotc/zdztiqrxwb/ReadVariableOp_22^
-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp-umdyqemnpr/conv1d/ExpandDims_1/ReadVariableOp2l
4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp4umdyqemnpr/squeeze_batch_dims/BiasAdd/ReadVariableOp2F
!ycxcgxamrr/BiasAdd/ReadVariableOp!ycxcgxamrr/BiasAdd/ReadVariableOp2D
 ycxcgxamrr/MatMul/ReadVariableOp ycxcgxamrr/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯F
ê
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1064354

inputs%
nqcjuhnaut_1064255:	 %
nqcjuhnaut_1064257:	 !
nqcjuhnaut_1064259:	 
nqcjuhnaut_1064261:  
nqcjuhnaut_1064263:  
nqcjuhnaut_1064265: 
identity¢"nqcjuhnaut/StatefulPartitionedCall¢whileD
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
"nqcjuhnaut/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0nqcjuhnaut_1064255nqcjuhnaut_1064257nqcjuhnaut_1064259nqcjuhnaut_1064261nqcjuhnaut_1064263nqcjuhnaut_1064265*
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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_10641782$
"nqcjuhnaut/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0nqcjuhnaut_1064255nqcjuhnaut_1064257nqcjuhnaut_1064259nqcjuhnaut_1064261nqcjuhnaut_1064263nqcjuhnaut_1064265*
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
while_body_1064274*
condR
while_cond_1064273*Q
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
IdentityIdentitystrided_slice_3:output:0#^nqcjuhnaut/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"nqcjuhnaut/StatefulPartitionedCall"nqcjuhnaut/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û

,__inference_subsmtgotc_layer_call_fn_1066895

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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_10649062
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
p
Ê
subsmtgotc_while_body_10664972
.subsmtgotc_while_subsmtgotc_while_loop_counter8
4subsmtgotc_while_subsmtgotc_while_maximum_iterations 
subsmtgotc_while_placeholder"
subsmtgotc_while_placeholder_1"
subsmtgotc_while_placeholder_2"
subsmtgotc_while_placeholder_31
-subsmtgotc_while_subsmtgotc_strided_slice_1_0m
isubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0O
<subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0:	Q
>subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 L
=subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0:	C
5subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0: E
7subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0: E
7subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0: 
subsmtgotc_while_identity
subsmtgotc_while_identity_1
subsmtgotc_while_identity_2
subsmtgotc_while_identity_3
subsmtgotc_while_identity_4
subsmtgotc_while_identity_5/
+subsmtgotc_while_subsmtgotc_strided_slice_1k
gsubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensorM
:subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource:	O
<subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource:	 J
;subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource:	A
3subsmtgotc_while_zdztiqrxwb_readvariableop_resource: C
5subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource: C
5subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource: ¢2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp¢1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp¢3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp¢*subsmtgotc/while/zdztiqrxwb/ReadVariableOp¢,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1¢,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2Ù
Bsubsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bsubsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4subsmtgotc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemisubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0subsmtgotc_while_placeholderKsubsmtgotc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4subsmtgotc/while/TensorArrayV2Read/TensorListGetItemä
1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp<subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOpý
"subsmtgotc/while/zdztiqrxwb/MatMulMatMul;subsmtgotc/while/TensorArrayV2Read/TensorListGetItem:item:09subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"subsmtgotc/while/zdztiqrxwb/MatMulê
3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp>subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOpæ
$subsmtgotc/while/zdztiqrxwb/MatMul_1MatMulsubsmtgotc_while_placeholder_2;subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$subsmtgotc/while/zdztiqrxwb/MatMul_1Ü
subsmtgotc/while/zdztiqrxwb/addAddV2,subsmtgotc/while/zdztiqrxwb/MatMul:product:0.subsmtgotc/while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
subsmtgotc/while/zdztiqrxwb/addã
2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp=subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOpé
#subsmtgotc/while/zdztiqrxwb/BiasAddBiasAdd#subsmtgotc/while/zdztiqrxwb/add:z:0:subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#subsmtgotc/while/zdztiqrxwb/BiasAdd
+subsmtgotc/while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+subsmtgotc/while/zdztiqrxwb/split/split_dim¯
!subsmtgotc/while/zdztiqrxwb/splitSplit4subsmtgotc/while/zdztiqrxwb/split/split_dim:output:0,subsmtgotc/while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!subsmtgotc/while/zdztiqrxwb/splitÊ
*subsmtgotc/while/zdztiqrxwb/ReadVariableOpReadVariableOp5subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*subsmtgotc/while/zdztiqrxwb/ReadVariableOpÏ
subsmtgotc/while/zdztiqrxwb/mulMul2subsmtgotc/while/zdztiqrxwb/ReadVariableOp:value:0subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
subsmtgotc/while/zdztiqrxwb/mulÒ
!subsmtgotc/while/zdztiqrxwb/add_1AddV2*subsmtgotc/while/zdztiqrxwb/split:output:0#subsmtgotc/while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_1®
#subsmtgotc/while/zdztiqrxwb/SigmoidSigmoid%subsmtgotc/while/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#subsmtgotc/while/zdztiqrxwb/SigmoidÐ
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp7subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1Õ
!subsmtgotc/while/zdztiqrxwb/mul_1Mul4subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1:value:0subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_1Ô
!subsmtgotc/while/zdztiqrxwb/add_2AddV2*subsmtgotc/while/zdztiqrxwb/split:output:1%subsmtgotc/while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_2²
%subsmtgotc/while/zdztiqrxwb/Sigmoid_1Sigmoid%subsmtgotc/while/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%subsmtgotc/while/zdztiqrxwb/Sigmoid_1Ê
!subsmtgotc/while/zdztiqrxwb/mul_2Mul)subsmtgotc/while/zdztiqrxwb/Sigmoid_1:y:0subsmtgotc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_2ª
 subsmtgotc/while/zdztiqrxwb/TanhTanh*subsmtgotc/while/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 subsmtgotc/while/zdztiqrxwb/TanhÎ
!subsmtgotc/while/zdztiqrxwb/mul_3Mul'subsmtgotc/while/zdztiqrxwb/Sigmoid:y:0$subsmtgotc/while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_3Ï
!subsmtgotc/while/zdztiqrxwb/add_3AddV2%subsmtgotc/while/zdztiqrxwb/mul_2:z:0%subsmtgotc/while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_3Ð
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp7subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2Ü
!subsmtgotc/while/zdztiqrxwb/mul_4Mul4subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2:value:0%subsmtgotc/while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_4Ô
!subsmtgotc/while/zdztiqrxwb/add_4AddV2*subsmtgotc/while/zdztiqrxwb/split:output:3%subsmtgotc/while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/add_4²
%subsmtgotc/while/zdztiqrxwb/Sigmoid_2Sigmoid%subsmtgotc/while/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%subsmtgotc/while/zdztiqrxwb/Sigmoid_2©
"subsmtgotc/while/zdztiqrxwb/Tanh_1Tanh%subsmtgotc/while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"subsmtgotc/while/zdztiqrxwb/Tanh_1Ò
!subsmtgotc/while/zdztiqrxwb/mul_5Mul)subsmtgotc/while/zdztiqrxwb/Sigmoid_2:y:0&subsmtgotc/while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!subsmtgotc/while/zdztiqrxwb/mul_5
5subsmtgotc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemsubsmtgotc_while_placeholder_1subsmtgotc_while_placeholder%subsmtgotc/while/zdztiqrxwb/mul_5:z:0*
_output_shapes
: *
element_dtype027
5subsmtgotc/while/TensorArrayV2Write/TensorListSetItemr
subsmtgotc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
subsmtgotc/while/add/y
subsmtgotc/while/addAddV2subsmtgotc_while_placeholdersubsmtgotc/while/add/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/while/addv
subsmtgotc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
subsmtgotc/while/add_1/y­
subsmtgotc/while/add_1AddV2.subsmtgotc_while_subsmtgotc_while_loop_counter!subsmtgotc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
subsmtgotc/while/add_1©
subsmtgotc/while/IdentityIdentitysubsmtgotc/while/add_1:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/IdentityÇ
subsmtgotc/while/Identity_1Identity4subsmtgotc_while_subsmtgotc_while_maximum_iterations3^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/Identity_1«
subsmtgotc/while/Identity_2Identitysubsmtgotc/while/add:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/Identity_2Ø
subsmtgotc/while/Identity_3IdentityEsubsmtgotc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
subsmtgotc/while/Identity_3É
subsmtgotc/while/Identity_4Identity%subsmtgotc/while/zdztiqrxwb/mul_5:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/while/Identity_4É
subsmtgotc/while/Identity_5Identity%subsmtgotc/while/zdztiqrxwb/add_3:z:03^subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2^subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp4^subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp+^subsmtgotc/while/zdztiqrxwb/ReadVariableOp-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1-^subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subsmtgotc/while/Identity_5"?
subsmtgotc_while_identity"subsmtgotc/while/Identity:output:0"C
subsmtgotc_while_identity_1$subsmtgotc/while/Identity_1:output:0"C
subsmtgotc_while_identity_2$subsmtgotc/while/Identity_2:output:0"C
subsmtgotc_while_identity_3$subsmtgotc/while/Identity_3:output:0"C
subsmtgotc_while_identity_4$subsmtgotc/while/Identity_4:output:0"C
subsmtgotc_while_identity_5$subsmtgotc/while/Identity_5:output:0"\
+subsmtgotc_while_subsmtgotc_strided_slice_1-subsmtgotc_while_subsmtgotc_strided_slice_1_0"Ô
gsubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensorisubsmtgotc_while_tensorarrayv2read_tensorlistgetitem_subsmtgotc_tensorarrayunstack_tensorlistfromtensor_0"|
;subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource=subsmtgotc_while_zdztiqrxwb_biasadd_readvariableop_resource_0"~
<subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource>subsmtgotc_while_zdztiqrxwb_matmul_1_readvariableop_resource_0"z
:subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource<subsmtgotc_while_zdztiqrxwb_matmul_readvariableop_resource_0"p
5subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource7subsmtgotc_while_zdztiqrxwb_readvariableop_1_resource_0"p
5subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource7subsmtgotc_while_zdztiqrxwb_readvariableop_2_resource_0"l
3subsmtgotc_while_zdztiqrxwb_readvariableop_resource5subsmtgotc_while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2subsmtgotc/while/zdztiqrxwb/BiasAdd/ReadVariableOp2f
1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp1subsmtgotc/while/zdztiqrxwb/MatMul/ReadVariableOp2j
3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp3subsmtgotc/while/zdztiqrxwb/MatMul_1/ReadVariableOp2X
*subsmtgotc/while/zdztiqrxwb/ReadVariableOp*subsmtgotc/while/zdztiqrxwb/ReadVariableOp2\
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_1,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_12\
,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2,subsmtgotc/while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
¥
©	
(sequential_quyyatshey_while_cond_1063038H
Dsequential_quyyatshey_while_sequential_quyyatshey_while_loop_counterN
Jsequential_quyyatshey_while_sequential_quyyatshey_while_maximum_iterations+
'sequential_quyyatshey_while_placeholder-
)sequential_quyyatshey_while_placeholder_1-
)sequential_quyyatshey_while_placeholder_2-
)sequential_quyyatshey_while_placeholder_3J
Fsequential_quyyatshey_while_less_sequential_quyyatshey_strided_slice_1a
]sequential_quyyatshey_while_sequential_quyyatshey_while_cond_1063038___redundant_placeholder0a
]sequential_quyyatshey_while_sequential_quyyatshey_while_cond_1063038___redundant_placeholder1a
]sequential_quyyatshey_while_sequential_quyyatshey_while_cond_1063038___redundant_placeholder2a
]sequential_quyyatshey_while_sequential_quyyatshey_while_cond_1063038___redundant_placeholder3a
]sequential_quyyatshey_while_sequential_quyyatshey_while_cond_1063038___redundant_placeholder4a
]sequential_quyyatshey_while_sequential_quyyatshey_while_cond_1063038___redundant_placeholder5a
]sequential_quyyatshey_while_sequential_quyyatshey_while_cond_1063038___redundant_placeholder6(
$sequential_quyyatshey_while_identity
Þ
 sequential/quyyatshey/while/LessLess'sequential_quyyatshey_while_placeholderFsequential_quyyatshey_while_less_sequential_quyyatshey_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/quyyatshey/while/Less
$sequential/quyyatshey/while/IdentityIdentity$sequential/quyyatshey/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/quyyatshey/while/Identity"U
$sequential_quyyatshey_while_identity-sequential/quyyatshey/while/Identity:output:0*(
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
while_body_1067779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_nqcjuhnaut_matmul_readvariableop_resource_0:	 F
3while_nqcjuhnaut_matmul_1_readvariableop_resource_0:	 A
2while_nqcjuhnaut_biasadd_readvariableop_resource_0:	8
*while_nqcjuhnaut_readvariableop_resource_0: :
,while_nqcjuhnaut_readvariableop_1_resource_0: :
,while_nqcjuhnaut_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_nqcjuhnaut_matmul_readvariableop_resource:	 D
1while_nqcjuhnaut_matmul_1_readvariableop_resource:	 ?
0while_nqcjuhnaut_biasadd_readvariableop_resource:	6
(while_nqcjuhnaut_readvariableop_resource: 8
*while_nqcjuhnaut_readvariableop_1_resource: 8
*while_nqcjuhnaut_readvariableop_2_resource: ¢'while/nqcjuhnaut/BiasAdd/ReadVariableOp¢&while/nqcjuhnaut/MatMul/ReadVariableOp¢(while/nqcjuhnaut/MatMul_1/ReadVariableOp¢while/nqcjuhnaut/ReadVariableOp¢!while/nqcjuhnaut/ReadVariableOp_1¢!while/nqcjuhnaut/ReadVariableOp_2Ã
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
&while/nqcjuhnaut/MatMul/ReadVariableOpReadVariableOp1while_nqcjuhnaut_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/nqcjuhnaut/MatMul/ReadVariableOpÑ
while/nqcjuhnaut/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/nqcjuhnaut/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMulÉ
(while/nqcjuhnaut/MatMul_1/ReadVariableOpReadVariableOp3while_nqcjuhnaut_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/nqcjuhnaut/MatMul_1/ReadVariableOpº
while/nqcjuhnaut/MatMul_1MatMulwhile_placeholder_20while/nqcjuhnaut/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/MatMul_1°
while/nqcjuhnaut/addAddV2!while/nqcjuhnaut/MatMul:product:0#while/nqcjuhnaut/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/addÂ
'while/nqcjuhnaut/BiasAdd/ReadVariableOpReadVariableOp2while_nqcjuhnaut_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/nqcjuhnaut/BiasAdd/ReadVariableOp½
while/nqcjuhnaut/BiasAddBiasAddwhile/nqcjuhnaut/add:z:0/while/nqcjuhnaut/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/nqcjuhnaut/BiasAdd
 while/nqcjuhnaut/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/nqcjuhnaut/split/split_dim
while/nqcjuhnaut/splitSplit)while/nqcjuhnaut/split/split_dim:output:0!while/nqcjuhnaut/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/nqcjuhnaut/split©
while/nqcjuhnaut/ReadVariableOpReadVariableOp*while_nqcjuhnaut_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/nqcjuhnaut/ReadVariableOp£
while/nqcjuhnaut/mulMul'while/nqcjuhnaut/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul¦
while/nqcjuhnaut/add_1AddV2while/nqcjuhnaut/split:output:0while/nqcjuhnaut/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_1
while/nqcjuhnaut/SigmoidSigmoidwhile/nqcjuhnaut/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid¯
!while/nqcjuhnaut/ReadVariableOp_1ReadVariableOp,while_nqcjuhnaut_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_1©
while/nqcjuhnaut/mul_1Mul)while/nqcjuhnaut/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_1¨
while/nqcjuhnaut/add_2AddV2while/nqcjuhnaut/split:output:1while/nqcjuhnaut/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_2
while/nqcjuhnaut/Sigmoid_1Sigmoidwhile/nqcjuhnaut/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_1
while/nqcjuhnaut/mul_2Mulwhile/nqcjuhnaut/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_2
while/nqcjuhnaut/TanhTanhwhile/nqcjuhnaut/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh¢
while/nqcjuhnaut/mul_3Mulwhile/nqcjuhnaut/Sigmoid:y:0while/nqcjuhnaut/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_3£
while/nqcjuhnaut/add_3AddV2while/nqcjuhnaut/mul_2:z:0while/nqcjuhnaut/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_3¯
!while/nqcjuhnaut/ReadVariableOp_2ReadVariableOp,while_nqcjuhnaut_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/nqcjuhnaut/ReadVariableOp_2°
while/nqcjuhnaut/mul_4Mul)while/nqcjuhnaut/ReadVariableOp_2:value:0while/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_4¨
while/nqcjuhnaut/add_4AddV2while/nqcjuhnaut/split:output:3while/nqcjuhnaut/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/add_4
while/nqcjuhnaut/Sigmoid_2Sigmoidwhile/nqcjuhnaut/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Sigmoid_2
while/nqcjuhnaut/Tanh_1Tanhwhile/nqcjuhnaut/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/Tanh_1¦
while/nqcjuhnaut/mul_5Mulwhile/nqcjuhnaut/Sigmoid_2:y:0while/nqcjuhnaut/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/nqcjuhnaut/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/nqcjuhnaut/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/nqcjuhnaut/mul_5:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/nqcjuhnaut/add_3:z:0(^while/nqcjuhnaut/BiasAdd/ReadVariableOp'^while/nqcjuhnaut/MatMul/ReadVariableOp)^while/nqcjuhnaut/MatMul_1/ReadVariableOp ^while/nqcjuhnaut/ReadVariableOp"^while/nqcjuhnaut/ReadVariableOp_1"^while/nqcjuhnaut/ReadVariableOp_2*
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
0while_nqcjuhnaut_biasadd_readvariableop_resource2while_nqcjuhnaut_biasadd_readvariableop_resource_0"h
1while_nqcjuhnaut_matmul_1_readvariableop_resource3while_nqcjuhnaut_matmul_1_readvariableop_resource_0"d
/while_nqcjuhnaut_matmul_readvariableop_resource1while_nqcjuhnaut_matmul_readvariableop_resource_0"Z
*while_nqcjuhnaut_readvariableop_1_resource,while_nqcjuhnaut_readvariableop_1_resource_0"Z
*while_nqcjuhnaut_readvariableop_2_resource,while_nqcjuhnaut_readvariableop_2_resource_0"V
(while_nqcjuhnaut_readvariableop_resource*while_nqcjuhnaut_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/nqcjuhnaut/BiasAdd/ReadVariableOp'while/nqcjuhnaut/BiasAdd/ReadVariableOp2P
&while/nqcjuhnaut/MatMul/ReadVariableOp&while/nqcjuhnaut/MatMul/ReadVariableOp2T
(while/nqcjuhnaut/MatMul_1/ReadVariableOp(while/nqcjuhnaut/MatMul_1/ReadVariableOp2B
while/nqcjuhnaut/ReadVariableOpwhile/nqcjuhnaut/ReadVariableOp2F
!while/nqcjuhnaut/ReadVariableOp_1!while/nqcjuhnaut/ReadVariableOp_12F
!while/nqcjuhnaut/ReadVariableOp_2!while/nqcjuhnaut/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1064804
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1064804___redundant_placeholder05
1while_while_cond_1064804___redundant_placeholder15
1while_while_cond_1064804___redundant_placeholder25
1while_while_cond_1064804___redundant_placeholder35
1while_while_cond_1064804___redundant_placeholder45
1while_while_cond_1064804___redundant_placeholder55
1while_while_cond_1064804___redundant_placeholder6
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
while_body_1064805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_zdztiqrxwb_matmul_readvariableop_resource_0:	F
3while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 A
2while_zdztiqrxwb_biasadd_readvariableop_resource_0:	8
*while_zdztiqrxwb_readvariableop_resource_0: :
,while_zdztiqrxwb_readvariableop_1_resource_0: :
,while_zdztiqrxwb_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_zdztiqrxwb_matmul_readvariableop_resource:	D
1while_zdztiqrxwb_matmul_1_readvariableop_resource:	 ?
0while_zdztiqrxwb_biasadd_readvariableop_resource:	6
(while_zdztiqrxwb_readvariableop_resource: 8
*while_zdztiqrxwb_readvariableop_1_resource: 8
*while_zdztiqrxwb_readvariableop_2_resource: ¢'while/zdztiqrxwb/BiasAdd/ReadVariableOp¢&while/zdztiqrxwb/MatMul/ReadVariableOp¢(while/zdztiqrxwb/MatMul_1/ReadVariableOp¢while/zdztiqrxwb/ReadVariableOp¢!while/zdztiqrxwb/ReadVariableOp_1¢!while/zdztiqrxwb/ReadVariableOp_2Ã
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
&while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp1while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/zdztiqrxwb/MatMul/ReadVariableOpÑ
while/zdztiqrxwb/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMulÉ
(while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp3while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/zdztiqrxwb/MatMul_1/ReadVariableOpº
while/zdztiqrxwb/MatMul_1MatMulwhile_placeholder_20while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMul_1°
while/zdztiqrxwb/addAddV2!while/zdztiqrxwb/MatMul:product:0#while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/addÂ
'while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp2while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/zdztiqrxwb/BiasAdd/ReadVariableOp½
while/zdztiqrxwb/BiasAddBiasAddwhile/zdztiqrxwb/add:z:0/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/BiasAdd
 while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/zdztiqrxwb/split/split_dim
while/zdztiqrxwb/splitSplit)while/zdztiqrxwb/split/split_dim:output:0!while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/zdztiqrxwb/split©
while/zdztiqrxwb/ReadVariableOpReadVariableOp*while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/zdztiqrxwb/ReadVariableOp£
while/zdztiqrxwb/mulMul'while/zdztiqrxwb/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul¦
while/zdztiqrxwb/add_1AddV2while/zdztiqrxwb/split:output:0while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_1
while/zdztiqrxwb/SigmoidSigmoidwhile/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid¯
!while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp,while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_1©
while/zdztiqrxwb/mul_1Mul)while/zdztiqrxwb/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_1¨
while/zdztiqrxwb/add_2AddV2while/zdztiqrxwb/split:output:1while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_2
while/zdztiqrxwb/Sigmoid_1Sigmoidwhile/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_1
while/zdztiqrxwb/mul_2Mulwhile/zdztiqrxwb/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_2
while/zdztiqrxwb/TanhTanhwhile/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh¢
while/zdztiqrxwb/mul_3Mulwhile/zdztiqrxwb/Sigmoid:y:0while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_3£
while/zdztiqrxwb/add_3AddV2while/zdztiqrxwb/mul_2:z:0while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_3¯
!while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp,while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_2°
while/zdztiqrxwb/mul_4Mul)while/zdztiqrxwb/ReadVariableOp_2:value:0while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_4¨
while/zdztiqrxwb/add_4AddV2while/zdztiqrxwb/split:output:3while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_4
while/zdztiqrxwb/Sigmoid_2Sigmoidwhile/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_2
while/zdztiqrxwb/Tanh_1Tanhwhile/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh_1¦
while/zdztiqrxwb/mul_5Mulwhile/zdztiqrxwb/Sigmoid_2:y:0while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/zdztiqrxwb/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/zdztiqrxwb/mul_5:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/zdztiqrxwb/add_3:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
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
0while_zdztiqrxwb_biasadd_readvariableop_resource2while_zdztiqrxwb_biasadd_readvariableop_resource_0"h
1while_zdztiqrxwb_matmul_1_readvariableop_resource3while_zdztiqrxwb_matmul_1_readvariableop_resource_0"d
/while_zdztiqrxwb_matmul_readvariableop_resource1while_zdztiqrxwb_matmul_readvariableop_resource_0"Z
*while_zdztiqrxwb_readvariableop_1_resource,while_zdztiqrxwb_readvariableop_1_resource_0"Z
*while_zdztiqrxwb_readvariableop_2_resource,while_zdztiqrxwb_readvariableop_2_resource_0"V
(while_zdztiqrxwb_readvariableop_resource*while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/zdztiqrxwb/BiasAdd/ReadVariableOp'while/zdztiqrxwb/BiasAdd/ReadVariableOp2P
&while/zdztiqrxwb/MatMul/ReadVariableOp&while/zdztiqrxwb/MatMul/ReadVariableOp2T
(while/zdztiqrxwb/MatMul_1/ReadVariableOp(while/zdztiqrxwb/MatMul_1/ReadVariableOp2B
while/zdztiqrxwb/ReadVariableOpwhile/zdztiqrxwb/ReadVariableOp2F
!while/zdztiqrxwb/ReadVariableOp_1!while/zdztiqrxwb/ReadVariableOp_12F
!while/zdztiqrxwb/ReadVariableOp_2!while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
subsmtgotc_while_cond_10660922
.subsmtgotc_while_subsmtgotc_while_loop_counter8
4subsmtgotc_while_subsmtgotc_while_maximum_iterations 
subsmtgotc_while_placeholder"
subsmtgotc_while_placeholder_1"
subsmtgotc_while_placeholder_2"
subsmtgotc_while_placeholder_34
0subsmtgotc_while_less_subsmtgotc_strided_slice_1K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066092___redundant_placeholder0K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066092___redundant_placeholder1K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066092___redundant_placeholder2K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066092___redundant_placeholder3K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066092___redundant_placeholder4K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066092___redundant_placeholder5K
Gsubsmtgotc_while_subsmtgotc_while_cond_1066092___redundant_placeholder6
subsmtgotc_while_identity
§
subsmtgotc/while/LessLesssubsmtgotc_while_placeholder0subsmtgotc_while_less_subsmtgotc_strided_slice_1*
T0*
_output_shapes
: 2
subsmtgotc/while/Less~
subsmtgotc/while/IdentityIdentitysubsmtgotc/while/Less:z:0*
T0
*
_output_shapes
: 2
subsmtgotc/while/Identity"?
subsmtgotc_while_identity"subsmtgotc/while/Identity:output:0*(
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
while_cond_1067530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1067530___redundant_placeholder05
1while_while_cond_1067530___redundant_placeholder15
1while_while_cond_1067530___redundant_placeholder25
1while_while_cond_1067530___redundant_placeholder35
1while_while_cond_1067530___redundant_placeholder45
1while_while_cond_1067530___redundant_placeholder55
1while_while_cond_1067530___redundant_placeholder6
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
while_body_1067351
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_zdztiqrxwb_matmul_readvariableop_resource_0:	F
3while_zdztiqrxwb_matmul_1_readvariableop_resource_0:	 A
2while_zdztiqrxwb_biasadd_readvariableop_resource_0:	8
*while_zdztiqrxwb_readvariableop_resource_0: :
,while_zdztiqrxwb_readvariableop_1_resource_0: :
,while_zdztiqrxwb_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_zdztiqrxwb_matmul_readvariableop_resource:	D
1while_zdztiqrxwb_matmul_1_readvariableop_resource:	 ?
0while_zdztiqrxwb_biasadd_readvariableop_resource:	6
(while_zdztiqrxwb_readvariableop_resource: 8
*while_zdztiqrxwb_readvariableop_1_resource: 8
*while_zdztiqrxwb_readvariableop_2_resource: ¢'while/zdztiqrxwb/BiasAdd/ReadVariableOp¢&while/zdztiqrxwb/MatMul/ReadVariableOp¢(while/zdztiqrxwb/MatMul_1/ReadVariableOp¢while/zdztiqrxwb/ReadVariableOp¢!while/zdztiqrxwb/ReadVariableOp_1¢!while/zdztiqrxwb/ReadVariableOp_2Ã
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
&while/zdztiqrxwb/MatMul/ReadVariableOpReadVariableOp1while_zdztiqrxwb_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/zdztiqrxwb/MatMul/ReadVariableOpÑ
while/zdztiqrxwb/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/zdztiqrxwb/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMulÉ
(while/zdztiqrxwb/MatMul_1/ReadVariableOpReadVariableOp3while_zdztiqrxwb_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/zdztiqrxwb/MatMul_1/ReadVariableOpº
while/zdztiqrxwb/MatMul_1MatMulwhile_placeholder_20while/zdztiqrxwb/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/MatMul_1°
while/zdztiqrxwb/addAddV2!while/zdztiqrxwb/MatMul:product:0#while/zdztiqrxwb/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/addÂ
'while/zdztiqrxwb/BiasAdd/ReadVariableOpReadVariableOp2while_zdztiqrxwb_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/zdztiqrxwb/BiasAdd/ReadVariableOp½
while/zdztiqrxwb/BiasAddBiasAddwhile/zdztiqrxwb/add:z:0/while/zdztiqrxwb/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/zdztiqrxwb/BiasAdd
 while/zdztiqrxwb/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/zdztiqrxwb/split/split_dim
while/zdztiqrxwb/splitSplit)while/zdztiqrxwb/split/split_dim:output:0!while/zdztiqrxwb/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/zdztiqrxwb/split©
while/zdztiqrxwb/ReadVariableOpReadVariableOp*while_zdztiqrxwb_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/zdztiqrxwb/ReadVariableOp£
while/zdztiqrxwb/mulMul'while/zdztiqrxwb/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul¦
while/zdztiqrxwb/add_1AddV2while/zdztiqrxwb/split:output:0while/zdztiqrxwb/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_1
while/zdztiqrxwb/SigmoidSigmoidwhile/zdztiqrxwb/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid¯
!while/zdztiqrxwb/ReadVariableOp_1ReadVariableOp,while_zdztiqrxwb_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_1©
while/zdztiqrxwb/mul_1Mul)while/zdztiqrxwb/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_1¨
while/zdztiqrxwb/add_2AddV2while/zdztiqrxwb/split:output:1while/zdztiqrxwb/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_2
while/zdztiqrxwb/Sigmoid_1Sigmoidwhile/zdztiqrxwb/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_1
while/zdztiqrxwb/mul_2Mulwhile/zdztiqrxwb/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_2
while/zdztiqrxwb/TanhTanhwhile/zdztiqrxwb/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh¢
while/zdztiqrxwb/mul_3Mulwhile/zdztiqrxwb/Sigmoid:y:0while/zdztiqrxwb/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_3£
while/zdztiqrxwb/add_3AddV2while/zdztiqrxwb/mul_2:z:0while/zdztiqrxwb/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_3¯
!while/zdztiqrxwb/ReadVariableOp_2ReadVariableOp,while_zdztiqrxwb_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/zdztiqrxwb/ReadVariableOp_2°
while/zdztiqrxwb/mul_4Mul)while/zdztiqrxwb/ReadVariableOp_2:value:0while/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_4¨
while/zdztiqrxwb/add_4AddV2while/zdztiqrxwb/split:output:3while/zdztiqrxwb/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/add_4
while/zdztiqrxwb/Sigmoid_2Sigmoidwhile/zdztiqrxwb/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Sigmoid_2
while/zdztiqrxwb/Tanh_1Tanhwhile/zdztiqrxwb/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/Tanh_1¦
while/zdztiqrxwb/mul_5Mulwhile/zdztiqrxwb/Sigmoid_2:y:0while/zdztiqrxwb/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/zdztiqrxwb/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/zdztiqrxwb/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/zdztiqrxwb/mul_5:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/zdztiqrxwb/add_3:z:0(^while/zdztiqrxwb/BiasAdd/ReadVariableOp'^while/zdztiqrxwb/MatMul/ReadVariableOp)^while/zdztiqrxwb/MatMul_1/ReadVariableOp ^while/zdztiqrxwb/ReadVariableOp"^while/zdztiqrxwb/ReadVariableOp_1"^while/zdztiqrxwb/ReadVariableOp_2*
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
0while_zdztiqrxwb_biasadd_readvariableop_resource2while_zdztiqrxwb_biasadd_readvariableop_resource_0"h
1while_zdztiqrxwb_matmul_1_readvariableop_resource3while_zdztiqrxwb_matmul_1_readvariableop_resource_0"d
/while_zdztiqrxwb_matmul_readvariableop_resource1while_zdztiqrxwb_matmul_readvariableop_resource_0"Z
*while_zdztiqrxwb_readvariableop_1_resource,while_zdztiqrxwb_readvariableop_1_resource_0"Z
*while_zdztiqrxwb_readvariableop_2_resource,while_zdztiqrxwb_readvariableop_2_resource_0"V
(while_zdztiqrxwb_readvariableop_resource*while_zdztiqrxwb_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/zdztiqrxwb/BiasAdd/ReadVariableOp'while/zdztiqrxwb/BiasAdd/ReadVariableOp2P
&while/zdztiqrxwb/MatMul/ReadVariableOp&while/zdztiqrxwb/MatMul/ReadVariableOp2T
(while/zdztiqrxwb/MatMul_1/ReadVariableOp(while/zdztiqrxwb/MatMul_1/ReadVariableOp2B
while/zdztiqrxwb/ReadVariableOpwhile/zdztiqrxwb/ReadVariableOp2F
!while/zdztiqrxwb/ReadVariableOp_1!while/zdztiqrxwb/ReadVariableOp_12F
!while/zdztiqrxwb/ReadVariableOp_2!while/zdztiqrxwb/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_quyyatshey_layer_call_fn_1067666
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_10643542
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
È

,__inference_umdyqemnpr_layer_call_fn_1066789

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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_10647062
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
Ó	
ø
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_1065123

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
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
I

mkdkkixskm;
serving_default_mkdkkixskm:0ÿÿÿÿÿÿÿÿÿ>

ycxcgxamrr0
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
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "mkdkkixskm"}}, {"class_name": "Conv1D", "config": {"name": "umdyqemnpr", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "unxqeixodn", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "subsmtgotc", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "zdztiqrxwb", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "quyyatshey", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "nqcjuhnaut", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "ycxcgxamrr", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "mkdkkixskm"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "mkdkkixskm"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "umdyqemnpr", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "unxqeixodn", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "subsmtgotc", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "zdztiqrxwb", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "quyyatshey", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "nqcjuhnaut", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "ycxcgxamrr", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
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
{"name": "umdyqemnpr", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "umdyqemnpr", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ÿ
_tf_keras_layerå{"name": "unxqeixodn", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "unxqeixodn", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
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
_tf_keras_rnn_layerä{"name": "subsmtgotc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "subsmtgotc", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "zdztiqrxwb", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
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
_tf_keras_rnn_layerê{"name": "quyyatshey", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "quyyatshey", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "nqcjuhnaut", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
__call__
+&call_and_return_all_conditional_losses"²
_tf_keras_layer{"name": "ycxcgxamrr", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "ycxcgxamrr", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
':%2umdyqemnpr/kernel
:2umdyqemnpr/bias
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
_tf_keras_layer¼{"name": "zdztiqrxwb", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "zdztiqrxwb", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
_tf_keras_layerÀ{"name": "nqcjuhnaut", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "nqcjuhnaut", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
#:! 2ycxcgxamrr/kernel
:2ycxcgxamrr/bias
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
/:-	2subsmtgotc/zdztiqrxwb/kernel
9:7	 2&subsmtgotc/zdztiqrxwb/recurrent_kernel
):'2subsmtgotc/zdztiqrxwb/bias
?:= 21subsmtgotc/zdztiqrxwb/input_gate_peephole_weights
@:> 22subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights
@:> 22subsmtgotc/zdztiqrxwb/output_gate_peephole_weights
/:-	 2quyyatshey/nqcjuhnaut/kernel
9:7	 2&quyyatshey/nqcjuhnaut/recurrent_kernel
):'2quyyatshey/nqcjuhnaut/bias
?:= 21quyyatshey/nqcjuhnaut/input_gate_peephole_weights
@:> 22quyyatshey/nqcjuhnaut/forget_gate_peephole_weights
@:> 22quyyatshey/nqcjuhnaut/output_gate_peephole_weights
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
1:/2RMSprop/umdyqemnpr/kernel/rms
':%2RMSprop/umdyqemnpr/bias/rms
-:+ 2RMSprop/ycxcgxamrr/kernel/rms
':%2RMSprop/ycxcgxamrr/bias/rms
9:7	2(RMSprop/subsmtgotc/zdztiqrxwb/kernel/rms
C:A	 22RMSprop/subsmtgotc/zdztiqrxwb/recurrent_kernel/rms
3:12&RMSprop/subsmtgotc/zdztiqrxwb/bias/rms
I:G 2=RMSprop/subsmtgotc/zdztiqrxwb/input_gate_peephole_weights/rms
J:H 2>RMSprop/subsmtgotc/zdztiqrxwb/forget_gate_peephole_weights/rms
J:H 2>RMSprop/subsmtgotc/zdztiqrxwb/output_gate_peephole_weights/rms
9:7	 2(RMSprop/quyyatshey/nqcjuhnaut/kernel/rms
C:A	 22RMSprop/quyyatshey/nqcjuhnaut/recurrent_kernel/rms
3:12&RMSprop/quyyatshey/nqcjuhnaut/bias/rms
I:G 2=RMSprop/quyyatshey/nqcjuhnaut/input_gate_peephole_weights/rms
J:H 2>RMSprop/quyyatshey/nqcjuhnaut/forget_gate_peephole_weights/rms
J:H 2>RMSprop/quyyatshey/nqcjuhnaut/output_gate_peephole_weights/rms
þ2û
,__inference_sequential_layer_call_fn_1065165
,__inference_sequential_layer_call_fn_1065935
,__inference_sequential_layer_call_fn_1065972
,__inference_sequential_layer_call_fn_1065771À
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
G__inference_sequential_layer_call_and_return_conditional_losses_1066376
G__inference_sequential_layer_call_and_return_conditional_losses_1066780
G__inference_sequential_layer_call_and_return_conditional_losses_1065812
G__inference_sequential_layer_call_and_return_conditional_losses_1065853À
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
"__inference__wrapped_model_1063146Á
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

mkdkkixskmÿÿÿÿÿÿÿÿÿ
Ö2Ó
,__inference_umdyqemnpr_layer_call_fn_1066789¢
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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_1066826¢
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
,__inference_unxqeixodn_layer_call_fn_1066831¢
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
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_1066844¢
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
,__inference_subsmtgotc_layer_call_fn_1066861
,__inference_subsmtgotc_layer_call_fn_1066878
,__inference_subsmtgotc_layer_call_fn_1066895
,__inference_subsmtgotc_layer_call_fn_1066912æ
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067092
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067272
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067452
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067632æ
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
,__inference_quyyatshey_layer_call_fn_1067649
,__inference_quyyatshey_layer_call_fn_1067666
,__inference_quyyatshey_layer_call_fn_1067683
,__inference_quyyatshey_layer_call_fn_1067700æ
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1067880
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068060
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068240
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068420æ
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
,__inference_ycxcgxamrr_layer_call_fn_1068429¢
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
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_1068439¢
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
%__inference_signature_wrapper_1065898
mkdkkixskm"
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
,__inference_zdztiqrxwb_layer_call_fn_1068462
,__inference_zdztiqrxwb_layer_call_fn_1068485¾
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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1068529
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1068573¾
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
,__inference_nqcjuhnaut_layer_call_fn_1068596
,__inference_nqcjuhnaut_layer_call_fn_1068619¾
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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1068663
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1068707¾
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
"__inference__wrapped_model_1063146-./012345678"#;¢8
1¢.
,)

mkdkkixskmÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

ycxcgxamrr$!

ycxcgxamrrÿÿÿÿÿÿÿÿÿÌ
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1068663345678¢}
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
G__inference_nqcjuhnaut_layer_call_and_return_conditional_losses_1068707345678¢}
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
,__inference_nqcjuhnaut_layer_call_fn_1068596ð345678¢}
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
,__inference_nqcjuhnaut_layer_call_fn_1068619ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ð
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1067880345678S¢P
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068060345678S¢P
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068240t345678C¢@
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
G__inference_quyyatshey_layer_call_and_return_conditional_losses_1068420t345678C¢@
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
,__inference_quyyatshey_layer_call_fn_1067649w345678S¢P
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
,__inference_quyyatshey_layer_call_fn_1067666w345678S¢P
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
,__inference_quyyatshey_layer_call_fn_1067683g345678C¢@
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
,__inference_quyyatshey_layer_call_fn_1067700g345678C¢@
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
G__inference_sequential_layer_call_and_return_conditional_losses_1065812~-./012345678"#C¢@
9¢6
,)

mkdkkixskmÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_layer_call_and_return_conditional_losses_1065853~-./012345678"#C¢@
9¢6
,)

mkdkkixskmÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_1066376z-./012345678"#?¢<
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
G__inference_sequential_layer_call_and_return_conditional_losses_1066780z-./012345678"#?¢<
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
,__inference_sequential_layer_call_fn_1065165q-./012345678"#C¢@
9¢6
,)

mkdkkixskmÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_layer_call_fn_1065771q-./012345678"#C¢@
9¢6
,)

mkdkkixskmÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1065935m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1065972m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
%__inference_signature_wrapper_1065898-./012345678"#I¢F
¢ 
?ª<
:

mkdkkixskm,)

mkdkkixskmÿÿÿÿÿÿÿÿÿ"7ª4
2

ycxcgxamrr$!

ycxcgxamrrÿÿÿÿÿÿÿÿÿÝ
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067092-./012S¢P
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067272-./012S¢P
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067452x-./012C¢@
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
G__inference_subsmtgotc_layer_call_and_return_conditional_losses_1067632x-./012C¢@
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
,__inference_subsmtgotc_layer_call_fn_1066861-./012S¢P
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
,__inference_subsmtgotc_layer_call_fn_1066878-./012S¢P
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
,__inference_subsmtgotc_layer_call_fn_1066895k-./012C¢@
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
,__inference_subsmtgotc_layer_call_fn_1066912k-./012C¢@
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
G__inference_umdyqemnpr_layer_call_and_return_conditional_losses_1066826l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_umdyqemnpr_layer_call_fn_1066789_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¯
G__inference_unxqeixodn_layer_call_and_return_conditional_losses_1066844d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_unxqeixodn_layer_call_fn_1066831W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_ycxcgxamrr_layer_call_and_return_conditional_losses_1068439\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_ycxcgxamrr_layer_call_fn_1068429O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÌ
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1068529-./012¢}
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
G__inference_zdztiqrxwb_layer_call_and_return_conditional_losses_1068573-./012¢}
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
,__inference_zdztiqrxwb_layer_call_fn_1068462ð-./012¢}
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
,__inference_zdztiqrxwb_layer_call_fn_1068485ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ 