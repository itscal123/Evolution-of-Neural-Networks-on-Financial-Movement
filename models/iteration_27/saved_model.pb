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
fethhjgisa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefethhjgisa/kernel
{
%fethhjgisa/kernel/Read/ReadVariableOpReadVariableOpfethhjgisa/kernel*"
_output_shapes
:*
dtype0
v
fethhjgisa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefethhjgisa/bias
o
#fethhjgisa/bias/Read/ReadVariableOpReadVariableOpfethhjgisa/bias*
_output_shapes
:*
dtype0
~
boyhyiogqf/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameboyhyiogqf/kernel
w
%boyhyiogqf/kernel/Read/ReadVariableOpReadVariableOpboyhyiogqf/kernel*
_output_shapes

: *
dtype0
v
boyhyiogqf/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameboyhyiogqf/bias
o
#boyhyiogqf/bias/Read/ReadVariableOpReadVariableOpboyhyiogqf/bias*
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
oohztcvkwo/ammytzqwsz/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameoohztcvkwo/ammytzqwsz/kernel

0oohztcvkwo/ammytzqwsz/kernel/Read/ReadVariableOpReadVariableOpoohztcvkwo/ammytzqwsz/kernel*
_output_shapes
:	*
dtype0
©
&oohztcvkwo/ammytzqwsz/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&oohztcvkwo/ammytzqwsz/recurrent_kernel
¢
:oohztcvkwo/ammytzqwsz/recurrent_kernel/Read/ReadVariableOpReadVariableOp&oohztcvkwo/ammytzqwsz/recurrent_kernel*
_output_shapes
:	 *
dtype0

oohztcvkwo/ammytzqwsz/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameoohztcvkwo/ammytzqwsz/bias

.oohztcvkwo/ammytzqwsz/bias/Read/ReadVariableOpReadVariableOpoohztcvkwo/ammytzqwsz/bias*
_output_shapes	
:*
dtype0
º
1oohztcvkwo/ammytzqwsz/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31oohztcvkwo/ammytzqwsz/input_gate_peephole_weights
³
Eoohztcvkwo/ammytzqwsz/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1oohztcvkwo/ammytzqwsz/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2oohztcvkwo/ammytzqwsz/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights
µ
Foohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2oohztcvkwo/ammytzqwsz/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42oohztcvkwo/ammytzqwsz/output_gate_peephole_weights
µ
Foohztcvkwo/ammytzqwsz/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2oohztcvkwo/ammytzqwsz/output_gate_peephole_weights*
_output_shapes
: *
dtype0

mfdtyewult/aflyrndiyz/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namemfdtyewult/aflyrndiyz/kernel

0mfdtyewult/aflyrndiyz/kernel/Read/ReadVariableOpReadVariableOpmfdtyewult/aflyrndiyz/kernel*
_output_shapes
:	 *
dtype0
©
&mfdtyewult/aflyrndiyz/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&mfdtyewult/aflyrndiyz/recurrent_kernel
¢
:mfdtyewult/aflyrndiyz/recurrent_kernel/Read/ReadVariableOpReadVariableOp&mfdtyewult/aflyrndiyz/recurrent_kernel*
_output_shapes
:	 *
dtype0

mfdtyewult/aflyrndiyz/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namemfdtyewult/aflyrndiyz/bias

.mfdtyewult/aflyrndiyz/bias/Read/ReadVariableOpReadVariableOpmfdtyewult/aflyrndiyz/bias*
_output_shapes	
:*
dtype0
º
1mfdtyewult/aflyrndiyz/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31mfdtyewult/aflyrndiyz/input_gate_peephole_weights
³
Emfdtyewult/aflyrndiyz/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1mfdtyewult/aflyrndiyz/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2mfdtyewult/aflyrndiyz/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42mfdtyewult/aflyrndiyz/forget_gate_peephole_weights
µ
Fmfdtyewult/aflyrndiyz/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2mfdtyewult/aflyrndiyz/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2mfdtyewult/aflyrndiyz/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42mfdtyewult/aflyrndiyz/output_gate_peephole_weights
µ
Fmfdtyewult/aflyrndiyz/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2mfdtyewult/aflyrndiyz/output_gate_peephole_weights*
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
RMSprop/fethhjgisa/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/fethhjgisa/kernel/rms

1RMSprop/fethhjgisa/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/fethhjgisa/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/fethhjgisa/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/fethhjgisa/bias/rms

/RMSprop/fethhjgisa/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/fethhjgisa/bias/rms*
_output_shapes
:*
dtype0

RMSprop/boyhyiogqf/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/boyhyiogqf/kernel/rms

1RMSprop/boyhyiogqf/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/boyhyiogqf/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/boyhyiogqf/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/boyhyiogqf/bias/rms

/RMSprop/boyhyiogqf/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/boyhyiogqf/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/oohztcvkwo/ammytzqwsz/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/oohztcvkwo/ammytzqwsz/kernel/rms
¦
<RMSprop/oohztcvkwo/ammytzqwsz/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/oohztcvkwo/ammytzqwsz/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rms
º
FRMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/oohztcvkwo/ammytzqwsz/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/oohztcvkwo/ammytzqwsz/bias/rms

:RMSprop/oohztcvkwo/ammytzqwsz/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/oohztcvkwo/ammytzqwsz/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rms
Ë
QRMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rms
Í
RRMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rms
Í
RRMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/mfdtyewult/aflyrndiyz/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/mfdtyewult/aflyrndiyz/kernel/rms
¦
<RMSprop/mfdtyewult/aflyrndiyz/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/mfdtyewult/aflyrndiyz/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rms
º
FRMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/mfdtyewult/aflyrndiyz/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/mfdtyewult/aflyrndiyz/bias/rms

:RMSprop/mfdtyewult/aflyrndiyz/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/mfdtyewult/aflyrndiyz/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rms
Ë
QRMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rms
Í
RRMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rms
Í
RRMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rms*
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
VARIABLE_VALUEfethhjgisa/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEfethhjgisa/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEboyhyiogqf/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEboyhyiogqf/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEoohztcvkwo/ammytzqwsz/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&oohztcvkwo/ammytzqwsz/recurrent_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEoohztcvkwo/ammytzqwsz/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1oohztcvkwo/ammytzqwsz/input_gate_peephole_weights0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2oohztcvkwo/ammytzqwsz/output_gate_peephole_weights0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEmfdtyewult/aflyrndiyz/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&mfdtyewult/aflyrndiyz/recurrent_kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEmfdtyewult/aflyrndiyz/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1mfdtyewult/aflyrndiyz/input_gate_peephole_weights1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2mfdtyewult/aflyrndiyz/forget_gate_peephole_weights1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2mfdtyewult/aflyrndiyz/output_gate_peephole_weights1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUERMSprop/fethhjgisa/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/fethhjgisa/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/boyhyiogqf/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/boyhyiogqf/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/oohztcvkwo/ammytzqwsz/kernel/rmsNtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rmsNtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/oohztcvkwo/ammytzqwsz/bias/rmsNtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¢
VARIABLE_VALUE=RMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rmsNtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE>RMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rmsNtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE>RMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rmsNtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/mfdtyewult/aflyrndiyz/kernel/rmsNtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rmsNtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/mfdtyewult/aflyrndiyz/bias/rmsOtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUE=RMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rmsOtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¤¡
VARIABLE_VALUE>RMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rmsOtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
¤¡
VARIABLE_VALUE>RMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rmsOtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_liksrhmmuxPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_liksrhmmuxfethhjgisa/kernelfethhjgisa/biasoohztcvkwo/ammytzqwsz/kernel&oohztcvkwo/ammytzqwsz/recurrent_kerneloohztcvkwo/ammytzqwsz/bias1oohztcvkwo/ammytzqwsz/input_gate_peephole_weights2oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights2oohztcvkwo/ammytzqwsz/output_gate_peephole_weightsmfdtyewult/aflyrndiyz/kernel&mfdtyewult/aflyrndiyz/recurrent_kernelmfdtyewult/aflyrndiyz/bias1mfdtyewult/aflyrndiyz/input_gate_peephole_weights2mfdtyewult/aflyrndiyz/forget_gate_peephole_weights2mfdtyewult/aflyrndiyz/output_gate_peephole_weightsboyhyiogqf/kernelboyhyiogqf/bias*
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
%__inference_signature_wrapper_1821602
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%fethhjgisa/kernel/Read/ReadVariableOp#fethhjgisa/bias/Read/ReadVariableOp%boyhyiogqf/kernel/Read/ReadVariableOp#boyhyiogqf/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0oohztcvkwo/ammytzqwsz/kernel/Read/ReadVariableOp:oohztcvkwo/ammytzqwsz/recurrent_kernel/Read/ReadVariableOp.oohztcvkwo/ammytzqwsz/bias/Read/ReadVariableOpEoohztcvkwo/ammytzqwsz/input_gate_peephole_weights/Read/ReadVariableOpFoohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/Read/ReadVariableOpFoohztcvkwo/ammytzqwsz/output_gate_peephole_weights/Read/ReadVariableOp0mfdtyewult/aflyrndiyz/kernel/Read/ReadVariableOp:mfdtyewult/aflyrndiyz/recurrent_kernel/Read/ReadVariableOp.mfdtyewult/aflyrndiyz/bias/Read/ReadVariableOpEmfdtyewult/aflyrndiyz/input_gate_peephole_weights/Read/ReadVariableOpFmfdtyewult/aflyrndiyz/forget_gate_peephole_weights/Read/ReadVariableOpFmfdtyewult/aflyrndiyz/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/fethhjgisa/kernel/rms/Read/ReadVariableOp/RMSprop/fethhjgisa/bias/rms/Read/ReadVariableOp1RMSprop/boyhyiogqf/kernel/rms/Read/ReadVariableOp/RMSprop/boyhyiogqf/bias/rms/Read/ReadVariableOp<RMSprop/oohztcvkwo/ammytzqwsz/kernel/rms/Read/ReadVariableOpFRMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/oohztcvkwo/ammytzqwsz/bias/rms/Read/ReadVariableOpQRMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/mfdtyewult/aflyrndiyz/kernel/rms/Read/ReadVariableOpFRMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/mfdtyewult/aflyrndiyz/bias/rms/Read/ReadVariableOpQRMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
 __inference__traced_save_1824551
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefethhjgisa/kernelfethhjgisa/biasboyhyiogqf/kernelboyhyiogqf/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhooohztcvkwo/ammytzqwsz/kernel&oohztcvkwo/ammytzqwsz/recurrent_kerneloohztcvkwo/ammytzqwsz/bias1oohztcvkwo/ammytzqwsz/input_gate_peephole_weights2oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights2oohztcvkwo/ammytzqwsz/output_gate_peephole_weightsmfdtyewult/aflyrndiyz/kernel&mfdtyewult/aflyrndiyz/recurrent_kernelmfdtyewult/aflyrndiyz/bias1mfdtyewult/aflyrndiyz/input_gate_peephole_weights2mfdtyewult/aflyrndiyz/forget_gate_peephole_weights2mfdtyewult/aflyrndiyz/output_gate_peephole_weightstotalcountRMSprop/fethhjgisa/kernel/rmsRMSprop/fethhjgisa/bias/rmsRMSprop/boyhyiogqf/kernel/rmsRMSprop/boyhyiogqf/bias/rms(RMSprop/oohztcvkwo/ammytzqwsz/kernel/rms2RMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rms&RMSprop/oohztcvkwo/ammytzqwsz/bias/rms=RMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rms>RMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rms>RMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rms(RMSprop/mfdtyewult/aflyrndiyz/kernel/rms2RMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rms&RMSprop/mfdtyewult/aflyrndiyz/bias/rms=RMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rms>RMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rms>RMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rms*3
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
#__inference__traced_restore_1824678¢ä-
¹'
µ
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1824231

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
p
Ê
mfdtyewult_while_body_18223032
.mfdtyewult_while_mfdtyewult_while_loop_counter8
4mfdtyewult_while_mfdtyewult_while_maximum_iterations 
mfdtyewult_while_placeholder"
mfdtyewult_while_placeholder_1"
mfdtyewult_while_placeholder_2"
mfdtyewult_while_placeholder_31
-mfdtyewult_while_mfdtyewult_strided_slice_1_0m
imfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0O
<mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0:	 Q
>mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 L
=mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0:	C
5mfdtyewult_while_aflyrndiyz_readvariableop_resource_0: E
7mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0: E
7mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0: 
mfdtyewult_while_identity
mfdtyewult_while_identity_1
mfdtyewult_while_identity_2
mfdtyewult_while_identity_3
mfdtyewult_while_identity_4
mfdtyewult_while_identity_5/
+mfdtyewult_while_mfdtyewult_strided_slice_1k
gmfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensorM
:mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource:	 O
<mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource:	 J
;mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource:	A
3mfdtyewult_while_aflyrndiyz_readvariableop_resource: C
5mfdtyewult_while_aflyrndiyz_readvariableop_1_resource: C
5mfdtyewult_while_aflyrndiyz_readvariableop_2_resource: ¢2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp¢1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp¢3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp¢*mfdtyewult/while/aflyrndiyz/ReadVariableOp¢,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1¢,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2Ù
Bmfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bmfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shape
4mfdtyewult/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemimfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0mfdtyewult_while_placeholderKmfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4mfdtyewult/while/TensorArrayV2Read/TensorListGetItemä
1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp<mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOpý
"mfdtyewult/while/aflyrndiyz/MatMulMatMul;mfdtyewult/while/TensorArrayV2Read/TensorListGetItem:item:09mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"mfdtyewult/while/aflyrndiyz/MatMulê
3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp>mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOpæ
$mfdtyewult/while/aflyrndiyz/MatMul_1MatMulmfdtyewult_while_placeholder_2;mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$mfdtyewult/while/aflyrndiyz/MatMul_1Ü
mfdtyewult/while/aflyrndiyz/addAddV2,mfdtyewult/while/aflyrndiyz/MatMul:product:0.mfdtyewult/while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
mfdtyewult/while/aflyrndiyz/addã
2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp=mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOpé
#mfdtyewult/while/aflyrndiyz/BiasAddBiasAdd#mfdtyewult/while/aflyrndiyz/add:z:0:mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#mfdtyewult/while/aflyrndiyz/BiasAdd
+mfdtyewult/while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+mfdtyewult/while/aflyrndiyz/split/split_dim¯
!mfdtyewult/while/aflyrndiyz/splitSplit4mfdtyewult/while/aflyrndiyz/split/split_dim:output:0,mfdtyewult/while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!mfdtyewult/while/aflyrndiyz/splitÊ
*mfdtyewult/while/aflyrndiyz/ReadVariableOpReadVariableOp5mfdtyewult_while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*mfdtyewult/while/aflyrndiyz/ReadVariableOpÏ
mfdtyewult/while/aflyrndiyz/mulMul2mfdtyewult/while/aflyrndiyz/ReadVariableOp:value:0mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
mfdtyewult/while/aflyrndiyz/mulÒ
!mfdtyewult/while/aflyrndiyz/add_1AddV2*mfdtyewult/while/aflyrndiyz/split:output:0#mfdtyewult/while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_1®
#mfdtyewult/while/aflyrndiyz/SigmoidSigmoid%mfdtyewult/while/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#mfdtyewult/while/aflyrndiyz/SigmoidÐ
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1ReadVariableOp7mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1Õ
!mfdtyewult/while/aflyrndiyz/mul_1Mul4mfdtyewult/while/aflyrndiyz/ReadVariableOp_1:value:0mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_1Ô
!mfdtyewult/while/aflyrndiyz/add_2AddV2*mfdtyewult/while/aflyrndiyz/split:output:1%mfdtyewult/while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_2²
%mfdtyewult/while/aflyrndiyz/Sigmoid_1Sigmoid%mfdtyewult/while/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%mfdtyewult/while/aflyrndiyz/Sigmoid_1Ê
!mfdtyewult/while/aflyrndiyz/mul_2Mul)mfdtyewult/while/aflyrndiyz/Sigmoid_1:y:0mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_2ª
 mfdtyewult/while/aflyrndiyz/TanhTanh*mfdtyewult/while/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 mfdtyewult/while/aflyrndiyz/TanhÎ
!mfdtyewult/while/aflyrndiyz/mul_3Mul'mfdtyewult/while/aflyrndiyz/Sigmoid:y:0$mfdtyewult/while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_3Ï
!mfdtyewult/while/aflyrndiyz/add_3AddV2%mfdtyewult/while/aflyrndiyz/mul_2:z:0%mfdtyewult/while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_3Ð
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2ReadVariableOp7mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2Ü
!mfdtyewult/while/aflyrndiyz/mul_4Mul4mfdtyewult/while/aflyrndiyz/ReadVariableOp_2:value:0%mfdtyewult/while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_4Ô
!mfdtyewult/while/aflyrndiyz/add_4AddV2*mfdtyewult/while/aflyrndiyz/split:output:3%mfdtyewult/while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_4²
%mfdtyewult/while/aflyrndiyz/Sigmoid_2Sigmoid%mfdtyewult/while/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%mfdtyewult/while/aflyrndiyz/Sigmoid_2©
"mfdtyewult/while/aflyrndiyz/Tanh_1Tanh%mfdtyewult/while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"mfdtyewult/while/aflyrndiyz/Tanh_1Ò
!mfdtyewult/while/aflyrndiyz/mul_5Mul)mfdtyewult/while/aflyrndiyz/Sigmoid_2:y:0&mfdtyewult/while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_5
5mfdtyewult/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmfdtyewult_while_placeholder_1mfdtyewult_while_placeholder%mfdtyewult/while/aflyrndiyz/mul_5:z:0*
_output_shapes
: *
element_dtype027
5mfdtyewult/while/TensorArrayV2Write/TensorListSetItemr
mfdtyewult/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
mfdtyewult/while/add/y
mfdtyewult/while/addAddV2mfdtyewult_while_placeholdermfdtyewult/while/add/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/while/addv
mfdtyewult/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
mfdtyewult/while/add_1/y­
mfdtyewult/while/add_1AddV2.mfdtyewult_while_mfdtyewult_while_loop_counter!mfdtyewult/while/add_1/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/while/add_1©
mfdtyewult/while/IdentityIdentitymfdtyewult/while/add_1:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/IdentityÇ
mfdtyewult/while/Identity_1Identity4mfdtyewult_while_mfdtyewult_while_maximum_iterations3^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/Identity_1«
mfdtyewult/while/Identity_2Identitymfdtyewult/while/add:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/Identity_2Ø
mfdtyewult/while/Identity_3IdentityEmfdtyewult/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/Identity_3É
mfdtyewult/while/Identity_4Identity%mfdtyewult/while/aflyrndiyz/mul_5:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/while/Identity_4É
mfdtyewult/while/Identity_5Identity%mfdtyewult/while/aflyrndiyz/add_3:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/while/Identity_5"|
;mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource=mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0"~
<mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource>mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0"z
:mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource<mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0"p
5mfdtyewult_while_aflyrndiyz_readvariableop_1_resource7mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0"p
5mfdtyewult_while_aflyrndiyz_readvariableop_2_resource7mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0"l
3mfdtyewult_while_aflyrndiyz_readvariableop_resource5mfdtyewult_while_aflyrndiyz_readvariableop_resource_0"?
mfdtyewult_while_identity"mfdtyewult/while/Identity:output:0"C
mfdtyewult_while_identity_1$mfdtyewult/while/Identity_1:output:0"C
mfdtyewult_while_identity_2$mfdtyewult/while/Identity_2:output:0"C
mfdtyewult_while_identity_3$mfdtyewult/while/Identity_3:output:0"C
mfdtyewult_while_identity_4$mfdtyewult/while/Identity_4:output:0"C
mfdtyewult_while_identity_5$mfdtyewult/while/Identity_5:output:0"\
+mfdtyewult_while_mfdtyewult_strided_slice_1-mfdtyewult_while_mfdtyewult_strided_slice_1_0"Ô
gmfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensorimfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2f
1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp2j
3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp2X
*mfdtyewult/while/aflyrndiyz/ReadVariableOp*mfdtyewult/while/aflyrndiyz/ReadVariableOp2\
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1,mfdtyewult/while/aflyrndiyz/ReadVariableOp_12\
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_1822987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ammytzqwsz_matmul_readvariableop_resource_0:	F
3while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 A
2while_ammytzqwsz_biasadd_readvariableop_resource_0:	8
*while_ammytzqwsz_readvariableop_resource_0: :
,while_ammytzqwsz_readvariableop_1_resource_0: :
,while_ammytzqwsz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ammytzqwsz_matmul_readvariableop_resource:	D
1while_ammytzqwsz_matmul_1_readvariableop_resource:	 ?
0while_ammytzqwsz_biasadd_readvariableop_resource:	6
(while_ammytzqwsz_readvariableop_resource: 8
*while_ammytzqwsz_readvariableop_1_resource: 8
*while_ammytzqwsz_readvariableop_2_resource: ¢'while/ammytzqwsz/BiasAdd/ReadVariableOp¢&while/ammytzqwsz/MatMul/ReadVariableOp¢(while/ammytzqwsz/MatMul_1/ReadVariableOp¢while/ammytzqwsz/ReadVariableOp¢!while/ammytzqwsz/ReadVariableOp_1¢!while/ammytzqwsz/ReadVariableOp_2Ã
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
&while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp1while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ammytzqwsz/MatMul/ReadVariableOpÑ
while/ammytzqwsz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMulÉ
(while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp3while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ammytzqwsz/MatMul_1/ReadVariableOpº
while/ammytzqwsz/MatMul_1MatMulwhile_placeholder_20while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMul_1°
while/ammytzqwsz/addAddV2!while/ammytzqwsz/MatMul:product:0#while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/addÂ
'while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp2while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ammytzqwsz/BiasAdd/ReadVariableOp½
while/ammytzqwsz/BiasAddBiasAddwhile/ammytzqwsz/add:z:0/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/BiasAdd
 while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ammytzqwsz/split/split_dim
while/ammytzqwsz/splitSplit)while/ammytzqwsz/split/split_dim:output:0!while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ammytzqwsz/split©
while/ammytzqwsz/ReadVariableOpReadVariableOp*while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ammytzqwsz/ReadVariableOp£
while/ammytzqwsz/mulMul'while/ammytzqwsz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul¦
while/ammytzqwsz/add_1AddV2while/ammytzqwsz/split:output:0while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_1
while/ammytzqwsz/SigmoidSigmoidwhile/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid¯
!while/ammytzqwsz/ReadVariableOp_1ReadVariableOp,while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_1©
while/ammytzqwsz/mul_1Mul)while/ammytzqwsz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_1¨
while/ammytzqwsz/add_2AddV2while/ammytzqwsz/split:output:1while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_2
while/ammytzqwsz/Sigmoid_1Sigmoidwhile/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_1
while/ammytzqwsz/mul_2Mulwhile/ammytzqwsz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_2
while/ammytzqwsz/TanhTanhwhile/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh¢
while/ammytzqwsz/mul_3Mulwhile/ammytzqwsz/Sigmoid:y:0while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_3£
while/ammytzqwsz/add_3AddV2while/ammytzqwsz/mul_2:z:0while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_3¯
!while/ammytzqwsz/ReadVariableOp_2ReadVariableOp,while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_2°
while/ammytzqwsz/mul_4Mul)while/ammytzqwsz/ReadVariableOp_2:value:0while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_4¨
while/ammytzqwsz/add_4AddV2while/ammytzqwsz/split:output:3while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_4
while/ammytzqwsz/Sigmoid_2Sigmoidwhile/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_2
while/ammytzqwsz/Tanh_1Tanhwhile/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh_1¦
while/ammytzqwsz/mul_5Mulwhile/ammytzqwsz/Sigmoid_2:y:0while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ammytzqwsz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ammytzqwsz/mul_5:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ammytzqwsz/add_3:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ammytzqwsz_biasadd_readvariableop_resource2while_ammytzqwsz_biasadd_readvariableop_resource_0"h
1while_ammytzqwsz_matmul_1_readvariableop_resource3while_ammytzqwsz_matmul_1_readvariableop_resource_0"d
/while_ammytzqwsz_matmul_readvariableop_resource1while_ammytzqwsz_matmul_readvariableop_resource_0"Z
*while_ammytzqwsz_readvariableop_1_resource,while_ammytzqwsz_readvariableop_1_resource_0"Z
*while_ammytzqwsz_readvariableop_2_resource,while_ammytzqwsz_readvariableop_2_resource_0"V
(while_ammytzqwsz_readvariableop_resource*while_ammytzqwsz_readvariableop_resource_0")
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
'while/ammytzqwsz/BiasAdd/ReadVariableOp'while/ammytzqwsz/BiasAdd/ReadVariableOp2P
&while/ammytzqwsz/MatMul/ReadVariableOp&while/ammytzqwsz/MatMul/ReadVariableOp2T
(while/ammytzqwsz/MatMul_1/ReadVariableOp(while/ammytzqwsz/MatMul_1/ReadVariableOp2B
while/ammytzqwsz/ReadVariableOpwhile/ammytzqwsz/ReadVariableOp2F
!while/ammytzqwsz/ReadVariableOp_1!while/ammytzqwsz/ReadVariableOp_12F
!while/ammytzqwsz/ReadVariableOp_2!while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1819219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1819219___redundant_placeholder05
1while_while_cond_1819219___redundant_placeholder15
1while_while_cond_1819219___redundant_placeholder25
1while_while_cond_1819219___redundant_placeholder35
1while_while_cond_1819219___redundant_placeholder45
1while_while_cond_1819219___redundant_placeholder55
1while_while_cond_1819219___redundant_placeholder6
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
while_body_1822627
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ammytzqwsz_matmul_readvariableop_resource_0:	F
3while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 A
2while_ammytzqwsz_biasadd_readvariableop_resource_0:	8
*while_ammytzqwsz_readvariableop_resource_0: :
,while_ammytzqwsz_readvariableop_1_resource_0: :
,while_ammytzqwsz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ammytzqwsz_matmul_readvariableop_resource:	D
1while_ammytzqwsz_matmul_1_readvariableop_resource:	 ?
0while_ammytzqwsz_biasadd_readvariableop_resource:	6
(while_ammytzqwsz_readvariableop_resource: 8
*while_ammytzqwsz_readvariableop_1_resource: 8
*while_ammytzqwsz_readvariableop_2_resource: ¢'while/ammytzqwsz/BiasAdd/ReadVariableOp¢&while/ammytzqwsz/MatMul/ReadVariableOp¢(while/ammytzqwsz/MatMul_1/ReadVariableOp¢while/ammytzqwsz/ReadVariableOp¢!while/ammytzqwsz/ReadVariableOp_1¢!while/ammytzqwsz/ReadVariableOp_2Ã
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
&while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp1while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ammytzqwsz/MatMul/ReadVariableOpÑ
while/ammytzqwsz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMulÉ
(while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp3while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ammytzqwsz/MatMul_1/ReadVariableOpº
while/ammytzqwsz/MatMul_1MatMulwhile_placeholder_20while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMul_1°
while/ammytzqwsz/addAddV2!while/ammytzqwsz/MatMul:product:0#while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/addÂ
'while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp2while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ammytzqwsz/BiasAdd/ReadVariableOp½
while/ammytzqwsz/BiasAddBiasAddwhile/ammytzqwsz/add:z:0/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/BiasAdd
 while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ammytzqwsz/split/split_dim
while/ammytzqwsz/splitSplit)while/ammytzqwsz/split/split_dim:output:0!while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ammytzqwsz/split©
while/ammytzqwsz/ReadVariableOpReadVariableOp*while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ammytzqwsz/ReadVariableOp£
while/ammytzqwsz/mulMul'while/ammytzqwsz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul¦
while/ammytzqwsz/add_1AddV2while/ammytzqwsz/split:output:0while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_1
while/ammytzqwsz/SigmoidSigmoidwhile/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid¯
!while/ammytzqwsz/ReadVariableOp_1ReadVariableOp,while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_1©
while/ammytzqwsz/mul_1Mul)while/ammytzqwsz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_1¨
while/ammytzqwsz/add_2AddV2while/ammytzqwsz/split:output:1while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_2
while/ammytzqwsz/Sigmoid_1Sigmoidwhile/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_1
while/ammytzqwsz/mul_2Mulwhile/ammytzqwsz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_2
while/ammytzqwsz/TanhTanhwhile/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh¢
while/ammytzqwsz/mul_3Mulwhile/ammytzqwsz/Sigmoid:y:0while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_3£
while/ammytzqwsz/add_3AddV2while/ammytzqwsz/mul_2:z:0while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_3¯
!while/ammytzqwsz/ReadVariableOp_2ReadVariableOp,while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_2°
while/ammytzqwsz/mul_4Mul)while/ammytzqwsz/ReadVariableOp_2:value:0while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_4¨
while/ammytzqwsz/add_4AddV2while/ammytzqwsz/split:output:3while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_4
while/ammytzqwsz/Sigmoid_2Sigmoidwhile/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_2
while/ammytzqwsz/Tanh_1Tanhwhile/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh_1¦
while/ammytzqwsz/mul_5Mulwhile/ammytzqwsz/Sigmoid_2:y:0while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ammytzqwsz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ammytzqwsz/mul_5:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ammytzqwsz/add_3:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ammytzqwsz_biasadd_readvariableop_resource2while_ammytzqwsz_biasadd_readvariableop_resource_0"h
1while_ammytzqwsz_matmul_1_readvariableop_resource3while_ammytzqwsz_matmul_1_readvariableop_resource_0"d
/while_ammytzqwsz_matmul_readvariableop_resource1while_ammytzqwsz_matmul_readvariableop_resource_0"Z
*while_ammytzqwsz_readvariableop_1_resource,while_ammytzqwsz_readvariableop_1_resource_0"Z
*while_ammytzqwsz_readvariableop_2_resource,while_ammytzqwsz_readvariableop_2_resource_0"V
(while_ammytzqwsz_readvariableop_resource*while_ammytzqwsz_readvariableop_resource_0")
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
'while/ammytzqwsz/BiasAdd/ReadVariableOp'while/ammytzqwsz/BiasAdd/ReadVariableOp2P
&while/ammytzqwsz/MatMul/ReadVariableOp&while/ammytzqwsz/MatMul/ReadVariableOp2T
(while/ammytzqwsz/MatMul_1/ReadVariableOp(while/ammytzqwsz/MatMul_1/ReadVariableOp2B
while/ammytzqwsz/ReadVariableOpwhile/ammytzqwsz/ReadVariableOp2F
!while/ammytzqwsz/ReadVariableOp_1!while/ammytzqwsz/ReadVariableOp_12F
!while/ammytzqwsz/ReadVariableOp_2!while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_1821516

liksrhmmux(
fethhjgisa_1821478: 
fethhjgisa_1821480:%
oohztcvkwo_1821484:	%
oohztcvkwo_1821486:	 !
oohztcvkwo_1821488:	 
oohztcvkwo_1821490:  
oohztcvkwo_1821492:  
oohztcvkwo_1821494: %
mfdtyewult_1821497:	 %
mfdtyewult_1821499:	 !
mfdtyewult_1821501:	 
mfdtyewult_1821503:  
mfdtyewult_1821505:  
mfdtyewult_1821507: $
boyhyiogqf_1821510:  
boyhyiogqf_1821512:
identity¢"boyhyiogqf/StatefulPartitionedCall¢"fethhjgisa/StatefulPartitionedCall¢"mfdtyewult/StatefulPartitionedCall¢"oohztcvkwo/StatefulPartitionedCall°
"fethhjgisa/StatefulPartitionedCallStatefulPartitionedCall
liksrhmmuxfethhjgisa_1821478fethhjgisa_1821480*
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
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_18204102$
"fethhjgisa/StatefulPartitionedCall
ohvvfduigw/PartitionedCallPartitionedCall+fethhjgisa/StatefulPartitionedCall:output:0*
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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_18204292
ohvvfduigw/PartitionedCall
"oohztcvkwo/StatefulPartitionedCallStatefulPartitionedCall#ohvvfduigw/PartitionedCall:output:0oohztcvkwo_1821484oohztcvkwo_1821486oohztcvkwo_1821488oohztcvkwo_1821490oohztcvkwo_1821492oohztcvkwo_1821494*
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18206102$
"oohztcvkwo/StatefulPartitionedCall¡
"mfdtyewult/StatefulPartitionedCallStatefulPartitionedCall+oohztcvkwo/StatefulPartitionedCall:output:0mfdtyewult_1821497mfdtyewult_1821499mfdtyewult_1821501mfdtyewult_1821503mfdtyewult_1821505mfdtyewult_1821507*
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18208032$
"mfdtyewult/StatefulPartitionedCallÉ
"boyhyiogqf/StatefulPartitionedCallStatefulPartitionedCall+mfdtyewult/StatefulPartitionedCall:output:0boyhyiogqf_1821510boyhyiogqf_1821512*
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
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_18208272$
"boyhyiogqf/StatefulPartitionedCall
IdentityIdentity+boyhyiogqf/StatefulPartitionedCall:output:0#^boyhyiogqf/StatefulPartitionedCall#^fethhjgisa/StatefulPartitionedCall#^mfdtyewult/StatefulPartitionedCall#^oohztcvkwo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"boyhyiogqf/StatefulPartitionedCall"boyhyiogqf/StatefulPartitionedCall2H
"fethhjgisa/StatefulPartitionedCall"fethhjgisa/StatefulPartitionedCall2H
"mfdtyewult/StatefulPartitionedCall"mfdtyewult/StatefulPartitionedCall2H
"oohztcvkwo/StatefulPartitionedCall"oohztcvkwo/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
liksrhmmux
Û

,__inference_oohztcvkwo_layer_call_fn_1823336

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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18212922
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
©0
¼
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_1820410

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
±'
³
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1818937

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
¡h

G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1821292

inputs<
)ammytzqwsz_matmul_readvariableop_resource:	>
+ammytzqwsz_matmul_1_readvariableop_resource:	 9
*ammytzqwsz_biasadd_readvariableop_resource:	0
"ammytzqwsz_readvariableop_resource: 2
$ammytzqwsz_readvariableop_1_resource: 2
$ammytzqwsz_readvariableop_2_resource: 
identity¢!ammytzqwsz/BiasAdd/ReadVariableOp¢ ammytzqwsz/MatMul/ReadVariableOp¢"ammytzqwsz/MatMul_1/ReadVariableOp¢ammytzqwsz/ReadVariableOp¢ammytzqwsz/ReadVariableOp_1¢ammytzqwsz/ReadVariableOp_2¢whileD
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
 ammytzqwsz/MatMul/ReadVariableOpReadVariableOp)ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ammytzqwsz/MatMul/ReadVariableOp§
ammytzqwsz/MatMulMatMulstrided_slice_2:output:0(ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMulµ
"ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp+ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ammytzqwsz/MatMul_1/ReadVariableOp£
ammytzqwsz/MatMul_1MatMulzeros:output:0*ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMul_1
ammytzqwsz/addAddV2ammytzqwsz/MatMul:product:0ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/add®
!ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp*ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ammytzqwsz/BiasAdd/ReadVariableOp¥
ammytzqwsz/BiasAddBiasAddammytzqwsz/add:z:0)ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/BiasAddz
ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ammytzqwsz/split/split_dimë
ammytzqwsz/splitSplit#ammytzqwsz/split/split_dim:output:0ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ammytzqwsz/split
ammytzqwsz/ReadVariableOpReadVariableOp"ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp
ammytzqwsz/mulMul!ammytzqwsz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul
ammytzqwsz/add_1AddV2ammytzqwsz/split:output:0ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_1{
ammytzqwsz/SigmoidSigmoidammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid
ammytzqwsz/ReadVariableOp_1ReadVariableOp$ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_1
ammytzqwsz/mul_1Mul#ammytzqwsz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_1
ammytzqwsz/add_2AddV2ammytzqwsz/split:output:1ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_2
ammytzqwsz/Sigmoid_1Sigmoidammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_1
ammytzqwsz/mul_2Mulammytzqwsz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_2w
ammytzqwsz/TanhTanhammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh
ammytzqwsz/mul_3Mulammytzqwsz/Sigmoid:y:0ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_3
ammytzqwsz/add_3AddV2ammytzqwsz/mul_2:z:0ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_3
ammytzqwsz/ReadVariableOp_2ReadVariableOp$ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_2
ammytzqwsz/mul_4Mul#ammytzqwsz/ReadVariableOp_2:value:0ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_4
ammytzqwsz/add_4AddV2ammytzqwsz/split:output:3ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_4
ammytzqwsz/Sigmoid_2Sigmoidammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_2v
ammytzqwsz/Tanh_1Tanhammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh_1
ammytzqwsz/mul_5Mulammytzqwsz/Sigmoid_2:y:0ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ammytzqwsz_matmul_readvariableop_resource+ammytzqwsz_matmul_1_readvariableop_resource*ammytzqwsz_biasadd_readvariableop_resource"ammytzqwsz_readvariableop_resource$ammytzqwsz_readvariableop_1_resource$ammytzqwsz_readvariableop_2_resource*
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
while_body_1821191*
condR
while_cond_1821190*Q
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
IdentityIdentitytranspose_1:y:0"^ammytzqwsz/BiasAdd/ReadVariableOp!^ammytzqwsz/MatMul/ReadVariableOp#^ammytzqwsz/MatMul_1/ReadVariableOp^ammytzqwsz/ReadVariableOp^ammytzqwsz/ReadVariableOp_1^ammytzqwsz/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ammytzqwsz/BiasAdd/ReadVariableOp!ammytzqwsz/BiasAdd/ReadVariableOp2D
 ammytzqwsz/MatMul/ReadVariableOp ammytzqwsz/MatMul/ReadVariableOp2H
"ammytzqwsz/MatMul_1/ReadVariableOp"ammytzqwsz/MatMul_1/ReadVariableOp26
ammytzqwsz/ReadVariableOpammytzqwsz/ReadVariableOp2:
ammytzqwsz/ReadVariableOp_1ammytzqwsz/ReadVariableOp_12:
ammytzqwsz/ReadVariableOp_2ammytzqwsz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àY

while_body_1820702
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aflyrndiyz_matmul_readvariableop_resource_0:	 F
3while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 A
2while_aflyrndiyz_biasadd_readvariableop_resource_0:	8
*while_aflyrndiyz_readvariableop_resource_0: :
,while_aflyrndiyz_readvariableop_1_resource_0: :
,while_aflyrndiyz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aflyrndiyz_matmul_readvariableop_resource:	 D
1while_aflyrndiyz_matmul_1_readvariableop_resource:	 ?
0while_aflyrndiyz_biasadd_readvariableop_resource:	6
(while_aflyrndiyz_readvariableop_resource: 8
*while_aflyrndiyz_readvariableop_1_resource: 8
*while_aflyrndiyz_readvariableop_2_resource: ¢'while/aflyrndiyz/BiasAdd/ReadVariableOp¢&while/aflyrndiyz/MatMul/ReadVariableOp¢(while/aflyrndiyz/MatMul_1/ReadVariableOp¢while/aflyrndiyz/ReadVariableOp¢!while/aflyrndiyz/ReadVariableOp_1¢!while/aflyrndiyz/ReadVariableOp_2Ã
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
&while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp1while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aflyrndiyz/MatMul/ReadVariableOpÑ
while/aflyrndiyz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMulÉ
(while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp3while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aflyrndiyz/MatMul_1/ReadVariableOpº
while/aflyrndiyz/MatMul_1MatMulwhile_placeholder_20while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMul_1°
while/aflyrndiyz/addAddV2!while/aflyrndiyz/MatMul:product:0#while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/addÂ
'while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp2while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aflyrndiyz/BiasAdd/ReadVariableOp½
while/aflyrndiyz/BiasAddBiasAddwhile/aflyrndiyz/add:z:0/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/BiasAdd
 while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aflyrndiyz/split/split_dim
while/aflyrndiyz/splitSplit)while/aflyrndiyz/split/split_dim:output:0!while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aflyrndiyz/split©
while/aflyrndiyz/ReadVariableOpReadVariableOp*while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aflyrndiyz/ReadVariableOp£
while/aflyrndiyz/mulMul'while/aflyrndiyz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul¦
while/aflyrndiyz/add_1AddV2while/aflyrndiyz/split:output:0while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_1
while/aflyrndiyz/SigmoidSigmoidwhile/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid¯
!while/aflyrndiyz/ReadVariableOp_1ReadVariableOp,while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_1©
while/aflyrndiyz/mul_1Mul)while/aflyrndiyz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_1¨
while/aflyrndiyz/add_2AddV2while/aflyrndiyz/split:output:1while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_2
while/aflyrndiyz/Sigmoid_1Sigmoidwhile/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_1
while/aflyrndiyz/mul_2Mulwhile/aflyrndiyz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_2
while/aflyrndiyz/TanhTanhwhile/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh¢
while/aflyrndiyz/mul_3Mulwhile/aflyrndiyz/Sigmoid:y:0while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_3£
while/aflyrndiyz/add_3AddV2while/aflyrndiyz/mul_2:z:0while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_3¯
!while/aflyrndiyz/ReadVariableOp_2ReadVariableOp,while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_2°
while/aflyrndiyz/mul_4Mul)while/aflyrndiyz/ReadVariableOp_2:value:0while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_4¨
while/aflyrndiyz/add_4AddV2while/aflyrndiyz/split:output:3while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_4
while/aflyrndiyz/Sigmoid_2Sigmoidwhile/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_2
while/aflyrndiyz/Tanh_1Tanhwhile/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh_1¦
while/aflyrndiyz/mul_5Mulwhile/aflyrndiyz/Sigmoid_2:y:0while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aflyrndiyz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aflyrndiyz/mul_5:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aflyrndiyz/add_3:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aflyrndiyz_biasadd_readvariableop_resource2while_aflyrndiyz_biasadd_readvariableop_resource_0"h
1while_aflyrndiyz_matmul_1_readvariableop_resource3while_aflyrndiyz_matmul_1_readvariableop_resource_0"d
/while_aflyrndiyz_matmul_readvariableop_resource1while_aflyrndiyz_matmul_readvariableop_resource_0"Z
*while_aflyrndiyz_readvariableop_1_resource,while_aflyrndiyz_readvariableop_1_resource_0"Z
*while_aflyrndiyz_readvariableop_2_resource,while_aflyrndiyz_readvariableop_2_resource_0"V
(while_aflyrndiyz_readvariableop_resource*while_aflyrndiyz_readvariableop_resource_0")
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
'while/aflyrndiyz/BiasAdd/ReadVariableOp'while/aflyrndiyz/BiasAdd/ReadVariableOp2P
&while/aflyrndiyz/MatMul/ReadVariableOp&while/aflyrndiyz/MatMul/ReadVariableOp2T
(while/aflyrndiyz/MatMul_1/ReadVariableOp(while/aflyrndiyz/MatMul_1/ReadVariableOp2B
while/aflyrndiyz/ReadVariableOpwhile/aflyrndiyz/ReadVariableOp2F
!while/aflyrndiyz/ReadVariableOp_1!while/aflyrndiyz/ReadVariableOp_12F
!while/aflyrndiyz/ReadVariableOp_2!while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1822626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1822626___redundant_placeholder05
1while_while_cond_1822626___redundant_placeholder15
1while_while_cond_1822626___redundant_placeholder25
1while_while_cond_1822626___redundant_placeholder35
1while_while_cond_1822626___redundant_placeholder45
1while_while_cond_1822626___redundant_placeholder55
1while_while_cond_1822626___redundant_placeholder6
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
,__inference_ammytzqwsz_layer_call_fn_1824254

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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_18189372
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
while_body_1823167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ammytzqwsz_matmul_readvariableop_resource_0:	F
3while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 A
2while_ammytzqwsz_biasadd_readvariableop_resource_0:	8
*while_ammytzqwsz_readvariableop_resource_0: :
,while_ammytzqwsz_readvariableop_1_resource_0: :
,while_ammytzqwsz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ammytzqwsz_matmul_readvariableop_resource:	D
1while_ammytzqwsz_matmul_1_readvariableop_resource:	 ?
0while_ammytzqwsz_biasadd_readvariableop_resource:	6
(while_ammytzqwsz_readvariableop_resource: 8
*while_ammytzqwsz_readvariableop_1_resource: 8
*while_ammytzqwsz_readvariableop_2_resource: ¢'while/ammytzqwsz/BiasAdd/ReadVariableOp¢&while/ammytzqwsz/MatMul/ReadVariableOp¢(while/ammytzqwsz/MatMul_1/ReadVariableOp¢while/ammytzqwsz/ReadVariableOp¢!while/ammytzqwsz/ReadVariableOp_1¢!while/ammytzqwsz/ReadVariableOp_2Ã
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
&while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp1while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ammytzqwsz/MatMul/ReadVariableOpÑ
while/ammytzqwsz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMulÉ
(while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp3while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ammytzqwsz/MatMul_1/ReadVariableOpº
while/ammytzqwsz/MatMul_1MatMulwhile_placeholder_20while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMul_1°
while/ammytzqwsz/addAddV2!while/ammytzqwsz/MatMul:product:0#while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/addÂ
'while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp2while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ammytzqwsz/BiasAdd/ReadVariableOp½
while/ammytzqwsz/BiasAddBiasAddwhile/ammytzqwsz/add:z:0/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/BiasAdd
 while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ammytzqwsz/split/split_dim
while/ammytzqwsz/splitSplit)while/ammytzqwsz/split/split_dim:output:0!while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ammytzqwsz/split©
while/ammytzqwsz/ReadVariableOpReadVariableOp*while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ammytzqwsz/ReadVariableOp£
while/ammytzqwsz/mulMul'while/ammytzqwsz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul¦
while/ammytzqwsz/add_1AddV2while/ammytzqwsz/split:output:0while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_1
while/ammytzqwsz/SigmoidSigmoidwhile/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid¯
!while/ammytzqwsz/ReadVariableOp_1ReadVariableOp,while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_1©
while/ammytzqwsz/mul_1Mul)while/ammytzqwsz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_1¨
while/ammytzqwsz/add_2AddV2while/ammytzqwsz/split:output:1while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_2
while/ammytzqwsz/Sigmoid_1Sigmoidwhile/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_1
while/ammytzqwsz/mul_2Mulwhile/ammytzqwsz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_2
while/ammytzqwsz/TanhTanhwhile/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh¢
while/ammytzqwsz/mul_3Mulwhile/ammytzqwsz/Sigmoid:y:0while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_3£
while/ammytzqwsz/add_3AddV2while/ammytzqwsz/mul_2:z:0while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_3¯
!while/ammytzqwsz/ReadVariableOp_2ReadVariableOp,while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_2°
while/ammytzqwsz/mul_4Mul)while/ammytzqwsz/ReadVariableOp_2:value:0while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_4¨
while/ammytzqwsz/add_4AddV2while/ammytzqwsz/split:output:3while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_4
while/ammytzqwsz/Sigmoid_2Sigmoidwhile/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_2
while/ammytzqwsz/Tanh_1Tanhwhile/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh_1¦
while/ammytzqwsz/mul_5Mulwhile/ammytzqwsz/Sigmoid_2:y:0while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ammytzqwsz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ammytzqwsz/mul_5:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ammytzqwsz/add_3:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ammytzqwsz_biasadd_readvariableop_resource2while_ammytzqwsz_biasadd_readvariableop_resource_0"h
1while_ammytzqwsz_matmul_1_readvariableop_resource3while_ammytzqwsz_matmul_1_readvariableop_resource_0"d
/while_ammytzqwsz_matmul_readvariableop_resource1while_ammytzqwsz_matmul_readvariableop_resource_0"Z
*while_ammytzqwsz_readvariableop_1_resource,while_ammytzqwsz_readvariableop_1_resource_0"Z
*while_ammytzqwsz_readvariableop_2_resource,while_ammytzqwsz_readvariableop_2_resource_0"V
(while_ammytzqwsz_readvariableop_resource*while_ammytzqwsz_readvariableop_resource_0")
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
'while/ammytzqwsz/BiasAdd/ReadVariableOp'while/ammytzqwsz/BiasAdd/ReadVariableOp2P
&while/ammytzqwsz/MatMul/ReadVariableOp&while/ammytzqwsz/MatMul/ReadVariableOp2T
(while/ammytzqwsz/MatMul_1/ReadVariableOp(while/ammytzqwsz/MatMul_1/ReadVariableOp2B
while/ammytzqwsz/ReadVariableOpwhile/ammytzqwsz/ReadVariableOp2F
!while/ammytzqwsz/ReadVariableOp_1!while/ammytzqwsz/ReadVariableOp_12F
!while/ammytzqwsz/ReadVariableOp_2!while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_1822543

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
,__inference_aflyrndiyz_layer_call_fn_1824411

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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_18198822
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
Ü

(sequential_mfdtyewult_while_body_1818743H
Dsequential_mfdtyewult_while_sequential_mfdtyewult_while_loop_counterN
Jsequential_mfdtyewult_while_sequential_mfdtyewult_while_maximum_iterations+
'sequential_mfdtyewult_while_placeholder-
)sequential_mfdtyewult_while_placeholder_1-
)sequential_mfdtyewult_while_placeholder_2-
)sequential_mfdtyewult_while_placeholder_3G
Csequential_mfdtyewult_while_sequential_mfdtyewult_strided_slice_1_0
sequential_mfdtyewult_while_tensorarrayv2read_tensorlistgetitem_sequential_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0:	 \
Isequential_mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 W
Hsequential_mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0:	N
@sequential_mfdtyewult_while_aflyrndiyz_readvariableop_resource_0: P
Bsequential_mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0: P
Bsequential_mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0: (
$sequential_mfdtyewult_while_identity*
&sequential_mfdtyewult_while_identity_1*
&sequential_mfdtyewult_while_identity_2*
&sequential_mfdtyewult_while_identity_3*
&sequential_mfdtyewult_while_identity_4*
&sequential_mfdtyewult_while_identity_5E
Asequential_mfdtyewult_while_sequential_mfdtyewult_strided_slice_1
}sequential_mfdtyewult_while_tensorarrayv2read_tensorlistgetitem_sequential_mfdtyewult_tensorarrayunstack_tensorlistfromtensorX
Esequential_mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource:	 Z
Gsequential_mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource:	 U
Fsequential_mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource:	L
>sequential_mfdtyewult_while_aflyrndiyz_readvariableop_resource: N
@sequential_mfdtyewult_while_aflyrndiyz_readvariableop_1_resource: N
@sequential_mfdtyewult_while_aflyrndiyz_readvariableop_2_resource: ¢=sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp¢<sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp¢>sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp¢5sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp¢7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_1¢7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2ï
Msequential/mfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/mfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/mfdtyewult/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_mfdtyewult_while_tensorarrayv2read_tensorlistgetitem_sequential_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0'sequential_mfdtyewult_while_placeholderVsequential/mfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/mfdtyewult/while/TensorArrayV2Read/TensorListGetItem
<sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOpGsequential_mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp©
-sequential/mfdtyewult/while/aflyrndiyz/MatMulMatMulFsequential/mfdtyewult/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/mfdtyewult/while/aflyrndiyz/MatMul
>sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOpIsequential_mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp
/sequential/mfdtyewult/while/aflyrndiyz/MatMul_1MatMul)sequential_mfdtyewult_while_placeholder_2Fsequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/mfdtyewult/while/aflyrndiyz/MatMul_1
*sequential/mfdtyewult/while/aflyrndiyz/addAddV27sequential/mfdtyewult/while/aflyrndiyz/MatMul:product:09sequential/mfdtyewult/while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/mfdtyewult/while/aflyrndiyz/add
=sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOpHsequential_mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp
.sequential/mfdtyewult/while/aflyrndiyz/BiasAddBiasAdd.sequential/mfdtyewult/while/aflyrndiyz/add:z:0Esequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/mfdtyewult/while/aflyrndiyz/BiasAdd²
6sequential/mfdtyewult/while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/mfdtyewult/while/aflyrndiyz/split/split_dimÛ
,sequential/mfdtyewult/while/aflyrndiyz/splitSplit?sequential/mfdtyewult/while/aflyrndiyz/split/split_dim:output:07sequential/mfdtyewult/while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/mfdtyewult/while/aflyrndiyz/splitë
5sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOpReadVariableOp@sequential_mfdtyewult_while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOpû
*sequential/mfdtyewult/while/aflyrndiyz/mulMul=sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp:value:0)sequential_mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/mfdtyewult/while/aflyrndiyz/mulþ
,sequential/mfdtyewult/while/aflyrndiyz/add_1AddV25sequential/mfdtyewult/while/aflyrndiyz/split:output:0.sequential/mfdtyewult/while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/add_1Ï
.sequential/mfdtyewult/while/aflyrndiyz/SigmoidSigmoid0sequential/mfdtyewult/while/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/mfdtyewult/while/aflyrndiyz/Sigmoidñ
7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_1ReadVariableOpBsequential_mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_1
,sequential/mfdtyewult/while/aflyrndiyz/mul_1Mul?sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_1:value:0)sequential_mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/mul_1
,sequential/mfdtyewult/while/aflyrndiyz/add_2AddV25sequential/mfdtyewult/while/aflyrndiyz/split:output:10sequential/mfdtyewult/while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/add_2Ó
0sequential/mfdtyewult/while/aflyrndiyz/Sigmoid_1Sigmoid0sequential/mfdtyewult/while/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/mfdtyewult/while/aflyrndiyz/Sigmoid_1ö
,sequential/mfdtyewult/while/aflyrndiyz/mul_2Mul4sequential/mfdtyewult/while/aflyrndiyz/Sigmoid_1:y:0)sequential_mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/mul_2Ë
+sequential/mfdtyewult/while/aflyrndiyz/TanhTanh5sequential/mfdtyewult/while/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/mfdtyewult/while/aflyrndiyz/Tanhú
,sequential/mfdtyewult/while/aflyrndiyz/mul_3Mul2sequential/mfdtyewult/while/aflyrndiyz/Sigmoid:y:0/sequential/mfdtyewult/while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/mul_3û
,sequential/mfdtyewult/while/aflyrndiyz/add_3AddV20sequential/mfdtyewult/while/aflyrndiyz/mul_2:z:00sequential/mfdtyewult/while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/add_3ñ
7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2ReadVariableOpBsequential_mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2
,sequential/mfdtyewult/while/aflyrndiyz/mul_4Mul?sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2:value:00sequential/mfdtyewult/while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/mul_4
,sequential/mfdtyewult/while/aflyrndiyz/add_4AddV25sequential/mfdtyewult/while/aflyrndiyz/split:output:30sequential/mfdtyewult/while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/add_4Ó
0sequential/mfdtyewult/while/aflyrndiyz/Sigmoid_2Sigmoid0sequential/mfdtyewult/while/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/mfdtyewult/while/aflyrndiyz/Sigmoid_2Ê
-sequential/mfdtyewult/while/aflyrndiyz/Tanh_1Tanh0sequential/mfdtyewult/while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/mfdtyewult/while/aflyrndiyz/Tanh_1þ
,sequential/mfdtyewult/while/aflyrndiyz/mul_5Mul4sequential/mfdtyewult/while/aflyrndiyz/Sigmoid_2:y:01sequential/mfdtyewult/while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/mfdtyewult/while/aflyrndiyz/mul_5Ì
@sequential/mfdtyewult/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_mfdtyewult_while_placeholder_1'sequential_mfdtyewult_while_placeholder0sequential/mfdtyewult/while/aflyrndiyz/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/mfdtyewult/while/TensorArrayV2Write/TensorListSetItem
!sequential/mfdtyewult/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/mfdtyewult/while/add/yÁ
sequential/mfdtyewult/while/addAddV2'sequential_mfdtyewult_while_placeholder*sequential/mfdtyewult/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/mfdtyewult/while/add
#sequential/mfdtyewult/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/mfdtyewult/while/add_1/yä
!sequential/mfdtyewult/while/add_1AddV2Dsequential_mfdtyewult_while_sequential_mfdtyewult_while_loop_counter,sequential/mfdtyewult/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/mfdtyewult/while/add_1
$sequential/mfdtyewult/while/IdentityIdentity%sequential/mfdtyewult/while/add_1:z:0>^sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp=^sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp?^sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp6^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp8^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_18^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/mfdtyewult/while/Identityµ
&sequential/mfdtyewult/while/Identity_1IdentityJsequential_mfdtyewult_while_sequential_mfdtyewult_while_maximum_iterations>^sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp=^sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp?^sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp6^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp8^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_18^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/mfdtyewult/while/Identity_1
&sequential/mfdtyewult/while/Identity_2Identity#sequential/mfdtyewult/while/add:z:0>^sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp=^sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp?^sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp6^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp8^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_18^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/mfdtyewult/while/Identity_2»
&sequential/mfdtyewult/while/Identity_3IdentityPsequential/mfdtyewult/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp=^sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp?^sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp6^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp8^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_18^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/mfdtyewult/while/Identity_3¬
&sequential/mfdtyewult/while/Identity_4Identity0sequential/mfdtyewult/while/aflyrndiyz/mul_5:z:0>^sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp=^sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp?^sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp6^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp8^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_18^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/while/Identity_4¬
&sequential/mfdtyewult/while/Identity_5Identity0sequential/mfdtyewult/while/aflyrndiyz/add_3:z:0>^sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp=^sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp?^sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp6^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp8^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_18^sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/while/Identity_5"
Fsequential_mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resourceHsequential_mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0"
Gsequential_mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resourceIsequential_mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0"
Esequential_mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resourceGsequential_mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0"
@sequential_mfdtyewult_while_aflyrndiyz_readvariableop_1_resourceBsequential_mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0"
@sequential_mfdtyewult_while_aflyrndiyz_readvariableop_2_resourceBsequential_mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0"
>sequential_mfdtyewult_while_aflyrndiyz_readvariableop_resource@sequential_mfdtyewult_while_aflyrndiyz_readvariableop_resource_0"U
$sequential_mfdtyewult_while_identity-sequential/mfdtyewult/while/Identity:output:0"Y
&sequential_mfdtyewult_while_identity_1/sequential/mfdtyewult/while/Identity_1:output:0"Y
&sequential_mfdtyewult_while_identity_2/sequential/mfdtyewult/while/Identity_2:output:0"Y
&sequential_mfdtyewult_while_identity_3/sequential/mfdtyewult/while/Identity_3:output:0"Y
&sequential_mfdtyewult_while_identity_4/sequential/mfdtyewult/while/Identity_4:output:0"Y
&sequential_mfdtyewult_while_identity_5/sequential/mfdtyewult/while/Identity_5:output:0"
Asequential_mfdtyewult_while_sequential_mfdtyewult_strided_slice_1Csequential_mfdtyewult_while_sequential_mfdtyewult_strided_slice_1_0"
}sequential_mfdtyewult_while_tensorarrayv2read_tensorlistgetitem_sequential_mfdtyewult_tensorarrayunstack_tensorlistfromtensorsequential_mfdtyewult_while_tensorarrayv2read_tensorlistgetitem_sequential_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp=sequential/mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2|
<sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp<sequential/mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp2
>sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp>sequential/mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp2n
5sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp5sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp2r
7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_17sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_12r
7sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_27sequential/mfdtyewult/while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1823268

inputs<
)ammytzqwsz_matmul_readvariableop_resource:	>
+ammytzqwsz_matmul_1_readvariableop_resource:	 9
*ammytzqwsz_biasadd_readvariableop_resource:	0
"ammytzqwsz_readvariableop_resource: 2
$ammytzqwsz_readvariableop_1_resource: 2
$ammytzqwsz_readvariableop_2_resource: 
identity¢!ammytzqwsz/BiasAdd/ReadVariableOp¢ ammytzqwsz/MatMul/ReadVariableOp¢"ammytzqwsz/MatMul_1/ReadVariableOp¢ammytzqwsz/ReadVariableOp¢ammytzqwsz/ReadVariableOp_1¢ammytzqwsz/ReadVariableOp_2¢whileD
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
 ammytzqwsz/MatMul/ReadVariableOpReadVariableOp)ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ammytzqwsz/MatMul/ReadVariableOp§
ammytzqwsz/MatMulMatMulstrided_slice_2:output:0(ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMulµ
"ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp+ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ammytzqwsz/MatMul_1/ReadVariableOp£
ammytzqwsz/MatMul_1MatMulzeros:output:0*ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMul_1
ammytzqwsz/addAddV2ammytzqwsz/MatMul:product:0ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/add®
!ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp*ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ammytzqwsz/BiasAdd/ReadVariableOp¥
ammytzqwsz/BiasAddBiasAddammytzqwsz/add:z:0)ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/BiasAddz
ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ammytzqwsz/split/split_dimë
ammytzqwsz/splitSplit#ammytzqwsz/split/split_dim:output:0ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ammytzqwsz/split
ammytzqwsz/ReadVariableOpReadVariableOp"ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp
ammytzqwsz/mulMul!ammytzqwsz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul
ammytzqwsz/add_1AddV2ammytzqwsz/split:output:0ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_1{
ammytzqwsz/SigmoidSigmoidammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid
ammytzqwsz/ReadVariableOp_1ReadVariableOp$ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_1
ammytzqwsz/mul_1Mul#ammytzqwsz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_1
ammytzqwsz/add_2AddV2ammytzqwsz/split:output:1ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_2
ammytzqwsz/Sigmoid_1Sigmoidammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_1
ammytzqwsz/mul_2Mulammytzqwsz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_2w
ammytzqwsz/TanhTanhammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh
ammytzqwsz/mul_3Mulammytzqwsz/Sigmoid:y:0ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_3
ammytzqwsz/add_3AddV2ammytzqwsz/mul_2:z:0ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_3
ammytzqwsz/ReadVariableOp_2ReadVariableOp$ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_2
ammytzqwsz/mul_4Mul#ammytzqwsz/ReadVariableOp_2:value:0ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_4
ammytzqwsz/add_4AddV2ammytzqwsz/split:output:3ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_4
ammytzqwsz/Sigmoid_2Sigmoidammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_2v
ammytzqwsz/Tanh_1Tanhammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh_1
ammytzqwsz/mul_5Mulammytzqwsz/Sigmoid_2:y:0ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ammytzqwsz_matmul_readvariableop_resource+ammytzqwsz_matmul_1_readvariableop_resource*ammytzqwsz_biasadd_readvariableop_resource"ammytzqwsz_readvariableop_resource$ammytzqwsz_readvariableop_1_resource$ammytzqwsz_readvariableop_2_resource*
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
while_body_1823167*
condR
while_cond_1823166*Q
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
IdentityIdentitytranspose_1:y:0"^ammytzqwsz/BiasAdd/ReadVariableOp!^ammytzqwsz/MatMul/ReadVariableOp#^ammytzqwsz/MatMul_1/ReadVariableOp^ammytzqwsz/ReadVariableOp^ammytzqwsz/ReadVariableOp_1^ammytzqwsz/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ammytzqwsz/BiasAdd/ReadVariableOp!ammytzqwsz/BiasAdd/ReadVariableOp2D
 ammytzqwsz/MatMul/ReadVariableOp ammytzqwsz/MatMul/ReadVariableOp2H
"ammytzqwsz/MatMul_1/ReadVariableOp"ammytzqwsz/MatMul_1/ReadVariableOp26
ammytzqwsz/ReadVariableOpammytzqwsz/ReadVariableOp2:
ammytzqwsz/ReadVariableOp_1ammytzqwsz/ReadVariableOp_12:
ammytzqwsz/ReadVariableOp_2ammytzqwsz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àY

while_body_1822807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ammytzqwsz_matmul_readvariableop_resource_0:	F
3while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 A
2while_ammytzqwsz_biasadd_readvariableop_resource_0:	8
*while_ammytzqwsz_readvariableop_resource_0: :
,while_ammytzqwsz_readvariableop_1_resource_0: :
,while_ammytzqwsz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ammytzqwsz_matmul_readvariableop_resource:	D
1while_ammytzqwsz_matmul_1_readvariableop_resource:	 ?
0while_ammytzqwsz_biasadd_readvariableop_resource:	6
(while_ammytzqwsz_readvariableop_resource: 8
*while_ammytzqwsz_readvariableop_1_resource: 8
*while_ammytzqwsz_readvariableop_2_resource: ¢'while/ammytzqwsz/BiasAdd/ReadVariableOp¢&while/ammytzqwsz/MatMul/ReadVariableOp¢(while/ammytzqwsz/MatMul_1/ReadVariableOp¢while/ammytzqwsz/ReadVariableOp¢!while/ammytzqwsz/ReadVariableOp_1¢!while/ammytzqwsz/ReadVariableOp_2Ã
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
&while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp1while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ammytzqwsz/MatMul/ReadVariableOpÑ
while/ammytzqwsz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMulÉ
(while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp3while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ammytzqwsz/MatMul_1/ReadVariableOpº
while/ammytzqwsz/MatMul_1MatMulwhile_placeholder_20while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMul_1°
while/ammytzqwsz/addAddV2!while/ammytzqwsz/MatMul:product:0#while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/addÂ
'while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp2while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ammytzqwsz/BiasAdd/ReadVariableOp½
while/ammytzqwsz/BiasAddBiasAddwhile/ammytzqwsz/add:z:0/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/BiasAdd
 while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ammytzqwsz/split/split_dim
while/ammytzqwsz/splitSplit)while/ammytzqwsz/split/split_dim:output:0!while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ammytzqwsz/split©
while/ammytzqwsz/ReadVariableOpReadVariableOp*while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ammytzqwsz/ReadVariableOp£
while/ammytzqwsz/mulMul'while/ammytzqwsz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul¦
while/ammytzqwsz/add_1AddV2while/ammytzqwsz/split:output:0while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_1
while/ammytzqwsz/SigmoidSigmoidwhile/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid¯
!while/ammytzqwsz/ReadVariableOp_1ReadVariableOp,while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_1©
while/ammytzqwsz/mul_1Mul)while/ammytzqwsz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_1¨
while/ammytzqwsz/add_2AddV2while/ammytzqwsz/split:output:1while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_2
while/ammytzqwsz/Sigmoid_1Sigmoidwhile/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_1
while/ammytzqwsz/mul_2Mulwhile/ammytzqwsz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_2
while/ammytzqwsz/TanhTanhwhile/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh¢
while/ammytzqwsz/mul_3Mulwhile/ammytzqwsz/Sigmoid:y:0while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_3£
while/ammytzqwsz/add_3AddV2while/ammytzqwsz/mul_2:z:0while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_3¯
!while/ammytzqwsz/ReadVariableOp_2ReadVariableOp,while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_2°
while/ammytzqwsz/mul_4Mul)while/ammytzqwsz/ReadVariableOp_2:value:0while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_4¨
while/ammytzqwsz/add_4AddV2while/ammytzqwsz/split:output:3while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_4
while/ammytzqwsz/Sigmoid_2Sigmoidwhile/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_2
while/ammytzqwsz/Tanh_1Tanhwhile/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh_1¦
while/ammytzqwsz/mul_5Mulwhile/ammytzqwsz/Sigmoid_2:y:0while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ammytzqwsz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ammytzqwsz/mul_5:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ammytzqwsz/add_3:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ammytzqwsz_biasadd_readvariableop_resource2while_ammytzqwsz_biasadd_readvariableop_resource_0"h
1while_ammytzqwsz_matmul_1_readvariableop_resource3while_ammytzqwsz_matmul_1_readvariableop_resource_0"d
/while_ammytzqwsz_matmul_readvariableop_resource1while_ammytzqwsz_matmul_readvariableop_resource_0"Z
*while_ammytzqwsz_readvariableop_1_resource,while_ammytzqwsz_readvariableop_1_resource_0"Z
*while_ammytzqwsz_readvariableop_2_resource,while_ammytzqwsz_readvariableop_2_resource_0"V
(while_ammytzqwsz_readvariableop_resource*while_ammytzqwsz_readvariableop_resource_0")
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
'while/ammytzqwsz/BiasAdd/ReadVariableOp'while/ammytzqwsz/BiasAdd/ReadVariableOp2P
&while/ammytzqwsz/MatMul/ReadVariableOp&while/ammytzqwsz/MatMul/ReadVariableOp2T
(while/ammytzqwsz/MatMul_1/ReadVariableOp(while/ammytzqwsz/MatMul_1/ReadVariableOp2B
while/ammytzqwsz/ReadVariableOpwhile/ammytzqwsz/ReadVariableOp2F
!while/ammytzqwsz/ReadVariableOp_1!while/ammytzqwsz/ReadVariableOp_12F
!while/ammytzqwsz/ReadVariableOp_2!while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_1821191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ammytzqwsz_matmul_readvariableop_resource_0:	F
3while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 A
2while_ammytzqwsz_biasadd_readvariableop_resource_0:	8
*while_ammytzqwsz_readvariableop_resource_0: :
,while_ammytzqwsz_readvariableop_1_resource_0: :
,while_ammytzqwsz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ammytzqwsz_matmul_readvariableop_resource:	D
1while_ammytzqwsz_matmul_1_readvariableop_resource:	 ?
0while_ammytzqwsz_biasadd_readvariableop_resource:	6
(while_ammytzqwsz_readvariableop_resource: 8
*while_ammytzqwsz_readvariableop_1_resource: 8
*while_ammytzqwsz_readvariableop_2_resource: ¢'while/ammytzqwsz/BiasAdd/ReadVariableOp¢&while/ammytzqwsz/MatMul/ReadVariableOp¢(while/ammytzqwsz/MatMul_1/ReadVariableOp¢while/ammytzqwsz/ReadVariableOp¢!while/ammytzqwsz/ReadVariableOp_1¢!while/ammytzqwsz/ReadVariableOp_2Ã
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
&while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp1while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ammytzqwsz/MatMul/ReadVariableOpÑ
while/ammytzqwsz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMulÉ
(while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp3while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ammytzqwsz/MatMul_1/ReadVariableOpº
while/ammytzqwsz/MatMul_1MatMulwhile_placeholder_20while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMul_1°
while/ammytzqwsz/addAddV2!while/ammytzqwsz/MatMul:product:0#while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/addÂ
'while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp2while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ammytzqwsz/BiasAdd/ReadVariableOp½
while/ammytzqwsz/BiasAddBiasAddwhile/ammytzqwsz/add:z:0/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/BiasAdd
 while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ammytzqwsz/split/split_dim
while/ammytzqwsz/splitSplit)while/ammytzqwsz/split/split_dim:output:0!while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ammytzqwsz/split©
while/ammytzqwsz/ReadVariableOpReadVariableOp*while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ammytzqwsz/ReadVariableOp£
while/ammytzqwsz/mulMul'while/ammytzqwsz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul¦
while/ammytzqwsz/add_1AddV2while/ammytzqwsz/split:output:0while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_1
while/ammytzqwsz/SigmoidSigmoidwhile/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid¯
!while/ammytzqwsz/ReadVariableOp_1ReadVariableOp,while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_1©
while/ammytzqwsz/mul_1Mul)while/ammytzqwsz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_1¨
while/ammytzqwsz/add_2AddV2while/ammytzqwsz/split:output:1while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_2
while/ammytzqwsz/Sigmoid_1Sigmoidwhile/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_1
while/ammytzqwsz/mul_2Mulwhile/ammytzqwsz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_2
while/ammytzqwsz/TanhTanhwhile/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh¢
while/ammytzqwsz/mul_3Mulwhile/ammytzqwsz/Sigmoid:y:0while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_3£
while/ammytzqwsz/add_3AddV2while/ammytzqwsz/mul_2:z:0while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_3¯
!while/ammytzqwsz/ReadVariableOp_2ReadVariableOp,while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_2°
while/ammytzqwsz/mul_4Mul)while/ammytzqwsz/ReadVariableOp_2:value:0while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_4¨
while/ammytzqwsz/add_4AddV2while/ammytzqwsz/split:output:3while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_4
while/ammytzqwsz/Sigmoid_2Sigmoidwhile/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_2
while/ammytzqwsz/Tanh_1Tanhwhile/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh_1¦
while/ammytzqwsz/mul_5Mulwhile/ammytzqwsz/Sigmoid_2:y:0while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ammytzqwsz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ammytzqwsz/mul_5:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ammytzqwsz/add_3:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ammytzqwsz_biasadd_readvariableop_resource2while_ammytzqwsz_biasadd_readvariableop_resource_0"h
1while_ammytzqwsz_matmul_1_readvariableop_resource3while_ammytzqwsz_matmul_1_readvariableop_resource_0"d
/while_ammytzqwsz_matmul_readvariableop_resource1while_ammytzqwsz_matmul_readvariableop_resource_0"Z
*while_ammytzqwsz_readvariableop_1_resource,while_ammytzqwsz_readvariableop_1_resource_0"Z
*while_ammytzqwsz_readvariableop_2_resource,while_ammytzqwsz_readvariableop_2_resource_0"V
(while_ammytzqwsz_readvariableop_resource*while_ammytzqwsz_readvariableop_resource_0")
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
'while/ammytzqwsz/BiasAdd/ReadVariableOp'while/ammytzqwsz/BiasAdd/ReadVariableOp2P
&while/ammytzqwsz/MatMul/ReadVariableOp&while/ammytzqwsz/MatMul/ReadVariableOp2T
(while/ammytzqwsz/MatMul_1/ReadVariableOp(while/ammytzqwsz/MatMul_1/ReadVariableOp2B
while/ammytzqwsz/ReadVariableOpwhile/ammytzqwsz/ReadVariableOp2F
!while/ammytzqwsz/ReadVariableOp_1!while/ammytzqwsz/ReadVariableOp_12F
!while/ammytzqwsz/ReadVariableOp_2!while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_mfdtyewult_layer_call_fn_1824073
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18197952
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
while_cond_1823414
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1823414___redundant_placeholder05
1while_while_cond_1823414___redundant_placeholder15
1while_while_cond_1823414___redundant_placeholder25
1while_while_cond_1823414___redundant_placeholder35
1while_while_cond_1823414___redundant_placeholder45
1while_while_cond_1823414___redundant_placeholder55
1while_while_cond_1823414___redundant_placeholder6
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
mfdtyewult_while_body_18218992
.mfdtyewult_while_mfdtyewult_while_loop_counter8
4mfdtyewult_while_mfdtyewult_while_maximum_iterations 
mfdtyewult_while_placeholder"
mfdtyewult_while_placeholder_1"
mfdtyewult_while_placeholder_2"
mfdtyewult_while_placeholder_31
-mfdtyewult_while_mfdtyewult_strided_slice_1_0m
imfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0O
<mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0:	 Q
>mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 L
=mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0:	C
5mfdtyewult_while_aflyrndiyz_readvariableop_resource_0: E
7mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0: E
7mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0: 
mfdtyewult_while_identity
mfdtyewult_while_identity_1
mfdtyewult_while_identity_2
mfdtyewult_while_identity_3
mfdtyewult_while_identity_4
mfdtyewult_while_identity_5/
+mfdtyewult_while_mfdtyewult_strided_slice_1k
gmfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensorM
:mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource:	 O
<mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource:	 J
;mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource:	A
3mfdtyewult_while_aflyrndiyz_readvariableop_resource: C
5mfdtyewult_while_aflyrndiyz_readvariableop_1_resource: C
5mfdtyewult_while_aflyrndiyz_readvariableop_2_resource: ¢2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp¢1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp¢3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp¢*mfdtyewult/while/aflyrndiyz/ReadVariableOp¢,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1¢,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2Ù
Bmfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bmfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shape
4mfdtyewult/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemimfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0mfdtyewult_while_placeholderKmfdtyewult/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4mfdtyewult/while/TensorArrayV2Read/TensorListGetItemä
1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp<mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOpý
"mfdtyewult/while/aflyrndiyz/MatMulMatMul;mfdtyewult/while/TensorArrayV2Read/TensorListGetItem:item:09mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"mfdtyewult/while/aflyrndiyz/MatMulê
3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp>mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOpæ
$mfdtyewult/while/aflyrndiyz/MatMul_1MatMulmfdtyewult_while_placeholder_2;mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$mfdtyewult/while/aflyrndiyz/MatMul_1Ü
mfdtyewult/while/aflyrndiyz/addAddV2,mfdtyewult/while/aflyrndiyz/MatMul:product:0.mfdtyewult/while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
mfdtyewult/while/aflyrndiyz/addã
2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp=mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOpé
#mfdtyewult/while/aflyrndiyz/BiasAddBiasAdd#mfdtyewult/while/aflyrndiyz/add:z:0:mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#mfdtyewult/while/aflyrndiyz/BiasAdd
+mfdtyewult/while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+mfdtyewult/while/aflyrndiyz/split/split_dim¯
!mfdtyewult/while/aflyrndiyz/splitSplit4mfdtyewult/while/aflyrndiyz/split/split_dim:output:0,mfdtyewult/while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!mfdtyewult/while/aflyrndiyz/splitÊ
*mfdtyewult/while/aflyrndiyz/ReadVariableOpReadVariableOp5mfdtyewult_while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*mfdtyewult/while/aflyrndiyz/ReadVariableOpÏ
mfdtyewult/while/aflyrndiyz/mulMul2mfdtyewult/while/aflyrndiyz/ReadVariableOp:value:0mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
mfdtyewult/while/aflyrndiyz/mulÒ
!mfdtyewult/while/aflyrndiyz/add_1AddV2*mfdtyewult/while/aflyrndiyz/split:output:0#mfdtyewult/while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_1®
#mfdtyewult/while/aflyrndiyz/SigmoidSigmoid%mfdtyewult/while/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#mfdtyewult/while/aflyrndiyz/SigmoidÐ
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1ReadVariableOp7mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1Õ
!mfdtyewult/while/aflyrndiyz/mul_1Mul4mfdtyewult/while/aflyrndiyz/ReadVariableOp_1:value:0mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_1Ô
!mfdtyewult/while/aflyrndiyz/add_2AddV2*mfdtyewult/while/aflyrndiyz/split:output:1%mfdtyewult/while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_2²
%mfdtyewult/while/aflyrndiyz/Sigmoid_1Sigmoid%mfdtyewult/while/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%mfdtyewult/while/aflyrndiyz/Sigmoid_1Ê
!mfdtyewult/while/aflyrndiyz/mul_2Mul)mfdtyewult/while/aflyrndiyz/Sigmoid_1:y:0mfdtyewult_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_2ª
 mfdtyewult/while/aflyrndiyz/TanhTanh*mfdtyewult/while/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 mfdtyewult/while/aflyrndiyz/TanhÎ
!mfdtyewult/while/aflyrndiyz/mul_3Mul'mfdtyewult/while/aflyrndiyz/Sigmoid:y:0$mfdtyewult/while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_3Ï
!mfdtyewult/while/aflyrndiyz/add_3AddV2%mfdtyewult/while/aflyrndiyz/mul_2:z:0%mfdtyewult/while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_3Ð
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2ReadVariableOp7mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2Ü
!mfdtyewult/while/aflyrndiyz/mul_4Mul4mfdtyewult/while/aflyrndiyz/ReadVariableOp_2:value:0%mfdtyewult/while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_4Ô
!mfdtyewult/while/aflyrndiyz/add_4AddV2*mfdtyewult/while/aflyrndiyz/split:output:3%mfdtyewult/while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/add_4²
%mfdtyewult/while/aflyrndiyz/Sigmoid_2Sigmoid%mfdtyewult/while/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%mfdtyewult/while/aflyrndiyz/Sigmoid_2©
"mfdtyewult/while/aflyrndiyz/Tanh_1Tanh%mfdtyewult/while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"mfdtyewult/while/aflyrndiyz/Tanh_1Ò
!mfdtyewult/while/aflyrndiyz/mul_5Mul)mfdtyewult/while/aflyrndiyz/Sigmoid_2:y:0&mfdtyewult/while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!mfdtyewult/while/aflyrndiyz/mul_5
5mfdtyewult/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmfdtyewult_while_placeholder_1mfdtyewult_while_placeholder%mfdtyewult/while/aflyrndiyz/mul_5:z:0*
_output_shapes
: *
element_dtype027
5mfdtyewult/while/TensorArrayV2Write/TensorListSetItemr
mfdtyewult/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
mfdtyewult/while/add/y
mfdtyewult/while/addAddV2mfdtyewult_while_placeholdermfdtyewult/while/add/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/while/addv
mfdtyewult/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
mfdtyewult/while/add_1/y­
mfdtyewult/while/add_1AddV2.mfdtyewult_while_mfdtyewult_while_loop_counter!mfdtyewult/while/add_1/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/while/add_1©
mfdtyewult/while/IdentityIdentitymfdtyewult/while/add_1:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/IdentityÇ
mfdtyewult/while/Identity_1Identity4mfdtyewult_while_mfdtyewult_while_maximum_iterations3^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/Identity_1«
mfdtyewult/while/Identity_2Identitymfdtyewult/while/add:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/Identity_2Ø
mfdtyewult/while/Identity_3IdentityEmfdtyewult/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
mfdtyewult/while/Identity_3É
mfdtyewult/while/Identity_4Identity%mfdtyewult/while/aflyrndiyz/mul_5:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/while/Identity_4É
mfdtyewult/while/Identity_5Identity%mfdtyewult/while/aflyrndiyz/add_3:z:03^mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2^mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp4^mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp+^mfdtyewult/while/aflyrndiyz/ReadVariableOp-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_1-^mfdtyewult/while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/while/Identity_5"|
;mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource=mfdtyewult_while_aflyrndiyz_biasadd_readvariableop_resource_0"~
<mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource>mfdtyewult_while_aflyrndiyz_matmul_1_readvariableop_resource_0"z
:mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource<mfdtyewult_while_aflyrndiyz_matmul_readvariableop_resource_0"p
5mfdtyewult_while_aflyrndiyz_readvariableop_1_resource7mfdtyewult_while_aflyrndiyz_readvariableop_1_resource_0"p
5mfdtyewult_while_aflyrndiyz_readvariableop_2_resource7mfdtyewult_while_aflyrndiyz_readvariableop_2_resource_0"l
3mfdtyewult_while_aflyrndiyz_readvariableop_resource5mfdtyewult_while_aflyrndiyz_readvariableop_resource_0"?
mfdtyewult_while_identity"mfdtyewult/while/Identity:output:0"C
mfdtyewult_while_identity_1$mfdtyewult/while/Identity_1:output:0"C
mfdtyewult_while_identity_2$mfdtyewult/while/Identity_2:output:0"C
mfdtyewult_while_identity_3$mfdtyewult/while/Identity_3:output:0"C
mfdtyewult_while_identity_4$mfdtyewult/while/Identity_4:output:0"C
mfdtyewult_while_identity_5$mfdtyewult/while/Identity_5:output:0"\
+mfdtyewult_while_mfdtyewult_strided_slice_1-mfdtyewult_while_mfdtyewult_strided_slice_1_0"Ô
gmfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensorimfdtyewult_while_tensorarrayv2read_tensorlistgetitem_mfdtyewult_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2mfdtyewult/while/aflyrndiyz/BiasAdd/ReadVariableOp2f
1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp1mfdtyewult/while/aflyrndiyz/MatMul/ReadVariableOp2j
3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp3mfdtyewult/while/aflyrndiyz/MatMul_1/ReadVariableOp2X
*mfdtyewult/while/aflyrndiyz/ReadVariableOp*mfdtyewult/while/aflyrndiyz/ReadVariableOp2\
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_1,mfdtyewult/while/aflyrndiyz/ReadVariableOp_12\
,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2,mfdtyewult/while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1821078

inputs<
)aflyrndiyz_matmul_readvariableop_resource:	 >
+aflyrndiyz_matmul_1_readvariableop_resource:	 9
*aflyrndiyz_biasadd_readvariableop_resource:	0
"aflyrndiyz_readvariableop_resource: 2
$aflyrndiyz_readvariableop_1_resource: 2
$aflyrndiyz_readvariableop_2_resource: 
identity¢!aflyrndiyz/BiasAdd/ReadVariableOp¢ aflyrndiyz/MatMul/ReadVariableOp¢"aflyrndiyz/MatMul_1/ReadVariableOp¢aflyrndiyz/ReadVariableOp¢aflyrndiyz/ReadVariableOp_1¢aflyrndiyz/ReadVariableOp_2¢whileD
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
 aflyrndiyz/MatMul/ReadVariableOpReadVariableOp)aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aflyrndiyz/MatMul/ReadVariableOp§
aflyrndiyz/MatMulMatMulstrided_slice_2:output:0(aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMulµ
"aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp+aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aflyrndiyz/MatMul_1/ReadVariableOp£
aflyrndiyz/MatMul_1MatMulzeros:output:0*aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMul_1
aflyrndiyz/addAddV2aflyrndiyz/MatMul:product:0aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/add®
!aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp*aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aflyrndiyz/BiasAdd/ReadVariableOp¥
aflyrndiyz/BiasAddBiasAddaflyrndiyz/add:z:0)aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/BiasAddz
aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aflyrndiyz/split/split_dimë
aflyrndiyz/splitSplit#aflyrndiyz/split/split_dim:output:0aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aflyrndiyz/split
aflyrndiyz/ReadVariableOpReadVariableOp"aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp
aflyrndiyz/mulMul!aflyrndiyz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul
aflyrndiyz/add_1AddV2aflyrndiyz/split:output:0aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_1{
aflyrndiyz/SigmoidSigmoidaflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid
aflyrndiyz/ReadVariableOp_1ReadVariableOp$aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_1
aflyrndiyz/mul_1Mul#aflyrndiyz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_1
aflyrndiyz/add_2AddV2aflyrndiyz/split:output:1aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_2
aflyrndiyz/Sigmoid_1Sigmoidaflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_1
aflyrndiyz/mul_2Mulaflyrndiyz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_2w
aflyrndiyz/TanhTanhaflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh
aflyrndiyz/mul_3Mulaflyrndiyz/Sigmoid:y:0aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_3
aflyrndiyz/add_3AddV2aflyrndiyz/mul_2:z:0aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_3
aflyrndiyz/ReadVariableOp_2ReadVariableOp$aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_2
aflyrndiyz/mul_4Mul#aflyrndiyz/ReadVariableOp_2:value:0aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_4
aflyrndiyz/add_4AddV2aflyrndiyz/split:output:3aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_4
aflyrndiyz/Sigmoid_2Sigmoidaflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_2v
aflyrndiyz/Tanh_1Tanhaflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh_1
aflyrndiyz/mul_5Mulaflyrndiyz/Sigmoid_2:y:0aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aflyrndiyz_matmul_readvariableop_resource+aflyrndiyz_matmul_1_readvariableop_resource*aflyrndiyz_biasadd_readvariableop_resource"aflyrndiyz_readvariableop_resource$aflyrndiyz_readvariableop_1_resource$aflyrndiyz_readvariableop_2_resource*
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
while_body_1820977*
condR
while_cond_1820976*Q
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
IdentityIdentitystrided_slice_3:output:0"^aflyrndiyz/BiasAdd/ReadVariableOp!^aflyrndiyz/MatMul/ReadVariableOp#^aflyrndiyz/MatMul_1/ReadVariableOp^aflyrndiyz/ReadVariableOp^aflyrndiyz/ReadVariableOp_1^aflyrndiyz/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aflyrndiyz/BiasAdd/ReadVariableOp!aflyrndiyz/BiasAdd/ReadVariableOp2D
 aflyrndiyz/MatMul/ReadVariableOp aflyrndiyz/MatMul/ReadVariableOp2H
"aflyrndiyz/MatMul_1/ReadVariableOp"aflyrndiyz/MatMul_1/ReadVariableOp26
aflyrndiyz/ReadVariableOpaflyrndiyz/ReadVariableOp2:
aflyrndiyz/ReadVariableOp_1aflyrndiyz/ReadVariableOp_12:
aflyrndiyz/ReadVariableOp_2aflyrndiyz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±'
³
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1819695

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
Ü

(sequential_oohztcvkwo_while_body_1818567H
Dsequential_oohztcvkwo_while_sequential_oohztcvkwo_while_loop_counterN
Jsequential_oohztcvkwo_while_sequential_oohztcvkwo_while_maximum_iterations+
'sequential_oohztcvkwo_while_placeholder-
)sequential_oohztcvkwo_while_placeholder_1-
)sequential_oohztcvkwo_while_placeholder_2-
)sequential_oohztcvkwo_while_placeholder_3G
Csequential_oohztcvkwo_while_sequential_oohztcvkwo_strided_slice_1_0
sequential_oohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_sequential_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0:	\
Isequential_oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 W
Hsequential_oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0:	N
@sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0: P
Bsequential_oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0: P
Bsequential_oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0: (
$sequential_oohztcvkwo_while_identity*
&sequential_oohztcvkwo_while_identity_1*
&sequential_oohztcvkwo_while_identity_2*
&sequential_oohztcvkwo_while_identity_3*
&sequential_oohztcvkwo_while_identity_4*
&sequential_oohztcvkwo_while_identity_5E
Asequential_oohztcvkwo_while_sequential_oohztcvkwo_strided_slice_1
}sequential_oohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_sequential_oohztcvkwo_tensorarrayunstack_tensorlistfromtensorX
Esequential_oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource:	Z
Gsequential_oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource:	 U
Fsequential_oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource:	L
>sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_resource: N
@sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource: N
@sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource: ¢=sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp¢<sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp¢>sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp¢5sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp¢7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1¢7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2ï
Msequential/oohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/oohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/oohztcvkwo/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_oohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_sequential_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0'sequential_oohztcvkwo_while_placeholderVsequential/oohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/oohztcvkwo/while/TensorArrayV2Read/TensorListGetItem
<sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOpGsequential_oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp©
-sequential/oohztcvkwo/while/ammytzqwsz/MatMulMatMulFsequential/oohztcvkwo/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/oohztcvkwo/while/ammytzqwsz/MatMul
>sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOpIsequential_oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp
/sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1MatMul)sequential_oohztcvkwo_while_placeholder_2Fsequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1
*sequential/oohztcvkwo/while/ammytzqwsz/addAddV27sequential/oohztcvkwo/while/ammytzqwsz/MatMul:product:09sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/oohztcvkwo/while/ammytzqwsz/add
=sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOpHsequential_oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp
.sequential/oohztcvkwo/while/ammytzqwsz/BiasAddBiasAdd.sequential/oohztcvkwo/while/ammytzqwsz/add:z:0Esequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd²
6sequential/oohztcvkwo/while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/oohztcvkwo/while/ammytzqwsz/split/split_dimÛ
,sequential/oohztcvkwo/while/ammytzqwsz/splitSplit?sequential/oohztcvkwo/while/ammytzqwsz/split/split_dim:output:07sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/oohztcvkwo/while/ammytzqwsz/splitë
5sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOpReadVariableOp@sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOpû
*sequential/oohztcvkwo/while/ammytzqwsz/mulMul=sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp:value:0)sequential_oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/oohztcvkwo/while/ammytzqwsz/mulþ
,sequential/oohztcvkwo/while/ammytzqwsz/add_1AddV25sequential/oohztcvkwo/while/ammytzqwsz/split:output:0.sequential/oohztcvkwo/while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/add_1Ï
.sequential/oohztcvkwo/while/ammytzqwsz/SigmoidSigmoid0sequential/oohztcvkwo/while/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/oohztcvkwo/while/ammytzqwsz/Sigmoidñ
7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1ReadVariableOpBsequential_oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1
,sequential/oohztcvkwo/while/ammytzqwsz/mul_1Mul?sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1:value:0)sequential_oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/mul_1
,sequential/oohztcvkwo/while/ammytzqwsz/add_2AddV25sequential/oohztcvkwo/while/ammytzqwsz/split:output:10sequential/oohztcvkwo/while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/add_2Ó
0sequential/oohztcvkwo/while/ammytzqwsz/Sigmoid_1Sigmoid0sequential/oohztcvkwo/while/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/oohztcvkwo/while/ammytzqwsz/Sigmoid_1ö
,sequential/oohztcvkwo/while/ammytzqwsz/mul_2Mul4sequential/oohztcvkwo/while/ammytzqwsz/Sigmoid_1:y:0)sequential_oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/mul_2Ë
+sequential/oohztcvkwo/while/ammytzqwsz/TanhTanh5sequential/oohztcvkwo/while/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/oohztcvkwo/while/ammytzqwsz/Tanhú
,sequential/oohztcvkwo/while/ammytzqwsz/mul_3Mul2sequential/oohztcvkwo/while/ammytzqwsz/Sigmoid:y:0/sequential/oohztcvkwo/while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/mul_3û
,sequential/oohztcvkwo/while/ammytzqwsz/add_3AddV20sequential/oohztcvkwo/while/ammytzqwsz/mul_2:z:00sequential/oohztcvkwo/while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/add_3ñ
7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2ReadVariableOpBsequential_oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2
,sequential/oohztcvkwo/while/ammytzqwsz/mul_4Mul?sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2:value:00sequential/oohztcvkwo/while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/mul_4
,sequential/oohztcvkwo/while/ammytzqwsz/add_4AddV25sequential/oohztcvkwo/while/ammytzqwsz/split:output:30sequential/oohztcvkwo/while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/add_4Ó
0sequential/oohztcvkwo/while/ammytzqwsz/Sigmoid_2Sigmoid0sequential/oohztcvkwo/while/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/oohztcvkwo/while/ammytzqwsz/Sigmoid_2Ê
-sequential/oohztcvkwo/while/ammytzqwsz/Tanh_1Tanh0sequential/oohztcvkwo/while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/oohztcvkwo/while/ammytzqwsz/Tanh_1þ
,sequential/oohztcvkwo/while/ammytzqwsz/mul_5Mul4sequential/oohztcvkwo/while/ammytzqwsz/Sigmoid_2:y:01sequential/oohztcvkwo/while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/oohztcvkwo/while/ammytzqwsz/mul_5Ì
@sequential/oohztcvkwo/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_oohztcvkwo_while_placeholder_1'sequential_oohztcvkwo_while_placeholder0sequential/oohztcvkwo/while/ammytzqwsz/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/oohztcvkwo/while/TensorArrayV2Write/TensorListSetItem
!sequential/oohztcvkwo/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/oohztcvkwo/while/add/yÁ
sequential/oohztcvkwo/while/addAddV2'sequential_oohztcvkwo_while_placeholder*sequential/oohztcvkwo/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/oohztcvkwo/while/add
#sequential/oohztcvkwo/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/oohztcvkwo/while/add_1/yä
!sequential/oohztcvkwo/while/add_1AddV2Dsequential_oohztcvkwo_while_sequential_oohztcvkwo_while_loop_counter,sequential/oohztcvkwo/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/oohztcvkwo/while/add_1
$sequential/oohztcvkwo/while/IdentityIdentity%sequential/oohztcvkwo/while/add_1:z:0>^sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp=^sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp?^sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp6^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp8^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_18^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/oohztcvkwo/while/Identityµ
&sequential/oohztcvkwo/while/Identity_1IdentityJsequential_oohztcvkwo_while_sequential_oohztcvkwo_while_maximum_iterations>^sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp=^sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp?^sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp6^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp8^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_18^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/oohztcvkwo/while/Identity_1
&sequential/oohztcvkwo/while/Identity_2Identity#sequential/oohztcvkwo/while/add:z:0>^sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp=^sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp?^sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp6^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp8^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_18^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/oohztcvkwo/while/Identity_2»
&sequential/oohztcvkwo/while/Identity_3IdentityPsequential/oohztcvkwo/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp=^sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp?^sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp6^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp8^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_18^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/oohztcvkwo/while/Identity_3¬
&sequential/oohztcvkwo/while/Identity_4Identity0sequential/oohztcvkwo/while/ammytzqwsz/mul_5:z:0>^sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp=^sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp?^sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp6^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp8^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_18^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/while/Identity_4¬
&sequential/oohztcvkwo/while/Identity_5Identity0sequential/oohztcvkwo/while/ammytzqwsz/add_3:z:0>^sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp=^sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp?^sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp6^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp8^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_18^sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/while/Identity_5"
Fsequential_oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resourceHsequential_oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0"
Gsequential_oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resourceIsequential_oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0"
Esequential_oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resourceGsequential_oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0"
@sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_1_resourceBsequential_oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0"
@sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_2_resourceBsequential_oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0"
>sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_resource@sequential_oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0"U
$sequential_oohztcvkwo_while_identity-sequential/oohztcvkwo/while/Identity:output:0"Y
&sequential_oohztcvkwo_while_identity_1/sequential/oohztcvkwo/while/Identity_1:output:0"Y
&sequential_oohztcvkwo_while_identity_2/sequential/oohztcvkwo/while/Identity_2:output:0"Y
&sequential_oohztcvkwo_while_identity_3/sequential/oohztcvkwo/while/Identity_3:output:0"Y
&sequential_oohztcvkwo_while_identity_4/sequential/oohztcvkwo/while/Identity_4:output:0"Y
&sequential_oohztcvkwo_while_identity_5/sequential/oohztcvkwo/while/Identity_5:output:0"
Asequential_oohztcvkwo_while_sequential_oohztcvkwo_strided_slice_1Csequential_oohztcvkwo_while_sequential_oohztcvkwo_strided_slice_1_0"
}sequential_oohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_sequential_oohztcvkwo_tensorarrayunstack_tensorlistfromtensorsequential_oohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_sequential_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp=sequential/oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2|
<sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp<sequential/oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp2
>sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp>sequential/oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp2n
5sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp5sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp2r
7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_17sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_12r
7sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_27sequential/oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_sequential_layer_call_fn_1822484

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
G__inference_sequential_layer_call_and_return_conditional_losses_18214032
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
Üh

G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823516
inputs_0<
)aflyrndiyz_matmul_readvariableop_resource:	 >
+aflyrndiyz_matmul_1_readvariableop_resource:	 9
*aflyrndiyz_biasadd_readvariableop_resource:	0
"aflyrndiyz_readvariableop_resource: 2
$aflyrndiyz_readvariableop_1_resource: 2
$aflyrndiyz_readvariableop_2_resource: 
identity¢!aflyrndiyz/BiasAdd/ReadVariableOp¢ aflyrndiyz/MatMul/ReadVariableOp¢"aflyrndiyz/MatMul_1/ReadVariableOp¢aflyrndiyz/ReadVariableOp¢aflyrndiyz/ReadVariableOp_1¢aflyrndiyz/ReadVariableOp_2¢whileF
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
 aflyrndiyz/MatMul/ReadVariableOpReadVariableOp)aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aflyrndiyz/MatMul/ReadVariableOp§
aflyrndiyz/MatMulMatMulstrided_slice_2:output:0(aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMulµ
"aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp+aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aflyrndiyz/MatMul_1/ReadVariableOp£
aflyrndiyz/MatMul_1MatMulzeros:output:0*aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMul_1
aflyrndiyz/addAddV2aflyrndiyz/MatMul:product:0aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/add®
!aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp*aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aflyrndiyz/BiasAdd/ReadVariableOp¥
aflyrndiyz/BiasAddBiasAddaflyrndiyz/add:z:0)aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/BiasAddz
aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aflyrndiyz/split/split_dimë
aflyrndiyz/splitSplit#aflyrndiyz/split/split_dim:output:0aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aflyrndiyz/split
aflyrndiyz/ReadVariableOpReadVariableOp"aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp
aflyrndiyz/mulMul!aflyrndiyz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul
aflyrndiyz/add_1AddV2aflyrndiyz/split:output:0aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_1{
aflyrndiyz/SigmoidSigmoidaflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid
aflyrndiyz/ReadVariableOp_1ReadVariableOp$aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_1
aflyrndiyz/mul_1Mul#aflyrndiyz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_1
aflyrndiyz/add_2AddV2aflyrndiyz/split:output:1aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_2
aflyrndiyz/Sigmoid_1Sigmoidaflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_1
aflyrndiyz/mul_2Mulaflyrndiyz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_2w
aflyrndiyz/TanhTanhaflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh
aflyrndiyz/mul_3Mulaflyrndiyz/Sigmoid:y:0aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_3
aflyrndiyz/add_3AddV2aflyrndiyz/mul_2:z:0aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_3
aflyrndiyz/ReadVariableOp_2ReadVariableOp$aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_2
aflyrndiyz/mul_4Mul#aflyrndiyz/ReadVariableOp_2:value:0aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_4
aflyrndiyz/add_4AddV2aflyrndiyz/split:output:3aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_4
aflyrndiyz/Sigmoid_2Sigmoidaflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_2v
aflyrndiyz/Tanh_1Tanhaflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh_1
aflyrndiyz/mul_5Mulaflyrndiyz/Sigmoid_2:y:0aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aflyrndiyz_matmul_readvariableop_resource+aflyrndiyz_matmul_1_readvariableop_resource*aflyrndiyz_biasadd_readvariableop_resource"aflyrndiyz_readvariableop_resource$aflyrndiyz_readvariableop_1_resource$aflyrndiyz_readvariableop_2_resource*
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
while_body_1823415*
condR
while_cond_1823414*Q
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
IdentityIdentitystrided_slice_3:output:0"^aflyrndiyz/BiasAdd/ReadVariableOp!^aflyrndiyz/MatMul/ReadVariableOp#^aflyrndiyz/MatMul_1/ReadVariableOp^aflyrndiyz/ReadVariableOp^aflyrndiyz/ReadVariableOp_1^aflyrndiyz/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aflyrndiyz/BiasAdd/ReadVariableOp!aflyrndiyz/BiasAdd/ReadVariableOp2D
 aflyrndiyz/MatMul/ReadVariableOp aflyrndiyz/MatMul/ReadVariableOp2H
"aflyrndiyz/MatMul_1/ReadVariableOp"aflyrndiyz/MatMul_1/ReadVariableOp26
aflyrndiyz/ReadVariableOpaflyrndiyz/ReadVariableOp2:
aflyrndiyz/ReadVariableOp_1aflyrndiyz/ReadVariableOp_12:
aflyrndiyz/ReadVariableOp_2aflyrndiyz/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


í
while_cond_1819714
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1819714___redundant_placeholder05
1while_while_cond_1819714___redundant_placeholder15
1while_while_cond_1819714___redundant_placeholder25
1while_while_cond_1819714___redundant_placeholder35
1while_while_cond_1819714___redundant_placeholder45
1while_while_cond_1819714___redundant_placeholder55
1while_while_cond_1819714___redundant_placeholder6
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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1822006

inputsL
6fethhjgisa_conv1d_expanddims_1_readvariableop_resource:K
=fethhjgisa_squeeze_batch_dims_biasadd_readvariableop_resource:G
4oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource:	I
6oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource:	 D
5oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource:	;
-oohztcvkwo_ammytzqwsz_readvariableop_resource: =
/oohztcvkwo_ammytzqwsz_readvariableop_1_resource: =
/oohztcvkwo_ammytzqwsz_readvariableop_2_resource: G
4mfdtyewult_aflyrndiyz_matmul_readvariableop_resource:	 I
6mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource:	 D
5mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource:	;
-mfdtyewult_aflyrndiyz_readvariableop_resource: =
/mfdtyewult_aflyrndiyz_readvariableop_1_resource: =
/mfdtyewult_aflyrndiyz_readvariableop_2_resource: ;
)boyhyiogqf_matmul_readvariableop_resource: 8
*boyhyiogqf_biasadd_readvariableop_resource:
identity¢!boyhyiogqf/BiasAdd/ReadVariableOp¢ boyhyiogqf/MatMul/ReadVariableOp¢-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp¢4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp¢+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp¢-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp¢$mfdtyewult/aflyrndiyz/ReadVariableOp¢&mfdtyewult/aflyrndiyz/ReadVariableOp_1¢&mfdtyewult/aflyrndiyz/ReadVariableOp_2¢mfdtyewult/while¢,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp¢+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp¢-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp¢$oohztcvkwo/ammytzqwsz/ReadVariableOp¢&oohztcvkwo/ammytzqwsz/ReadVariableOp_1¢&oohztcvkwo/ammytzqwsz/ReadVariableOp_2¢oohztcvkwo/while
 fethhjgisa/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 fethhjgisa/conv1d/ExpandDims/dim»
fethhjgisa/conv1d/ExpandDims
ExpandDimsinputs)fethhjgisa/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/ExpandDimsÙ
-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6fethhjgisa_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp
"fethhjgisa/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"fethhjgisa/conv1d/ExpandDims_1/dimã
fethhjgisa/conv1d/ExpandDims_1
ExpandDims5fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp:value:0+fethhjgisa/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
fethhjgisa/conv1d/ExpandDims_1
fethhjgisa/conv1d/ShapeShape%fethhjgisa/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
fethhjgisa/conv1d/Shape
%fethhjgisa/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%fethhjgisa/conv1d/strided_slice/stack¥
'fethhjgisa/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'fethhjgisa/conv1d/strided_slice/stack_1
'fethhjgisa/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'fethhjgisa/conv1d/strided_slice/stack_2Ì
fethhjgisa/conv1d/strided_sliceStridedSlice fethhjgisa/conv1d/Shape:output:0.fethhjgisa/conv1d/strided_slice/stack:output:00fethhjgisa/conv1d/strided_slice/stack_1:output:00fethhjgisa/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
fethhjgisa/conv1d/strided_slice
fethhjgisa/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
fethhjgisa/conv1d/Reshape/shapeÌ
fethhjgisa/conv1d/ReshapeReshape%fethhjgisa/conv1d/ExpandDims:output:0(fethhjgisa/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/Reshapeî
fethhjgisa/conv1d/Conv2DConv2D"fethhjgisa/conv1d/Reshape:output:0'fethhjgisa/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
fethhjgisa/conv1d/Conv2D
!fethhjgisa/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!fethhjgisa/conv1d/concat/values_1
fethhjgisa/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/concat/axisì
fethhjgisa/conv1d/concatConcatV2(fethhjgisa/conv1d/strided_slice:output:0*fethhjgisa/conv1d/concat/values_1:output:0&fethhjgisa/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fethhjgisa/conv1d/concatÉ
fethhjgisa/conv1d/Reshape_1Reshape!fethhjgisa/conv1d/Conv2D:output:0!fethhjgisa/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/Reshape_1Á
fethhjgisa/conv1d/SqueezeSqueeze$fethhjgisa/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/Squeeze
#fethhjgisa/squeeze_batch_dims/ShapeShape"fethhjgisa/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#fethhjgisa/squeeze_batch_dims/Shape°
1fethhjgisa/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1fethhjgisa/squeeze_batch_dims/strided_slice/stack½
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_1´
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_2
+fethhjgisa/squeeze_batch_dims/strided_sliceStridedSlice,fethhjgisa/squeeze_batch_dims/Shape:output:0:fethhjgisa/squeeze_batch_dims/strided_slice/stack:output:0<fethhjgisa/squeeze_batch_dims/strided_slice/stack_1:output:0<fethhjgisa/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+fethhjgisa/squeeze_batch_dims/strided_slice¯
+fethhjgisa/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+fethhjgisa/squeeze_batch_dims/Reshape/shapeé
%fethhjgisa/squeeze_batch_dims/ReshapeReshape"fethhjgisa/conv1d/Squeeze:output:04fethhjgisa/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%fethhjgisa/squeeze_batch_dims/Reshapeæ
4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=fethhjgisa_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%fethhjgisa/squeeze_batch_dims/BiasAddBiasAdd.fethhjgisa/squeeze_batch_dims/Reshape:output:0<fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%fethhjgisa/squeeze_batch_dims/BiasAdd¯
-fethhjgisa/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-fethhjgisa/squeeze_batch_dims/concat/values_1¡
)fethhjgisa/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)fethhjgisa/squeeze_batch_dims/concat/axis¨
$fethhjgisa/squeeze_batch_dims/concatConcatV24fethhjgisa/squeeze_batch_dims/strided_slice:output:06fethhjgisa/squeeze_batch_dims/concat/values_1:output:02fethhjgisa/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$fethhjgisa/squeeze_batch_dims/concatö
'fethhjgisa/squeeze_batch_dims/Reshape_1Reshape.fethhjgisa/squeeze_batch_dims/BiasAdd:output:0-fethhjgisa/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'fethhjgisa/squeeze_batch_dims/Reshape_1
ohvvfduigw/ShapeShape0fethhjgisa/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ohvvfduigw/Shape
ohvvfduigw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ohvvfduigw/strided_slice/stack
 ohvvfduigw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ohvvfduigw/strided_slice/stack_1
 ohvvfduigw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ohvvfduigw/strided_slice/stack_2¤
ohvvfduigw/strided_sliceStridedSliceohvvfduigw/Shape:output:0'ohvvfduigw/strided_slice/stack:output:0)ohvvfduigw/strided_slice/stack_1:output:0)ohvvfduigw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ohvvfduigw/strided_slicez
ohvvfduigw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ohvvfduigw/Reshape/shape/1z
ohvvfduigw/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ohvvfduigw/Reshape/shape/2×
ohvvfduigw/Reshape/shapePack!ohvvfduigw/strided_slice:output:0#ohvvfduigw/Reshape/shape/1:output:0#ohvvfduigw/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ohvvfduigw/Reshape/shape¾
ohvvfduigw/ReshapeReshape0fethhjgisa/squeeze_batch_dims/Reshape_1:output:0!ohvvfduigw/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ohvvfduigw/Reshapeo
oohztcvkwo/ShapeShapeohvvfduigw/Reshape:output:0*
T0*
_output_shapes
:2
oohztcvkwo/Shape
oohztcvkwo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
oohztcvkwo/strided_slice/stack
 oohztcvkwo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 oohztcvkwo/strided_slice/stack_1
 oohztcvkwo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 oohztcvkwo/strided_slice/stack_2¤
oohztcvkwo/strided_sliceStridedSliceoohztcvkwo/Shape:output:0'oohztcvkwo/strided_slice/stack:output:0)oohztcvkwo/strided_slice/stack_1:output:0)oohztcvkwo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
oohztcvkwo/strided_slicer
oohztcvkwo/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros/mul/y
oohztcvkwo/zeros/mulMul!oohztcvkwo/strided_slice:output:0oohztcvkwo/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros/mulu
oohztcvkwo/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
oohztcvkwo/zeros/Less/y
oohztcvkwo/zeros/LessLessoohztcvkwo/zeros/mul:z:0 oohztcvkwo/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros/Lessx
oohztcvkwo/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros/packed/1¯
oohztcvkwo/zeros/packedPack!oohztcvkwo/strided_slice:output:0"oohztcvkwo/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
oohztcvkwo/zeros/packedu
oohztcvkwo/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
oohztcvkwo/zeros/Const¡
oohztcvkwo/zerosFill oohztcvkwo/zeros/packed:output:0oohztcvkwo/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/zerosv
oohztcvkwo/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros_1/mul/y
oohztcvkwo/zeros_1/mulMul!oohztcvkwo/strided_slice:output:0!oohztcvkwo/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros_1/muly
oohztcvkwo/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
oohztcvkwo/zeros_1/Less/y
oohztcvkwo/zeros_1/LessLessoohztcvkwo/zeros_1/mul:z:0"oohztcvkwo/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros_1/Less|
oohztcvkwo/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros_1/packed/1µ
oohztcvkwo/zeros_1/packedPack!oohztcvkwo/strided_slice:output:0$oohztcvkwo/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
oohztcvkwo/zeros_1/packedy
oohztcvkwo/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
oohztcvkwo/zeros_1/Const©
oohztcvkwo/zeros_1Fill"oohztcvkwo/zeros_1/packed:output:0!oohztcvkwo/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/zeros_1
oohztcvkwo/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
oohztcvkwo/transpose/perm°
oohztcvkwo/transpose	Transposeohvvfduigw/Reshape:output:0"oohztcvkwo/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/transposep
oohztcvkwo/Shape_1Shapeoohztcvkwo/transpose:y:0*
T0*
_output_shapes
:2
oohztcvkwo/Shape_1
 oohztcvkwo/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 oohztcvkwo/strided_slice_1/stack
"oohztcvkwo/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_1/stack_1
"oohztcvkwo/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_1/stack_2°
oohztcvkwo/strided_slice_1StridedSliceoohztcvkwo/Shape_1:output:0)oohztcvkwo/strided_slice_1/stack:output:0+oohztcvkwo/strided_slice_1/stack_1:output:0+oohztcvkwo/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
oohztcvkwo/strided_slice_1
&oohztcvkwo/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&oohztcvkwo/TensorArrayV2/element_shapeÞ
oohztcvkwo/TensorArrayV2TensorListReserve/oohztcvkwo/TensorArrayV2/element_shape:output:0#oohztcvkwo/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
oohztcvkwo/TensorArrayV2Õ
@oohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@oohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2oohztcvkwo/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoroohztcvkwo/transpose:y:0Ioohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2oohztcvkwo/TensorArrayUnstack/TensorListFromTensor
 oohztcvkwo/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 oohztcvkwo/strided_slice_2/stack
"oohztcvkwo/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_2/stack_1
"oohztcvkwo/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_2/stack_2¾
oohztcvkwo/strided_slice_2StridedSliceoohztcvkwo/transpose:y:0)oohztcvkwo/strided_slice_2/stack:output:0+oohztcvkwo/strided_slice_2/stack_1:output:0+oohztcvkwo/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
oohztcvkwo/strided_slice_2Ð
+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp4oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOpÓ
oohztcvkwo/ammytzqwsz/MatMulMatMul#oohztcvkwo/strided_slice_2:output:03oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/ammytzqwsz/MatMulÖ
-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp6oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOpÏ
oohztcvkwo/ammytzqwsz/MatMul_1MatMuloohztcvkwo/zeros:output:05oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
oohztcvkwo/ammytzqwsz/MatMul_1Ä
oohztcvkwo/ammytzqwsz/addAddV2&oohztcvkwo/ammytzqwsz/MatMul:product:0(oohztcvkwo/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/ammytzqwsz/addÏ
,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp5oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOpÑ
oohztcvkwo/ammytzqwsz/BiasAddBiasAddoohztcvkwo/ammytzqwsz/add:z:04oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/ammytzqwsz/BiasAdd
%oohztcvkwo/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%oohztcvkwo/ammytzqwsz/split/split_dim
oohztcvkwo/ammytzqwsz/splitSplit.oohztcvkwo/ammytzqwsz/split/split_dim:output:0&oohztcvkwo/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
oohztcvkwo/ammytzqwsz/split¶
$oohztcvkwo/ammytzqwsz/ReadVariableOpReadVariableOp-oohztcvkwo_ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02&
$oohztcvkwo/ammytzqwsz/ReadVariableOpº
oohztcvkwo/ammytzqwsz/mulMul,oohztcvkwo/ammytzqwsz/ReadVariableOp:value:0oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mulº
oohztcvkwo/ammytzqwsz/add_1AddV2$oohztcvkwo/ammytzqwsz/split:output:0oohztcvkwo/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_1
oohztcvkwo/ammytzqwsz/SigmoidSigmoidoohztcvkwo/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/Sigmoid¼
&oohztcvkwo/ammytzqwsz/ReadVariableOp_1ReadVariableOp/oohztcvkwo_ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&oohztcvkwo/ammytzqwsz/ReadVariableOp_1À
oohztcvkwo/ammytzqwsz/mul_1Mul.oohztcvkwo/ammytzqwsz/ReadVariableOp_1:value:0oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_1¼
oohztcvkwo/ammytzqwsz/add_2AddV2$oohztcvkwo/ammytzqwsz/split:output:1oohztcvkwo/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_2 
oohztcvkwo/ammytzqwsz/Sigmoid_1Sigmoidoohztcvkwo/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
oohztcvkwo/ammytzqwsz/Sigmoid_1µ
oohztcvkwo/ammytzqwsz/mul_2Mul#oohztcvkwo/ammytzqwsz/Sigmoid_1:y:0oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_2
oohztcvkwo/ammytzqwsz/TanhTanh$oohztcvkwo/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/Tanh¶
oohztcvkwo/ammytzqwsz/mul_3Mul!oohztcvkwo/ammytzqwsz/Sigmoid:y:0oohztcvkwo/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_3·
oohztcvkwo/ammytzqwsz/add_3AddV2oohztcvkwo/ammytzqwsz/mul_2:z:0oohztcvkwo/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_3¼
&oohztcvkwo/ammytzqwsz/ReadVariableOp_2ReadVariableOp/oohztcvkwo_ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&oohztcvkwo/ammytzqwsz/ReadVariableOp_2Ä
oohztcvkwo/ammytzqwsz/mul_4Mul.oohztcvkwo/ammytzqwsz/ReadVariableOp_2:value:0oohztcvkwo/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_4¼
oohztcvkwo/ammytzqwsz/add_4AddV2$oohztcvkwo/ammytzqwsz/split:output:3oohztcvkwo/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_4 
oohztcvkwo/ammytzqwsz/Sigmoid_2Sigmoidoohztcvkwo/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
oohztcvkwo/ammytzqwsz/Sigmoid_2
oohztcvkwo/ammytzqwsz/Tanh_1Tanhoohztcvkwo/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/Tanh_1º
oohztcvkwo/ammytzqwsz/mul_5Mul#oohztcvkwo/ammytzqwsz/Sigmoid_2:y:0 oohztcvkwo/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_5¥
(oohztcvkwo/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(oohztcvkwo/TensorArrayV2_1/element_shapeä
oohztcvkwo/TensorArrayV2_1TensorListReserve1oohztcvkwo/TensorArrayV2_1/element_shape:output:0#oohztcvkwo/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
oohztcvkwo/TensorArrayV2_1d
oohztcvkwo/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/time
#oohztcvkwo/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#oohztcvkwo/while/maximum_iterations
oohztcvkwo/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/while/loop_counter²
oohztcvkwo/whileWhile&oohztcvkwo/while/loop_counter:output:0,oohztcvkwo/while/maximum_iterations:output:0oohztcvkwo/time:output:0#oohztcvkwo/TensorArrayV2_1:handle:0oohztcvkwo/zeros:output:0oohztcvkwo/zeros_1:output:0#oohztcvkwo/strided_slice_1:output:0Boohztcvkwo/TensorArrayUnstack/TensorListFromTensor:output_handle:04oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource6oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource5oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource-oohztcvkwo_ammytzqwsz_readvariableop_resource/oohztcvkwo_ammytzqwsz_readvariableop_1_resource/oohztcvkwo_ammytzqwsz_readvariableop_2_resource*
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
oohztcvkwo_while_body_1821723*)
cond!R
oohztcvkwo_while_cond_1821722*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
oohztcvkwo/whileË
;oohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;oohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shape
-oohztcvkwo/TensorArrayV2Stack/TensorListStackTensorListStackoohztcvkwo/while:output:3Doohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-oohztcvkwo/TensorArrayV2Stack/TensorListStack
 oohztcvkwo/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 oohztcvkwo/strided_slice_3/stack
"oohztcvkwo/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"oohztcvkwo/strided_slice_3/stack_1
"oohztcvkwo/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_3/stack_2Ü
oohztcvkwo/strided_slice_3StridedSlice6oohztcvkwo/TensorArrayV2Stack/TensorListStack:tensor:0)oohztcvkwo/strided_slice_3/stack:output:0+oohztcvkwo/strided_slice_3/stack_1:output:0+oohztcvkwo/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
oohztcvkwo/strided_slice_3
oohztcvkwo/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
oohztcvkwo/transpose_1/permÑ
oohztcvkwo/transpose_1	Transpose6oohztcvkwo/TensorArrayV2Stack/TensorListStack:tensor:0$oohztcvkwo/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/transpose_1n
mfdtyewult/ShapeShapeoohztcvkwo/transpose_1:y:0*
T0*
_output_shapes
:2
mfdtyewult/Shape
mfdtyewult/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
mfdtyewult/strided_slice/stack
 mfdtyewult/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 mfdtyewult/strided_slice/stack_1
 mfdtyewult/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 mfdtyewult/strided_slice/stack_2¤
mfdtyewult/strided_sliceStridedSlicemfdtyewult/Shape:output:0'mfdtyewult/strided_slice/stack:output:0)mfdtyewult/strided_slice/stack_1:output:0)mfdtyewult/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mfdtyewult/strided_slicer
mfdtyewult/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros/mul/y
mfdtyewult/zeros/mulMul!mfdtyewult/strided_slice:output:0mfdtyewult/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros/mulu
mfdtyewult/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
mfdtyewult/zeros/Less/y
mfdtyewult/zeros/LessLessmfdtyewult/zeros/mul:z:0 mfdtyewult/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros/Lessx
mfdtyewult/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros/packed/1¯
mfdtyewult/zeros/packedPack!mfdtyewult/strided_slice:output:0"mfdtyewult/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
mfdtyewult/zeros/packedu
mfdtyewult/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mfdtyewult/zeros/Const¡
mfdtyewult/zerosFill mfdtyewult/zeros/packed:output:0mfdtyewult/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/zerosv
mfdtyewult/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros_1/mul/y
mfdtyewult/zeros_1/mulMul!mfdtyewult/strided_slice:output:0!mfdtyewult/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros_1/muly
mfdtyewult/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
mfdtyewult/zeros_1/Less/y
mfdtyewult/zeros_1/LessLessmfdtyewult/zeros_1/mul:z:0"mfdtyewult/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros_1/Less|
mfdtyewult/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros_1/packed/1µ
mfdtyewult/zeros_1/packedPack!mfdtyewult/strided_slice:output:0$mfdtyewult/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
mfdtyewult/zeros_1/packedy
mfdtyewult/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mfdtyewult/zeros_1/Const©
mfdtyewult/zeros_1Fill"mfdtyewult/zeros_1/packed:output:0!mfdtyewult/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/zeros_1
mfdtyewult/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mfdtyewult/transpose/perm¯
mfdtyewult/transpose	Transposeoohztcvkwo/transpose_1:y:0"mfdtyewult/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/transposep
mfdtyewult/Shape_1Shapemfdtyewult/transpose:y:0*
T0*
_output_shapes
:2
mfdtyewult/Shape_1
 mfdtyewult/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 mfdtyewult/strided_slice_1/stack
"mfdtyewult/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_1/stack_1
"mfdtyewult/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_1/stack_2°
mfdtyewult/strided_slice_1StridedSlicemfdtyewult/Shape_1:output:0)mfdtyewult/strided_slice_1/stack:output:0+mfdtyewult/strided_slice_1/stack_1:output:0+mfdtyewult/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mfdtyewult/strided_slice_1
&mfdtyewult/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&mfdtyewult/TensorArrayV2/element_shapeÞ
mfdtyewult/TensorArrayV2TensorListReserve/mfdtyewult/TensorArrayV2/element_shape:output:0#mfdtyewult/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
mfdtyewult/TensorArrayV2Õ
@mfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@mfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2mfdtyewult/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormfdtyewult/transpose:y:0Imfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2mfdtyewult/TensorArrayUnstack/TensorListFromTensor
 mfdtyewult/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 mfdtyewult/strided_slice_2/stack
"mfdtyewult/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_2/stack_1
"mfdtyewult/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_2/stack_2¾
mfdtyewult/strided_slice_2StridedSlicemfdtyewult/transpose:y:0)mfdtyewult/strided_slice_2/stack:output:0+mfdtyewult/strided_slice_2/stack_1:output:0+mfdtyewult/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
mfdtyewult/strided_slice_2Ð
+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp4mfdtyewult_aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOpÓ
mfdtyewult/aflyrndiyz/MatMulMatMul#mfdtyewult/strided_slice_2:output:03mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mfdtyewult/aflyrndiyz/MatMulÖ
-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp6mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOpÏ
mfdtyewult/aflyrndiyz/MatMul_1MatMulmfdtyewult/zeros:output:05mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
mfdtyewult/aflyrndiyz/MatMul_1Ä
mfdtyewult/aflyrndiyz/addAddV2&mfdtyewult/aflyrndiyz/MatMul:product:0(mfdtyewult/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mfdtyewult/aflyrndiyz/addÏ
,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp5mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOpÑ
mfdtyewult/aflyrndiyz/BiasAddBiasAddmfdtyewult/aflyrndiyz/add:z:04mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mfdtyewult/aflyrndiyz/BiasAdd
%mfdtyewult/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%mfdtyewult/aflyrndiyz/split/split_dim
mfdtyewult/aflyrndiyz/splitSplit.mfdtyewult/aflyrndiyz/split/split_dim:output:0&mfdtyewult/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
mfdtyewult/aflyrndiyz/split¶
$mfdtyewult/aflyrndiyz/ReadVariableOpReadVariableOp-mfdtyewult_aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02&
$mfdtyewult/aflyrndiyz/ReadVariableOpº
mfdtyewult/aflyrndiyz/mulMul,mfdtyewult/aflyrndiyz/ReadVariableOp:value:0mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mulº
mfdtyewult/aflyrndiyz/add_1AddV2$mfdtyewult/aflyrndiyz/split:output:0mfdtyewult/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_1
mfdtyewult/aflyrndiyz/SigmoidSigmoidmfdtyewult/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/Sigmoid¼
&mfdtyewult/aflyrndiyz/ReadVariableOp_1ReadVariableOp/mfdtyewult_aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&mfdtyewult/aflyrndiyz/ReadVariableOp_1À
mfdtyewult/aflyrndiyz/mul_1Mul.mfdtyewult/aflyrndiyz/ReadVariableOp_1:value:0mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_1¼
mfdtyewult/aflyrndiyz/add_2AddV2$mfdtyewult/aflyrndiyz/split:output:1mfdtyewult/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_2 
mfdtyewult/aflyrndiyz/Sigmoid_1Sigmoidmfdtyewult/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
mfdtyewult/aflyrndiyz/Sigmoid_1µ
mfdtyewult/aflyrndiyz/mul_2Mul#mfdtyewult/aflyrndiyz/Sigmoid_1:y:0mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_2
mfdtyewult/aflyrndiyz/TanhTanh$mfdtyewult/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/Tanh¶
mfdtyewult/aflyrndiyz/mul_3Mul!mfdtyewult/aflyrndiyz/Sigmoid:y:0mfdtyewult/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_3·
mfdtyewult/aflyrndiyz/add_3AddV2mfdtyewult/aflyrndiyz/mul_2:z:0mfdtyewult/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_3¼
&mfdtyewult/aflyrndiyz/ReadVariableOp_2ReadVariableOp/mfdtyewult_aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&mfdtyewult/aflyrndiyz/ReadVariableOp_2Ä
mfdtyewult/aflyrndiyz/mul_4Mul.mfdtyewult/aflyrndiyz/ReadVariableOp_2:value:0mfdtyewult/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_4¼
mfdtyewult/aflyrndiyz/add_4AddV2$mfdtyewult/aflyrndiyz/split:output:3mfdtyewult/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_4 
mfdtyewult/aflyrndiyz/Sigmoid_2Sigmoidmfdtyewult/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
mfdtyewult/aflyrndiyz/Sigmoid_2
mfdtyewult/aflyrndiyz/Tanh_1Tanhmfdtyewult/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/Tanh_1º
mfdtyewult/aflyrndiyz/mul_5Mul#mfdtyewult/aflyrndiyz/Sigmoid_2:y:0 mfdtyewult/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_5¥
(mfdtyewult/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(mfdtyewult/TensorArrayV2_1/element_shapeä
mfdtyewult/TensorArrayV2_1TensorListReserve1mfdtyewult/TensorArrayV2_1/element_shape:output:0#mfdtyewult/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
mfdtyewult/TensorArrayV2_1d
mfdtyewult/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/time
#mfdtyewult/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#mfdtyewult/while/maximum_iterations
mfdtyewult/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/while/loop_counter²
mfdtyewult/whileWhile&mfdtyewult/while/loop_counter:output:0,mfdtyewult/while/maximum_iterations:output:0mfdtyewult/time:output:0#mfdtyewult/TensorArrayV2_1:handle:0mfdtyewult/zeros:output:0mfdtyewult/zeros_1:output:0#mfdtyewult/strided_slice_1:output:0Bmfdtyewult/TensorArrayUnstack/TensorListFromTensor:output_handle:04mfdtyewult_aflyrndiyz_matmul_readvariableop_resource6mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource5mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource-mfdtyewult_aflyrndiyz_readvariableop_resource/mfdtyewult_aflyrndiyz_readvariableop_1_resource/mfdtyewult_aflyrndiyz_readvariableop_2_resource*
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
mfdtyewult_while_body_1821899*)
cond!R
mfdtyewult_while_cond_1821898*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
mfdtyewult/whileË
;mfdtyewult/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;mfdtyewult/TensorArrayV2Stack/TensorListStack/element_shape
-mfdtyewult/TensorArrayV2Stack/TensorListStackTensorListStackmfdtyewult/while:output:3Dmfdtyewult/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-mfdtyewult/TensorArrayV2Stack/TensorListStack
 mfdtyewult/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 mfdtyewult/strided_slice_3/stack
"mfdtyewult/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"mfdtyewult/strided_slice_3/stack_1
"mfdtyewult/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_3/stack_2Ü
mfdtyewult/strided_slice_3StridedSlice6mfdtyewult/TensorArrayV2Stack/TensorListStack:tensor:0)mfdtyewult/strided_slice_3/stack:output:0+mfdtyewult/strided_slice_3/stack_1:output:0+mfdtyewult/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
mfdtyewult/strided_slice_3
mfdtyewult/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mfdtyewult/transpose_1/permÑ
mfdtyewult/transpose_1	Transpose6mfdtyewult/TensorArrayV2Stack/TensorListStack:tensor:0$mfdtyewult/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/transpose_1®
 boyhyiogqf/MatMul/ReadVariableOpReadVariableOp)boyhyiogqf_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 boyhyiogqf/MatMul/ReadVariableOp±
boyhyiogqf/MatMulMatMul#mfdtyewult/strided_slice_3:output:0(boyhyiogqf/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
boyhyiogqf/MatMul­
!boyhyiogqf/BiasAdd/ReadVariableOpReadVariableOp*boyhyiogqf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!boyhyiogqf/BiasAdd/ReadVariableOp­
boyhyiogqf/BiasAddBiasAddboyhyiogqf/MatMul:product:0)boyhyiogqf/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
boyhyiogqf/BiasAddÏ
IdentityIdentityboyhyiogqf/BiasAdd:output:0"^boyhyiogqf/BiasAdd/ReadVariableOp!^boyhyiogqf/MatMul/ReadVariableOp.^fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp5^fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp-^mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp,^mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp.^mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp%^mfdtyewult/aflyrndiyz/ReadVariableOp'^mfdtyewult/aflyrndiyz/ReadVariableOp_1'^mfdtyewult/aflyrndiyz/ReadVariableOp_2^mfdtyewult/while-^oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp,^oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp.^oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp%^oohztcvkwo/ammytzqwsz/ReadVariableOp'^oohztcvkwo/ammytzqwsz/ReadVariableOp_1'^oohztcvkwo/ammytzqwsz/ReadVariableOp_2^oohztcvkwo/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2F
!boyhyiogqf/BiasAdd/ReadVariableOp!boyhyiogqf/BiasAdd/ReadVariableOp2D
 boyhyiogqf/MatMul/ReadVariableOp boyhyiogqf/MatMul/ReadVariableOp2^
-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp2l
4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp2Z
+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp2^
-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp2L
$mfdtyewult/aflyrndiyz/ReadVariableOp$mfdtyewult/aflyrndiyz/ReadVariableOp2P
&mfdtyewult/aflyrndiyz/ReadVariableOp_1&mfdtyewult/aflyrndiyz/ReadVariableOp_12P
&mfdtyewult/aflyrndiyz/ReadVariableOp_2&mfdtyewult/aflyrndiyz/ReadVariableOp_22$
mfdtyewult/whilemfdtyewult/while2\
,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp2Z
+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp2^
-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp2L
$oohztcvkwo/ammytzqwsz/ReadVariableOp$oohztcvkwo/ammytzqwsz/ReadVariableOp2P
&oohztcvkwo/ammytzqwsz/ReadVariableOp_1&oohztcvkwo/ammytzqwsz/ReadVariableOp_12P
&oohztcvkwo/ammytzqwsz/ReadVariableOp_2&oohztcvkwo/ammytzqwsz/ReadVariableOp_22$
oohztcvkwo/whileoohztcvkwo/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³F
ê
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1819037

inputs%
ammytzqwsz_1818938:	%
ammytzqwsz_1818940:	 !
ammytzqwsz_1818942:	 
ammytzqwsz_1818944:  
ammytzqwsz_1818946:  
ammytzqwsz_1818948: 
identity¢"ammytzqwsz/StatefulPartitionedCall¢whileD
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
"ammytzqwsz/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0ammytzqwsz_1818938ammytzqwsz_1818940ammytzqwsz_1818942ammytzqwsz_1818944ammytzqwsz_1818946ammytzqwsz_1818948*
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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_18189372$
"ammytzqwsz/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0ammytzqwsz_1818938ammytzqwsz_1818940ammytzqwsz_1818942ammytzqwsz_1818944ammytzqwsz_1818946ammytzqwsz_1818948*
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
while_body_1818957*
condR
while_cond_1818956*Q
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
IdentityIdentitytranspose_1:y:0#^ammytzqwsz/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"ammytzqwsz/StatefulPartitionedCall"ammytzqwsz/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

,__inference_oohztcvkwo_layer_call_fn_1823285
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18190372
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
¥
©	
(sequential_mfdtyewult_while_cond_1818742H
Dsequential_mfdtyewult_while_sequential_mfdtyewult_while_loop_counterN
Jsequential_mfdtyewult_while_sequential_mfdtyewult_while_maximum_iterations+
'sequential_mfdtyewult_while_placeholder-
)sequential_mfdtyewult_while_placeholder_1-
)sequential_mfdtyewult_while_placeholder_2-
)sequential_mfdtyewult_while_placeholder_3J
Fsequential_mfdtyewult_while_less_sequential_mfdtyewult_strided_slice_1a
]sequential_mfdtyewult_while_sequential_mfdtyewult_while_cond_1818742___redundant_placeholder0a
]sequential_mfdtyewult_while_sequential_mfdtyewult_while_cond_1818742___redundant_placeholder1a
]sequential_mfdtyewult_while_sequential_mfdtyewult_while_cond_1818742___redundant_placeholder2a
]sequential_mfdtyewult_while_sequential_mfdtyewult_while_cond_1818742___redundant_placeholder3a
]sequential_mfdtyewult_while_sequential_mfdtyewult_while_cond_1818742___redundant_placeholder4a
]sequential_mfdtyewult_while_sequential_mfdtyewult_while_cond_1818742___redundant_placeholder5a
]sequential_mfdtyewult_while_sequential_mfdtyewult_while_cond_1818742___redundant_placeholder6(
$sequential_mfdtyewult_while_identity
Þ
 sequential/mfdtyewult/while/LessLess'sequential_mfdtyewult_while_placeholderFsequential_mfdtyewult_while_less_sequential_mfdtyewult_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/mfdtyewult/while/Less
$sequential/mfdtyewult/while/IdentityIdentity$sequential/mfdtyewult/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/mfdtyewult/while/Identity"U
$sequential_mfdtyewult_while_identity-sequential/mfdtyewult/while/Identity:output:0*(
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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1819882

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
³
×
"__inference__wrapped_model_1818850

liksrhmmuxW
Asequential_fethhjgisa_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_fethhjgisa_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource:	T
Asequential_oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource:	 O
@sequential_oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource:	F
8sequential_oohztcvkwo_ammytzqwsz_readvariableop_resource: H
:sequential_oohztcvkwo_ammytzqwsz_readvariableop_1_resource: H
:sequential_oohztcvkwo_ammytzqwsz_readvariableop_2_resource: R
?sequential_mfdtyewult_aflyrndiyz_matmul_readvariableop_resource:	 T
Asequential_mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource:	 O
@sequential_mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource:	F
8sequential_mfdtyewult_aflyrndiyz_readvariableop_resource: H
:sequential_mfdtyewult_aflyrndiyz_readvariableop_1_resource: H
:sequential_mfdtyewult_aflyrndiyz_readvariableop_2_resource: F
4sequential_boyhyiogqf_matmul_readvariableop_resource: C
5sequential_boyhyiogqf_biasadd_readvariableop_resource:
identity¢,sequential/boyhyiogqf/BiasAdd/ReadVariableOp¢+sequential/boyhyiogqf/MatMul/ReadVariableOp¢8sequential/fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7sequential/mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp¢6sequential/mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp¢8sequential/mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp¢/sequential/mfdtyewult/aflyrndiyz/ReadVariableOp¢1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_1¢1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_2¢sequential/mfdtyewult/while¢7sequential/oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp¢6sequential/oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp¢8sequential/oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp¢/sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp¢1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_1¢1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_2¢sequential/oohztcvkwo/while¥
+sequential/fethhjgisa/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/fethhjgisa/conv1d/ExpandDims/dimà
'sequential/fethhjgisa/conv1d/ExpandDims
ExpandDims
liksrhmmux4sequential/fethhjgisa/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/fethhjgisa/conv1d/ExpandDimsú
8sequential/fethhjgisa/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_fethhjgisa_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/fethhjgisa/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/fethhjgisa/conv1d/ExpandDims_1/dim
)sequential/fethhjgisa/conv1d/ExpandDims_1
ExpandDims@sequential/fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/fethhjgisa/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/fethhjgisa/conv1d/ExpandDims_1¨
"sequential/fethhjgisa/conv1d/ShapeShape0sequential/fethhjgisa/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/fethhjgisa/conv1d/Shape®
0sequential/fethhjgisa/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/fethhjgisa/conv1d/strided_slice/stack»
2sequential/fethhjgisa/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/fethhjgisa/conv1d/strided_slice/stack_1²
2sequential/fethhjgisa/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/fethhjgisa/conv1d/strided_slice/stack_2
*sequential/fethhjgisa/conv1d/strided_sliceStridedSlice+sequential/fethhjgisa/conv1d/Shape:output:09sequential/fethhjgisa/conv1d/strided_slice/stack:output:0;sequential/fethhjgisa/conv1d/strided_slice/stack_1:output:0;sequential/fethhjgisa/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/fethhjgisa/conv1d/strided_slice±
*sequential/fethhjgisa/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/fethhjgisa/conv1d/Reshape/shapeø
$sequential/fethhjgisa/conv1d/ReshapeReshape0sequential/fethhjgisa/conv1d/ExpandDims:output:03sequential/fethhjgisa/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/fethhjgisa/conv1d/Reshape
#sequential/fethhjgisa/conv1d/Conv2DConv2D-sequential/fethhjgisa/conv1d/Reshape:output:02sequential/fethhjgisa/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/fethhjgisa/conv1d/Conv2D±
,sequential/fethhjgisa/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/fethhjgisa/conv1d/concat/values_1
(sequential/fethhjgisa/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/fethhjgisa/conv1d/concat/axis£
#sequential/fethhjgisa/conv1d/concatConcatV23sequential/fethhjgisa/conv1d/strided_slice:output:05sequential/fethhjgisa/conv1d/concat/values_1:output:01sequential/fethhjgisa/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/fethhjgisa/conv1d/concatõ
&sequential/fethhjgisa/conv1d/Reshape_1Reshape,sequential/fethhjgisa/conv1d/Conv2D:output:0,sequential/fethhjgisa/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/fethhjgisa/conv1d/Reshape_1â
$sequential/fethhjgisa/conv1d/SqueezeSqueeze/sequential/fethhjgisa/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/fethhjgisa/conv1d/Squeeze½
.sequential/fethhjgisa/squeeze_batch_dims/ShapeShape-sequential/fethhjgisa/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/fethhjgisa/squeeze_batch_dims/ShapeÆ
<sequential/fethhjgisa/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/fethhjgisa/squeeze_batch_dims/strided_slice/stackÓ
>sequential/fethhjgisa/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/fethhjgisa/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/fethhjgisa/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/fethhjgisa/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/fethhjgisa/squeeze_batch_dims/strided_sliceStridedSlice7sequential/fethhjgisa/squeeze_batch_dims/Shape:output:0Esequential/fethhjgisa/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/fethhjgisa/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/fethhjgisa/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/fethhjgisa/squeeze_batch_dims/strided_sliceÅ
6sequential/fethhjgisa/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/fethhjgisa/squeeze_batch_dims/Reshape/shape
0sequential/fethhjgisa/squeeze_batch_dims/ReshapeReshape-sequential/fethhjgisa/conv1d/Squeeze:output:0?sequential/fethhjgisa/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/fethhjgisa/squeeze_batch_dims/Reshape
?sequential/fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_fethhjgisa_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/fethhjgisa/squeeze_batch_dims/BiasAddBiasAdd9sequential/fethhjgisa/squeeze_batch_dims/Reshape:output:0Gsequential/fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/fethhjgisa/squeeze_batch_dims/BiasAddÅ
8sequential/fethhjgisa/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/fethhjgisa/squeeze_batch_dims/concat/values_1·
4sequential/fethhjgisa/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/fethhjgisa/squeeze_batch_dims/concat/axisß
/sequential/fethhjgisa/squeeze_batch_dims/concatConcatV2?sequential/fethhjgisa/squeeze_batch_dims/strided_slice:output:0Asequential/fethhjgisa/squeeze_batch_dims/concat/values_1:output:0=sequential/fethhjgisa/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/fethhjgisa/squeeze_batch_dims/concat¢
2sequential/fethhjgisa/squeeze_batch_dims/Reshape_1Reshape9sequential/fethhjgisa/squeeze_batch_dims/BiasAdd:output:08sequential/fethhjgisa/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/fethhjgisa/squeeze_batch_dims/Reshape_1¥
sequential/ohvvfduigw/ShapeShape;sequential/fethhjgisa/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/ohvvfduigw/Shape 
)sequential/ohvvfduigw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/ohvvfduigw/strided_slice/stack¤
+sequential/ohvvfduigw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ohvvfduigw/strided_slice/stack_1¤
+sequential/ohvvfduigw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ohvvfduigw/strided_slice/stack_2æ
#sequential/ohvvfduigw/strided_sliceStridedSlice$sequential/ohvvfduigw/Shape:output:02sequential/ohvvfduigw/strided_slice/stack:output:04sequential/ohvvfduigw/strided_slice/stack_1:output:04sequential/ohvvfduigw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/ohvvfduigw/strided_slice
%sequential/ohvvfduigw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ohvvfduigw/Reshape/shape/1
%sequential/ohvvfduigw/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ohvvfduigw/Reshape/shape/2
#sequential/ohvvfduigw/Reshape/shapePack,sequential/ohvvfduigw/strided_slice:output:0.sequential/ohvvfduigw/Reshape/shape/1:output:0.sequential/ohvvfduigw/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/ohvvfduigw/Reshape/shapeê
sequential/ohvvfduigw/ReshapeReshape;sequential/fethhjgisa/squeeze_batch_dims/Reshape_1:output:0,sequential/ohvvfduigw/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/ohvvfduigw/Reshape
sequential/oohztcvkwo/ShapeShape&sequential/ohvvfduigw/Reshape:output:0*
T0*
_output_shapes
:2
sequential/oohztcvkwo/Shape 
)sequential/oohztcvkwo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/oohztcvkwo/strided_slice/stack¤
+sequential/oohztcvkwo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/oohztcvkwo/strided_slice/stack_1¤
+sequential/oohztcvkwo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/oohztcvkwo/strided_slice/stack_2æ
#sequential/oohztcvkwo/strided_sliceStridedSlice$sequential/oohztcvkwo/Shape:output:02sequential/oohztcvkwo/strided_slice/stack:output:04sequential/oohztcvkwo/strided_slice/stack_1:output:04sequential/oohztcvkwo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/oohztcvkwo/strided_slice
!sequential/oohztcvkwo/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/oohztcvkwo/zeros/mul/yÄ
sequential/oohztcvkwo/zeros/mulMul,sequential/oohztcvkwo/strided_slice:output:0*sequential/oohztcvkwo/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/oohztcvkwo/zeros/mul
"sequential/oohztcvkwo/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/oohztcvkwo/zeros/Less/y¿
 sequential/oohztcvkwo/zeros/LessLess#sequential/oohztcvkwo/zeros/mul:z:0+sequential/oohztcvkwo/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/oohztcvkwo/zeros/Less
$sequential/oohztcvkwo/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/oohztcvkwo/zeros/packed/1Û
"sequential/oohztcvkwo/zeros/packedPack,sequential/oohztcvkwo/strided_slice:output:0-sequential/oohztcvkwo/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/oohztcvkwo/zeros/packed
!sequential/oohztcvkwo/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/oohztcvkwo/zeros/ConstÍ
sequential/oohztcvkwo/zerosFill+sequential/oohztcvkwo/zeros/packed:output:0*sequential/oohztcvkwo/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/oohztcvkwo/zeros
#sequential/oohztcvkwo/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/oohztcvkwo/zeros_1/mul/yÊ
!sequential/oohztcvkwo/zeros_1/mulMul,sequential/oohztcvkwo/strided_slice:output:0,sequential/oohztcvkwo/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/oohztcvkwo/zeros_1/mul
$sequential/oohztcvkwo/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/oohztcvkwo/zeros_1/Less/yÇ
"sequential/oohztcvkwo/zeros_1/LessLess%sequential/oohztcvkwo/zeros_1/mul:z:0-sequential/oohztcvkwo/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/oohztcvkwo/zeros_1/Less
&sequential/oohztcvkwo/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/oohztcvkwo/zeros_1/packed/1á
$sequential/oohztcvkwo/zeros_1/packedPack,sequential/oohztcvkwo/strided_slice:output:0/sequential/oohztcvkwo/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/oohztcvkwo/zeros_1/packed
#sequential/oohztcvkwo/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/oohztcvkwo/zeros_1/ConstÕ
sequential/oohztcvkwo/zeros_1Fill-sequential/oohztcvkwo/zeros_1/packed:output:0,sequential/oohztcvkwo/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/oohztcvkwo/zeros_1¡
$sequential/oohztcvkwo/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/oohztcvkwo/transpose/permÜ
sequential/oohztcvkwo/transpose	Transpose&sequential/ohvvfduigw/Reshape:output:0-sequential/oohztcvkwo/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/oohztcvkwo/transpose
sequential/oohztcvkwo/Shape_1Shape#sequential/oohztcvkwo/transpose:y:0*
T0*
_output_shapes
:2
sequential/oohztcvkwo/Shape_1¤
+sequential/oohztcvkwo/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/oohztcvkwo/strided_slice_1/stack¨
-sequential/oohztcvkwo/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/oohztcvkwo/strided_slice_1/stack_1¨
-sequential/oohztcvkwo/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/oohztcvkwo/strided_slice_1/stack_2ò
%sequential/oohztcvkwo/strided_slice_1StridedSlice&sequential/oohztcvkwo/Shape_1:output:04sequential/oohztcvkwo/strided_slice_1/stack:output:06sequential/oohztcvkwo/strided_slice_1/stack_1:output:06sequential/oohztcvkwo/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/oohztcvkwo/strided_slice_1±
1sequential/oohztcvkwo/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/oohztcvkwo/TensorArrayV2/element_shape
#sequential/oohztcvkwo/TensorArrayV2TensorListReserve:sequential/oohztcvkwo/TensorArrayV2/element_shape:output:0.sequential/oohztcvkwo/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/oohztcvkwo/TensorArrayV2ë
Ksequential/oohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/oohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/oohztcvkwo/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/oohztcvkwo/transpose:y:0Tsequential/oohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/oohztcvkwo/TensorArrayUnstack/TensorListFromTensor¤
+sequential/oohztcvkwo/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/oohztcvkwo/strided_slice_2/stack¨
-sequential/oohztcvkwo/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/oohztcvkwo/strided_slice_2/stack_1¨
-sequential/oohztcvkwo/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/oohztcvkwo/strided_slice_2/stack_2
%sequential/oohztcvkwo/strided_slice_2StridedSlice#sequential/oohztcvkwo/transpose:y:04sequential/oohztcvkwo/strided_slice_2/stack:output:06sequential/oohztcvkwo/strided_slice_2/stack_1:output:06sequential/oohztcvkwo/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/oohztcvkwo/strided_slice_2ñ
6sequential/oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp?sequential_oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOpÿ
'sequential/oohztcvkwo/ammytzqwsz/MatMulMatMul.sequential/oohztcvkwo/strided_slice_2:output:0>sequential/oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/oohztcvkwo/ammytzqwsz/MatMul÷
8sequential/oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOpAsequential_oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOpû
)sequential/oohztcvkwo/ammytzqwsz/MatMul_1MatMul$sequential/oohztcvkwo/zeros:output:0@sequential/oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/oohztcvkwo/ammytzqwsz/MatMul_1ð
$sequential/oohztcvkwo/ammytzqwsz/addAddV21sequential/oohztcvkwo/ammytzqwsz/MatMul:product:03sequential/oohztcvkwo/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/oohztcvkwo/ammytzqwsz/addð
7sequential/oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp@sequential_oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOpý
(sequential/oohztcvkwo/ammytzqwsz/BiasAddBiasAdd(sequential/oohztcvkwo/ammytzqwsz/add:z:0?sequential/oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/oohztcvkwo/ammytzqwsz/BiasAdd¦
0sequential/oohztcvkwo/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/oohztcvkwo/ammytzqwsz/split/split_dimÃ
&sequential/oohztcvkwo/ammytzqwsz/splitSplit9sequential/oohztcvkwo/ammytzqwsz/split/split_dim:output:01sequential/oohztcvkwo/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/oohztcvkwo/ammytzqwsz/split×
/sequential/oohztcvkwo/ammytzqwsz/ReadVariableOpReadVariableOp8sequential_oohztcvkwo_ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/oohztcvkwo/ammytzqwsz/ReadVariableOpæ
$sequential/oohztcvkwo/ammytzqwsz/mulMul7sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp:value:0&sequential/oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/oohztcvkwo/ammytzqwsz/mulæ
&sequential/oohztcvkwo/ammytzqwsz/add_1AddV2/sequential/oohztcvkwo/ammytzqwsz/split:output:0(sequential/oohztcvkwo/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/add_1½
(sequential/oohztcvkwo/ammytzqwsz/SigmoidSigmoid*sequential/oohztcvkwo/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/oohztcvkwo/ammytzqwsz/SigmoidÝ
1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_1ReadVariableOp:sequential_oohztcvkwo_ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_1ì
&sequential/oohztcvkwo/ammytzqwsz/mul_1Mul9sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_1:value:0&sequential/oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/mul_1è
&sequential/oohztcvkwo/ammytzqwsz/add_2AddV2/sequential/oohztcvkwo/ammytzqwsz/split:output:1*sequential/oohztcvkwo/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/add_2Á
*sequential/oohztcvkwo/ammytzqwsz/Sigmoid_1Sigmoid*sequential/oohztcvkwo/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/oohztcvkwo/ammytzqwsz/Sigmoid_1á
&sequential/oohztcvkwo/ammytzqwsz/mul_2Mul.sequential/oohztcvkwo/ammytzqwsz/Sigmoid_1:y:0&sequential/oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/mul_2¹
%sequential/oohztcvkwo/ammytzqwsz/TanhTanh/sequential/oohztcvkwo/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/oohztcvkwo/ammytzqwsz/Tanhâ
&sequential/oohztcvkwo/ammytzqwsz/mul_3Mul,sequential/oohztcvkwo/ammytzqwsz/Sigmoid:y:0)sequential/oohztcvkwo/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/mul_3ã
&sequential/oohztcvkwo/ammytzqwsz/add_3AddV2*sequential/oohztcvkwo/ammytzqwsz/mul_2:z:0*sequential/oohztcvkwo/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/add_3Ý
1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_2ReadVariableOp:sequential_oohztcvkwo_ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_2ð
&sequential/oohztcvkwo/ammytzqwsz/mul_4Mul9sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_2:value:0*sequential/oohztcvkwo/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/mul_4è
&sequential/oohztcvkwo/ammytzqwsz/add_4AddV2/sequential/oohztcvkwo/ammytzqwsz/split:output:3*sequential/oohztcvkwo/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/add_4Á
*sequential/oohztcvkwo/ammytzqwsz/Sigmoid_2Sigmoid*sequential/oohztcvkwo/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/oohztcvkwo/ammytzqwsz/Sigmoid_2¸
'sequential/oohztcvkwo/ammytzqwsz/Tanh_1Tanh*sequential/oohztcvkwo/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/oohztcvkwo/ammytzqwsz/Tanh_1æ
&sequential/oohztcvkwo/ammytzqwsz/mul_5Mul.sequential/oohztcvkwo/ammytzqwsz/Sigmoid_2:y:0+sequential/oohztcvkwo/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/oohztcvkwo/ammytzqwsz/mul_5»
3sequential/oohztcvkwo/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/oohztcvkwo/TensorArrayV2_1/element_shape
%sequential/oohztcvkwo/TensorArrayV2_1TensorListReserve<sequential/oohztcvkwo/TensorArrayV2_1/element_shape:output:0.sequential/oohztcvkwo/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/oohztcvkwo/TensorArrayV2_1z
sequential/oohztcvkwo/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/oohztcvkwo/time«
.sequential/oohztcvkwo/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/oohztcvkwo/while/maximum_iterations
(sequential/oohztcvkwo/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/oohztcvkwo/while/loop_counterø	
sequential/oohztcvkwo/whileWhile1sequential/oohztcvkwo/while/loop_counter:output:07sequential/oohztcvkwo/while/maximum_iterations:output:0#sequential/oohztcvkwo/time:output:0.sequential/oohztcvkwo/TensorArrayV2_1:handle:0$sequential/oohztcvkwo/zeros:output:0&sequential/oohztcvkwo/zeros_1:output:0.sequential/oohztcvkwo/strided_slice_1:output:0Msequential/oohztcvkwo/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_oohztcvkwo_ammytzqwsz_matmul_readvariableop_resourceAsequential_oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource@sequential_oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource8sequential_oohztcvkwo_ammytzqwsz_readvariableop_resource:sequential_oohztcvkwo_ammytzqwsz_readvariableop_1_resource:sequential_oohztcvkwo_ammytzqwsz_readvariableop_2_resource*
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
(sequential_oohztcvkwo_while_body_1818567*4
cond,R*
(sequential_oohztcvkwo_while_cond_1818566*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/oohztcvkwo/whileá
Fsequential/oohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/oohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/oohztcvkwo/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/oohztcvkwo/while:output:3Osequential/oohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/oohztcvkwo/TensorArrayV2Stack/TensorListStack­
+sequential/oohztcvkwo/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/oohztcvkwo/strided_slice_3/stack¨
-sequential/oohztcvkwo/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/oohztcvkwo/strided_slice_3/stack_1¨
-sequential/oohztcvkwo/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/oohztcvkwo/strided_slice_3/stack_2
%sequential/oohztcvkwo/strided_slice_3StridedSliceAsequential/oohztcvkwo/TensorArrayV2Stack/TensorListStack:tensor:04sequential/oohztcvkwo/strided_slice_3/stack:output:06sequential/oohztcvkwo/strided_slice_3/stack_1:output:06sequential/oohztcvkwo/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/oohztcvkwo/strided_slice_3¥
&sequential/oohztcvkwo/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/oohztcvkwo/transpose_1/permý
!sequential/oohztcvkwo/transpose_1	TransposeAsequential/oohztcvkwo/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/oohztcvkwo/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/oohztcvkwo/transpose_1
sequential/mfdtyewult/ShapeShape%sequential/oohztcvkwo/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/mfdtyewult/Shape 
)sequential/mfdtyewult/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/mfdtyewult/strided_slice/stack¤
+sequential/mfdtyewult/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/mfdtyewult/strided_slice/stack_1¤
+sequential/mfdtyewult/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/mfdtyewult/strided_slice/stack_2æ
#sequential/mfdtyewult/strided_sliceStridedSlice$sequential/mfdtyewult/Shape:output:02sequential/mfdtyewult/strided_slice/stack:output:04sequential/mfdtyewult/strided_slice/stack_1:output:04sequential/mfdtyewult/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/mfdtyewult/strided_slice
!sequential/mfdtyewult/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/mfdtyewult/zeros/mul/yÄ
sequential/mfdtyewult/zeros/mulMul,sequential/mfdtyewult/strided_slice:output:0*sequential/mfdtyewult/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/mfdtyewult/zeros/mul
"sequential/mfdtyewult/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/mfdtyewult/zeros/Less/y¿
 sequential/mfdtyewult/zeros/LessLess#sequential/mfdtyewult/zeros/mul:z:0+sequential/mfdtyewult/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/mfdtyewult/zeros/Less
$sequential/mfdtyewult/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/mfdtyewult/zeros/packed/1Û
"sequential/mfdtyewult/zeros/packedPack,sequential/mfdtyewult/strided_slice:output:0-sequential/mfdtyewult/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/mfdtyewult/zeros/packed
!sequential/mfdtyewult/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/mfdtyewult/zeros/ConstÍ
sequential/mfdtyewult/zerosFill+sequential/mfdtyewult/zeros/packed:output:0*sequential/mfdtyewult/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/mfdtyewult/zeros
#sequential/mfdtyewult/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/mfdtyewult/zeros_1/mul/yÊ
!sequential/mfdtyewult/zeros_1/mulMul,sequential/mfdtyewult/strided_slice:output:0,sequential/mfdtyewult/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/mfdtyewult/zeros_1/mul
$sequential/mfdtyewult/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/mfdtyewult/zeros_1/Less/yÇ
"sequential/mfdtyewult/zeros_1/LessLess%sequential/mfdtyewult/zeros_1/mul:z:0-sequential/mfdtyewult/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/mfdtyewult/zeros_1/Less
&sequential/mfdtyewult/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/mfdtyewult/zeros_1/packed/1á
$sequential/mfdtyewult/zeros_1/packedPack,sequential/mfdtyewult/strided_slice:output:0/sequential/mfdtyewult/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/mfdtyewult/zeros_1/packed
#sequential/mfdtyewult/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/mfdtyewult/zeros_1/ConstÕ
sequential/mfdtyewult/zeros_1Fill-sequential/mfdtyewult/zeros_1/packed:output:0,sequential/mfdtyewult/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/mfdtyewult/zeros_1¡
$sequential/mfdtyewult/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/mfdtyewult/transpose/permÛ
sequential/mfdtyewult/transpose	Transpose%sequential/oohztcvkwo/transpose_1:y:0-sequential/mfdtyewult/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/mfdtyewult/transpose
sequential/mfdtyewult/Shape_1Shape#sequential/mfdtyewult/transpose:y:0*
T0*
_output_shapes
:2
sequential/mfdtyewult/Shape_1¤
+sequential/mfdtyewult/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/mfdtyewult/strided_slice_1/stack¨
-sequential/mfdtyewult/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/mfdtyewult/strided_slice_1/stack_1¨
-sequential/mfdtyewult/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/mfdtyewult/strided_slice_1/stack_2ò
%sequential/mfdtyewult/strided_slice_1StridedSlice&sequential/mfdtyewult/Shape_1:output:04sequential/mfdtyewult/strided_slice_1/stack:output:06sequential/mfdtyewult/strided_slice_1/stack_1:output:06sequential/mfdtyewult/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/mfdtyewult/strided_slice_1±
1sequential/mfdtyewult/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/mfdtyewult/TensorArrayV2/element_shape
#sequential/mfdtyewult/TensorArrayV2TensorListReserve:sequential/mfdtyewult/TensorArrayV2/element_shape:output:0.sequential/mfdtyewult/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/mfdtyewult/TensorArrayV2ë
Ksequential/mfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/mfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/mfdtyewult/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/mfdtyewult/transpose:y:0Tsequential/mfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/mfdtyewult/TensorArrayUnstack/TensorListFromTensor¤
+sequential/mfdtyewult/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/mfdtyewult/strided_slice_2/stack¨
-sequential/mfdtyewult/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/mfdtyewult/strided_slice_2/stack_1¨
-sequential/mfdtyewult/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/mfdtyewult/strided_slice_2/stack_2
%sequential/mfdtyewult/strided_slice_2StridedSlice#sequential/mfdtyewult/transpose:y:04sequential/mfdtyewult/strided_slice_2/stack:output:06sequential/mfdtyewult/strided_slice_2/stack_1:output:06sequential/mfdtyewult/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/mfdtyewult/strided_slice_2ñ
6sequential/mfdtyewult/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp?sequential_mfdtyewult_aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/mfdtyewult/aflyrndiyz/MatMul/ReadVariableOpÿ
'sequential/mfdtyewult/aflyrndiyz/MatMulMatMul.sequential/mfdtyewult/strided_slice_2:output:0>sequential/mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/mfdtyewult/aflyrndiyz/MatMul÷
8sequential/mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOpAsequential_mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOpû
)sequential/mfdtyewult/aflyrndiyz/MatMul_1MatMul$sequential/mfdtyewult/zeros:output:0@sequential/mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/mfdtyewult/aflyrndiyz/MatMul_1ð
$sequential/mfdtyewult/aflyrndiyz/addAddV21sequential/mfdtyewult/aflyrndiyz/MatMul:product:03sequential/mfdtyewult/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/mfdtyewult/aflyrndiyz/addð
7sequential/mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp@sequential_mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOpý
(sequential/mfdtyewult/aflyrndiyz/BiasAddBiasAdd(sequential/mfdtyewult/aflyrndiyz/add:z:0?sequential/mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/mfdtyewult/aflyrndiyz/BiasAdd¦
0sequential/mfdtyewult/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/mfdtyewult/aflyrndiyz/split/split_dimÃ
&sequential/mfdtyewult/aflyrndiyz/splitSplit9sequential/mfdtyewult/aflyrndiyz/split/split_dim:output:01sequential/mfdtyewult/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/mfdtyewult/aflyrndiyz/split×
/sequential/mfdtyewult/aflyrndiyz/ReadVariableOpReadVariableOp8sequential_mfdtyewult_aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/mfdtyewult/aflyrndiyz/ReadVariableOpæ
$sequential/mfdtyewult/aflyrndiyz/mulMul7sequential/mfdtyewult/aflyrndiyz/ReadVariableOp:value:0&sequential/mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/mfdtyewult/aflyrndiyz/mulæ
&sequential/mfdtyewult/aflyrndiyz/add_1AddV2/sequential/mfdtyewult/aflyrndiyz/split:output:0(sequential/mfdtyewult/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/add_1½
(sequential/mfdtyewult/aflyrndiyz/SigmoidSigmoid*sequential/mfdtyewult/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/mfdtyewult/aflyrndiyz/SigmoidÝ
1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_1ReadVariableOp:sequential_mfdtyewult_aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_1ì
&sequential/mfdtyewult/aflyrndiyz/mul_1Mul9sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_1:value:0&sequential/mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/mul_1è
&sequential/mfdtyewult/aflyrndiyz/add_2AddV2/sequential/mfdtyewult/aflyrndiyz/split:output:1*sequential/mfdtyewult/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/add_2Á
*sequential/mfdtyewult/aflyrndiyz/Sigmoid_1Sigmoid*sequential/mfdtyewult/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/mfdtyewult/aflyrndiyz/Sigmoid_1á
&sequential/mfdtyewult/aflyrndiyz/mul_2Mul.sequential/mfdtyewult/aflyrndiyz/Sigmoid_1:y:0&sequential/mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/mul_2¹
%sequential/mfdtyewult/aflyrndiyz/TanhTanh/sequential/mfdtyewult/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/mfdtyewult/aflyrndiyz/Tanhâ
&sequential/mfdtyewult/aflyrndiyz/mul_3Mul,sequential/mfdtyewult/aflyrndiyz/Sigmoid:y:0)sequential/mfdtyewult/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/mul_3ã
&sequential/mfdtyewult/aflyrndiyz/add_3AddV2*sequential/mfdtyewult/aflyrndiyz/mul_2:z:0*sequential/mfdtyewult/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/add_3Ý
1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_2ReadVariableOp:sequential_mfdtyewult_aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_2ð
&sequential/mfdtyewult/aflyrndiyz/mul_4Mul9sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_2:value:0*sequential/mfdtyewult/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/mul_4è
&sequential/mfdtyewult/aflyrndiyz/add_4AddV2/sequential/mfdtyewult/aflyrndiyz/split:output:3*sequential/mfdtyewult/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/add_4Á
*sequential/mfdtyewult/aflyrndiyz/Sigmoid_2Sigmoid*sequential/mfdtyewult/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/mfdtyewult/aflyrndiyz/Sigmoid_2¸
'sequential/mfdtyewult/aflyrndiyz/Tanh_1Tanh*sequential/mfdtyewult/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/mfdtyewult/aflyrndiyz/Tanh_1æ
&sequential/mfdtyewult/aflyrndiyz/mul_5Mul.sequential/mfdtyewult/aflyrndiyz/Sigmoid_2:y:0+sequential/mfdtyewult/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/mfdtyewult/aflyrndiyz/mul_5»
3sequential/mfdtyewult/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/mfdtyewult/TensorArrayV2_1/element_shape
%sequential/mfdtyewult/TensorArrayV2_1TensorListReserve<sequential/mfdtyewult/TensorArrayV2_1/element_shape:output:0.sequential/mfdtyewult/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/mfdtyewult/TensorArrayV2_1z
sequential/mfdtyewult/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/mfdtyewult/time«
.sequential/mfdtyewult/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/mfdtyewult/while/maximum_iterations
(sequential/mfdtyewult/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/mfdtyewult/while/loop_counterø	
sequential/mfdtyewult/whileWhile1sequential/mfdtyewult/while/loop_counter:output:07sequential/mfdtyewult/while/maximum_iterations:output:0#sequential/mfdtyewult/time:output:0.sequential/mfdtyewult/TensorArrayV2_1:handle:0$sequential/mfdtyewult/zeros:output:0&sequential/mfdtyewult/zeros_1:output:0.sequential/mfdtyewult/strided_slice_1:output:0Msequential/mfdtyewult/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_mfdtyewult_aflyrndiyz_matmul_readvariableop_resourceAsequential_mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource@sequential_mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource8sequential_mfdtyewult_aflyrndiyz_readvariableop_resource:sequential_mfdtyewult_aflyrndiyz_readvariableop_1_resource:sequential_mfdtyewult_aflyrndiyz_readvariableop_2_resource*
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
(sequential_mfdtyewult_while_body_1818743*4
cond,R*
(sequential_mfdtyewult_while_cond_1818742*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/mfdtyewult/whileá
Fsequential/mfdtyewult/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/mfdtyewult/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/mfdtyewult/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/mfdtyewult/while:output:3Osequential/mfdtyewult/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/mfdtyewult/TensorArrayV2Stack/TensorListStack­
+sequential/mfdtyewult/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/mfdtyewult/strided_slice_3/stack¨
-sequential/mfdtyewult/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/mfdtyewult/strided_slice_3/stack_1¨
-sequential/mfdtyewult/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/mfdtyewult/strided_slice_3/stack_2
%sequential/mfdtyewult/strided_slice_3StridedSliceAsequential/mfdtyewult/TensorArrayV2Stack/TensorListStack:tensor:04sequential/mfdtyewult/strided_slice_3/stack:output:06sequential/mfdtyewult/strided_slice_3/stack_1:output:06sequential/mfdtyewult/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/mfdtyewult/strided_slice_3¥
&sequential/mfdtyewult/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/mfdtyewult/transpose_1/permý
!sequential/mfdtyewult/transpose_1	TransposeAsequential/mfdtyewult/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/mfdtyewult/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/mfdtyewult/transpose_1Ï
+sequential/boyhyiogqf/MatMul/ReadVariableOpReadVariableOp4sequential_boyhyiogqf_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/boyhyiogqf/MatMul/ReadVariableOpÝ
sequential/boyhyiogqf/MatMulMatMul.sequential/mfdtyewult/strided_slice_3:output:03sequential/boyhyiogqf/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/boyhyiogqf/MatMulÎ
,sequential/boyhyiogqf/BiasAdd/ReadVariableOpReadVariableOp5sequential_boyhyiogqf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/boyhyiogqf/BiasAdd/ReadVariableOpÙ
sequential/boyhyiogqf/BiasAddBiasAdd&sequential/boyhyiogqf/MatMul:product:04sequential/boyhyiogqf/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/boyhyiogqf/BiasAdd 
IdentityIdentity&sequential/boyhyiogqf/BiasAdd:output:0-^sequential/boyhyiogqf/BiasAdd/ReadVariableOp,^sequential/boyhyiogqf/MatMul/ReadVariableOp9^sequential/fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp@^sequential/fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp8^sequential/mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp7^sequential/mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp9^sequential/mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp0^sequential/mfdtyewult/aflyrndiyz/ReadVariableOp2^sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_12^sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_2^sequential/mfdtyewult/while8^sequential/oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp7^sequential/oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp9^sequential/oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp0^sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp2^sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_12^sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_2^sequential/oohztcvkwo/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,sequential/boyhyiogqf/BiasAdd/ReadVariableOp,sequential/boyhyiogqf/BiasAdd/ReadVariableOp2Z
+sequential/boyhyiogqf/MatMul/ReadVariableOp+sequential/boyhyiogqf/MatMul/ReadVariableOp2t
8sequential/fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp8sequential/fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7sequential/mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp7sequential/mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp2p
6sequential/mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp6sequential/mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp2t
8sequential/mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp8sequential/mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp2b
/sequential/mfdtyewult/aflyrndiyz/ReadVariableOp/sequential/mfdtyewult/aflyrndiyz/ReadVariableOp2f
1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_11sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_12f
1sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_21sequential/mfdtyewult/aflyrndiyz/ReadVariableOp_22:
sequential/mfdtyewult/whilesequential/mfdtyewult/while2r
7sequential/oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp7sequential/oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp2p
6sequential/oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp6sequential/oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp2t
8sequential/oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp8sequential/oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp2b
/sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp/sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp2f
1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_11sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_12f
1sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_21sequential/oohztcvkwo/ammytzqwsz/ReadVariableOp_22:
sequential/oohztcvkwo/whilesequential/oohztcvkwo/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
liksrhmmux
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_1821557

liksrhmmux(
fethhjgisa_1821519: 
fethhjgisa_1821521:%
oohztcvkwo_1821525:	%
oohztcvkwo_1821527:	 !
oohztcvkwo_1821529:	 
oohztcvkwo_1821531:  
oohztcvkwo_1821533:  
oohztcvkwo_1821535: %
mfdtyewult_1821538:	 %
mfdtyewult_1821540:	 !
mfdtyewult_1821542:	 
mfdtyewult_1821544:  
mfdtyewult_1821546:  
mfdtyewult_1821548: $
boyhyiogqf_1821551:  
boyhyiogqf_1821553:
identity¢"boyhyiogqf/StatefulPartitionedCall¢"fethhjgisa/StatefulPartitionedCall¢"mfdtyewult/StatefulPartitionedCall¢"oohztcvkwo/StatefulPartitionedCall°
"fethhjgisa/StatefulPartitionedCallStatefulPartitionedCall
liksrhmmuxfethhjgisa_1821519fethhjgisa_1821521*
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
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_18204102$
"fethhjgisa/StatefulPartitionedCall
ohvvfduigw/PartitionedCallPartitionedCall+fethhjgisa/StatefulPartitionedCall:output:0*
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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_18204292
ohvvfduigw/PartitionedCall
"oohztcvkwo/StatefulPartitionedCallStatefulPartitionedCall#ohvvfduigw/PartitionedCall:output:0oohztcvkwo_1821525oohztcvkwo_1821527oohztcvkwo_1821529oohztcvkwo_1821531oohztcvkwo_1821533oohztcvkwo_1821535*
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18212922$
"oohztcvkwo/StatefulPartitionedCall¡
"mfdtyewult/StatefulPartitionedCallStatefulPartitionedCall+oohztcvkwo/StatefulPartitionedCall:output:0mfdtyewult_1821538mfdtyewult_1821540mfdtyewult_1821542mfdtyewult_1821544mfdtyewult_1821546mfdtyewult_1821548*
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18210782$
"mfdtyewult/StatefulPartitionedCallÉ
"boyhyiogqf/StatefulPartitionedCallStatefulPartitionedCall+mfdtyewult/StatefulPartitionedCall:output:0boyhyiogqf_1821551boyhyiogqf_1821553*
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
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_18208272$
"boyhyiogqf/StatefulPartitionedCall
IdentityIdentity+boyhyiogqf/StatefulPartitionedCall:output:0#^boyhyiogqf/StatefulPartitionedCall#^fethhjgisa/StatefulPartitionedCall#^mfdtyewult/StatefulPartitionedCall#^oohztcvkwo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"boyhyiogqf/StatefulPartitionedCall"boyhyiogqf/StatefulPartitionedCall2H
"fethhjgisa/StatefulPartitionedCall"fethhjgisa/StatefulPartitionedCall2H
"mfdtyewult/StatefulPartitionedCall"mfdtyewult/StatefulPartitionedCall2H
"oohztcvkwo/StatefulPartitionedCall"oohztcvkwo/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
liksrhmmux
àh

G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1822728
inputs_0<
)ammytzqwsz_matmul_readvariableop_resource:	>
+ammytzqwsz_matmul_1_readvariableop_resource:	 9
*ammytzqwsz_biasadd_readvariableop_resource:	0
"ammytzqwsz_readvariableop_resource: 2
$ammytzqwsz_readvariableop_1_resource: 2
$ammytzqwsz_readvariableop_2_resource: 
identity¢!ammytzqwsz/BiasAdd/ReadVariableOp¢ ammytzqwsz/MatMul/ReadVariableOp¢"ammytzqwsz/MatMul_1/ReadVariableOp¢ammytzqwsz/ReadVariableOp¢ammytzqwsz/ReadVariableOp_1¢ammytzqwsz/ReadVariableOp_2¢whileF
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
 ammytzqwsz/MatMul/ReadVariableOpReadVariableOp)ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ammytzqwsz/MatMul/ReadVariableOp§
ammytzqwsz/MatMulMatMulstrided_slice_2:output:0(ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMulµ
"ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp+ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ammytzqwsz/MatMul_1/ReadVariableOp£
ammytzqwsz/MatMul_1MatMulzeros:output:0*ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMul_1
ammytzqwsz/addAddV2ammytzqwsz/MatMul:product:0ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/add®
!ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp*ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ammytzqwsz/BiasAdd/ReadVariableOp¥
ammytzqwsz/BiasAddBiasAddammytzqwsz/add:z:0)ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/BiasAddz
ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ammytzqwsz/split/split_dimë
ammytzqwsz/splitSplit#ammytzqwsz/split/split_dim:output:0ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ammytzqwsz/split
ammytzqwsz/ReadVariableOpReadVariableOp"ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp
ammytzqwsz/mulMul!ammytzqwsz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul
ammytzqwsz/add_1AddV2ammytzqwsz/split:output:0ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_1{
ammytzqwsz/SigmoidSigmoidammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid
ammytzqwsz/ReadVariableOp_1ReadVariableOp$ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_1
ammytzqwsz/mul_1Mul#ammytzqwsz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_1
ammytzqwsz/add_2AddV2ammytzqwsz/split:output:1ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_2
ammytzqwsz/Sigmoid_1Sigmoidammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_1
ammytzqwsz/mul_2Mulammytzqwsz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_2w
ammytzqwsz/TanhTanhammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh
ammytzqwsz/mul_3Mulammytzqwsz/Sigmoid:y:0ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_3
ammytzqwsz/add_3AddV2ammytzqwsz/mul_2:z:0ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_3
ammytzqwsz/ReadVariableOp_2ReadVariableOp$ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_2
ammytzqwsz/mul_4Mul#ammytzqwsz/ReadVariableOp_2:value:0ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_4
ammytzqwsz/add_4AddV2ammytzqwsz/split:output:3ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_4
ammytzqwsz/Sigmoid_2Sigmoidammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_2v
ammytzqwsz/Tanh_1Tanhammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh_1
ammytzqwsz/mul_5Mulammytzqwsz/Sigmoid_2:y:0ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ammytzqwsz_matmul_readvariableop_resource+ammytzqwsz_matmul_1_readvariableop_resource*ammytzqwsz_biasadd_readvariableop_resource"ammytzqwsz_readvariableop_resource$ammytzqwsz_readvariableop_1_resource$ammytzqwsz_readvariableop_2_resource*
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
while_body_1822627*
condR
while_cond_1822626*Q
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
IdentityIdentitytranspose_1:y:0"^ammytzqwsz/BiasAdd/ReadVariableOp!^ammytzqwsz/MatMul/ReadVariableOp#^ammytzqwsz/MatMul_1/ReadVariableOp^ammytzqwsz/ReadVariableOp^ammytzqwsz/ReadVariableOp_1^ammytzqwsz/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ammytzqwsz/BiasAdd/ReadVariableOp!ammytzqwsz/BiasAdd/ReadVariableOp2D
 ammytzqwsz/MatMul/ReadVariableOp ammytzqwsz/MatMul/ReadVariableOp2H
"ammytzqwsz/MatMul_1/ReadVariableOp"ammytzqwsz/MatMul_1/ReadVariableOp26
ammytzqwsz/ReadVariableOpammytzqwsz/ReadVariableOp2:
ammytzqwsz/ReadVariableOp_1ammytzqwsz/ReadVariableOp_12:
ammytzqwsz/ReadVariableOp_2ammytzqwsz/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ç)
Ò
while_body_1819978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_aflyrndiyz_1820002_0:	 -
while_aflyrndiyz_1820004_0:	 )
while_aflyrndiyz_1820006_0:	(
while_aflyrndiyz_1820008_0: (
while_aflyrndiyz_1820010_0: (
while_aflyrndiyz_1820012_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_aflyrndiyz_1820002:	 +
while_aflyrndiyz_1820004:	 '
while_aflyrndiyz_1820006:	&
while_aflyrndiyz_1820008: &
while_aflyrndiyz_1820010: &
while_aflyrndiyz_1820012: ¢(while/aflyrndiyz/StatefulPartitionedCallÃ
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
(while/aflyrndiyz/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_aflyrndiyz_1820002_0while_aflyrndiyz_1820004_0while_aflyrndiyz_1820006_0while_aflyrndiyz_1820008_0while_aflyrndiyz_1820010_0while_aflyrndiyz_1820012_0*
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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_18198822*
(while/aflyrndiyz/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/aflyrndiyz/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/aflyrndiyz/StatefulPartitionedCall:output:1)^while/aflyrndiyz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/aflyrndiyz/StatefulPartitionedCall:output:2)^while/aflyrndiyz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_aflyrndiyz_1820002while_aflyrndiyz_1820002_0"6
while_aflyrndiyz_1820004while_aflyrndiyz_1820004_0"6
while_aflyrndiyz_1820006while_aflyrndiyz_1820006_0"6
while_aflyrndiyz_1820008while_aflyrndiyz_1820008_0"6
while_aflyrndiyz_1820010while_aflyrndiyz_1820010_0"6
while_aflyrndiyz_1820012while_aflyrndiyz_1820012_0")
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
(while/aflyrndiyz/StatefulPartitionedCall(while/aflyrndiyz/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_sequential_layer_call_fn_1820869

liksrhmmux
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
liksrhmmuxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_18208342
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
liksrhmmux


oohztcvkwo_while_cond_18221262
.oohztcvkwo_while_oohztcvkwo_while_loop_counter8
4oohztcvkwo_while_oohztcvkwo_while_maximum_iterations 
oohztcvkwo_while_placeholder"
oohztcvkwo_while_placeholder_1"
oohztcvkwo_while_placeholder_2"
oohztcvkwo_while_placeholder_34
0oohztcvkwo_while_less_oohztcvkwo_strided_slice_1K
Goohztcvkwo_while_oohztcvkwo_while_cond_1822126___redundant_placeholder0K
Goohztcvkwo_while_oohztcvkwo_while_cond_1822126___redundant_placeholder1K
Goohztcvkwo_while_oohztcvkwo_while_cond_1822126___redundant_placeholder2K
Goohztcvkwo_while_oohztcvkwo_while_cond_1822126___redundant_placeholder3K
Goohztcvkwo_while_oohztcvkwo_while_cond_1822126___redundant_placeholder4K
Goohztcvkwo_while_oohztcvkwo_while_cond_1822126___redundant_placeholder5K
Goohztcvkwo_while_oohztcvkwo_while_cond_1822126___redundant_placeholder6
oohztcvkwo_while_identity
§
oohztcvkwo/while/LessLessoohztcvkwo_while_placeholder0oohztcvkwo_while_less_oohztcvkwo_strided_slice_1*
T0*
_output_shapes
: 2
oohztcvkwo/while/Less~
oohztcvkwo/while/IdentityIdentityoohztcvkwo/while/Less:z:0*
T0
*
_output_shapes
: 2
oohztcvkwo/while/Identity"?
oohztcvkwo_while_identity"oohztcvkwo/while/Identity:output:0*(
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
oohztcvkwo_while_body_18221272
.oohztcvkwo_while_oohztcvkwo_while_loop_counter8
4oohztcvkwo_while_oohztcvkwo_while_maximum_iterations 
oohztcvkwo_while_placeholder"
oohztcvkwo_while_placeholder_1"
oohztcvkwo_while_placeholder_2"
oohztcvkwo_while_placeholder_31
-oohztcvkwo_while_oohztcvkwo_strided_slice_1_0m
ioohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0O
<oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0:	Q
>oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 L
=oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0:	C
5oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0: E
7oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0: E
7oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0: 
oohztcvkwo_while_identity
oohztcvkwo_while_identity_1
oohztcvkwo_while_identity_2
oohztcvkwo_while_identity_3
oohztcvkwo_while_identity_4
oohztcvkwo_while_identity_5/
+oohztcvkwo_while_oohztcvkwo_strided_slice_1k
goohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensorM
:oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource:	O
<oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource:	 J
;oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource:	A
3oohztcvkwo_while_ammytzqwsz_readvariableop_resource: C
5oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource: C
5oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource: ¢2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp¢1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp¢3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp¢*oohztcvkwo/while/ammytzqwsz/ReadVariableOp¢,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1¢,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2Ù
Boohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Boohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shape
4oohztcvkwo/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemioohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0oohztcvkwo_while_placeholderKoohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4oohztcvkwo/while/TensorArrayV2Read/TensorListGetItemä
1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp<oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOpý
"oohztcvkwo/while/ammytzqwsz/MatMulMatMul;oohztcvkwo/while/TensorArrayV2Read/TensorListGetItem:item:09oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"oohztcvkwo/while/ammytzqwsz/MatMulê
3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp>oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOpæ
$oohztcvkwo/while/ammytzqwsz/MatMul_1MatMuloohztcvkwo_while_placeholder_2;oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$oohztcvkwo/while/ammytzqwsz/MatMul_1Ü
oohztcvkwo/while/ammytzqwsz/addAddV2,oohztcvkwo/while/ammytzqwsz/MatMul:product:0.oohztcvkwo/while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
oohztcvkwo/while/ammytzqwsz/addã
2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp=oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOpé
#oohztcvkwo/while/ammytzqwsz/BiasAddBiasAdd#oohztcvkwo/while/ammytzqwsz/add:z:0:oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#oohztcvkwo/while/ammytzqwsz/BiasAdd
+oohztcvkwo/while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+oohztcvkwo/while/ammytzqwsz/split/split_dim¯
!oohztcvkwo/while/ammytzqwsz/splitSplit4oohztcvkwo/while/ammytzqwsz/split/split_dim:output:0,oohztcvkwo/while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!oohztcvkwo/while/ammytzqwsz/splitÊ
*oohztcvkwo/while/ammytzqwsz/ReadVariableOpReadVariableOp5oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*oohztcvkwo/while/ammytzqwsz/ReadVariableOpÏ
oohztcvkwo/while/ammytzqwsz/mulMul2oohztcvkwo/while/ammytzqwsz/ReadVariableOp:value:0oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
oohztcvkwo/while/ammytzqwsz/mulÒ
!oohztcvkwo/while/ammytzqwsz/add_1AddV2*oohztcvkwo/while/ammytzqwsz/split:output:0#oohztcvkwo/while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_1®
#oohztcvkwo/while/ammytzqwsz/SigmoidSigmoid%oohztcvkwo/while/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#oohztcvkwo/while/ammytzqwsz/SigmoidÐ
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1ReadVariableOp7oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1Õ
!oohztcvkwo/while/ammytzqwsz/mul_1Mul4oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1:value:0oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_1Ô
!oohztcvkwo/while/ammytzqwsz/add_2AddV2*oohztcvkwo/while/ammytzqwsz/split:output:1%oohztcvkwo/while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_2²
%oohztcvkwo/while/ammytzqwsz/Sigmoid_1Sigmoid%oohztcvkwo/while/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%oohztcvkwo/while/ammytzqwsz/Sigmoid_1Ê
!oohztcvkwo/while/ammytzqwsz/mul_2Mul)oohztcvkwo/while/ammytzqwsz/Sigmoid_1:y:0oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_2ª
 oohztcvkwo/while/ammytzqwsz/TanhTanh*oohztcvkwo/while/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 oohztcvkwo/while/ammytzqwsz/TanhÎ
!oohztcvkwo/while/ammytzqwsz/mul_3Mul'oohztcvkwo/while/ammytzqwsz/Sigmoid:y:0$oohztcvkwo/while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_3Ï
!oohztcvkwo/while/ammytzqwsz/add_3AddV2%oohztcvkwo/while/ammytzqwsz/mul_2:z:0%oohztcvkwo/while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_3Ð
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2ReadVariableOp7oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2Ü
!oohztcvkwo/while/ammytzqwsz/mul_4Mul4oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2:value:0%oohztcvkwo/while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_4Ô
!oohztcvkwo/while/ammytzqwsz/add_4AddV2*oohztcvkwo/while/ammytzqwsz/split:output:3%oohztcvkwo/while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_4²
%oohztcvkwo/while/ammytzqwsz/Sigmoid_2Sigmoid%oohztcvkwo/while/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%oohztcvkwo/while/ammytzqwsz/Sigmoid_2©
"oohztcvkwo/while/ammytzqwsz/Tanh_1Tanh%oohztcvkwo/while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"oohztcvkwo/while/ammytzqwsz/Tanh_1Ò
!oohztcvkwo/while/ammytzqwsz/mul_5Mul)oohztcvkwo/while/ammytzqwsz/Sigmoid_2:y:0&oohztcvkwo/while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_5
5oohztcvkwo/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemoohztcvkwo_while_placeholder_1oohztcvkwo_while_placeholder%oohztcvkwo/while/ammytzqwsz/mul_5:z:0*
_output_shapes
: *
element_dtype027
5oohztcvkwo/while/TensorArrayV2Write/TensorListSetItemr
oohztcvkwo/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
oohztcvkwo/while/add/y
oohztcvkwo/while/addAddV2oohztcvkwo_while_placeholderoohztcvkwo/while/add/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/while/addv
oohztcvkwo/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
oohztcvkwo/while/add_1/y­
oohztcvkwo/while/add_1AddV2.oohztcvkwo_while_oohztcvkwo_while_loop_counter!oohztcvkwo/while/add_1/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/while/add_1©
oohztcvkwo/while/IdentityIdentityoohztcvkwo/while/add_1:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/IdentityÇ
oohztcvkwo/while/Identity_1Identity4oohztcvkwo_while_oohztcvkwo_while_maximum_iterations3^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/Identity_1«
oohztcvkwo/while/Identity_2Identityoohztcvkwo/while/add:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/Identity_2Ø
oohztcvkwo/while/Identity_3IdentityEoohztcvkwo/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/Identity_3É
oohztcvkwo/while/Identity_4Identity%oohztcvkwo/while/ammytzqwsz/mul_5:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/while/Identity_4É
oohztcvkwo/while/Identity_5Identity%oohztcvkwo/while/ammytzqwsz/add_3:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/while/Identity_5"|
;oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource=oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0"~
<oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource>oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0"z
:oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource<oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0"p
5oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource7oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0"p
5oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource7oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0"l
3oohztcvkwo_while_ammytzqwsz_readvariableop_resource5oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0"?
oohztcvkwo_while_identity"oohztcvkwo/while/Identity:output:0"C
oohztcvkwo_while_identity_1$oohztcvkwo/while/Identity_1:output:0"C
oohztcvkwo_while_identity_2$oohztcvkwo/while/Identity_2:output:0"C
oohztcvkwo_while_identity_3$oohztcvkwo/while/Identity_3:output:0"C
oohztcvkwo_while_identity_4$oohztcvkwo/while/Identity_4:output:0"C
oohztcvkwo_while_identity_5$oohztcvkwo/while/Identity_5:output:0"\
+oohztcvkwo_while_oohztcvkwo_strided_slice_1-oohztcvkwo_while_oohztcvkwo_strided_slice_1_0"Ô
goohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensorioohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2f
1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp2j
3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp2X
*oohztcvkwo/while/ammytzqwsz/ReadVariableOp*oohztcvkwo/while/ammytzqwsz/ReadVariableOp2\
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_12\
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
oohztcvkwo_while_body_18217232
.oohztcvkwo_while_oohztcvkwo_while_loop_counter8
4oohztcvkwo_while_oohztcvkwo_while_maximum_iterations 
oohztcvkwo_while_placeholder"
oohztcvkwo_while_placeholder_1"
oohztcvkwo_while_placeholder_2"
oohztcvkwo_while_placeholder_31
-oohztcvkwo_while_oohztcvkwo_strided_slice_1_0m
ioohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0O
<oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0:	Q
>oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 L
=oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0:	C
5oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0: E
7oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0: E
7oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0: 
oohztcvkwo_while_identity
oohztcvkwo_while_identity_1
oohztcvkwo_while_identity_2
oohztcvkwo_while_identity_3
oohztcvkwo_while_identity_4
oohztcvkwo_while_identity_5/
+oohztcvkwo_while_oohztcvkwo_strided_slice_1k
goohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensorM
:oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource:	O
<oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource:	 J
;oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource:	A
3oohztcvkwo_while_ammytzqwsz_readvariableop_resource: C
5oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource: C
5oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource: ¢2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp¢1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp¢3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp¢*oohztcvkwo/while/ammytzqwsz/ReadVariableOp¢,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1¢,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2Ù
Boohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Boohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shape
4oohztcvkwo/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemioohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0oohztcvkwo_while_placeholderKoohztcvkwo/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4oohztcvkwo/while/TensorArrayV2Read/TensorListGetItemä
1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp<oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOpý
"oohztcvkwo/while/ammytzqwsz/MatMulMatMul;oohztcvkwo/while/TensorArrayV2Read/TensorListGetItem:item:09oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"oohztcvkwo/while/ammytzqwsz/MatMulê
3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp>oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOpæ
$oohztcvkwo/while/ammytzqwsz/MatMul_1MatMuloohztcvkwo_while_placeholder_2;oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$oohztcvkwo/while/ammytzqwsz/MatMul_1Ü
oohztcvkwo/while/ammytzqwsz/addAddV2,oohztcvkwo/while/ammytzqwsz/MatMul:product:0.oohztcvkwo/while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
oohztcvkwo/while/ammytzqwsz/addã
2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp=oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOpé
#oohztcvkwo/while/ammytzqwsz/BiasAddBiasAdd#oohztcvkwo/while/ammytzqwsz/add:z:0:oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#oohztcvkwo/while/ammytzqwsz/BiasAdd
+oohztcvkwo/while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+oohztcvkwo/while/ammytzqwsz/split/split_dim¯
!oohztcvkwo/while/ammytzqwsz/splitSplit4oohztcvkwo/while/ammytzqwsz/split/split_dim:output:0,oohztcvkwo/while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!oohztcvkwo/while/ammytzqwsz/splitÊ
*oohztcvkwo/while/ammytzqwsz/ReadVariableOpReadVariableOp5oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*oohztcvkwo/while/ammytzqwsz/ReadVariableOpÏ
oohztcvkwo/while/ammytzqwsz/mulMul2oohztcvkwo/while/ammytzqwsz/ReadVariableOp:value:0oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
oohztcvkwo/while/ammytzqwsz/mulÒ
!oohztcvkwo/while/ammytzqwsz/add_1AddV2*oohztcvkwo/while/ammytzqwsz/split:output:0#oohztcvkwo/while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_1®
#oohztcvkwo/while/ammytzqwsz/SigmoidSigmoid%oohztcvkwo/while/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#oohztcvkwo/while/ammytzqwsz/SigmoidÐ
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1ReadVariableOp7oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1Õ
!oohztcvkwo/while/ammytzqwsz/mul_1Mul4oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1:value:0oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_1Ô
!oohztcvkwo/while/ammytzqwsz/add_2AddV2*oohztcvkwo/while/ammytzqwsz/split:output:1%oohztcvkwo/while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_2²
%oohztcvkwo/while/ammytzqwsz/Sigmoid_1Sigmoid%oohztcvkwo/while/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%oohztcvkwo/while/ammytzqwsz/Sigmoid_1Ê
!oohztcvkwo/while/ammytzqwsz/mul_2Mul)oohztcvkwo/while/ammytzqwsz/Sigmoid_1:y:0oohztcvkwo_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_2ª
 oohztcvkwo/while/ammytzqwsz/TanhTanh*oohztcvkwo/while/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 oohztcvkwo/while/ammytzqwsz/TanhÎ
!oohztcvkwo/while/ammytzqwsz/mul_3Mul'oohztcvkwo/while/ammytzqwsz/Sigmoid:y:0$oohztcvkwo/while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_3Ï
!oohztcvkwo/while/ammytzqwsz/add_3AddV2%oohztcvkwo/while/ammytzqwsz/mul_2:z:0%oohztcvkwo/while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_3Ð
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2ReadVariableOp7oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2Ü
!oohztcvkwo/while/ammytzqwsz/mul_4Mul4oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2:value:0%oohztcvkwo/while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_4Ô
!oohztcvkwo/while/ammytzqwsz/add_4AddV2*oohztcvkwo/while/ammytzqwsz/split:output:3%oohztcvkwo/while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/add_4²
%oohztcvkwo/while/ammytzqwsz/Sigmoid_2Sigmoid%oohztcvkwo/while/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%oohztcvkwo/while/ammytzqwsz/Sigmoid_2©
"oohztcvkwo/while/ammytzqwsz/Tanh_1Tanh%oohztcvkwo/while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"oohztcvkwo/while/ammytzqwsz/Tanh_1Ò
!oohztcvkwo/while/ammytzqwsz/mul_5Mul)oohztcvkwo/while/ammytzqwsz/Sigmoid_2:y:0&oohztcvkwo/while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!oohztcvkwo/while/ammytzqwsz/mul_5
5oohztcvkwo/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemoohztcvkwo_while_placeholder_1oohztcvkwo_while_placeholder%oohztcvkwo/while/ammytzqwsz/mul_5:z:0*
_output_shapes
: *
element_dtype027
5oohztcvkwo/while/TensorArrayV2Write/TensorListSetItemr
oohztcvkwo/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
oohztcvkwo/while/add/y
oohztcvkwo/while/addAddV2oohztcvkwo_while_placeholderoohztcvkwo/while/add/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/while/addv
oohztcvkwo/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
oohztcvkwo/while/add_1/y­
oohztcvkwo/while/add_1AddV2.oohztcvkwo_while_oohztcvkwo_while_loop_counter!oohztcvkwo/while/add_1/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/while/add_1©
oohztcvkwo/while/IdentityIdentityoohztcvkwo/while/add_1:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/IdentityÇ
oohztcvkwo/while/Identity_1Identity4oohztcvkwo_while_oohztcvkwo_while_maximum_iterations3^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/Identity_1«
oohztcvkwo/while/Identity_2Identityoohztcvkwo/while/add:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/Identity_2Ø
oohztcvkwo/while/Identity_3IdentityEoohztcvkwo/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
oohztcvkwo/while/Identity_3É
oohztcvkwo/while/Identity_4Identity%oohztcvkwo/while/ammytzqwsz/mul_5:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/while/Identity_4É
oohztcvkwo/while/Identity_5Identity%oohztcvkwo/while/ammytzqwsz/add_3:z:03^oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2^oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp4^oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp+^oohztcvkwo/while/ammytzqwsz/ReadVariableOp-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1-^oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/while/Identity_5"|
;oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource=oohztcvkwo_while_ammytzqwsz_biasadd_readvariableop_resource_0"~
<oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource>oohztcvkwo_while_ammytzqwsz_matmul_1_readvariableop_resource_0"z
:oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource<oohztcvkwo_while_ammytzqwsz_matmul_readvariableop_resource_0"p
5oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource7oohztcvkwo_while_ammytzqwsz_readvariableop_1_resource_0"p
5oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource7oohztcvkwo_while_ammytzqwsz_readvariableop_2_resource_0"l
3oohztcvkwo_while_ammytzqwsz_readvariableop_resource5oohztcvkwo_while_ammytzqwsz_readvariableop_resource_0"?
oohztcvkwo_while_identity"oohztcvkwo/while/Identity:output:0"C
oohztcvkwo_while_identity_1$oohztcvkwo/while/Identity_1:output:0"C
oohztcvkwo_while_identity_2$oohztcvkwo/while/Identity_2:output:0"C
oohztcvkwo_while_identity_3$oohztcvkwo/while/Identity_3:output:0"C
oohztcvkwo_while_identity_4$oohztcvkwo/while/Identity_4:output:0"C
oohztcvkwo_while_identity_5$oohztcvkwo/while/Identity_5:output:0"\
+oohztcvkwo_while_oohztcvkwo_strided_slice_1-oohztcvkwo_while_oohztcvkwo_strided_slice_1_0"Ô
goohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensorioohztcvkwo_while_tensorarrayv2read_tensorlistgetitem_oohztcvkwo_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2oohztcvkwo/while/ammytzqwsz/BiasAdd/ReadVariableOp2f
1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp1oohztcvkwo/while/ammytzqwsz/MatMul/ReadVariableOp2j
3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp3oohztcvkwo/while/ammytzqwsz/MatMul_1/ReadVariableOp2X
*oohztcvkwo/while/ammytzqwsz/ReadVariableOp*oohztcvkwo/while/ammytzqwsz/ReadVariableOp2\
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_1,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_12\
,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2,oohztcvkwo/while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_1823595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aflyrndiyz_matmul_readvariableop_resource_0:	 F
3while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 A
2while_aflyrndiyz_biasadd_readvariableop_resource_0:	8
*while_aflyrndiyz_readvariableop_resource_0: :
,while_aflyrndiyz_readvariableop_1_resource_0: :
,while_aflyrndiyz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aflyrndiyz_matmul_readvariableop_resource:	 D
1while_aflyrndiyz_matmul_1_readvariableop_resource:	 ?
0while_aflyrndiyz_biasadd_readvariableop_resource:	6
(while_aflyrndiyz_readvariableop_resource: 8
*while_aflyrndiyz_readvariableop_1_resource: 8
*while_aflyrndiyz_readvariableop_2_resource: ¢'while/aflyrndiyz/BiasAdd/ReadVariableOp¢&while/aflyrndiyz/MatMul/ReadVariableOp¢(while/aflyrndiyz/MatMul_1/ReadVariableOp¢while/aflyrndiyz/ReadVariableOp¢!while/aflyrndiyz/ReadVariableOp_1¢!while/aflyrndiyz/ReadVariableOp_2Ã
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
&while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp1while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aflyrndiyz/MatMul/ReadVariableOpÑ
while/aflyrndiyz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMulÉ
(while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp3while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aflyrndiyz/MatMul_1/ReadVariableOpº
while/aflyrndiyz/MatMul_1MatMulwhile_placeholder_20while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMul_1°
while/aflyrndiyz/addAddV2!while/aflyrndiyz/MatMul:product:0#while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/addÂ
'while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp2while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aflyrndiyz/BiasAdd/ReadVariableOp½
while/aflyrndiyz/BiasAddBiasAddwhile/aflyrndiyz/add:z:0/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/BiasAdd
 while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aflyrndiyz/split/split_dim
while/aflyrndiyz/splitSplit)while/aflyrndiyz/split/split_dim:output:0!while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aflyrndiyz/split©
while/aflyrndiyz/ReadVariableOpReadVariableOp*while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aflyrndiyz/ReadVariableOp£
while/aflyrndiyz/mulMul'while/aflyrndiyz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul¦
while/aflyrndiyz/add_1AddV2while/aflyrndiyz/split:output:0while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_1
while/aflyrndiyz/SigmoidSigmoidwhile/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid¯
!while/aflyrndiyz/ReadVariableOp_1ReadVariableOp,while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_1©
while/aflyrndiyz/mul_1Mul)while/aflyrndiyz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_1¨
while/aflyrndiyz/add_2AddV2while/aflyrndiyz/split:output:1while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_2
while/aflyrndiyz/Sigmoid_1Sigmoidwhile/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_1
while/aflyrndiyz/mul_2Mulwhile/aflyrndiyz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_2
while/aflyrndiyz/TanhTanhwhile/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh¢
while/aflyrndiyz/mul_3Mulwhile/aflyrndiyz/Sigmoid:y:0while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_3£
while/aflyrndiyz/add_3AddV2while/aflyrndiyz/mul_2:z:0while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_3¯
!while/aflyrndiyz/ReadVariableOp_2ReadVariableOp,while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_2°
while/aflyrndiyz/mul_4Mul)while/aflyrndiyz/ReadVariableOp_2:value:0while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_4¨
while/aflyrndiyz/add_4AddV2while/aflyrndiyz/split:output:3while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_4
while/aflyrndiyz/Sigmoid_2Sigmoidwhile/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_2
while/aflyrndiyz/Tanh_1Tanhwhile/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh_1¦
while/aflyrndiyz/mul_5Mulwhile/aflyrndiyz/Sigmoid_2:y:0while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aflyrndiyz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aflyrndiyz/mul_5:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aflyrndiyz/add_3:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aflyrndiyz_biasadd_readvariableop_resource2while_aflyrndiyz_biasadd_readvariableop_resource_0"h
1while_aflyrndiyz_matmul_1_readvariableop_resource3while_aflyrndiyz_matmul_1_readvariableop_resource_0"d
/while_aflyrndiyz_matmul_readvariableop_resource1while_aflyrndiyz_matmul_readvariableop_resource_0"Z
*while_aflyrndiyz_readvariableop_1_resource,while_aflyrndiyz_readvariableop_1_resource_0"Z
*while_aflyrndiyz_readvariableop_2_resource,while_aflyrndiyz_readvariableop_2_resource_0"V
(while_aflyrndiyz_readvariableop_resource*while_aflyrndiyz_readvariableop_resource_0")
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
'while/aflyrndiyz/BiasAdd/ReadVariableOp'while/aflyrndiyz/BiasAdd/ReadVariableOp2P
&while/aflyrndiyz/MatMul/ReadVariableOp&while/aflyrndiyz/MatMul/ReadVariableOp2T
(while/aflyrndiyz/MatMul_1/ReadVariableOp(while/aflyrndiyz/MatMul_1/ReadVariableOp2B
while/aflyrndiyz/ReadVariableOpwhile/aflyrndiyz/ReadVariableOp2F
!while/aflyrndiyz/ReadVariableOp_1!while/aflyrndiyz/ReadVariableOp_12F
!while/aflyrndiyz/ReadVariableOp_2!while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1820058

inputs%
aflyrndiyz_1819959:	 %
aflyrndiyz_1819961:	 !
aflyrndiyz_1819963:	 
aflyrndiyz_1819965:  
aflyrndiyz_1819967:  
aflyrndiyz_1819969: 
identity¢"aflyrndiyz/StatefulPartitionedCall¢whileD
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
"aflyrndiyz/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0aflyrndiyz_1819959aflyrndiyz_1819961aflyrndiyz_1819963aflyrndiyz_1819965aflyrndiyz_1819967aflyrndiyz_1819969*
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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_18198822$
"aflyrndiyz/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0aflyrndiyz_1819959aflyrndiyz_1819961aflyrndiyz_1819963aflyrndiyz_1819965aflyrndiyz_1819967aflyrndiyz_1819969*
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
while_body_1819978*
condR
while_cond_1819977*Q
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
IdentityIdentitystrided_slice_3:output:0#^aflyrndiyz/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"aflyrndiyz/StatefulPartitionedCall"aflyrndiyz/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û

,__inference_oohztcvkwo_layer_call_fn_1823319

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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18206102
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
¹'
µ
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1824365

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
ë

,__inference_mfdtyewult_layer_call_fn_1824090
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18200582
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
while_cond_1822806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1822806___redundant_placeholder05
1while_while_cond_1822806___redundant_placeholder15
1while_while_cond_1822806___redundant_placeholder25
1while_while_cond_1822806___redundant_placeholder35
1while_while_cond_1822806___redundant_placeholder45
1while_while_cond_1822806___redundant_placeholder55
1while_while_cond_1822806___redundant_placeholder6
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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1819124

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

c
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_1820429

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
mfdtyewult_while_cond_18218982
.mfdtyewult_while_mfdtyewult_while_loop_counter8
4mfdtyewult_while_mfdtyewult_while_maximum_iterations 
mfdtyewult_while_placeholder"
mfdtyewult_while_placeholder_1"
mfdtyewult_while_placeholder_2"
mfdtyewult_while_placeholder_34
0mfdtyewult_while_less_mfdtyewult_strided_slice_1K
Gmfdtyewult_while_mfdtyewult_while_cond_1821898___redundant_placeholder0K
Gmfdtyewult_while_mfdtyewult_while_cond_1821898___redundant_placeholder1K
Gmfdtyewult_while_mfdtyewult_while_cond_1821898___redundant_placeholder2K
Gmfdtyewult_while_mfdtyewult_while_cond_1821898___redundant_placeholder3K
Gmfdtyewult_while_mfdtyewult_while_cond_1821898___redundant_placeholder4K
Gmfdtyewult_while_mfdtyewult_while_cond_1821898___redundant_placeholder5K
Gmfdtyewult_while_mfdtyewult_while_cond_1821898___redundant_placeholder6
mfdtyewult_while_identity
§
mfdtyewult/while/LessLessmfdtyewult_while_placeholder0mfdtyewult_while_less_mfdtyewult_strided_slice_1*
T0*
_output_shapes
: 2
mfdtyewult/while/Less~
mfdtyewult/while/IdentityIdentitymfdtyewult/while/Less:z:0*
T0
*
_output_shapes
: 2
mfdtyewult/while/Identity"?
mfdtyewult_while_identity"mfdtyewult/while/Identity:output:0*(
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
while_cond_1821190
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1821190___redundant_placeholder05
1while_while_cond_1821190___redundant_placeholder15
1while_while_cond_1821190___redundant_placeholder25
1while_while_cond_1821190___redundant_placeholder35
1while_while_cond_1821190___redundant_placeholder45
1while_while_cond_1821190___redundant_placeholder55
1while_while_cond_1821190___redundant_placeholder6
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
while_cond_1823774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1823774___redundant_placeholder05
1while_while_cond_1823774___redundant_placeholder15
1while_while_cond_1823774___redundant_placeholder25
1while_while_cond_1823774___redundant_placeholder35
1while_while_cond_1823774___redundant_placeholder45
1while_while_cond_1823774___redundant_placeholder55
1while_while_cond_1823774___redundant_placeholder6
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1824056

inputs<
)aflyrndiyz_matmul_readvariableop_resource:	 >
+aflyrndiyz_matmul_1_readvariableop_resource:	 9
*aflyrndiyz_biasadd_readvariableop_resource:	0
"aflyrndiyz_readvariableop_resource: 2
$aflyrndiyz_readvariableop_1_resource: 2
$aflyrndiyz_readvariableop_2_resource: 
identity¢!aflyrndiyz/BiasAdd/ReadVariableOp¢ aflyrndiyz/MatMul/ReadVariableOp¢"aflyrndiyz/MatMul_1/ReadVariableOp¢aflyrndiyz/ReadVariableOp¢aflyrndiyz/ReadVariableOp_1¢aflyrndiyz/ReadVariableOp_2¢whileD
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
 aflyrndiyz/MatMul/ReadVariableOpReadVariableOp)aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aflyrndiyz/MatMul/ReadVariableOp§
aflyrndiyz/MatMulMatMulstrided_slice_2:output:0(aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMulµ
"aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp+aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aflyrndiyz/MatMul_1/ReadVariableOp£
aflyrndiyz/MatMul_1MatMulzeros:output:0*aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMul_1
aflyrndiyz/addAddV2aflyrndiyz/MatMul:product:0aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/add®
!aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp*aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aflyrndiyz/BiasAdd/ReadVariableOp¥
aflyrndiyz/BiasAddBiasAddaflyrndiyz/add:z:0)aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/BiasAddz
aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aflyrndiyz/split/split_dimë
aflyrndiyz/splitSplit#aflyrndiyz/split/split_dim:output:0aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aflyrndiyz/split
aflyrndiyz/ReadVariableOpReadVariableOp"aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp
aflyrndiyz/mulMul!aflyrndiyz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul
aflyrndiyz/add_1AddV2aflyrndiyz/split:output:0aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_1{
aflyrndiyz/SigmoidSigmoidaflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid
aflyrndiyz/ReadVariableOp_1ReadVariableOp$aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_1
aflyrndiyz/mul_1Mul#aflyrndiyz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_1
aflyrndiyz/add_2AddV2aflyrndiyz/split:output:1aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_2
aflyrndiyz/Sigmoid_1Sigmoidaflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_1
aflyrndiyz/mul_2Mulaflyrndiyz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_2w
aflyrndiyz/TanhTanhaflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh
aflyrndiyz/mul_3Mulaflyrndiyz/Sigmoid:y:0aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_3
aflyrndiyz/add_3AddV2aflyrndiyz/mul_2:z:0aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_3
aflyrndiyz/ReadVariableOp_2ReadVariableOp$aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_2
aflyrndiyz/mul_4Mul#aflyrndiyz/ReadVariableOp_2:value:0aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_4
aflyrndiyz/add_4AddV2aflyrndiyz/split:output:3aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_4
aflyrndiyz/Sigmoid_2Sigmoidaflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_2v
aflyrndiyz/Tanh_1Tanhaflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh_1
aflyrndiyz/mul_5Mulaflyrndiyz/Sigmoid_2:y:0aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aflyrndiyz_matmul_readvariableop_resource+aflyrndiyz_matmul_1_readvariableop_resource*aflyrndiyz_biasadd_readvariableop_resource"aflyrndiyz_readvariableop_resource$aflyrndiyz_readvariableop_1_resource$aflyrndiyz_readvariableop_2_resource*
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
while_body_1823955*
condR
while_cond_1823954*Q
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
IdentityIdentitystrided_slice_3:output:0"^aflyrndiyz/BiasAdd/ReadVariableOp!^aflyrndiyz/MatMul/ReadVariableOp#^aflyrndiyz/MatMul_1/ReadVariableOp^aflyrndiyz/ReadVariableOp^aflyrndiyz/ReadVariableOp_1^aflyrndiyz/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aflyrndiyz/BiasAdd/ReadVariableOp!aflyrndiyz/BiasAdd/ReadVariableOp2D
 aflyrndiyz/MatMul/ReadVariableOp aflyrndiyz/MatMul/ReadVariableOp2H
"aflyrndiyz/MatMul_1/ReadVariableOp"aflyrndiyz/MatMul_1/ReadVariableOp26
aflyrndiyz/ReadVariableOpaflyrndiyz/ReadVariableOp2:
aflyrndiyz/ReadVariableOp_1aflyrndiyz/ReadVariableOp_12:
aflyrndiyz/ReadVariableOp_2aflyrndiyz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç)
Ò
while_body_1819220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_ammytzqwsz_1819244_0:	-
while_ammytzqwsz_1819246_0:	 )
while_ammytzqwsz_1819248_0:	(
while_ammytzqwsz_1819250_0: (
while_ammytzqwsz_1819252_0: (
while_ammytzqwsz_1819254_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_ammytzqwsz_1819244:	+
while_ammytzqwsz_1819246:	 '
while_ammytzqwsz_1819248:	&
while_ammytzqwsz_1819250: &
while_ammytzqwsz_1819252: &
while_ammytzqwsz_1819254: ¢(while/ammytzqwsz/StatefulPartitionedCallÃ
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
(while/ammytzqwsz/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_ammytzqwsz_1819244_0while_ammytzqwsz_1819246_0while_ammytzqwsz_1819248_0while_ammytzqwsz_1819250_0while_ammytzqwsz_1819252_0while_ammytzqwsz_1819254_0*
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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_18191242*
(while/ammytzqwsz/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/ammytzqwsz/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/ammytzqwsz/StatefulPartitionedCall:output:1)^while/ammytzqwsz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/ammytzqwsz/StatefulPartitionedCall:output:2)^while/ammytzqwsz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_ammytzqwsz_1819244while_ammytzqwsz_1819244_0"6
while_ammytzqwsz_1819246while_ammytzqwsz_1819246_0"6
while_ammytzqwsz_1819248while_ammytzqwsz_1819248_0"6
while_ammytzqwsz_1819250while_ammytzqwsz_1819250_0"6
while_ammytzqwsz_1819252while_ammytzqwsz_1819252_0"6
while_ammytzqwsz_1819254while_ammytzqwsz_1819254_0")
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
(while/ammytzqwsz/StatefulPartitionedCall(while/ammytzqwsz/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1823088

inputs<
)ammytzqwsz_matmul_readvariableop_resource:	>
+ammytzqwsz_matmul_1_readvariableop_resource:	 9
*ammytzqwsz_biasadd_readvariableop_resource:	0
"ammytzqwsz_readvariableop_resource: 2
$ammytzqwsz_readvariableop_1_resource: 2
$ammytzqwsz_readvariableop_2_resource: 
identity¢!ammytzqwsz/BiasAdd/ReadVariableOp¢ ammytzqwsz/MatMul/ReadVariableOp¢"ammytzqwsz/MatMul_1/ReadVariableOp¢ammytzqwsz/ReadVariableOp¢ammytzqwsz/ReadVariableOp_1¢ammytzqwsz/ReadVariableOp_2¢whileD
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
 ammytzqwsz/MatMul/ReadVariableOpReadVariableOp)ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ammytzqwsz/MatMul/ReadVariableOp§
ammytzqwsz/MatMulMatMulstrided_slice_2:output:0(ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMulµ
"ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp+ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ammytzqwsz/MatMul_1/ReadVariableOp£
ammytzqwsz/MatMul_1MatMulzeros:output:0*ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMul_1
ammytzqwsz/addAddV2ammytzqwsz/MatMul:product:0ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/add®
!ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp*ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ammytzqwsz/BiasAdd/ReadVariableOp¥
ammytzqwsz/BiasAddBiasAddammytzqwsz/add:z:0)ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/BiasAddz
ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ammytzqwsz/split/split_dimë
ammytzqwsz/splitSplit#ammytzqwsz/split/split_dim:output:0ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ammytzqwsz/split
ammytzqwsz/ReadVariableOpReadVariableOp"ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp
ammytzqwsz/mulMul!ammytzqwsz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul
ammytzqwsz/add_1AddV2ammytzqwsz/split:output:0ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_1{
ammytzqwsz/SigmoidSigmoidammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid
ammytzqwsz/ReadVariableOp_1ReadVariableOp$ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_1
ammytzqwsz/mul_1Mul#ammytzqwsz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_1
ammytzqwsz/add_2AddV2ammytzqwsz/split:output:1ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_2
ammytzqwsz/Sigmoid_1Sigmoidammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_1
ammytzqwsz/mul_2Mulammytzqwsz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_2w
ammytzqwsz/TanhTanhammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh
ammytzqwsz/mul_3Mulammytzqwsz/Sigmoid:y:0ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_3
ammytzqwsz/add_3AddV2ammytzqwsz/mul_2:z:0ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_3
ammytzqwsz/ReadVariableOp_2ReadVariableOp$ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_2
ammytzqwsz/mul_4Mul#ammytzqwsz/ReadVariableOp_2:value:0ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_4
ammytzqwsz/add_4AddV2ammytzqwsz/split:output:3ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_4
ammytzqwsz/Sigmoid_2Sigmoidammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_2v
ammytzqwsz/Tanh_1Tanhammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh_1
ammytzqwsz/mul_5Mulammytzqwsz/Sigmoid_2:y:0ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ammytzqwsz_matmul_readvariableop_resource+ammytzqwsz_matmul_1_readvariableop_resource*ammytzqwsz_biasadd_readvariableop_resource"ammytzqwsz_readvariableop_resource$ammytzqwsz_readvariableop_1_resource$ammytzqwsz_readvariableop_2_resource*
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
while_body_1822987*
condR
while_cond_1822986*Q
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
IdentityIdentitytranspose_1:y:0"^ammytzqwsz/BiasAdd/ReadVariableOp!^ammytzqwsz/MatMul/ReadVariableOp#^ammytzqwsz/MatMul_1/ReadVariableOp^ammytzqwsz/ReadVariableOp^ammytzqwsz/ReadVariableOp_1^ammytzqwsz/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ammytzqwsz/BiasAdd/ReadVariableOp!ammytzqwsz/BiasAdd/ReadVariableOp2D
 ammytzqwsz/MatMul/ReadVariableOp ammytzqwsz/MatMul/ReadVariableOp2H
"ammytzqwsz/MatMul_1/ReadVariableOp"ammytzqwsz/MatMul_1/ReadVariableOp26
ammytzqwsz/ReadVariableOpammytzqwsz/ReadVariableOp2:
ammytzqwsz/ReadVariableOp_1ammytzqwsz/ReadVariableOp_12:
ammytzqwsz/ReadVariableOp_2ammytzqwsz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹'
µ
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1824187

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

À
,__inference_aflyrndiyz_layer_call_fn_1824388

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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_18196952
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
¤

,__inference_boyhyiogqf_layer_call_fn_1824143

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
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_18208272
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


í
while_cond_1819977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1819977___redundant_placeholder05
1while_while_cond_1819977___redundant_placeholder15
1while_while_cond_1819977___redundant_placeholder25
1while_while_cond_1819977___redundant_placeholder35
1while_while_cond_1819977___redundant_placeholder45
1while_while_cond_1819977___redundant_placeholder55
1while_while_cond_1819977___redundant_placeholder6
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
while_body_1819715
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_aflyrndiyz_1819739_0:	 -
while_aflyrndiyz_1819741_0:	 )
while_aflyrndiyz_1819743_0:	(
while_aflyrndiyz_1819745_0: (
while_aflyrndiyz_1819747_0: (
while_aflyrndiyz_1819749_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_aflyrndiyz_1819739:	 +
while_aflyrndiyz_1819741:	 '
while_aflyrndiyz_1819743:	&
while_aflyrndiyz_1819745: &
while_aflyrndiyz_1819747: &
while_aflyrndiyz_1819749: ¢(while/aflyrndiyz/StatefulPartitionedCallÃ
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
(while/aflyrndiyz/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_aflyrndiyz_1819739_0while_aflyrndiyz_1819741_0while_aflyrndiyz_1819743_0while_aflyrndiyz_1819745_0while_aflyrndiyz_1819747_0while_aflyrndiyz_1819749_0*
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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_18196952*
(while/aflyrndiyz/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/aflyrndiyz/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/aflyrndiyz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/aflyrndiyz/StatefulPartitionedCall:output:1)^while/aflyrndiyz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/aflyrndiyz/StatefulPartitionedCall:output:2)^while/aflyrndiyz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_aflyrndiyz_1819739while_aflyrndiyz_1819739_0"6
while_aflyrndiyz_1819741while_aflyrndiyz_1819741_0"6
while_aflyrndiyz_1819743while_aflyrndiyz_1819743_0"6
while_aflyrndiyz_1819745while_aflyrndiyz_1819745_0"6
while_aflyrndiyz_1819747while_aflyrndiyz_1819747_0"6
while_aflyrndiyz_1819749while_aflyrndiyz_1819749_0")
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
(while/aflyrndiyz/StatefulPartitionedCall(while/aflyrndiyz/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_mfdtyewult_layer_call_fn_1824107

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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18208032
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


í
while_cond_1822986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1822986___redundant_placeholder05
1while_while_cond_1822986___redundant_placeholder15
1while_while_cond_1822986___redundant_placeholder25
1while_while_cond_1822986___redundant_placeholder35
1while_while_cond_1822986___redundant_placeholder45
1while_while_cond_1822986___redundant_placeholder55
1while_while_cond_1822986___redundant_placeholder6
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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_1822410

inputsL
6fethhjgisa_conv1d_expanddims_1_readvariableop_resource:K
=fethhjgisa_squeeze_batch_dims_biasadd_readvariableop_resource:G
4oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource:	I
6oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource:	 D
5oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource:	;
-oohztcvkwo_ammytzqwsz_readvariableop_resource: =
/oohztcvkwo_ammytzqwsz_readvariableop_1_resource: =
/oohztcvkwo_ammytzqwsz_readvariableop_2_resource: G
4mfdtyewult_aflyrndiyz_matmul_readvariableop_resource:	 I
6mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource:	 D
5mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource:	;
-mfdtyewult_aflyrndiyz_readvariableop_resource: =
/mfdtyewult_aflyrndiyz_readvariableop_1_resource: =
/mfdtyewult_aflyrndiyz_readvariableop_2_resource: ;
)boyhyiogqf_matmul_readvariableop_resource: 8
*boyhyiogqf_biasadd_readvariableop_resource:
identity¢!boyhyiogqf/BiasAdd/ReadVariableOp¢ boyhyiogqf/MatMul/ReadVariableOp¢-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp¢4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp¢+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp¢-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp¢$mfdtyewult/aflyrndiyz/ReadVariableOp¢&mfdtyewult/aflyrndiyz/ReadVariableOp_1¢&mfdtyewult/aflyrndiyz/ReadVariableOp_2¢mfdtyewult/while¢,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp¢+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp¢-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp¢$oohztcvkwo/ammytzqwsz/ReadVariableOp¢&oohztcvkwo/ammytzqwsz/ReadVariableOp_1¢&oohztcvkwo/ammytzqwsz/ReadVariableOp_2¢oohztcvkwo/while
 fethhjgisa/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 fethhjgisa/conv1d/ExpandDims/dim»
fethhjgisa/conv1d/ExpandDims
ExpandDimsinputs)fethhjgisa/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/ExpandDimsÙ
-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6fethhjgisa_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp
"fethhjgisa/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"fethhjgisa/conv1d/ExpandDims_1/dimã
fethhjgisa/conv1d/ExpandDims_1
ExpandDims5fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp:value:0+fethhjgisa/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
fethhjgisa/conv1d/ExpandDims_1
fethhjgisa/conv1d/ShapeShape%fethhjgisa/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
fethhjgisa/conv1d/Shape
%fethhjgisa/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%fethhjgisa/conv1d/strided_slice/stack¥
'fethhjgisa/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'fethhjgisa/conv1d/strided_slice/stack_1
'fethhjgisa/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'fethhjgisa/conv1d/strided_slice/stack_2Ì
fethhjgisa/conv1d/strided_sliceStridedSlice fethhjgisa/conv1d/Shape:output:0.fethhjgisa/conv1d/strided_slice/stack:output:00fethhjgisa/conv1d/strided_slice/stack_1:output:00fethhjgisa/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
fethhjgisa/conv1d/strided_slice
fethhjgisa/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
fethhjgisa/conv1d/Reshape/shapeÌ
fethhjgisa/conv1d/ReshapeReshape%fethhjgisa/conv1d/ExpandDims:output:0(fethhjgisa/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/Reshapeî
fethhjgisa/conv1d/Conv2DConv2D"fethhjgisa/conv1d/Reshape:output:0'fethhjgisa/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
fethhjgisa/conv1d/Conv2D
!fethhjgisa/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!fethhjgisa/conv1d/concat/values_1
fethhjgisa/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/concat/axisì
fethhjgisa/conv1d/concatConcatV2(fethhjgisa/conv1d/strided_slice:output:0*fethhjgisa/conv1d/concat/values_1:output:0&fethhjgisa/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
fethhjgisa/conv1d/concatÉ
fethhjgisa/conv1d/Reshape_1Reshape!fethhjgisa/conv1d/Conv2D:output:0!fethhjgisa/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/Reshape_1Á
fethhjgisa/conv1d/SqueezeSqueeze$fethhjgisa/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
fethhjgisa/conv1d/Squeeze
#fethhjgisa/squeeze_batch_dims/ShapeShape"fethhjgisa/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#fethhjgisa/squeeze_batch_dims/Shape°
1fethhjgisa/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1fethhjgisa/squeeze_batch_dims/strided_slice/stack½
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_1´
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3fethhjgisa/squeeze_batch_dims/strided_slice/stack_2
+fethhjgisa/squeeze_batch_dims/strided_sliceStridedSlice,fethhjgisa/squeeze_batch_dims/Shape:output:0:fethhjgisa/squeeze_batch_dims/strided_slice/stack:output:0<fethhjgisa/squeeze_batch_dims/strided_slice/stack_1:output:0<fethhjgisa/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+fethhjgisa/squeeze_batch_dims/strided_slice¯
+fethhjgisa/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+fethhjgisa/squeeze_batch_dims/Reshape/shapeé
%fethhjgisa/squeeze_batch_dims/ReshapeReshape"fethhjgisa/conv1d/Squeeze:output:04fethhjgisa/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%fethhjgisa/squeeze_batch_dims/Reshapeæ
4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=fethhjgisa_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%fethhjgisa/squeeze_batch_dims/BiasAddBiasAdd.fethhjgisa/squeeze_batch_dims/Reshape:output:0<fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%fethhjgisa/squeeze_batch_dims/BiasAdd¯
-fethhjgisa/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-fethhjgisa/squeeze_batch_dims/concat/values_1¡
)fethhjgisa/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)fethhjgisa/squeeze_batch_dims/concat/axis¨
$fethhjgisa/squeeze_batch_dims/concatConcatV24fethhjgisa/squeeze_batch_dims/strided_slice:output:06fethhjgisa/squeeze_batch_dims/concat/values_1:output:02fethhjgisa/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$fethhjgisa/squeeze_batch_dims/concatö
'fethhjgisa/squeeze_batch_dims/Reshape_1Reshape.fethhjgisa/squeeze_batch_dims/BiasAdd:output:0-fethhjgisa/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'fethhjgisa/squeeze_batch_dims/Reshape_1
ohvvfduigw/ShapeShape0fethhjgisa/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ohvvfduigw/Shape
ohvvfduigw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ohvvfduigw/strided_slice/stack
 ohvvfduigw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ohvvfduigw/strided_slice/stack_1
 ohvvfduigw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ohvvfduigw/strided_slice/stack_2¤
ohvvfduigw/strided_sliceStridedSliceohvvfduigw/Shape:output:0'ohvvfduigw/strided_slice/stack:output:0)ohvvfduigw/strided_slice/stack_1:output:0)ohvvfduigw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ohvvfduigw/strided_slicez
ohvvfduigw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ohvvfduigw/Reshape/shape/1z
ohvvfduigw/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ohvvfduigw/Reshape/shape/2×
ohvvfduigw/Reshape/shapePack!ohvvfduigw/strided_slice:output:0#ohvvfduigw/Reshape/shape/1:output:0#ohvvfduigw/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ohvvfduigw/Reshape/shape¾
ohvvfduigw/ReshapeReshape0fethhjgisa/squeeze_batch_dims/Reshape_1:output:0!ohvvfduigw/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ohvvfduigw/Reshapeo
oohztcvkwo/ShapeShapeohvvfduigw/Reshape:output:0*
T0*
_output_shapes
:2
oohztcvkwo/Shape
oohztcvkwo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
oohztcvkwo/strided_slice/stack
 oohztcvkwo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 oohztcvkwo/strided_slice/stack_1
 oohztcvkwo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 oohztcvkwo/strided_slice/stack_2¤
oohztcvkwo/strided_sliceStridedSliceoohztcvkwo/Shape:output:0'oohztcvkwo/strided_slice/stack:output:0)oohztcvkwo/strided_slice/stack_1:output:0)oohztcvkwo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
oohztcvkwo/strided_slicer
oohztcvkwo/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros/mul/y
oohztcvkwo/zeros/mulMul!oohztcvkwo/strided_slice:output:0oohztcvkwo/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros/mulu
oohztcvkwo/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
oohztcvkwo/zeros/Less/y
oohztcvkwo/zeros/LessLessoohztcvkwo/zeros/mul:z:0 oohztcvkwo/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros/Lessx
oohztcvkwo/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros/packed/1¯
oohztcvkwo/zeros/packedPack!oohztcvkwo/strided_slice:output:0"oohztcvkwo/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
oohztcvkwo/zeros/packedu
oohztcvkwo/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
oohztcvkwo/zeros/Const¡
oohztcvkwo/zerosFill oohztcvkwo/zeros/packed:output:0oohztcvkwo/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/zerosv
oohztcvkwo/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros_1/mul/y
oohztcvkwo/zeros_1/mulMul!oohztcvkwo/strided_slice:output:0!oohztcvkwo/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros_1/muly
oohztcvkwo/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
oohztcvkwo/zeros_1/Less/y
oohztcvkwo/zeros_1/LessLessoohztcvkwo/zeros_1/mul:z:0"oohztcvkwo/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
oohztcvkwo/zeros_1/Less|
oohztcvkwo/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/zeros_1/packed/1µ
oohztcvkwo/zeros_1/packedPack!oohztcvkwo/strided_slice:output:0$oohztcvkwo/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
oohztcvkwo/zeros_1/packedy
oohztcvkwo/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
oohztcvkwo/zeros_1/Const©
oohztcvkwo/zeros_1Fill"oohztcvkwo/zeros_1/packed:output:0!oohztcvkwo/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/zeros_1
oohztcvkwo/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
oohztcvkwo/transpose/perm°
oohztcvkwo/transpose	Transposeohvvfduigw/Reshape:output:0"oohztcvkwo/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/transposep
oohztcvkwo/Shape_1Shapeoohztcvkwo/transpose:y:0*
T0*
_output_shapes
:2
oohztcvkwo/Shape_1
 oohztcvkwo/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 oohztcvkwo/strided_slice_1/stack
"oohztcvkwo/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_1/stack_1
"oohztcvkwo/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_1/stack_2°
oohztcvkwo/strided_slice_1StridedSliceoohztcvkwo/Shape_1:output:0)oohztcvkwo/strided_slice_1/stack:output:0+oohztcvkwo/strided_slice_1/stack_1:output:0+oohztcvkwo/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
oohztcvkwo/strided_slice_1
&oohztcvkwo/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&oohztcvkwo/TensorArrayV2/element_shapeÞ
oohztcvkwo/TensorArrayV2TensorListReserve/oohztcvkwo/TensorArrayV2/element_shape:output:0#oohztcvkwo/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
oohztcvkwo/TensorArrayV2Õ
@oohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@oohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2oohztcvkwo/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoroohztcvkwo/transpose:y:0Ioohztcvkwo/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2oohztcvkwo/TensorArrayUnstack/TensorListFromTensor
 oohztcvkwo/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 oohztcvkwo/strided_slice_2/stack
"oohztcvkwo/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_2/stack_1
"oohztcvkwo/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_2/stack_2¾
oohztcvkwo/strided_slice_2StridedSliceoohztcvkwo/transpose:y:0)oohztcvkwo/strided_slice_2/stack:output:0+oohztcvkwo/strided_slice_2/stack_1:output:0+oohztcvkwo/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
oohztcvkwo/strided_slice_2Ð
+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp4oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOpÓ
oohztcvkwo/ammytzqwsz/MatMulMatMul#oohztcvkwo/strided_slice_2:output:03oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/ammytzqwsz/MatMulÖ
-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp6oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOpÏ
oohztcvkwo/ammytzqwsz/MatMul_1MatMuloohztcvkwo/zeros:output:05oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
oohztcvkwo/ammytzqwsz/MatMul_1Ä
oohztcvkwo/ammytzqwsz/addAddV2&oohztcvkwo/ammytzqwsz/MatMul:product:0(oohztcvkwo/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/ammytzqwsz/addÏ
,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp5oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOpÑ
oohztcvkwo/ammytzqwsz/BiasAddBiasAddoohztcvkwo/ammytzqwsz/add:z:04oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
oohztcvkwo/ammytzqwsz/BiasAdd
%oohztcvkwo/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%oohztcvkwo/ammytzqwsz/split/split_dim
oohztcvkwo/ammytzqwsz/splitSplit.oohztcvkwo/ammytzqwsz/split/split_dim:output:0&oohztcvkwo/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
oohztcvkwo/ammytzqwsz/split¶
$oohztcvkwo/ammytzqwsz/ReadVariableOpReadVariableOp-oohztcvkwo_ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02&
$oohztcvkwo/ammytzqwsz/ReadVariableOpº
oohztcvkwo/ammytzqwsz/mulMul,oohztcvkwo/ammytzqwsz/ReadVariableOp:value:0oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mulº
oohztcvkwo/ammytzqwsz/add_1AddV2$oohztcvkwo/ammytzqwsz/split:output:0oohztcvkwo/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_1
oohztcvkwo/ammytzqwsz/SigmoidSigmoidoohztcvkwo/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/Sigmoid¼
&oohztcvkwo/ammytzqwsz/ReadVariableOp_1ReadVariableOp/oohztcvkwo_ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&oohztcvkwo/ammytzqwsz/ReadVariableOp_1À
oohztcvkwo/ammytzqwsz/mul_1Mul.oohztcvkwo/ammytzqwsz/ReadVariableOp_1:value:0oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_1¼
oohztcvkwo/ammytzqwsz/add_2AddV2$oohztcvkwo/ammytzqwsz/split:output:1oohztcvkwo/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_2 
oohztcvkwo/ammytzqwsz/Sigmoid_1Sigmoidoohztcvkwo/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
oohztcvkwo/ammytzqwsz/Sigmoid_1µ
oohztcvkwo/ammytzqwsz/mul_2Mul#oohztcvkwo/ammytzqwsz/Sigmoid_1:y:0oohztcvkwo/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_2
oohztcvkwo/ammytzqwsz/TanhTanh$oohztcvkwo/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/Tanh¶
oohztcvkwo/ammytzqwsz/mul_3Mul!oohztcvkwo/ammytzqwsz/Sigmoid:y:0oohztcvkwo/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_3·
oohztcvkwo/ammytzqwsz/add_3AddV2oohztcvkwo/ammytzqwsz/mul_2:z:0oohztcvkwo/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_3¼
&oohztcvkwo/ammytzqwsz/ReadVariableOp_2ReadVariableOp/oohztcvkwo_ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&oohztcvkwo/ammytzqwsz/ReadVariableOp_2Ä
oohztcvkwo/ammytzqwsz/mul_4Mul.oohztcvkwo/ammytzqwsz/ReadVariableOp_2:value:0oohztcvkwo/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_4¼
oohztcvkwo/ammytzqwsz/add_4AddV2$oohztcvkwo/ammytzqwsz/split:output:3oohztcvkwo/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/add_4 
oohztcvkwo/ammytzqwsz/Sigmoid_2Sigmoidoohztcvkwo/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
oohztcvkwo/ammytzqwsz/Sigmoid_2
oohztcvkwo/ammytzqwsz/Tanh_1Tanhoohztcvkwo/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/Tanh_1º
oohztcvkwo/ammytzqwsz/mul_5Mul#oohztcvkwo/ammytzqwsz/Sigmoid_2:y:0 oohztcvkwo/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/ammytzqwsz/mul_5¥
(oohztcvkwo/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(oohztcvkwo/TensorArrayV2_1/element_shapeä
oohztcvkwo/TensorArrayV2_1TensorListReserve1oohztcvkwo/TensorArrayV2_1/element_shape:output:0#oohztcvkwo/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
oohztcvkwo/TensorArrayV2_1d
oohztcvkwo/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/time
#oohztcvkwo/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#oohztcvkwo/while/maximum_iterations
oohztcvkwo/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
oohztcvkwo/while/loop_counter²
oohztcvkwo/whileWhile&oohztcvkwo/while/loop_counter:output:0,oohztcvkwo/while/maximum_iterations:output:0oohztcvkwo/time:output:0#oohztcvkwo/TensorArrayV2_1:handle:0oohztcvkwo/zeros:output:0oohztcvkwo/zeros_1:output:0#oohztcvkwo/strided_slice_1:output:0Boohztcvkwo/TensorArrayUnstack/TensorListFromTensor:output_handle:04oohztcvkwo_ammytzqwsz_matmul_readvariableop_resource6oohztcvkwo_ammytzqwsz_matmul_1_readvariableop_resource5oohztcvkwo_ammytzqwsz_biasadd_readvariableop_resource-oohztcvkwo_ammytzqwsz_readvariableop_resource/oohztcvkwo_ammytzqwsz_readvariableop_1_resource/oohztcvkwo_ammytzqwsz_readvariableop_2_resource*
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
oohztcvkwo_while_body_1822127*)
cond!R
oohztcvkwo_while_cond_1822126*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
oohztcvkwo/whileË
;oohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;oohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shape
-oohztcvkwo/TensorArrayV2Stack/TensorListStackTensorListStackoohztcvkwo/while:output:3Doohztcvkwo/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-oohztcvkwo/TensorArrayV2Stack/TensorListStack
 oohztcvkwo/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 oohztcvkwo/strided_slice_3/stack
"oohztcvkwo/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"oohztcvkwo/strided_slice_3/stack_1
"oohztcvkwo/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"oohztcvkwo/strided_slice_3/stack_2Ü
oohztcvkwo/strided_slice_3StridedSlice6oohztcvkwo/TensorArrayV2Stack/TensorListStack:tensor:0)oohztcvkwo/strided_slice_3/stack:output:0+oohztcvkwo/strided_slice_3/stack_1:output:0+oohztcvkwo/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
oohztcvkwo/strided_slice_3
oohztcvkwo/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
oohztcvkwo/transpose_1/permÑ
oohztcvkwo/transpose_1	Transpose6oohztcvkwo/TensorArrayV2Stack/TensorListStack:tensor:0$oohztcvkwo/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
oohztcvkwo/transpose_1n
mfdtyewult/ShapeShapeoohztcvkwo/transpose_1:y:0*
T0*
_output_shapes
:2
mfdtyewult/Shape
mfdtyewult/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
mfdtyewult/strided_slice/stack
 mfdtyewult/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 mfdtyewult/strided_slice/stack_1
 mfdtyewult/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 mfdtyewult/strided_slice/stack_2¤
mfdtyewult/strided_sliceStridedSlicemfdtyewult/Shape:output:0'mfdtyewult/strided_slice/stack:output:0)mfdtyewult/strided_slice/stack_1:output:0)mfdtyewult/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mfdtyewult/strided_slicer
mfdtyewult/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros/mul/y
mfdtyewult/zeros/mulMul!mfdtyewult/strided_slice:output:0mfdtyewult/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros/mulu
mfdtyewult/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
mfdtyewult/zeros/Less/y
mfdtyewult/zeros/LessLessmfdtyewult/zeros/mul:z:0 mfdtyewult/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros/Lessx
mfdtyewult/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros/packed/1¯
mfdtyewult/zeros/packedPack!mfdtyewult/strided_slice:output:0"mfdtyewult/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
mfdtyewult/zeros/packedu
mfdtyewult/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mfdtyewult/zeros/Const¡
mfdtyewult/zerosFill mfdtyewult/zeros/packed:output:0mfdtyewult/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/zerosv
mfdtyewult/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros_1/mul/y
mfdtyewult/zeros_1/mulMul!mfdtyewult/strided_slice:output:0!mfdtyewult/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros_1/muly
mfdtyewult/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
mfdtyewult/zeros_1/Less/y
mfdtyewult/zeros_1/LessLessmfdtyewult/zeros_1/mul:z:0"mfdtyewult/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
mfdtyewult/zeros_1/Less|
mfdtyewult/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/zeros_1/packed/1µ
mfdtyewult/zeros_1/packedPack!mfdtyewult/strided_slice:output:0$mfdtyewult/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
mfdtyewult/zeros_1/packedy
mfdtyewult/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mfdtyewult/zeros_1/Const©
mfdtyewult/zeros_1Fill"mfdtyewult/zeros_1/packed:output:0!mfdtyewult/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/zeros_1
mfdtyewult/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mfdtyewult/transpose/perm¯
mfdtyewult/transpose	Transposeoohztcvkwo/transpose_1:y:0"mfdtyewult/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/transposep
mfdtyewult/Shape_1Shapemfdtyewult/transpose:y:0*
T0*
_output_shapes
:2
mfdtyewult/Shape_1
 mfdtyewult/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 mfdtyewult/strided_slice_1/stack
"mfdtyewult/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_1/stack_1
"mfdtyewult/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_1/stack_2°
mfdtyewult/strided_slice_1StridedSlicemfdtyewult/Shape_1:output:0)mfdtyewult/strided_slice_1/stack:output:0+mfdtyewult/strided_slice_1/stack_1:output:0+mfdtyewult/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
mfdtyewult/strided_slice_1
&mfdtyewult/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&mfdtyewult/TensorArrayV2/element_shapeÞ
mfdtyewult/TensorArrayV2TensorListReserve/mfdtyewult/TensorArrayV2/element_shape:output:0#mfdtyewult/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
mfdtyewult/TensorArrayV2Õ
@mfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@mfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2mfdtyewult/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormfdtyewult/transpose:y:0Imfdtyewult/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2mfdtyewult/TensorArrayUnstack/TensorListFromTensor
 mfdtyewult/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 mfdtyewult/strided_slice_2/stack
"mfdtyewult/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_2/stack_1
"mfdtyewult/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_2/stack_2¾
mfdtyewult/strided_slice_2StridedSlicemfdtyewult/transpose:y:0)mfdtyewult/strided_slice_2/stack:output:0+mfdtyewult/strided_slice_2/stack_1:output:0+mfdtyewult/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
mfdtyewult/strided_slice_2Ð
+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp4mfdtyewult_aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOpÓ
mfdtyewult/aflyrndiyz/MatMulMatMul#mfdtyewult/strided_slice_2:output:03mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mfdtyewult/aflyrndiyz/MatMulÖ
-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp6mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOpÏ
mfdtyewult/aflyrndiyz/MatMul_1MatMulmfdtyewult/zeros:output:05mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
mfdtyewult/aflyrndiyz/MatMul_1Ä
mfdtyewult/aflyrndiyz/addAddV2&mfdtyewult/aflyrndiyz/MatMul:product:0(mfdtyewult/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mfdtyewult/aflyrndiyz/addÏ
,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp5mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOpÑ
mfdtyewult/aflyrndiyz/BiasAddBiasAddmfdtyewult/aflyrndiyz/add:z:04mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mfdtyewult/aflyrndiyz/BiasAdd
%mfdtyewult/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%mfdtyewult/aflyrndiyz/split/split_dim
mfdtyewult/aflyrndiyz/splitSplit.mfdtyewult/aflyrndiyz/split/split_dim:output:0&mfdtyewult/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
mfdtyewult/aflyrndiyz/split¶
$mfdtyewult/aflyrndiyz/ReadVariableOpReadVariableOp-mfdtyewult_aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02&
$mfdtyewult/aflyrndiyz/ReadVariableOpº
mfdtyewult/aflyrndiyz/mulMul,mfdtyewult/aflyrndiyz/ReadVariableOp:value:0mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mulº
mfdtyewult/aflyrndiyz/add_1AddV2$mfdtyewult/aflyrndiyz/split:output:0mfdtyewult/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_1
mfdtyewult/aflyrndiyz/SigmoidSigmoidmfdtyewult/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/Sigmoid¼
&mfdtyewult/aflyrndiyz/ReadVariableOp_1ReadVariableOp/mfdtyewult_aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&mfdtyewult/aflyrndiyz/ReadVariableOp_1À
mfdtyewult/aflyrndiyz/mul_1Mul.mfdtyewult/aflyrndiyz/ReadVariableOp_1:value:0mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_1¼
mfdtyewult/aflyrndiyz/add_2AddV2$mfdtyewult/aflyrndiyz/split:output:1mfdtyewult/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_2 
mfdtyewult/aflyrndiyz/Sigmoid_1Sigmoidmfdtyewult/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
mfdtyewult/aflyrndiyz/Sigmoid_1µ
mfdtyewult/aflyrndiyz/mul_2Mul#mfdtyewult/aflyrndiyz/Sigmoid_1:y:0mfdtyewult/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_2
mfdtyewult/aflyrndiyz/TanhTanh$mfdtyewult/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/Tanh¶
mfdtyewult/aflyrndiyz/mul_3Mul!mfdtyewult/aflyrndiyz/Sigmoid:y:0mfdtyewult/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_3·
mfdtyewult/aflyrndiyz/add_3AddV2mfdtyewult/aflyrndiyz/mul_2:z:0mfdtyewult/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_3¼
&mfdtyewult/aflyrndiyz/ReadVariableOp_2ReadVariableOp/mfdtyewult_aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&mfdtyewult/aflyrndiyz/ReadVariableOp_2Ä
mfdtyewult/aflyrndiyz/mul_4Mul.mfdtyewult/aflyrndiyz/ReadVariableOp_2:value:0mfdtyewult/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_4¼
mfdtyewult/aflyrndiyz/add_4AddV2$mfdtyewult/aflyrndiyz/split:output:3mfdtyewult/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/add_4 
mfdtyewult/aflyrndiyz/Sigmoid_2Sigmoidmfdtyewult/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
mfdtyewult/aflyrndiyz/Sigmoid_2
mfdtyewult/aflyrndiyz/Tanh_1Tanhmfdtyewult/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/Tanh_1º
mfdtyewult/aflyrndiyz/mul_5Mul#mfdtyewult/aflyrndiyz/Sigmoid_2:y:0 mfdtyewult/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/aflyrndiyz/mul_5¥
(mfdtyewult/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(mfdtyewult/TensorArrayV2_1/element_shapeä
mfdtyewult/TensorArrayV2_1TensorListReserve1mfdtyewult/TensorArrayV2_1/element_shape:output:0#mfdtyewult/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
mfdtyewult/TensorArrayV2_1d
mfdtyewult/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/time
#mfdtyewult/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#mfdtyewult/while/maximum_iterations
mfdtyewult/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
mfdtyewult/while/loop_counter²
mfdtyewult/whileWhile&mfdtyewult/while/loop_counter:output:0,mfdtyewult/while/maximum_iterations:output:0mfdtyewult/time:output:0#mfdtyewult/TensorArrayV2_1:handle:0mfdtyewult/zeros:output:0mfdtyewult/zeros_1:output:0#mfdtyewult/strided_slice_1:output:0Bmfdtyewult/TensorArrayUnstack/TensorListFromTensor:output_handle:04mfdtyewult_aflyrndiyz_matmul_readvariableop_resource6mfdtyewult_aflyrndiyz_matmul_1_readvariableop_resource5mfdtyewult_aflyrndiyz_biasadd_readvariableop_resource-mfdtyewult_aflyrndiyz_readvariableop_resource/mfdtyewult_aflyrndiyz_readvariableop_1_resource/mfdtyewult_aflyrndiyz_readvariableop_2_resource*
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
mfdtyewult_while_body_1822303*)
cond!R
mfdtyewult_while_cond_1822302*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
mfdtyewult/whileË
;mfdtyewult/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;mfdtyewult/TensorArrayV2Stack/TensorListStack/element_shape
-mfdtyewult/TensorArrayV2Stack/TensorListStackTensorListStackmfdtyewult/while:output:3Dmfdtyewult/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-mfdtyewult/TensorArrayV2Stack/TensorListStack
 mfdtyewult/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 mfdtyewult/strided_slice_3/stack
"mfdtyewult/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"mfdtyewult/strided_slice_3/stack_1
"mfdtyewult/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"mfdtyewult/strided_slice_3/stack_2Ü
mfdtyewult/strided_slice_3StridedSlice6mfdtyewult/TensorArrayV2Stack/TensorListStack:tensor:0)mfdtyewult/strided_slice_3/stack:output:0+mfdtyewult/strided_slice_3/stack_1:output:0+mfdtyewult/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
mfdtyewult/strided_slice_3
mfdtyewult/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
mfdtyewult/transpose_1/permÑ
mfdtyewult/transpose_1	Transpose6mfdtyewult/TensorArrayV2Stack/TensorListStack:tensor:0$mfdtyewult/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mfdtyewult/transpose_1®
 boyhyiogqf/MatMul/ReadVariableOpReadVariableOp)boyhyiogqf_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 boyhyiogqf/MatMul/ReadVariableOp±
boyhyiogqf/MatMulMatMul#mfdtyewult/strided_slice_3:output:0(boyhyiogqf/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
boyhyiogqf/MatMul­
!boyhyiogqf/BiasAdd/ReadVariableOpReadVariableOp*boyhyiogqf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!boyhyiogqf/BiasAdd/ReadVariableOp­
boyhyiogqf/BiasAddBiasAddboyhyiogqf/MatMul:product:0)boyhyiogqf/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
boyhyiogqf/BiasAddÏ
IdentityIdentityboyhyiogqf/BiasAdd:output:0"^boyhyiogqf/BiasAdd/ReadVariableOp!^boyhyiogqf/MatMul/ReadVariableOp.^fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp5^fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp-^mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp,^mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp.^mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp%^mfdtyewult/aflyrndiyz/ReadVariableOp'^mfdtyewult/aflyrndiyz/ReadVariableOp_1'^mfdtyewult/aflyrndiyz/ReadVariableOp_2^mfdtyewult/while-^oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp,^oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp.^oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp%^oohztcvkwo/ammytzqwsz/ReadVariableOp'^oohztcvkwo/ammytzqwsz/ReadVariableOp_1'^oohztcvkwo/ammytzqwsz/ReadVariableOp_2^oohztcvkwo/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2F
!boyhyiogqf/BiasAdd/ReadVariableOp!boyhyiogqf/BiasAdd/ReadVariableOp2D
 boyhyiogqf/MatMul/ReadVariableOp boyhyiogqf/MatMul/ReadVariableOp2^
-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp-fethhjgisa/conv1d/ExpandDims_1/ReadVariableOp2l
4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp4fethhjgisa/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp,mfdtyewult/aflyrndiyz/BiasAdd/ReadVariableOp2Z
+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp+mfdtyewult/aflyrndiyz/MatMul/ReadVariableOp2^
-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp-mfdtyewult/aflyrndiyz/MatMul_1/ReadVariableOp2L
$mfdtyewult/aflyrndiyz/ReadVariableOp$mfdtyewult/aflyrndiyz/ReadVariableOp2P
&mfdtyewult/aflyrndiyz/ReadVariableOp_1&mfdtyewult/aflyrndiyz/ReadVariableOp_12P
&mfdtyewult/aflyrndiyz/ReadVariableOp_2&mfdtyewult/aflyrndiyz/ReadVariableOp_22$
mfdtyewult/whilemfdtyewult/while2\
,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp,oohztcvkwo/ammytzqwsz/BiasAdd/ReadVariableOp2Z
+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp+oohztcvkwo/ammytzqwsz/MatMul/ReadVariableOp2^
-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp-oohztcvkwo/ammytzqwsz/MatMul_1/ReadVariableOp2L
$oohztcvkwo/ammytzqwsz/ReadVariableOp$oohztcvkwo/ammytzqwsz/ReadVariableOp2P
&oohztcvkwo/ammytzqwsz/ReadVariableOp_1&oohztcvkwo/ammytzqwsz/ReadVariableOp_12P
&oohztcvkwo/ammytzqwsz/ReadVariableOp_2&oohztcvkwo/ammytzqwsz/ReadVariableOp_22$
oohztcvkwo/whileoohztcvkwo/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©0
¼
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_1822521

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
while_body_1820509
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_ammytzqwsz_matmul_readvariableop_resource_0:	F
3while_ammytzqwsz_matmul_1_readvariableop_resource_0:	 A
2while_ammytzqwsz_biasadd_readvariableop_resource_0:	8
*while_ammytzqwsz_readvariableop_resource_0: :
,while_ammytzqwsz_readvariableop_1_resource_0: :
,while_ammytzqwsz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_ammytzqwsz_matmul_readvariableop_resource:	D
1while_ammytzqwsz_matmul_1_readvariableop_resource:	 ?
0while_ammytzqwsz_biasadd_readvariableop_resource:	6
(while_ammytzqwsz_readvariableop_resource: 8
*while_ammytzqwsz_readvariableop_1_resource: 8
*while_ammytzqwsz_readvariableop_2_resource: ¢'while/ammytzqwsz/BiasAdd/ReadVariableOp¢&while/ammytzqwsz/MatMul/ReadVariableOp¢(while/ammytzqwsz/MatMul_1/ReadVariableOp¢while/ammytzqwsz/ReadVariableOp¢!while/ammytzqwsz/ReadVariableOp_1¢!while/ammytzqwsz/ReadVariableOp_2Ã
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
&while/ammytzqwsz/MatMul/ReadVariableOpReadVariableOp1while_ammytzqwsz_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/ammytzqwsz/MatMul/ReadVariableOpÑ
while/ammytzqwsz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMulÉ
(while/ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp3while_ammytzqwsz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/ammytzqwsz/MatMul_1/ReadVariableOpº
while/ammytzqwsz/MatMul_1MatMulwhile_placeholder_20while/ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/MatMul_1°
while/ammytzqwsz/addAddV2!while/ammytzqwsz/MatMul:product:0#while/ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/addÂ
'while/ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp2while_ammytzqwsz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/ammytzqwsz/BiasAdd/ReadVariableOp½
while/ammytzqwsz/BiasAddBiasAddwhile/ammytzqwsz/add:z:0/while/ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/ammytzqwsz/BiasAdd
 while/ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/ammytzqwsz/split/split_dim
while/ammytzqwsz/splitSplit)while/ammytzqwsz/split/split_dim:output:0!while/ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/ammytzqwsz/split©
while/ammytzqwsz/ReadVariableOpReadVariableOp*while_ammytzqwsz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/ammytzqwsz/ReadVariableOp£
while/ammytzqwsz/mulMul'while/ammytzqwsz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul¦
while/ammytzqwsz/add_1AddV2while/ammytzqwsz/split:output:0while/ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_1
while/ammytzqwsz/SigmoidSigmoidwhile/ammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid¯
!while/ammytzqwsz/ReadVariableOp_1ReadVariableOp,while_ammytzqwsz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_1©
while/ammytzqwsz/mul_1Mul)while/ammytzqwsz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_1¨
while/ammytzqwsz/add_2AddV2while/ammytzqwsz/split:output:1while/ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_2
while/ammytzqwsz/Sigmoid_1Sigmoidwhile/ammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_1
while/ammytzqwsz/mul_2Mulwhile/ammytzqwsz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_2
while/ammytzqwsz/TanhTanhwhile/ammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh¢
while/ammytzqwsz/mul_3Mulwhile/ammytzqwsz/Sigmoid:y:0while/ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_3£
while/ammytzqwsz/add_3AddV2while/ammytzqwsz/mul_2:z:0while/ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_3¯
!while/ammytzqwsz/ReadVariableOp_2ReadVariableOp,while_ammytzqwsz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/ammytzqwsz/ReadVariableOp_2°
while/ammytzqwsz/mul_4Mul)while/ammytzqwsz/ReadVariableOp_2:value:0while/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_4¨
while/ammytzqwsz/add_4AddV2while/ammytzqwsz/split:output:3while/ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/add_4
while/ammytzqwsz/Sigmoid_2Sigmoidwhile/ammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Sigmoid_2
while/ammytzqwsz/Tanh_1Tanhwhile/ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/Tanh_1¦
while/ammytzqwsz/mul_5Mulwhile/ammytzqwsz/Sigmoid_2:y:0while/ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/ammytzqwsz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/ammytzqwsz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/ammytzqwsz/mul_5:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/ammytzqwsz/add_3:z:0(^while/ammytzqwsz/BiasAdd/ReadVariableOp'^while/ammytzqwsz/MatMul/ReadVariableOp)^while/ammytzqwsz/MatMul_1/ReadVariableOp ^while/ammytzqwsz/ReadVariableOp"^while/ammytzqwsz/ReadVariableOp_1"^while/ammytzqwsz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_ammytzqwsz_biasadd_readvariableop_resource2while_ammytzqwsz_biasadd_readvariableop_resource_0"h
1while_ammytzqwsz_matmul_1_readvariableop_resource3while_ammytzqwsz_matmul_1_readvariableop_resource_0"d
/while_ammytzqwsz_matmul_readvariableop_resource1while_ammytzqwsz_matmul_readvariableop_resource_0"Z
*while_ammytzqwsz_readvariableop_1_resource,while_ammytzqwsz_readvariableop_1_resource_0"Z
*while_ammytzqwsz_readvariableop_2_resource,while_ammytzqwsz_readvariableop_2_resource_0"V
(while_ammytzqwsz_readvariableop_resource*while_ammytzqwsz_readvariableop_resource_0")
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
'while/ammytzqwsz/BiasAdd/ReadVariableOp'while/ammytzqwsz/BiasAdd/ReadVariableOp2P
&while/ammytzqwsz/MatMul/ReadVariableOp&while/ammytzqwsz/MatMul/ReadVariableOp2T
(while/ammytzqwsz/MatMul_1/ReadVariableOp(while/ammytzqwsz/MatMul_1/ReadVariableOp2B
while/ammytzqwsz/ReadVariableOpwhile/ammytzqwsz/ReadVariableOp2F
!while/ammytzqwsz/ReadVariableOp_1!while/ammytzqwsz/ReadVariableOp_12F
!while/ammytzqwsz/ReadVariableOp_2!while/ammytzqwsz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_1823415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aflyrndiyz_matmul_readvariableop_resource_0:	 F
3while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 A
2while_aflyrndiyz_biasadd_readvariableop_resource_0:	8
*while_aflyrndiyz_readvariableop_resource_0: :
,while_aflyrndiyz_readvariableop_1_resource_0: :
,while_aflyrndiyz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aflyrndiyz_matmul_readvariableop_resource:	 D
1while_aflyrndiyz_matmul_1_readvariableop_resource:	 ?
0while_aflyrndiyz_biasadd_readvariableop_resource:	6
(while_aflyrndiyz_readvariableop_resource: 8
*while_aflyrndiyz_readvariableop_1_resource: 8
*while_aflyrndiyz_readvariableop_2_resource: ¢'while/aflyrndiyz/BiasAdd/ReadVariableOp¢&while/aflyrndiyz/MatMul/ReadVariableOp¢(while/aflyrndiyz/MatMul_1/ReadVariableOp¢while/aflyrndiyz/ReadVariableOp¢!while/aflyrndiyz/ReadVariableOp_1¢!while/aflyrndiyz/ReadVariableOp_2Ã
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
&while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp1while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aflyrndiyz/MatMul/ReadVariableOpÑ
while/aflyrndiyz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMulÉ
(while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp3while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aflyrndiyz/MatMul_1/ReadVariableOpº
while/aflyrndiyz/MatMul_1MatMulwhile_placeholder_20while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMul_1°
while/aflyrndiyz/addAddV2!while/aflyrndiyz/MatMul:product:0#while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/addÂ
'while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp2while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aflyrndiyz/BiasAdd/ReadVariableOp½
while/aflyrndiyz/BiasAddBiasAddwhile/aflyrndiyz/add:z:0/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/BiasAdd
 while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aflyrndiyz/split/split_dim
while/aflyrndiyz/splitSplit)while/aflyrndiyz/split/split_dim:output:0!while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aflyrndiyz/split©
while/aflyrndiyz/ReadVariableOpReadVariableOp*while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aflyrndiyz/ReadVariableOp£
while/aflyrndiyz/mulMul'while/aflyrndiyz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul¦
while/aflyrndiyz/add_1AddV2while/aflyrndiyz/split:output:0while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_1
while/aflyrndiyz/SigmoidSigmoidwhile/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid¯
!while/aflyrndiyz/ReadVariableOp_1ReadVariableOp,while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_1©
while/aflyrndiyz/mul_1Mul)while/aflyrndiyz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_1¨
while/aflyrndiyz/add_2AddV2while/aflyrndiyz/split:output:1while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_2
while/aflyrndiyz/Sigmoid_1Sigmoidwhile/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_1
while/aflyrndiyz/mul_2Mulwhile/aflyrndiyz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_2
while/aflyrndiyz/TanhTanhwhile/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh¢
while/aflyrndiyz/mul_3Mulwhile/aflyrndiyz/Sigmoid:y:0while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_3£
while/aflyrndiyz/add_3AddV2while/aflyrndiyz/mul_2:z:0while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_3¯
!while/aflyrndiyz/ReadVariableOp_2ReadVariableOp,while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_2°
while/aflyrndiyz/mul_4Mul)while/aflyrndiyz/ReadVariableOp_2:value:0while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_4¨
while/aflyrndiyz/add_4AddV2while/aflyrndiyz/split:output:3while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_4
while/aflyrndiyz/Sigmoid_2Sigmoidwhile/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_2
while/aflyrndiyz/Tanh_1Tanhwhile/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh_1¦
while/aflyrndiyz/mul_5Mulwhile/aflyrndiyz/Sigmoid_2:y:0while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aflyrndiyz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aflyrndiyz/mul_5:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aflyrndiyz/add_3:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aflyrndiyz_biasadd_readvariableop_resource2while_aflyrndiyz_biasadd_readvariableop_resource_0"h
1while_aflyrndiyz_matmul_1_readvariableop_resource3while_aflyrndiyz_matmul_1_readvariableop_resource_0"d
/while_aflyrndiyz_matmul_readvariableop_resource1while_aflyrndiyz_matmul_readvariableop_resource_0"Z
*while_aflyrndiyz_readvariableop_1_resource,while_aflyrndiyz_readvariableop_1_resource_0"Z
*while_aflyrndiyz_readvariableop_2_resource,while_aflyrndiyz_readvariableop_2_resource_0"V
(while_aflyrndiyz_readvariableop_resource*while_aflyrndiyz_readvariableop_resource_0")
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
'while/aflyrndiyz/BiasAdd/ReadVariableOp'while/aflyrndiyz/BiasAdd/ReadVariableOp2P
&while/aflyrndiyz/MatMul/ReadVariableOp&while/aflyrndiyz/MatMul/ReadVariableOp2T
(while/aflyrndiyz/MatMul_1/ReadVariableOp(while/aflyrndiyz/MatMul_1/ReadVariableOp2B
while/aflyrndiyz/ReadVariableOpwhile/aflyrndiyz/ReadVariableOp2F
!while/aflyrndiyz/ReadVariableOp_1!while/aflyrndiyz/ReadVariableOp_12F
!while/aflyrndiyz/ReadVariableOp_2!while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_mfdtyewult_layer_call_fn_1824124

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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18210782
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
¡h

G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1820610

inputs<
)ammytzqwsz_matmul_readvariableop_resource:	>
+ammytzqwsz_matmul_1_readvariableop_resource:	 9
*ammytzqwsz_biasadd_readvariableop_resource:	0
"ammytzqwsz_readvariableop_resource: 2
$ammytzqwsz_readvariableop_1_resource: 2
$ammytzqwsz_readvariableop_2_resource: 
identity¢!ammytzqwsz/BiasAdd/ReadVariableOp¢ ammytzqwsz/MatMul/ReadVariableOp¢"ammytzqwsz/MatMul_1/ReadVariableOp¢ammytzqwsz/ReadVariableOp¢ammytzqwsz/ReadVariableOp_1¢ammytzqwsz/ReadVariableOp_2¢whileD
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
 ammytzqwsz/MatMul/ReadVariableOpReadVariableOp)ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ammytzqwsz/MatMul/ReadVariableOp§
ammytzqwsz/MatMulMatMulstrided_slice_2:output:0(ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMulµ
"ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp+ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ammytzqwsz/MatMul_1/ReadVariableOp£
ammytzqwsz/MatMul_1MatMulzeros:output:0*ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMul_1
ammytzqwsz/addAddV2ammytzqwsz/MatMul:product:0ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/add®
!ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp*ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ammytzqwsz/BiasAdd/ReadVariableOp¥
ammytzqwsz/BiasAddBiasAddammytzqwsz/add:z:0)ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/BiasAddz
ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ammytzqwsz/split/split_dimë
ammytzqwsz/splitSplit#ammytzqwsz/split/split_dim:output:0ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ammytzqwsz/split
ammytzqwsz/ReadVariableOpReadVariableOp"ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp
ammytzqwsz/mulMul!ammytzqwsz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul
ammytzqwsz/add_1AddV2ammytzqwsz/split:output:0ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_1{
ammytzqwsz/SigmoidSigmoidammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid
ammytzqwsz/ReadVariableOp_1ReadVariableOp$ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_1
ammytzqwsz/mul_1Mul#ammytzqwsz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_1
ammytzqwsz/add_2AddV2ammytzqwsz/split:output:1ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_2
ammytzqwsz/Sigmoid_1Sigmoidammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_1
ammytzqwsz/mul_2Mulammytzqwsz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_2w
ammytzqwsz/TanhTanhammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh
ammytzqwsz/mul_3Mulammytzqwsz/Sigmoid:y:0ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_3
ammytzqwsz/add_3AddV2ammytzqwsz/mul_2:z:0ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_3
ammytzqwsz/ReadVariableOp_2ReadVariableOp$ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_2
ammytzqwsz/mul_4Mul#ammytzqwsz/ReadVariableOp_2:value:0ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_4
ammytzqwsz/add_4AddV2ammytzqwsz/split:output:3ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_4
ammytzqwsz/Sigmoid_2Sigmoidammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_2v
ammytzqwsz/Tanh_1Tanhammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh_1
ammytzqwsz/mul_5Mulammytzqwsz/Sigmoid_2:y:0ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ammytzqwsz_matmul_readvariableop_resource+ammytzqwsz_matmul_1_readvariableop_resource*ammytzqwsz_biasadd_readvariableop_resource"ammytzqwsz_readvariableop_resource$ammytzqwsz_readvariableop_1_resource$ammytzqwsz_readvariableop_2_resource*
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
while_body_1820509*
condR
while_cond_1820508*Q
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
IdentityIdentitytranspose_1:y:0"^ammytzqwsz/BiasAdd/ReadVariableOp!^ammytzqwsz/MatMul/ReadVariableOp#^ammytzqwsz/MatMul_1/ReadVariableOp^ammytzqwsz/ReadVariableOp^ammytzqwsz/ReadVariableOp_1^ammytzqwsz/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ammytzqwsz/BiasAdd/ReadVariableOp!ammytzqwsz/BiasAdd/ReadVariableOp2D
 ammytzqwsz/MatMul/ReadVariableOp ammytzqwsz/MatMul/ReadVariableOp2H
"ammytzqwsz/MatMul_1/ReadVariableOp"ammytzqwsz/MatMul_1/ReadVariableOp26
ammytzqwsz/ReadVariableOpammytzqwsz/ReadVariableOp2:
ammytzqwsz/ReadVariableOp_1ammytzqwsz/ReadVariableOp_12:
ammytzqwsz/ReadVariableOp_2ammytzqwsz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_fethhjgisa_layer_call_fn_1822530

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
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_18204102
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
³F
ê
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1819300

inputs%
ammytzqwsz_1819201:	%
ammytzqwsz_1819203:	 !
ammytzqwsz_1819205:	 
ammytzqwsz_1819207:  
ammytzqwsz_1819209:  
ammytzqwsz_1819211: 
identity¢"ammytzqwsz/StatefulPartitionedCall¢whileD
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
"ammytzqwsz/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0ammytzqwsz_1819201ammytzqwsz_1819203ammytzqwsz_1819205ammytzqwsz_1819207ammytzqwsz_1819209ammytzqwsz_1819211*
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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_18191242$
"ammytzqwsz/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0ammytzqwsz_1819201ammytzqwsz_1819203ammytzqwsz_1819205ammytzqwsz_1819207ammytzqwsz_1819209ammytzqwsz_1819211*
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
while_body_1819220*
condR
while_cond_1819219*Q
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
IdentityIdentitytranspose_1:y:0#^ammytzqwsz/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"ammytzqwsz/StatefulPartitionedCall"ammytzqwsz/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

#__inference__traced_restore_1824678
file_prefix8
"assignvariableop_fethhjgisa_kernel:0
"assignvariableop_1_fethhjgisa_bias:6
$assignvariableop_2_boyhyiogqf_kernel: 0
"assignvariableop_3_boyhyiogqf_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_oohztcvkwo_ammytzqwsz_kernel:	M
:assignvariableop_10_oohztcvkwo_ammytzqwsz_recurrent_kernel:	 =
.assignvariableop_11_oohztcvkwo_ammytzqwsz_bias:	S
Eassignvariableop_12_oohztcvkwo_ammytzqwsz_input_gate_peephole_weights: T
Fassignvariableop_13_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weights: T
Fassignvariableop_14_oohztcvkwo_ammytzqwsz_output_gate_peephole_weights: C
0assignvariableop_15_mfdtyewult_aflyrndiyz_kernel:	 M
:assignvariableop_16_mfdtyewult_aflyrndiyz_recurrent_kernel:	 =
.assignvariableop_17_mfdtyewult_aflyrndiyz_bias:	S
Eassignvariableop_18_mfdtyewult_aflyrndiyz_input_gate_peephole_weights: T
Fassignvariableop_19_mfdtyewult_aflyrndiyz_forget_gate_peephole_weights: T
Fassignvariableop_20_mfdtyewult_aflyrndiyz_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_fethhjgisa_kernel_rms:=
/assignvariableop_24_rmsprop_fethhjgisa_bias_rms:C
1assignvariableop_25_rmsprop_boyhyiogqf_kernel_rms: =
/assignvariableop_26_rmsprop_boyhyiogqf_bias_rms:O
<assignvariableop_27_rmsprop_oohztcvkwo_ammytzqwsz_kernel_rms:	Y
Fassignvariableop_28_rmsprop_oohztcvkwo_ammytzqwsz_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_oohztcvkwo_ammytzqwsz_bias_rms:	_
Qassignvariableop_30_rmsprop_oohztcvkwo_ammytzqwsz_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_oohztcvkwo_ammytzqwsz_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_mfdtyewult_aflyrndiyz_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_mfdtyewult_aflyrndiyz_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_mfdtyewult_aflyrndiyz_bias_rms:	_
Qassignvariableop_36_rmsprop_mfdtyewult_aflyrndiyz_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_mfdtyewult_aflyrndiyz_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_mfdtyewult_aflyrndiyz_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_fethhjgisa_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_fethhjgisa_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_boyhyiogqf_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_boyhyiogqf_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_oohztcvkwo_ammytzqwsz_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_oohztcvkwo_ammytzqwsz_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_oohztcvkwo_ammytzqwsz_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_oohztcvkwo_ammytzqwsz_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_oohztcvkwo_ammytzqwsz_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_mfdtyewult_aflyrndiyz_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_mfdtyewult_aflyrndiyz_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_mfdtyewult_aflyrndiyz_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_mfdtyewult_aflyrndiyz_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_mfdtyewult_aflyrndiyz_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_mfdtyewult_aflyrndiyz_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_fethhjgisa_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_fethhjgisa_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_boyhyiogqf_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_boyhyiogqf_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_oohztcvkwo_ammytzqwsz_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_oohztcvkwo_ammytzqwsz_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_oohztcvkwo_ammytzqwsz_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_oohztcvkwo_ammytzqwsz_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_oohztcvkwo_ammytzqwsz_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_mfdtyewult_aflyrndiyz_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_mfdtyewult_aflyrndiyz_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_mfdtyewult_aflyrndiyz_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_mfdtyewult_aflyrndiyz_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_mfdtyewult_aflyrndiyz_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_mfdtyewult_aflyrndiyz_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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
while_cond_1823594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1823594___redundant_placeholder05
1while_while_cond_1823594___redundant_placeholder15
1while_while_cond_1823594___redundant_placeholder25
1while_while_cond_1823594___redundant_placeholder35
1while_while_cond_1823594___redundant_placeholder45
1while_while_cond_1823594___redundant_placeholder55
1while_while_cond_1823594___redundant_placeholder6
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823876

inputs<
)aflyrndiyz_matmul_readvariableop_resource:	 >
+aflyrndiyz_matmul_1_readvariableop_resource:	 9
*aflyrndiyz_biasadd_readvariableop_resource:	0
"aflyrndiyz_readvariableop_resource: 2
$aflyrndiyz_readvariableop_1_resource: 2
$aflyrndiyz_readvariableop_2_resource: 
identity¢!aflyrndiyz/BiasAdd/ReadVariableOp¢ aflyrndiyz/MatMul/ReadVariableOp¢"aflyrndiyz/MatMul_1/ReadVariableOp¢aflyrndiyz/ReadVariableOp¢aflyrndiyz/ReadVariableOp_1¢aflyrndiyz/ReadVariableOp_2¢whileD
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
 aflyrndiyz/MatMul/ReadVariableOpReadVariableOp)aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aflyrndiyz/MatMul/ReadVariableOp§
aflyrndiyz/MatMulMatMulstrided_slice_2:output:0(aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMulµ
"aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp+aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aflyrndiyz/MatMul_1/ReadVariableOp£
aflyrndiyz/MatMul_1MatMulzeros:output:0*aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMul_1
aflyrndiyz/addAddV2aflyrndiyz/MatMul:product:0aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/add®
!aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp*aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aflyrndiyz/BiasAdd/ReadVariableOp¥
aflyrndiyz/BiasAddBiasAddaflyrndiyz/add:z:0)aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/BiasAddz
aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aflyrndiyz/split/split_dimë
aflyrndiyz/splitSplit#aflyrndiyz/split/split_dim:output:0aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aflyrndiyz/split
aflyrndiyz/ReadVariableOpReadVariableOp"aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp
aflyrndiyz/mulMul!aflyrndiyz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul
aflyrndiyz/add_1AddV2aflyrndiyz/split:output:0aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_1{
aflyrndiyz/SigmoidSigmoidaflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid
aflyrndiyz/ReadVariableOp_1ReadVariableOp$aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_1
aflyrndiyz/mul_1Mul#aflyrndiyz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_1
aflyrndiyz/add_2AddV2aflyrndiyz/split:output:1aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_2
aflyrndiyz/Sigmoid_1Sigmoidaflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_1
aflyrndiyz/mul_2Mulaflyrndiyz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_2w
aflyrndiyz/TanhTanhaflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh
aflyrndiyz/mul_3Mulaflyrndiyz/Sigmoid:y:0aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_3
aflyrndiyz/add_3AddV2aflyrndiyz/mul_2:z:0aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_3
aflyrndiyz/ReadVariableOp_2ReadVariableOp$aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_2
aflyrndiyz/mul_4Mul#aflyrndiyz/ReadVariableOp_2:value:0aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_4
aflyrndiyz/add_4AddV2aflyrndiyz/split:output:3aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_4
aflyrndiyz/Sigmoid_2Sigmoidaflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_2v
aflyrndiyz/Tanh_1Tanhaflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh_1
aflyrndiyz/mul_5Mulaflyrndiyz/Sigmoid_2:y:0aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aflyrndiyz_matmul_readvariableop_resource+aflyrndiyz_matmul_1_readvariableop_resource*aflyrndiyz_biasadd_readvariableop_resource"aflyrndiyz_readvariableop_resource$aflyrndiyz_readvariableop_1_resource$aflyrndiyz_readvariableop_2_resource*
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
while_body_1823775*
condR
while_cond_1823774*Q
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
IdentityIdentitystrided_slice_3:output:0"^aflyrndiyz/BiasAdd/ReadVariableOp!^aflyrndiyz/MatMul/ReadVariableOp#^aflyrndiyz/MatMul_1/ReadVariableOp^aflyrndiyz/ReadVariableOp^aflyrndiyz/ReadVariableOp_1^aflyrndiyz/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aflyrndiyz/BiasAdd/ReadVariableOp!aflyrndiyz/BiasAdd/ReadVariableOp2D
 aflyrndiyz/MatMul/ReadVariableOp aflyrndiyz/MatMul/ReadVariableOp2H
"aflyrndiyz/MatMul_1/ReadVariableOp"aflyrndiyz/MatMul_1/ReadVariableOp26
aflyrndiyz/ReadVariableOpaflyrndiyz/ReadVariableOp2:
aflyrndiyz/ReadVariableOp_1aflyrndiyz/ReadVariableOp_12:
aflyrndiyz/ReadVariableOp_2aflyrndiyz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¦h

G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1820803

inputs<
)aflyrndiyz_matmul_readvariableop_resource:	 >
+aflyrndiyz_matmul_1_readvariableop_resource:	 9
*aflyrndiyz_biasadd_readvariableop_resource:	0
"aflyrndiyz_readvariableop_resource: 2
$aflyrndiyz_readvariableop_1_resource: 2
$aflyrndiyz_readvariableop_2_resource: 
identity¢!aflyrndiyz/BiasAdd/ReadVariableOp¢ aflyrndiyz/MatMul/ReadVariableOp¢"aflyrndiyz/MatMul_1/ReadVariableOp¢aflyrndiyz/ReadVariableOp¢aflyrndiyz/ReadVariableOp_1¢aflyrndiyz/ReadVariableOp_2¢whileD
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
 aflyrndiyz/MatMul/ReadVariableOpReadVariableOp)aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aflyrndiyz/MatMul/ReadVariableOp§
aflyrndiyz/MatMulMatMulstrided_slice_2:output:0(aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMulµ
"aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp+aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aflyrndiyz/MatMul_1/ReadVariableOp£
aflyrndiyz/MatMul_1MatMulzeros:output:0*aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMul_1
aflyrndiyz/addAddV2aflyrndiyz/MatMul:product:0aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/add®
!aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp*aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aflyrndiyz/BiasAdd/ReadVariableOp¥
aflyrndiyz/BiasAddBiasAddaflyrndiyz/add:z:0)aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/BiasAddz
aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aflyrndiyz/split/split_dimë
aflyrndiyz/splitSplit#aflyrndiyz/split/split_dim:output:0aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aflyrndiyz/split
aflyrndiyz/ReadVariableOpReadVariableOp"aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp
aflyrndiyz/mulMul!aflyrndiyz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul
aflyrndiyz/add_1AddV2aflyrndiyz/split:output:0aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_1{
aflyrndiyz/SigmoidSigmoidaflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid
aflyrndiyz/ReadVariableOp_1ReadVariableOp$aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_1
aflyrndiyz/mul_1Mul#aflyrndiyz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_1
aflyrndiyz/add_2AddV2aflyrndiyz/split:output:1aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_2
aflyrndiyz/Sigmoid_1Sigmoidaflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_1
aflyrndiyz/mul_2Mulaflyrndiyz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_2w
aflyrndiyz/TanhTanhaflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh
aflyrndiyz/mul_3Mulaflyrndiyz/Sigmoid:y:0aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_3
aflyrndiyz/add_3AddV2aflyrndiyz/mul_2:z:0aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_3
aflyrndiyz/ReadVariableOp_2ReadVariableOp$aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_2
aflyrndiyz/mul_4Mul#aflyrndiyz/ReadVariableOp_2:value:0aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_4
aflyrndiyz/add_4AddV2aflyrndiyz/split:output:3aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_4
aflyrndiyz/Sigmoid_2Sigmoidaflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_2v
aflyrndiyz/Tanh_1Tanhaflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh_1
aflyrndiyz/mul_5Mulaflyrndiyz/Sigmoid_2:y:0aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aflyrndiyz_matmul_readvariableop_resource+aflyrndiyz_matmul_1_readvariableop_resource*aflyrndiyz_biasadd_readvariableop_resource"aflyrndiyz_readvariableop_resource$aflyrndiyz_readvariableop_1_resource$aflyrndiyz_readvariableop_2_resource*
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
while_body_1820702*
condR
while_cond_1820701*Q
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
IdentityIdentitystrided_slice_3:output:0"^aflyrndiyz/BiasAdd/ReadVariableOp!^aflyrndiyz/MatMul/ReadVariableOp#^aflyrndiyz/MatMul_1/ReadVariableOp^aflyrndiyz/ReadVariableOp^aflyrndiyz/ReadVariableOp_1^aflyrndiyz/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aflyrndiyz/BiasAdd/ReadVariableOp!aflyrndiyz/BiasAdd/ReadVariableOp2D
 aflyrndiyz/MatMul/ReadVariableOp aflyrndiyz/MatMul/ReadVariableOp2H
"aflyrndiyz/MatMul_1/ReadVariableOp"aflyrndiyz/MatMul_1/ReadVariableOp26
aflyrndiyz/ReadVariableOpaflyrndiyz/ReadVariableOp2:
aflyrndiyz/ReadVariableOp_1aflyrndiyz/ReadVariableOp_12:
aflyrndiyz/ReadVariableOp_2aflyrndiyz/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
©	
(sequential_oohztcvkwo_while_cond_1818566H
Dsequential_oohztcvkwo_while_sequential_oohztcvkwo_while_loop_counterN
Jsequential_oohztcvkwo_while_sequential_oohztcvkwo_while_maximum_iterations+
'sequential_oohztcvkwo_while_placeholder-
)sequential_oohztcvkwo_while_placeholder_1-
)sequential_oohztcvkwo_while_placeholder_2-
)sequential_oohztcvkwo_while_placeholder_3J
Fsequential_oohztcvkwo_while_less_sequential_oohztcvkwo_strided_slice_1a
]sequential_oohztcvkwo_while_sequential_oohztcvkwo_while_cond_1818566___redundant_placeholder0a
]sequential_oohztcvkwo_while_sequential_oohztcvkwo_while_cond_1818566___redundant_placeholder1a
]sequential_oohztcvkwo_while_sequential_oohztcvkwo_while_cond_1818566___redundant_placeholder2a
]sequential_oohztcvkwo_while_sequential_oohztcvkwo_while_cond_1818566___redundant_placeholder3a
]sequential_oohztcvkwo_while_sequential_oohztcvkwo_while_cond_1818566___redundant_placeholder4a
]sequential_oohztcvkwo_while_sequential_oohztcvkwo_while_cond_1818566___redundant_placeholder5a
]sequential_oohztcvkwo_while_sequential_oohztcvkwo_while_cond_1818566___redundant_placeholder6(
$sequential_oohztcvkwo_while_identity
Þ
 sequential/oohztcvkwo/while/LessLess'sequential_oohztcvkwo_while_placeholderFsequential_oohztcvkwo_while_less_sequential_oohztcvkwo_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/oohztcvkwo/while/Less
$sequential/oohztcvkwo/while/IdentityIdentity$sequential/oohztcvkwo/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/oohztcvkwo/while/Identity"U
$sequential_oohztcvkwo_while_identity-sequential/oohztcvkwo/while/Identity:output:0*(
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
while_body_1818957
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_ammytzqwsz_1818981_0:	-
while_ammytzqwsz_1818983_0:	 )
while_ammytzqwsz_1818985_0:	(
while_ammytzqwsz_1818987_0: (
while_ammytzqwsz_1818989_0: (
while_ammytzqwsz_1818991_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_ammytzqwsz_1818981:	+
while_ammytzqwsz_1818983:	 '
while_ammytzqwsz_1818985:	&
while_ammytzqwsz_1818987: &
while_ammytzqwsz_1818989: &
while_ammytzqwsz_1818991: ¢(while/ammytzqwsz/StatefulPartitionedCallÃ
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
(while/ammytzqwsz/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_ammytzqwsz_1818981_0while_ammytzqwsz_1818983_0while_ammytzqwsz_1818985_0while_ammytzqwsz_1818987_0while_ammytzqwsz_1818989_0while_ammytzqwsz_1818991_0*
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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_18189372*
(while/ammytzqwsz/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/ammytzqwsz/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/ammytzqwsz/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/ammytzqwsz/StatefulPartitionedCall:output:1)^while/ammytzqwsz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/ammytzqwsz/StatefulPartitionedCall:output:2)^while/ammytzqwsz/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_ammytzqwsz_1818981while_ammytzqwsz_1818981_0"6
while_ammytzqwsz_1818983while_ammytzqwsz_1818983_0"6
while_ammytzqwsz_1818985while_ammytzqwsz_1818985_0"6
while_ammytzqwsz_1818987while_ammytzqwsz_1818987_0"6
while_ammytzqwsz_1818989while_ammytzqwsz_1818989_0"6
while_ammytzqwsz_1818991while_ammytzqwsz_1818991_0")
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
(while/ammytzqwsz/StatefulPartitionedCall(while/ammytzqwsz/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1820701
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1820701___redundant_placeholder05
1while_while_cond_1820701___redundant_placeholder15
1while_while_cond_1820701___redundant_placeholder25
1while_while_cond_1820701___redundant_placeholder35
1while_while_cond_1820701___redundant_placeholder45
1while_while_cond_1820701___redundant_placeholder55
1while_while_cond_1820701___redundant_placeholder6
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
¸
¼
G__inference_sequential_layer_call_and_return_conditional_losses_1820834

inputs(
fethhjgisa_1820411: 
fethhjgisa_1820413:%
oohztcvkwo_1820611:	%
oohztcvkwo_1820613:	 !
oohztcvkwo_1820615:	 
oohztcvkwo_1820617:  
oohztcvkwo_1820619:  
oohztcvkwo_1820621: %
mfdtyewult_1820804:	 %
mfdtyewult_1820806:	 !
mfdtyewult_1820808:	 
mfdtyewult_1820810:  
mfdtyewult_1820812:  
mfdtyewult_1820814: $
boyhyiogqf_1820828:  
boyhyiogqf_1820830:
identity¢"boyhyiogqf/StatefulPartitionedCall¢"fethhjgisa/StatefulPartitionedCall¢"mfdtyewult/StatefulPartitionedCall¢"oohztcvkwo/StatefulPartitionedCall¬
"fethhjgisa/StatefulPartitionedCallStatefulPartitionedCallinputsfethhjgisa_1820411fethhjgisa_1820413*
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
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_18204102$
"fethhjgisa/StatefulPartitionedCall
ohvvfduigw/PartitionedCallPartitionedCall+fethhjgisa/StatefulPartitionedCall:output:0*
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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_18204292
ohvvfduigw/PartitionedCall
"oohztcvkwo/StatefulPartitionedCallStatefulPartitionedCall#ohvvfduigw/PartitionedCall:output:0oohztcvkwo_1820611oohztcvkwo_1820613oohztcvkwo_1820615oohztcvkwo_1820617oohztcvkwo_1820619oohztcvkwo_1820621*
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18206102$
"oohztcvkwo/StatefulPartitionedCall¡
"mfdtyewult/StatefulPartitionedCallStatefulPartitionedCall+oohztcvkwo/StatefulPartitionedCall:output:0mfdtyewult_1820804mfdtyewult_1820806mfdtyewult_1820808mfdtyewult_1820810mfdtyewult_1820812mfdtyewult_1820814*
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18208032$
"mfdtyewult/StatefulPartitionedCallÉ
"boyhyiogqf/StatefulPartitionedCallStatefulPartitionedCall+mfdtyewult/StatefulPartitionedCall:output:0boyhyiogqf_1820828boyhyiogqf_1820830*
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
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_18208272$
"boyhyiogqf/StatefulPartitionedCall
IdentityIdentity+boyhyiogqf/StatefulPartitionedCall:output:0#^boyhyiogqf/StatefulPartitionedCall#^fethhjgisa/StatefulPartitionedCall#^mfdtyewult/StatefulPartitionedCall#^oohztcvkwo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"boyhyiogqf/StatefulPartitionedCall"boyhyiogqf/StatefulPartitionedCall2H
"fethhjgisa/StatefulPartitionedCall"fethhjgisa/StatefulPartitionedCall2H
"mfdtyewult/StatefulPartitionedCall"mfdtyewult/StatefulPartitionedCall2H
"oohztcvkwo/StatefulPartitionedCall"oohztcvkwo/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_sequential_layer_call_fn_1821475

liksrhmmux
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
liksrhmmuxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_18214032
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
liksrhmmux
àY

while_body_1823955
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aflyrndiyz_matmul_readvariableop_resource_0:	 F
3while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 A
2while_aflyrndiyz_biasadd_readvariableop_resource_0:	8
*while_aflyrndiyz_readvariableop_resource_0: :
,while_aflyrndiyz_readvariableop_1_resource_0: :
,while_aflyrndiyz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aflyrndiyz_matmul_readvariableop_resource:	 D
1while_aflyrndiyz_matmul_1_readvariableop_resource:	 ?
0while_aflyrndiyz_biasadd_readvariableop_resource:	6
(while_aflyrndiyz_readvariableop_resource: 8
*while_aflyrndiyz_readvariableop_1_resource: 8
*while_aflyrndiyz_readvariableop_2_resource: ¢'while/aflyrndiyz/BiasAdd/ReadVariableOp¢&while/aflyrndiyz/MatMul/ReadVariableOp¢(while/aflyrndiyz/MatMul_1/ReadVariableOp¢while/aflyrndiyz/ReadVariableOp¢!while/aflyrndiyz/ReadVariableOp_1¢!while/aflyrndiyz/ReadVariableOp_2Ã
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
&while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp1while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aflyrndiyz/MatMul/ReadVariableOpÑ
while/aflyrndiyz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMulÉ
(while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp3while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aflyrndiyz/MatMul_1/ReadVariableOpº
while/aflyrndiyz/MatMul_1MatMulwhile_placeholder_20while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMul_1°
while/aflyrndiyz/addAddV2!while/aflyrndiyz/MatMul:product:0#while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/addÂ
'while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp2while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aflyrndiyz/BiasAdd/ReadVariableOp½
while/aflyrndiyz/BiasAddBiasAddwhile/aflyrndiyz/add:z:0/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/BiasAdd
 while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aflyrndiyz/split/split_dim
while/aflyrndiyz/splitSplit)while/aflyrndiyz/split/split_dim:output:0!while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aflyrndiyz/split©
while/aflyrndiyz/ReadVariableOpReadVariableOp*while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aflyrndiyz/ReadVariableOp£
while/aflyrndiyz/mulMul'while/aflyrndiyz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul¦
while/aflyrndiyz/add_1AddV2while/aflyrndiyz/split:output:0while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_1
while/aflyrndiyz/SigmoidSigmoidwhile/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid¯
!while/aflyrndiyz/ReadVariableOp_1ReadVariableOp,while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_1©
while/aflyrndiyz/mul_1Mul)while/aflyrndiyz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_1¨
while/aflyrndiyz/add_2AddV2while/aflyrndiyz/split:output:1while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_2
while/aflyrndiyz/Sigmoid_1Sigmoidwhile/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_1
while/aflyrndiyz/mul_2Mulwhile/aflyrndiyz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_2
while/aflyrndiyz/TanhTanhwhile/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh¢
while/aflyrndiyz/mul_3Mulwhile/aflyrndiyz/Sigmoid:y:0while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_3£
while/aflyrndiyz/add_3AddV2while/aflyrndiyz/mul_2:z:0while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_3¯
!while/aflyrndiyz/ReadVariableOp_2ReadVariableOp,while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_2°
while/aflyrndiyz/mul_4Mul)while/aflyrndiyz/ReadVariableOp_2:value:0while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_4¨
while/aflyrndiyz/add_4AddV2while/aflyrndiyz/split:output:3while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_4
while/aflyrndiyz/Sigmoid_2Sigmoidwhile/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_2
while/aflyrndiyz/Tanh_1Tanhwhile/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh_1¦
while/aflyrndiyz/mul_5Mulwhile/aflyrndiyz/Sigmoid_2:y:0while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aflyrndiyz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aflyrndiyz/mul_5:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aflyrndiyz/add_3:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aflyrndiyz_biasadd_readvariableop_resource2while_aflyrndiyz_biasadd_readvariableop_resource_0"h
1while_aflyrndiyz_matmul_1_readvariableop_resource3while_aflyrndiyz_matmul_1_readvariableop_resource_0"d
/while_aflyrndiyz_matmul_readvariableop_resource1while_aflyrndiyz_matmul_readvariableop_resource_0"Z
*while_aflyrndiyz_readvariableop_1_resource,while_aflyrndiyz_readvariableop_1_resource_0"Z
*while_aflyrndiyz_readvariableop_2_resource,while_aflyrndiyz_readvariableop_2_resource_0"V
(while_aflyrndiyz_readvariableop_resource*while_aflyrndiyz_readvariableop_resource_0")
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
'while/aflyrndiyz/BiasAdd/ReadVariableOp'while/aflyrndiyz/BiasAdd/ReadVariableOp2P
&while/aflyrndiyz/MatMul/ReadVariableOp&while/aflyrndiyz/MatMul/ReadVariableOp2T
(while/aflyrndiyz/MatMul_1/ReadVariableOp(while/aflyrndiyz/MatMul_1/ReadVariableOp2B
while/aflyrndiyz/ReadVariableOpwhile/aflyrndiyz/ReadVariableOp2F
!while/aflyrndiyz/ReadVariableOp_1!while/aflyrndiyz/ReadVariableOp_12F
!while/aflyrndiyz/ReadVariableOp_2!while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_1820827

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
ú[
Ë
 __inference__traced_save_1824551
file_prefix0
,savev2_fethhjgisa_kernel_read_readvariableop.
*savev2_fethhjgisa_bias_read_readvariableop0
,savev2_boyhyiogqf_kernel_read_readvariableop.
*savev2_boyhyiogqf_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_oohztcvkwo_ammytzqwsz_kernel_read_readvariableopE
Asavev2_oohztcvkwo_ammytzqwsz_recurrent_kernel_read_readvariableop9
5savev2_oohztcvkwo_ammytzqwsz_bias_read_readvariableopP
Lsavev2_oohztcvkwo_ammytzqwsz_input_gate_peephole_weights_read_readvariableopQ
Msavev2_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_oohztcvkwo_ammytzqwsz_output_gate_peephole_weights_read_readvariableop;
7savev2_mfdtyewult_aflyrndiyz_kernel_read_readvariableopE
Asavev2_mfdtyewult_aflyrndiyz_recurrent_kernel_read_readvariableop9
5savev2_mfdtyewult_aflyrndiyz_bias_read_readvariableopP
Lsavev2_mfdtyewult_aflyrndiyz_input_gate_peephole_weights_read_readvariableopQ
Msavev2_mfdtyewult_aflyrndiyz_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_mfdtyewult_aflyrndiyz_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_fethhjgisa_kernel_rms_read_readvariableop:
6savev2_rmsprop_fethhjgisa_bias_rms_read_readvariableop<
8savev2_rmsprop_boyhyiogqf_kernel_rms_read_readvariableop:
6savev2_rmsprop_boyhyiogqf_bias_rms_read_readvariableopG
Csavev2_rmsprop_oohztcvkwo_ammytzqwsz_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_oohztcvkwo_ammytzqwsz_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_oohztcvkwo_ammytzqwsz_bias_rms_read_readvariableop\
Xsavev2_rmsprop_oohztcvkwo_ammytzqwsz_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_oohztcvkwo_ammytzqwsz_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_mfdtyewult_aflyrndiyz_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_mfdtyewult_aflyrndiyz_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_mfdtyewult_aflyrndiyz_bias_rms_read_readvariableop\
Xsavev2_rmsprop_mfdtyewult_aflyrndiyz_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_mfdtyewult_aflyrndiyz_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_mfdtyewult_aflyrndiyz_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_fethhjgisa_kernel_read_readvariableop*savev2_fethhjgisa_bias_read_readvariableop,savev2_boyhyiogqf_kernel_read_readvariableop*savev2_boyhyiogqf_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_oohztcvkwo_ammytzqwsz_kernel_read_readvariableopAsavev2_oohztcvkwo_ammytzqwsz_recurrent_kernel_read_readvariableop5savev2_oohztcvkwo_ammytzqwsz_bias_read_readvariableopLsavev2_oohztcvkwo_ammytzqwsz_input_gate_peephole_weights_read_readvariableopMsavev2_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weights_read_readvariableopMsavev2_oohztcvkwo_ammytzqwsz_output_gate_peephole_weights_read_readvariableop7savev2_mfdtyewult_aflyrndiyz_kernel_read_readvariableopAsavev2_mfdtyewult_aflyrndiyz_recurrent_kernel_read_readvariableop5savev2_mfdtyewult_aflyrndiyz_bias_read_readvariableopLsavev2_mfdtyewult_aflyrndiyz_input_gate_peephole_weights_read_readvariableopMsavev2_mfdtyewult_aflyrndiyz_forget_gate_peephole_weights_read_readvariableopMsavev2_mfdtyewult_aflyrndiyz_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_fethhjgisa_kernel_rms_read_readvariableop6savev2_rmsprop_fethhjgisa_bias_rms_read_readvariableop8savev2_rmsprop_boyhyiogqf_kernel_rms_read_readvariableop6savev2_rmsprop_boyhyiogqf_bias_rms_read_readvariableopCsavev2_rmsprop_oohztcvkwo_ammytzqwsz_kernel_rms_read_readvariableopMsavev2_rmsprop_oohztcvkwo_ammytzqwsz_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_oohztcvkwo_ammytzqwsz_bias_rms_read_readvariableopXsavev2_rmsprop_oohztcvkwo_ammytzqwsz_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_oohztcvkwo_ammytzqwsz_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_oohztcvkwo_ammytzqwsz_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_mfdtyewult_aflyrndiyz_kernel_rms_read_readvariableopMsavev2_rmsprop_mfdtyewult_aflyrndiyz_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_mfdtyewult_aflyrndiyz_bias_rms_read_readvariableopXsavev2_rmsprop_mfdtyewult_aflyrndiyz_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_mfdtyewult_aflyrndiyz_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_mfdtyewult_aflyrndiyz_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
ì

%__inference_signature_wrapper_1821602

liksrhmmux
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
liksrhmmuxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_18188502
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
liksrhmmux
¯F
ê
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1819795

inputs%
aflyrndiyz_1819696:	 %
aflyrndiyz_1819698:	 !
aflyrndiyz_1819700:	 
aflyrndiyz_1819702:  
aflyrndiyz_1819704:  
aflyrndiyz_1819706: 
identity¢"aflyrndiyz/StatefulPartitionedCall¢whileD
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
"aflyrndiyz/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0aflyrndiyz_1819696aflyrndiyz_1819698aflyrndiyz_1819700aflyrndiyz_1819702aflyrndiyz_1819704aflyrndiyz_1819706*
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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_18196952$
"aflyrndiyz/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0aflyrndiyz_1819696aflyrndiyz_1819698aflyrndiyz_1819700aflyrndiyz_1819702aflyrndiyz_1819704aflyrndiyz_1819706*
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
while_body_1819715*
condR
while_cond_1819714*Q
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
IdentityIdentitystrided_slice_3:output:0#^aflyrndiyz/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"aflyrndiyz/StatefulPartitionedCall"aflyrndiyz/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


mfdtyewult_while_cond_18223022
.mfdtyewult_while_mfdtyewult_while_loop_counter8
4mfdtyewult_while_mfdtyewult_while_maximum_iterations 
mfdtyewult_while_placeholder"
mfdtyewult_while_placeholder_1"
mfdtyewult_while_placeholder_2"
mfdtyewult_while_placeholder_34
0mfdtyewult_while_less_mfdtyewult_strided_slice_1K
Gmfdtyewult_while_mfdtyewult_while_cond_1822302___redundant_placeholder0K
Gmfdtyewult_while_mfdtyewult_while_cond_1822302___redundant_placeholder1K
Gmfdtyewult_while_mfdtyewult_while_cond_1822302___redundant_placeholder2K
Gmfdtyewult_while_mfdtyewult_while_cond_1822302___redundant_placeholder3K
Gmfdtyewult_while_mfdtyewult_while_cond_1822302___redundant_placeholder4K
Gmfdtyewult_while_mfdtyewult_while_cond_1822302___redundant_placeholder5K
Gmfdtyewult_while_mfdtyewult_while_cond_1822302___redundant_placeholder6
mfdtyewult_while_identity
§
mfdtyewult/while/LessLessmfdtyewult_while_placeholder0mfdtyewult_while_less_mfdtyewult_strided_slice_1*
T0*
_output_shapes
: 2
mfdtyewult/while/Less~
mfdtyewult/while/IdentityIdentitymfdtyewult/while/Less:z:0*
T0
*
_output_shapes
: 2
mfdtyewult/while/Identity"?
mfdtyewult_while_identity"mfdtyewult/while/Identity:output:0*(
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
while_cond_1820508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1820508___redundant_placeholder05
1while_while_cond_1820508___redundant_placeholder15
1while_while_cond_1820508___redundant_placeholder25
1while_while_cond_1820508___redundant_placeholder35
1while_while_cond_1820508___redundant_placeholder45
1while_while_cond_1820508___redundant_placeholder55
1while_while_cond_1820508___redundant_placeholder6
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
while_body_1823775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aflyrndiyz_matmul_readvariableop_resource_0:	 F
3while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 A
2while_aflyrndiyz_biasadd_readvariableop_resource_0:	8
*while_aflyrndiyz_readvariableop_resource_0: :
,while_aflyrndiyz_readvariableop_1_resource_0: :
,while_aflyrndiyz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aflyrndiyz_matmul_readvariableop_resource:	 D
1while_aflyrndiyz_matmul_1_readvariableop_resource:	 ?
0while_aflyrndiyz_biasadd_readvariableop_resource:	6
(while_aflyrndiyz_readvariableop_resource: 8
*while_aflyrndiyz_readvariableop_1_resource: 8
*while_aflyrndiyz_readvariableop_2_resource: ¢'while/aflyrndiyz/BiasAdd/ReadVariableOp¢&while/aflyrndiyz/MatMul/ReadVariableOp¢(while/aflyrndiyz/MatMul_1/ReadVariableOp¢while/aflyrndiyz/ReadVariableOp¢!while/aflyrndiyz/ReadVariableOp_1¢!while/aflyrndiyz/ReadVariableOp_2Ã
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
&while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp1while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aflyrndiyz/MatMul/ReadVariableOpÑ
while/aflyrndiyz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMulÉ
(while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp3while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aflyrndiyz/MatMul_1/ReadVariableOpº
while/aflyrndiyz/MatMul_1MatMulwhile_placeholder_20while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMul_1°
while/aflyrndiyz/addAddV2!while/aflyrndiyz/MatMul:product:0#while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/addÂ
'while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp2while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aflyrndiyz/BiasAdd/ReadVariableOp½
while/aflyrndiyz/BiasAddBiasAddwhile/aflyrndiyz/add:z:0/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/BiasAdd
 while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aflyrndiyz/split/split_dim
while/aflyrndiyz/splitSplit)while/aflyrndiyz/split/split_dim:output:0!while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aflyrndiyz/split©
while/aflyrndiyz/ReadVariableOpReadVariableOp*while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aflyrndiyz/ReadVariableOp£
while/aflyrndiyz/mulMul'while/aflyrndiyz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul¦
while/aflyrndiyz/add_1AddV2while/aflyrndiyz/split:output:0while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_1
while/aflyrndiyz/SigmoidSigmoidwhile/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid¯
!while/aflyrndiyz/ReadVariableOp_1ReadVariableOp,while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_1©
while/aflyrndiyz/mul_1Mul)while/aflyrndiyz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_1¨
while/aflyrndiyz/add_2AddV2while/aflyrndiyz/split:output:1while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_2
while/aflyrndiyz/Sigmoid_1Sigmoidwhile/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_1
while/aflyrndiyz/mul_2Mulwhile/aflyrndiyz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_2
while/aflyrndiyz/TanhTanhwhile/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh¢
while/aflyrndiyz/mul_3Mulwhile/aflyrndiyz/Sigmoid:y:0while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_3£
while/aflyrndiyz/add_3AddV2while/aflyrndiyz/mul_2:z:0while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_3¯
!while/aflyrndiyz/ReadVariableOp_2ReadVariableOp,while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_2°
while/aflyrndiyz/mul_4Mul)while/aflyrndiyz/ReadVariableOp_2:value:0while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_4¨
while/aflyrndiyz/add_4AddV2while/aflyrndiyz/split:output:3while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_4
while/aflyrndiyz/Sigmoid_2Sigmoidwhile/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_2
while/aflyrndiyz/Tanh_1Tanhwhile/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh_1¦
while/aflyrndiyz/mul_5Mulwhile/aflyrndiyz/Sigmoid_2:y:0while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aflyrndiyz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aflyrndiyz/mul_5:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aflyrndiyz/add_3:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aflyrndiyz_biasadd_readvariableop_resource2while_aflyrndiyz_biasadd_readvariableop_resource_0"h
1while_aflyrndiyz_matmul_1_readvariableop_resource3while_aflyrndiyz_matmul_1_readvariableop_resource_0"d
/while_aflyrndiyz_matmul_readvariableop_resource1while_aflyrndiyz_matmul_readvariableop_resource_0"Z
*while_aflyrndiyz_readvariableop_1_resource,while_aflyrndiyz_readvariableop_1_resource_0"Z
*while_aflyrndiyz_readvariableop_2_resource,while_aflyrndiyz_readvariableop_2_resource_0"V
(while_aflyrndiyz_readvariableop_resource*while_aflyrndiyz_readvariableop_resource_0")
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
'while/aflyrndiyz/BiasAdd/ReadVariableOp'while/aflyrndiyz/BiasAdd/ReadVariableOp2P
&while/aflyrndiyz/MatMul/ReadVariableOp&while/aflyrndiyz/MatMul/ReadVariableOp2T
(while/aflyrndiyz/MatMul_1/ReadVariableOp(while/aflyrndiyz/MatMul_1/ReadVariableOp2B
while/aflyrndiyz/ReadVariableOpwhile/aflyrndiyz/ReadVariableOp2F
!while/aflyrndiyz/ReadVariableOp_1!while/aflyrndiyz/ReadVariableOp_12F
!while/aflyrndiyz/ReadVariableOp_2!while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_1820976
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1820976___redundant_placeholder05
1while_while_cond_1820976___redundant_placeholder15
1while_while_cond_1820976___redundant_placeholder25
1while_while_cond_1820976___redundant_placeholder35
1while_while_cond_1820976___redundant_placeholder45
1while_while_cond_1820976___redundant_placeholder55
1while_while_cond_1820976___redundant_placeholder6
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
while_cond_1823166
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1823166___redundant_placeholder05
1while_while_cond_1823166___redundant_placeholder15
1while_while_cond_1823166___redundant_placeholder25
1while_while_cond_1823166___redundant_placeholder35
1while_while_cond_1823166___redundant_placeholder45
1while_while_cond_1823166___redundant_placeholder55
1while_while_cond_1823166___redundant_placeholder6
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823696
inputs_0<
)aflyrndiyz_matmul_readvariableop_resource:	 >
+aflyrndiyz_matmul_1_readvariableop_resource:	 9
*aflyrndiyz_biasadd_readvariableop_resource:	0
"aflyrndiyz_readvariableop_resource: 2
$aflyrndiyz_readvariableop_1_resource: 2
$aflyrndiyz_readvariableop_2_resource: 
identity¢!aflyrndiyz/BiasAdd/ReadVariableOp¢ aflyrndiyz/MatMul/ReadVariableOp¢"aflyrndiyz/MatMul_1/ReadVariableOp¢aflyrndiyz/ReadVariableOp¢aflyrndiyz/ReadVariableOp_1¢aflyrndiyz/ReadVariableOp_2¢whileF
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
 aflyrndiyz/MatMul/ReadVariableOpReadVariableOp)aflyrndiyz_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 aflyrndiyz/MatMul/ReadVariableOp§
aflyrndiyz/MatMulMatMulstrided_slice_2:output:0(aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMulµ
"aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp+aflyrndiyz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"aflyrndiyz/MatMul_1/ReadVariableOp£
aflyrndiyz/MatMul_1MatMulzeros:output:0*aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/MatMul_1
aflyrndiyz/addAddV2aflyrndiyz/MatMul:product:0aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/add®
!aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp*aflyrndiyz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!aflyrndiyz/BiasAdd/ReadVariableOp¥
aflyrndiyz/BiasAddBiasAddaflyrndiyz/add:z:0)aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
aflyrndiyz/BiasAddz
aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
aflyrndiyz/split/split_dimë
aflyrndiyz/splitSplit#aflyrndiyz/split/split_dim:output:0aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
aflyrndiyz/split
aflyrndiyz/ReadVariableOpReadVariableOp"aflyrndiyz_readvariableop_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp
aflyrndiyz/mulMul!aflyrndiyz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul
aflyrndiyz/add_1AddV2aflyrndiyz/split:output:0aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_1{
aflyrndiyz/SigmoidSigmoidaflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid
aflyrndiyz/ReadVariableOp_1ReadVariableOp$aflyrndiyz_readvariableop_1_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_1
aflyrndiyz/mul_1Mul#aflyrndiyz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_1
aflyrndiyz/add_2AddV2aflyrndiyz/split:output:1aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_2
aflyrndiyz/Sigmoid_1Sigmoidaflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_1
aflyrndiyz/mul_2Mulaflyrndiyz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_2w
aflyrndiyz/TanhTanhaflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh
aflyrndiyz/mul_3Mulaflyrndiyz/Sigmoid:y:0aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_3
aflyrndiyz/add_3AddV2aflyrndiyz/mul_2:z:0aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_3
aflyrndiyz/ReadVariableOp_2ReadVariableOp$aflyrndiyz_readvariableop_2_resource*
_output_shapes
: *
dtype02
aflyrndiyz/ReadVariableOp_2
aflyrndiyz/mul_4Mul#aflyrndiyz/ReadVariableOp_2:value:0aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_4
aflyrndiyz/add_4AddV2aflyrndiyz/split:output:3aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/add_4
aflyrndiyz/Sigmoid_2Sigmoidaflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Sigmoid_2v
aflyrndiyz/Tanh_1Tanhaflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/Tanh_1
aflyrndiyz/mul_5Mulaflyrndiyz/Sigmoid_2:y:0aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
aflyrndiyz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)aflyrndiyz_matmul_readvariableop_resource+aflyrndiyz_matmul_1_readvariableop_resource*aflyrndiyz_biasadd_readvariableop_resource"aflyrndiyz_readvariableop_resource$aflyrndiyz_readvariableop_1_resource$aflyrndiyz_readvariableop_2_resource*
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
while_body_1823595*
condR
while_cond_1823594*Q
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
IdentityIdentitystrided_slice_3:output:0"^aflyrndiyz/BiasAdd/ReadVariableOp!^aflyrndiyz/MatMul/ReadVariableOp#^aflyrndiyz/MatMul_1/ReadVariableOp^aflyrndiyz/ReadVariableOp^aflyrndiyz/ReadVariableOp_1^aflyrndiyz/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!aflyrndiyz/BiasAdd/ReadVariableOp!aflyrndiyz/BiasAdd/ReadVariableOp2D
 aflyrndiyz/MatMul/ReadVariableOp aflyrndiyz/MatMul/ReadVariableOp2H
"aflyrndiyz/MatMul_1/ReadVariableOp"aflyrndiyz/MatMul_1/ReadVariableOp26
aflyrndiyz/ReadVariableOpaflyrndiyz/ReadVariableOp2:
aflyrndiyz/ReadVariableOp_1aflyrndiyz/ReadVariableOp_12:
aflyrndiyz/ReadVariableOp_2aflyrndiyz/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


í
while_cond_1818956
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1818956___redundant_placeholder05
1while_while_cond_1818956___redundant_placeholder15
1while_while_cond_1818956___redundant_placeholder25
1while_while_cond_1818956___redundant_placeholder35
1while_while_cond_1818956___redundant_placeholder45
1while_while_cond_1818956___redundant_placeholder55
1while_while_cond_1818956___redundant_placeholder6
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
while_body_1820977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_aflyrndiyz_matmul_readvariableop_resource_0:	 F
3while_aflyrndiyz_matmul_1_readvariableop_resource_0:	 A
2while_aflyrndiyz_biasadd_readvariableop_resource_0:	8
*while_aflyrndiyz_readvariableop_resource_0: :
,while_aflyrndiyz_readvariableop_1_resource_0: :
,while_aflyrndiyz_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_aflyrndiyz_matmul_readvariableop_resource:	 D
1while_aflyrndiyz_matmul_1_readvariableop_resource:	 ?
0while_aflyrndiyz_biasadd_readvariableop_resource:	6
(while_aflyrndiyz_readvariableop_resource: 8
*while_aflyrndiyz_readvariableop_1_resource: 8
*while_aflyrndiyz_readvariableop_2_resource: ¢'while/aflyrndiyz/BiasAdd/ReadVariableOp¢&while/aflyrndiyz/MatMul/ReadVariableOp¢(while/aflyrndiyz/MatMul_1/ReadVariableOp¢while/aflyrndiyz/ReadVariableOp¢!while/aflyrndiyz/ReadVariableOp_1¢!while/aflyrndiyz/ReadVariableOp_2Ã
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
&while/aflyrndiyz/MatMul/ReadVariableOpReadVariableOp1while_aflyrndiyz_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/aflyrndiyz/MatMul/ReadVariableOpÑ
while/aflyrndiyz/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/aflyrndiyz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMulÉ
(while/aflyrndiyz/MatMul_1/ReadVariableOpReadVariableOp3while_aflyrndiyz_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/aflyrndiyz/MatMul_1/ReadVariableOpº
while/aflyrndiyz/MatMul_1MatMulwhile_placeholder_20while/aflyrndiyz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/MatMul_1°
while/aflyrndiyz/addAddV2!while/aflyrndiyz/MatMul:product:0#while/aflyrndiyz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/addÂ
'while/aflyrndiyz/BiasAdd/ReadVariableOpReadVariableOp2while_aflyrndiyz_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/aflyrndiyz/BiasAdd/ReadVariableOp½
while/aflyrndiyz/BiasAddBiasAddwhile/aflyrndiyz/add:z:0/while/aflyrndiyz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/aflyrndiyz/BiasAdd
 while/aflyrndiyz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/aflyrndiyz/split/split_dim
while/aflyrndiyz/splitSplit)while/aflyrndiyz/split/split_dim:output:0!while/aflyrndiyz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/aflyrndiyz/split©
while/aflyrndiyz/ReadVariableOpReadVariableOp*while_aflyrndiyz_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/aflyrndiyz/ReadVariableOp£
while/aflyrndiyz/mulMul'while/aflyrndiyz/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul¦
while/aflyrndiyz/add_1AddV2while/aflyrndiyz/split:output:0while/aflyrndiyz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_1
while/aflyrndiyz/SigmoidSigmoidwhile/aflyrndiyz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid¯
!while/aflyrndiyz/ReadVariableOp_1ReadVariableOp,while_aflyrndiyz_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_1©
while/aflyrndiyz/mul_1Mul)while/aflyrndiyz/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_1¨
while/aflyrndiyz/add_2AddV2while/aflyrndiyz/split:output:1while/aflyrndiyz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_2
while/aflyrndiyz/Sigmoid_1Sigmoidwhile/aflyrndiyz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_1
while/aflyrndiyz/mul_2Mulwhile/aflyrndiyz/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_2
while/aflyrndiyz/TanhTanhwhile/aflyrndiyz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh¢
while/aflyrndiyz/mul_3Mulwhile/aflyrndiyz/Sigmoid:y:0while/aflyrndiyz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_3£
while/aflyrndiyz/add_3AddV2while/aflyrndiyz/mul_2:z:0while/aflyrndiyz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_3¯
!while/aflyrndiyz/ReadVariableOp_2ReadVariableOp,while_aflyrndiyz_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/aflyrndiyz/ReadVariableOp_2°
while/aflyrndiyz/mul_4Mul)while/aflyrndiyz/ReadVariableOp_2:value:0while/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_4¨
while/aflyrndiyz/add_4AddV2while/aflyrndiyz/split:output:3while/aflyrndiyz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/add_4
while/aflyrndiyz/Sigmoid_2Sigmoidwhile/aflyrndiyz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Sigmoid_2
while/aflyrndiyz/Tanh_1Tanhwhile/aflyrndiyz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/Tanh_1¦
while/aflyrndiyz/mul_5Mulwhile/aflyrndiyz/Sigmoid_2:y:0while/aflyrndiyz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/aflyrndiyz/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/aflyrndiyz/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/aflyrndiyz/mul_5:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/aflyrndiyz/add_3:z:0(^while/aflyrndiyz/BiasAdd/ReadVariableOp'^while/aflyrndiyz/MatMul/ReadVariableOp)^while/aflyrndiyz/MatMul_1/ReadVariableOp ^while/aflyrndiyz/ReadVariableOp"^while/aflyrndiyz/ReadVariableOp_1"^while/aflyrndiyz/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_aflyrndiyz_biasadd_readvariableop_resource2while_aflyrndiyz_biasadd_readvariableop_resource_0"h
1while_aflyrndiyz_matmul_1_readvariableop_resource3while_aflyrndiyz_matmul_1_readvariableop_resource_0"d
/while_aflyrndiyz_matmul_readvariableop_resource1while_aflyrndiyz_matmul_readvariableop_resource_0"Z
*while_aflyrndiyz_readvariableop_1_resource,while_aflyrndiyz_readvariableop_1_resource_0"Z
*while_aflyrndiyz_readvariableop_2_resource,while_aflyrndiyz_readvariableop_2_resource_0"V
(while_aflyrndiyz_readvariableop_resource*while_aflyrndiyz_readvariableop_resource_0")
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
'while/aflyrndiyz/BiasAdd/ReadVariableOp'while/aflyrndiyz/BiasAdd/ReadVariableOp2P
&while/aflyrndiyz/MatMul/ReadVariableOp&while/aflyrndiyz/MatMul/ReadVariableOp2T
(while/aflyrndiyz/MatMul_1/ReadVariableOp(while/aflyrndiyz/MatMul_1/ReadVariableOp2B
while/aflyrndiyz/ReadVariableOpwhile/aflyrndiyz/ReadVariableOp2F
!while/aflyrndiyz/ReadVariableOp_1!while/aflyrndiyz/ReadVariableOp_12F
!while/aflyrndiyz/ReadVariableOp_2!while/aflyrndiyz/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
àh

G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1822908
inputs_0<
)ammytzqwsz_matmul_readvariableop_resource:	>
+ammytzqwsz_matmul_1_readvariableop_resource:	 9
*ammytzqwsz_biasadd_readvariableop_resource:	0
"ammytzqwsz_readvariableop_resource: 2
$ammytzqwsz_readvariableop_1_resource: 2
$ammytzqwsz_readvariableop_2_resource: 
identity¢!ammytzqwsz/BiasAdd/ReadVariableOp¢ ammytzqwsz/MatMul/ReadVariableOp¢"ammytzqwsz/MatMul_1/ReadVariableOp¢ammytzqwsz/ReadVariableOp¢ammytzqwsz/ReadVariableOp_1¢ammytzqwsz/ReadVariableOp_2¢whileF
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
 ammytzqwsz/MatMul/ReadVariableOpReadVariableOp)ammytzqwsz_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ammytzqwsz/MatMul/ReadVariableOp§
ammytzqwsz/MatMulMatMulstrided_slice_2:output:0(ammytzqwsz/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMulµ
"ammytzqwsz/MatMul_1/ReadVariableOpReadVariableOp+ammytzqwsz_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"ammytzqwsz/MatMul_1/ReadVariableOp£
ammytzqwsz/MatMul_1MatMulzeros:output:0*ammytzqwsz/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/MatMul_1
ammytzqwsz/addAddV2ammytzqwsz/MatMul:product:0ammytzqwsz/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/add®
!ammytzqwsz/BiasAdd/ReadVariableOpReadVariableOp*ammytzqwsz_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ammytzqwsz/BiasAdd/ReadVariableOp¥
ammytzqwsz/BiasAddBiasAddammytzqwsz/add:z:0)ammytzqwsz/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ammytzqwsz/BiasAddz
ammytzqwsz/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
ammytzqwsz/split/split_dimë
ammytzqwsz/splitSplit#ammytzqwsz/split/split_dim:output:0ammytzqwsz/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ammytzqwsz/split
ammytzqwsz/ReadVariableOpReadVariableOp"ammytzqwsz_readvariableop_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp
ammytzqwsz/mulMul!ammytzqwsz/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul
ammytzqwsz/add_1AddV2ammytzqwsz/split:output:0ammytzqwsz/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_1{
ammytzqwsz/SigmoidSigmoidammytzqwsz/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid
ammytzqwsz/ReadVariableOp_1ReadVariableOp$ammytzqwsz_readvariableop_1_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_1
ammytzqwsz/mul_1Mul#ammytzqwsz/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_1
ammytzqwsz/add_2AddV2ammytzqwsz/split:output:1ammytzqwsz/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_2
ammytzqwsz/Sigmoid_1Sigmoidammytzqwsz/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_1
ammytzqwsz/mul_2Mulammytzqwsz/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_2w
ammytzqwsz/TanhTanhammytzqwsz/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh
ammytzqwsz/mul_3Mulammytzqwsz/Sigmoid:y:0ammytzqwsz/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_3
ammytzqwsz/add_3AddV2ammytzqwsz/mul_2:z:0ammytzqwsz/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_3
ammytzqwsz/ReadVariableOp_2ReadVariableOp$ammytzqwsz_readvariableop_2_resource*
_output_shapes
: *
dtype02
ammytzqwsz/ReadVariableOp_2
ammytzqwsz/mul_4Mul#ammytzqwsz/ReadVariableOp_2:value:0ammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_4
ammytzqwsz/add_4AddV2ammytzqwsz/split:output:3ammytzqwsz/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/add_4
ammytzqwsz/Sigmoid_2Sigmoidammytzqwsz/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Sigmoid_2v
ammytzqwsz/Tanh_1Tanhammytzqwsz/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/Tanh_1
ammytzqwsz/mul_5Mulammytzqwsz/Sigmoid_2:y:0ammytzqwsz/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ammytzqwsz/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)ammytzqwsz_matmul_readvariableop_resource+ammytzqwsz_matmul_1_readvariableop_resource*ammytzqwsz_biasadd_readvariableop_resource"ammytzqwsz_readvariableop_resource$ammytzqwsz_readvariableop_1_resource$ammytzqwsz_readvariableop_2_resource*
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
while_body_1822807*
condR
while_cond_1822806*Q
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
IdentityIdentitytranspose_1:y:0"^ammytzqwsz/BiasAdd/ReadVariableOp!^ammytzqwsz/MatMul/ReadVariableOp#^ammytzqwsz/MatMul_1/ReadVariableOp^ammytzqwsz/ReadVariableOp^ammytzqwsz/ReadVariableOp_1^ammytzqwsz/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!ammytzqwsz/BiasAdd/ReadVariableOp!ammytzqwsz/BiasAdd/ReadVariableOp2D
 ammytzqwsz/MatMul/ReadVariableOp ammytzqwsz/MatMul/ReadVariableOp2H
"ammytzqwsz/MatMul_1/ReadVariableOp"ammytzqwsz/MatMul_1/ReadVariableOp26
ammytzqwsz/ReadVariableOpammytzqwsz/ReadVariableOp2:
ammytzqwsz/ReadVariableOp_1ammytzqwsz/ReadVariableOp_12:
ammytzqwsz/ReadVariableOp_2ammytzqwsz/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
	

,__inference_oohztcvkwo_layer_call_fn_1823302
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18193002
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


,__inference_sequential_layer_call_fn_1822447

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
G__inference_sequential_layer_call_and_return_conditional_losses_18208342
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
Ó	
ø
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_1824134

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
Ý
H
,__inference_ohvvfduigw_layer_call_fn_1822548

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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_18204292
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


oohztcvkwo_while_cond_18217222
.oohztcvkwo_while_oohztcvkwo_while_loop_counter8
4oohztcvkwo_while_oohztcvkwo_while_maximum_iterations 
oohztcvkwo_while_placeholder"
oohztcvkwo_while_placeholder_1"
oohztcvkwo_while_placeholder_2"
oohztcvkwo_while_placeholder_34
0oohztcvkwo_while_less_oohztcvkwo_strided_slice_1K
Goohztcvkwo_while_oohztcvkwo_while_cond_1821722___redundant_placeholder0K
Goohztcvkwo_while_oohztcvkwo_while_cond_1821722___redundant_placeholder1K
Goohztcvkwo_while_oohztcvkwo_while_cond_1821722___redundant_placeholder2K
Goohztcvkwo_while_oohztcvkwo_while_cond_1821722___redundant_placeholder3K
Goohztcvkwo_while_oohztcvkwo_while_cond_1821722___redundant_placeholder4K
Goohztcvkwo_while_oohztcvkwo_while_cond_1821722___redundant_placeholder5K
Goohztcvkwo_while_oohztcvkwo_while_cond_1821722___redundant_placeholder6
oohztcvkwo_while_identity
§
oohztcvkwo/while/LessLessoohztcvkwo_while_placeholder0oohztcvkwo_while_less_oohztcvkwo_strided_slice_1*
T0*
_output_shapes
: 2
oohztcvkwo/while/Less~
oohztcvkwo/while/IdentityIdentityoohztcvkwo/while/Less:z:0*
T0
*
_output_shapes
: 2
oohztcvkwo/while/Identity"?
oohztcvkwo_while_identity"oohztcvkwo/while/Identity:output:0*(
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
,__inference_ammytzqwsz_layer_call_fn_1824277

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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_18191242
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
¹'
µ
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1824321

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
while_cond_1823954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1823954___redundant_placeholder05
1while_while_cond_1823954___redundant_placeholder15
1while_while_cond_1823954___redundant_placeholder25
1while_while_cond_1823954___redundant_placeholder35
1while_while_cond_1823954___redundant_placeholder45
1while_while_cond_1823954___redundant_placeholder55
1while_while_cond_1823954___redundant_placeholder6
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
¸
¼
G__inference_sequential_layer_call_and_return_conditional_losses_1821403

inputs(
fethhjgisa_1821365: 
fethhjgisa_1821367:%
oohztcvkwo_1821371:	%
oohztcvkwo_1821373:	 !
oohztcvkwo_1821375:	 
oohztcvkwo_1821377:  
oohztcvkwo_1821379:  
oohztcvkwo_1821381: %
mfdtyewult_1821384:	 %
mfdtyewult_1821386:	 !
mfdtyewult_1821388:	 
mfdtyewult_1821390:  
mfdtyewult_1821392:  
mfdtyewult_1821394: $
boyhyiogqf_1821397:  
boyhyiogqf_1821399:
identity¢"boyhyiogqf/StatefulPartitionedCall¢"fethhjgisa/StatefulPartitionedCall¢"mfdtyewult/StatefulPartitionedCall¢"oohztcvkwo/StatefulPartitionedCall¬
"fethhjgisa/StatefulPartitionedCallStatefulPartitionedCallinputsfethhjgisa_1821365fethhjgisa_1821367*
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
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_18204102$
"fethhjgisa/StatefulPartitionedCall
ohvvfduigw/PartitionedCallPartitionedCall+fethhjgisa/StatefulPartitionedCall:output:0*
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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_18204292
ohvvfduigw/PartitionedCall
"oohztcvkwo/StatefulPartitionedCallStatefulPartitionedCall#ohvvfduigw/PartitionedCall:output:0oohztcvkwo_1821371oohztcvkwo_1821373oohztcvkwo_1821375oohztcvkwo_1821377oohztcvkwo_1821379oohztcvkwo_1821381*
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_18212922$
"oohztcvkwo/StatefulPartitionedCall¡
"mfdtyewult/StatefulPartitionedCallStatefulPartitionedCall+oohztcvkwo/StatefulPartitionedCall:output:0mfdtyewult_1821384mfdtyewult_1821386mfdtyewult_1821388mfdtyewult_1821390mfdtyewult_1821392mfdtyewult_1821394*
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_18210782$
"mfdtyewult/StatefulPartitionedCallÉ
"boyhyiogqf/StatefulPartitionedCallStatefulPartitionedCall+mfdtyewult/StatefulPartitionedCall:output:0boyhyiogqf_1821397boyhyiogqf_1821399*
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
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_18208272$
"boyhyiogqf/StatefulPartitionedCall
IdentityIdentity+boyhyiogqf/StatefulPartitionedCall:output:0#^boyhyiogqf/StatefulPartitionedCall#^fethhjgisa/StatefulPartitionedCall#^mfdtyewult/StatefulPartitionedCall#^oohztcvkwo/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"boyhyiogqf/StatefulPartitionedCall"boyhyiogqf/StatefulPartitionedCall2H
"fethhjgisa/StatefulPartitionedCall"fethhjgisa/StatefulPartitionedCall2H
"mfdtyewult/StatefulPartitionedCall"mfdtyewult/StatefulPartitionedCall2H
"oohztcvkwo/StatefulPartitionedCall"oohztcvkwo/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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

liksrhmmux;
serving_default_liksrhmmux:0ÿÿÿÿÿÿÿÿÿ>

boyhyiogqf0
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
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "liksrhmmux"}}, {"class_name": "Conv1D", "config": {"name": "fethhjgisa", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "ohvvfduigw", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "oohztcvkwo", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ammytzqwsz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "mfdtyewult", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aflyrndiyz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "boyhyiogqf", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "liksrhmmux"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "liksrhmmux"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "fethhjgisa", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "ohvvfduigw", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "oohztcvkwo", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ammytzqwsz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "mfdtyewult", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aflyrndiyz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "boyhyiogqf", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
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
{"name": "fethhjgisa", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "fethhjgisa", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"name": "ohvvfduigw", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "ohvvfduigw", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
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
_tf_keras_rnn_layerä{"name": "oohztcvkwo", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "oohztcvkwo", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ammytzqwsz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
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
_tf_keras_rnn_layerê{"name": "mfdtyewult", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "mfdtyewult", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aflyrndiyz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "boyhyiogqf", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "boyhyiogqf", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
':%2fethhjgisa/kernel
:2fethhjgisa/bias
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
_tf_keras_layer¼{"name": "ammytzqwsz", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "ammytzqwsz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
_tf_keras_layerÀ{"name": "aflyrndiyz", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "aflyrndiyz", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
#:! 2boyhyiogqf/kernel
:2boyhyiogqf/bias
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
/:-	2oohztcvkwo/ammytzqwsz/kernel
9:7	 2&oohztcvkwo/ammytzqwsz/recurrent_kernel
):'2oohztcvkwo/ammytzqwsz/bias
?:= 21oohztcvkwo/ammytzqwsz/input_gate_peephole_weights
@:> 22oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights
@:> 22oohztcvkwo/ammytzqwsz/output_gate_peephole_weights
/:-	 2mfdtyewult/aflyrndiyz/kernel
9:7	 2&mfdtyewult/aflyrndiyz/recurrent_kernel
):'2mfdtyewult/aflyrndiyz/bias
?:= 21mfdtyewult/aflyrndiyz/input_gate_peephole_weights
@:> 22mfdtyewult/aflyrndiyz/forget_gate_peephole_weights
@:> 22mfdtyewult/aflyrndiyz/output_gate_peephole_weights
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
1:/2RMSprop/fethhjgisa/kernel/rms
':%2RMSprop/fethhjgisa/bias/rms
-:+ 2RMSprop/boyhyiogqf/kernel/rms
':%2RMSprop/boyhyiogqf/bias/rms
9:7	2(RMSprop/oohztcvkwo/ammytzqwsz/kernel/rms
C:A	 22RMSprop/oohztcvkwo/ammytzqwsz/recurrent_kernel/rms
3:12&RMSprop/oohztcvkwo/ammytzqwsz/bias/rms
I:G 2=RMSprop/oohztcvkwo/ammytzqwsz/input_gate_peephole_weights/rms
J:H 2>RMSprop/oohztcvkwo/ammytzqwsz/forget_gate_peephole_weights/rms
J:H 2>RMSprop/oohztcvkwo/ammytzqwsz/output_gate_peephole_weights/rms
9:7	 2(RMSprop/mfdtyewult/aflyrndiyz/kernel/rms
C:A	 22RMSprop/mfdtyewult/aflyrndiyz/recurrent_kernel/rms
3:12&RMSprop/mfdtyewult/aflyrndiyz/bias/rms
I:G 2=RMSprop/mfdtyewult/aflyrndiyz/input_gate_peephole_weights/rms
J:H 2>RMSprop/mfdtyewult/aflyrndiyz/forget_gate_peephole_weights/rms
J:H 2>RMSprop/mfdtyewult/aflyrndiyz/output_gate_peephole_weights/rms
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_1822006
G__inference_sequential_layer_call_and_return_conditional_losses_1822410
G__inference_sequential_layer_call_and_return_conditional_losses_1821516
G__inference_sequential_layer_call_and_return_conditional_losses_1821557À
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
,__inference_sequential_layer_call_fn_1820869
,__inference_sequential_layer_call_fn_1822447
,__inference_sequential_layer_call_fn_1822484
,__inference_sequential_layer_call_fn_1821475À
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
"__inference__wrapped_model_1818850Á
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

liksrhmmuxÿÿÿÿÿÿÿÿÿ
ñ2î
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_1822521¢
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
,__inference_fethhjgisa_layer_call_fn_1822530¢
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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_1822543¢
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
,__inference_ohvvfduigw_layer_call_fn_1822548¢
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1822728
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1822908
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1823088
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1823268æ
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
,__inference_oohztcvkwo_layer_call_fn_1823285
,__inference_oohztcvkwo_layer_call_fn_1823302
,__inference_oohztcvkwo_layer_call_fn_1823319
,__inference_oohztcvkwo_layer_call_fn_1823336æ
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823516
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823696
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823876
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1824056æ
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
,__inference_mfdtyewult_layer_call_fn_1824073
,__inference_mfdtyewult_layer_call_fn_1824090
,__inference_mfdtyewult_layer_call_fn_1824107
,__inference_mfdtyewult_layer_call_fn_1824124æ
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
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_1824134¢
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
,__inference_boyhyiogqf_layer_call_fn_1824143¢
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
%__inference_signature_wrapper_1821602
liksrhmmux"
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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1824187
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1824231¾
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
,__inference_ammytzqwsz_layer_call_fn_1824254
,__inference_ammytzqwsz_layer_call_fn_1824277¾
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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1824321
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1824365¾
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
,__inference_aflyrndiyz_layer_call_fn_1824388
,__inference_aflyrndiyz_layer_call_fn_1824411¾
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
"__inference__wrapped_model_1818850-./012345678"#;¢8
1¢.
,)

liksrhmmuxÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

boyhyiogqf$!

boyhyiogqfÿÿÿÿÿÿÿÿÿÌ
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1824321345678¢}
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
G__inference_aflyrndiyz_layer_call_and_return_conditional_losses_1824365345678¢}
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
,__inference_aflyrndiyz_layer_call_fn_1824388ð345678¢}
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
,__inference_aflyrndiyz_layer_call_fn_1824411ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ì
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1824187-./012¢}
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
G__inference_ammytzqwsz_layer_call_and_return_conditional_losses_1824231-./012¢}
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
,__inference_ammytzqwsz_layer_call_fn_1824254ð-./012¢}
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
,__inference_ammytzqwsz_layer_call_fn_1824277ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ §
G__inference_boyhyiogqf_layer_call_and_return_conditional_losses_1824134\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_boyhyiogqf_layer_call_fn_1824143O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ·
G__inference_fethhjgisa_layer_call_and_return_conditional_losses_1822521l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_fethhjgisa_layer_call_fn_1822530_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÐ
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823516345678S¢P
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823696345678S¢P
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1823876t345678C¢@
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
G__inference_mfdtyewult_layer_call_and_return_conditional_losses_1824056t345678C¢@
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
,__inference_mfdtyewult_layer_call_fn_1824073w345678S¢P
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
,__inference_mfdtyewult_layer_call_fn_1824090w345678S¢P
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
,__inference_mfdtyewult_layer_call_fn_1824107g345678C¢@
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
,__inference_mfdtyewult_layer_call_fn_1824124g345678C¢@
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
G__inference_ohvvfduigw_layer_call_and_return_conditional_losses_1822543d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_ohvvfduigw_layer_call_fn_1822548W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÝ
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1822728-./012S¢P
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1822908-./012S¢P
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1823088x-./012C¢@
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
G__inference_oohztcvkwo_layer_call_and_return_conditional_losses_1823268x-./012C¢@
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
,__inference_oohztcvkwo_layer_call_fn_1823285-./012S¢P
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
,__inference_oohztcvkwo_layer_call_fn_1823302-./012S¢P
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
,__inference_oohztcvkwo_layer_call_fn_1823319k-./012C¢@
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
,__inference_oohztcvkwo_layer_call_fn_1823336k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ É
G__inference_sequential_layer_call_and_return_conditional_losses_1821516~-./012345678"#C¢@
9¢6
,)

liksrhmmuxÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_layer_call_and_return_conditional_losses_1821557~-./012345678"#C¢@
9¢6
,)

liksrhmmuxÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_1822006z-./012345678"#?¢<
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
G__inference_sequential_layer_call_and_return_conditional_losses_1822410z-./012345678"#?¢<
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
,__inference_sequential_layer_call_fn_1820869q-./012345678"#C¢@
9¢6
,)

liksrhmmuxÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_layer_call_fn_1821475q-./012345678"#C¢@
9¢6
,)

liksrhmmuxÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1822447m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_1822484m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
%__inference_signature_wrapper_1821602-./012345678"#I¢F
¢ 
?ª<
:

liksrhmmux,)

liksrhmmuxÿÿÿÿÿÿÿÿÿ"7ª4
2

boyhyiogqf$!

boyhyiogqfÿÿÿÿÿÿÿÿÿ