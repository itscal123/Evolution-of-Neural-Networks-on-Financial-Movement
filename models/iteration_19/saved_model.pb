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
vfwtupxpzf/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namevfwtupxpzf/kernel
{
%vfwtupxpzf/kernel/Read/ReadVariableOpReadVariableOpvfwtupxpzf/kernel*"
_output_shapes
:*
dtype0
v
vfwtupxpzf/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namevfwtupxpzf/bias
o
#vfwtupxpzf/bias/Read/ReadVariableOpReadVariableOpvfwtupxpzf/bias*
_output_shapes
:*
dtype0
~
kekwghyimt/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namekekwghyimt/kernel
w
%kekwghyimt/kernel/Read/ReadVariableOpReadVariableOpkekwghyimt/kernel*
_output_shapes

: *
dtype0
v
kekwghyimt/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namekekwghyimt/bias
o
#kekwghyimt/bias/Read/ReadVariableOpReadVariableOpkekwghyimt/bias*
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
vhacowjcza/jczmzyhsca/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namevhacowjcza/jczmzyhsca/kernel

0vhacowjcza/jczmzyhsca/kernel/Read/ReadVariableOpReadVariableOpvhacowjcza/jczmzyhsca/kernel*
_output_shapes
:	*
dtype0
©
&vhacowjcza/jczmzyhsca/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&vhacowjcza/jczmzyhsca/recurrent_kernel
¢
:vhacowjcza/jczmzyhsca/recurrent_kernel/Read/ReadVariableOpReadVariableOp&vhacowjcza/jczmzyhsca/recurrent_kernel*
_output_shapes
:	 *
dtype0

vhacowjcza/jczmzyhsca/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namevhacowjcza/jczmzyhsca/bias

.vhacowjcza/jczmzyhsca/bias/Read/ReadVariableOpReadVariableOpvhacowjcza/jczmzyhsca/bias*
_output_shapes	
:*
dtype0
º
1vhacowjcza/jczmzyhsca/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31vhacowjcza/jczmzyhsca/input_gate_peephole_weights
³
Evhacowjcza/jczmzyhsca/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1vhacowjcza/jczmzyhsca/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2vhacowjcza/jczmzyhsca/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42vhacowjcza/jczmzyhsca/forget_gate_peephole_weights
µ
Fvhacowjcza/jczmzyhsca/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2vhacowjcza/jczmzyhsca/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2vhacowjcza/jczmzyhsca/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42vhacowjcza/jczmzyhsca/output_gate_peephole_weights
µ
Fvhacowjcza/jczmzyhsca/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2vhacowjcza/jczmzyhsca/output_gate_peephole_weights*
_output_shapes
: *
dtype0

zgrdiwrovx/wdwulgrltk/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namezgrdiwrovx/wdwulgrltk/kernel

0zgrdiwrovx/wdwulgrltk/kernel/Read/ReadVariableOpReadVariableOpzgrdiwrovx/wdwulgrltk/kernel*
_output_shapes
:	 *
dtype0
©
&zgrdiwrovx/wdwulgrltk/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&zgrdiwrovx/wdwulgrltk/recurrent_kernel
¢
:zgrdiwrovx/wdwulgrltk/recurrent_kernel/Read/ReadVariableOpReadVariableOp&zgrdiwrovx/wdwulgrltk/recurrent_kernel*
_output_shapes
:	 *
dtype0

zgrdiwrovx/wdwulgrltk/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namezgrdiwrovx/wdwulgrltk/bias

.zgrdiwrovx/wdwulgrltk/bias/Read/ReadVariableOpReadVariableOpzgrdiwrovx/wdwulgrltk/bias*
_output_shapes	
:*
dtype0
º
1zgrdiwrovx/wdwulgrltk/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights
³
Ezgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights
µ
Fzgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2zgrdiwrovx/wdwulgrltk/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights
µ
Fzgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights*
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
RMSprop/vfwtupxpzf/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/vfwtupxpzf/kernel/rms

1RMSprop/vfwtupxpzf/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/vfwtupxpzf/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/vfwtupxpzf/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/vfwtupxpzf/bias/rms

/RMSprop/vfwtupxpzf/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/vfwtupxpzf/bias/rms*
_output_shapes
:*
dtype0

RMSprop/kekwghyimt/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/kekwghyimt/kernel/rms

1RMSprop/kekwghyimt/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/kekwghyimt/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/kekwghyimt/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/kekwghyimt/bias/rms

/RMSprop/kekwghyimt/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/kekwghyimt/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/vhacowjcza/jczmzyhsca/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/vhacowjcza/jczmzyhsca/kernel/rms
¦
<RMSprop/vhacowjcza/jczmzyhsca/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/vhacowjcza/jczmzyhsca/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rms
º
FRMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/vhacowjcza/jczmzyhsca/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/vhacowjcza/jczmzyhsca/bias/rms

:RMSprop/vhacowjcza/jczmzyhsca/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/vhacowjcza/jczmzyhsca/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rms
Ë
QRMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rms
Í
RRMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rms
Í
RRMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rms
¦
<RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rms
º
FRMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/zgrdiwrovx/wdwulgrltk/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/zgrdiwrovx/wdwulgrltk/bias/rms

:RMSprop/zgrdiwrovx/wdwulgrltk/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/zgrdiwrovx/wdwulgrltk/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rms
Ë
QRMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rms
Í
RRMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rms
Í
RRMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rms*
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
9non_trainable_variables
	variables

:layers
regularization_losses
;layer_metrics
<metrics
=layer_regularization_losses
	trainable_variables
 
][
VARIABLE_VALUEvfwtupxpzf/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEvfwtupxpzf/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
][
VARIABLE_VALUEkekwghyimt/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEkekwghyimt/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
XV
VARIABLE_VALUEvhacowjcza/jczmzyhsca/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&vhacowjcza/jczmzyhsca/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvhacowjcza/jczmzyhsca/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1vhacowjcza/jczmzyhsca/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2vhacowjcza/jczmzyhsca/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2vhacowjcza/jczmzyhsca/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEzgrdiwrovx/wdwulgrltk/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&zgrdiwrovx/wdwulgrltk/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEzgrdiwrovx/wdwulgrltk/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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

VARIABLE_VALUERMSprop/vfwtupxpzf/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/vfwtupxpzf/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/kekwghyimt/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/kekwghyimt/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/vhacowjcza/jczmzyhsca/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/vhacowjcza/jczmzyhsca/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/zgrdiwrovx/wdwulgrltk/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_aveeivcxurPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_aveeivcxurvfwtupxpzf/kernelvfwtupxpzf/biasvhacowjcza/jczmzyhsca/kernel&vhacowjcza/jczmzyhsca/recurrent_kernelvhacowjcza/jczmzyhsca/bias1vhacowjcza/jczmzyhsca/input_gate_peephole_weights2vhacowjcza/jczmzyhsca/forget_gate_peephole_weights2vhacowjcza/jczmzyhsca/output_gate_peephole_weightszgrdiwrovx/wdwulgrltk/kernel&zgrdiwrovx/wdwulgrltk/recurrent_kernelzgrdiwrovx/wdwulgrltk/bias1zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights2zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights2zgrdiwrovx/wdwulgrltk/output_gate_peephole_weightskekwghyimt/kernelkekwghyimt/bias*
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
%__inference_signature_wrapper_2160253
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%vfwtupxpzf/kernel/Read/ReadVariableOp#vfwtupxpzf/bias/Read/ReadVariableOp%kekwghyimt/kernel/Read/ReadVariableOp#kekwghyimt/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0vhacowjcza/jczmzyhsca/kernel/Read/ReadVariableOp:vhacowjcza/jczmzyhsca/recurrent_kernel/Read/ReadVariableOp.vhacowjcza/jczmzyhsca/bias/Read/ReadVariableOpEvhacowjcza/jczmzyhsca/input_gate_peephole_weights/Read/ReadVariableOpFvhacowjcza/jczmzyhsca/forget_gate_peephole_weights/Read/ReadVariableOpFvhacowjcza/jczmzyhsca/output_gate_peephole_weights/Read/ReadVariableOp0zgrdiwrovx/wdwulgrltk/kernel/Read/ReadVariableOp:zgrdiwrovx/wdwulgrltk/recurrent_kernel/Read/ReadVariableOp.zgrdiwrovx/wdwulgrltk/bias/Read/ReadVariableOpEzgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/Read/ReadVariableOpFzgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/Read/ReadVariableOpFzgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/vfwtupxpzf/kernel/rms/Read/ReadVariableOp/RMSprop/vfwtupxpzf/bias/rms/Read/ReadVariableOp1RMSprop/kekwghyimt/kernel/rms/Read/ReadVariableOp/RMSprop/kekwghyimt/bias/rms/Read/ReadVariableOp<RMSprop/vhacowjcza/jczmzyhsca/kernel/rms/Read/ReadVariableOpFRMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/vhacowjcza/jczmzyhsca/bias/rms/Read/ReadVariableOpQRMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rms/Read/ReadVariableOpFRMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/zgrdiwrovx/wdwulgrltk/bias/rms/Read/ReadVariableOpQRMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
 __inference__traced_save_2163202
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamevfwtupxpzf/kernelvfwtupxpzf/biaskekwghyimt/kernelkekwghyimt/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhovhacowjcza/jczmzyhsca/kernel&vhacowjcza/jczmzyhsca/recurrent_kernelvhacowjcza/jczmzyhsca/bias1vhacowjcza/jczmzyhsca/input_gate_peephole_weights2vhacowjcza/jczmzyhsca/forget_gate_peephole_weights2vhacowjcza/jczmzyhsca/output_gate_peephole_weightszgrdiwrovx/wdwulgrltk/kernel&zgrdiwrovx/wdwulgrltk/recurrent_kernelzgrdiwrovx/wdwulgrltk/bias1zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights2zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights2zgrdiwrovx/wdwulgrltk/output_gate_peephole_weightstotalcountRMSprop/vfwtupxpzf/kernel/rmsRMSprop/vfwtupxpzf/bias/rmsRMSprop/kekwghyimt/kernel/rmsRMSprop/kekwghyimt/bias/rms(RMSprop/vhacowjcza/jczmzyhsca/kernel/rms2RMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rms&RMSprop/vhacowjcza/jczmzyhsca/bias/rms=RMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rms>RMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rms>RMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rms(RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rms2RMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rms&RMSprop/zgrdiwrovx/wdwulgrltk/bias/rms=RMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rms>RMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rms>RMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rms*3
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
#__inference__traced_restore_2163329Âà-
àY

while_body_2159160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jczmzyhsca_matmul_readvariableop_resource_0:	F
3while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 A
2while_jczmzyhsca_biasadd_readvariableop_resource_0:	8
*while_jczmzyhsca_readvariableop_resource_0: :
,while_jczmzyhsca_readvariableop_1_resource_0: :
,while_jczmzyhsca_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jczmzyhsca_matmul_readvariableop_resource:	D
1while_jczmzyhsca_matmul_1_readvariableop_resource:	 ?
0while_jczmzyhsca_biasadd_readvariableop_resource:	6
(while_jczmzyhsca_readvariableop_resource: 8
*while_jczmzyhsca_readvariableop_1_resource: 8
*while_jczmzyhsca_readvariableop_2_resource: ¢'while/jczmzyhsca/BiasAdd/ReadVariableOp¢&while/jczmzyhsca/MatMul/ReadVariableOp¢(while/jczmzyhsca/MatMul_1/ReadVariableOp¢while/jczmzyhsca/ReadVariableOp¢!while/jczmzyhsca/ReadVariableOp_1¢!while/jczmzyhsca/ReadVariableOp_2Ã
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
&while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp1while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jczmzyhsca/MatMul/ReadVariableOpÑ
while/jczmzyhsca/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMulÉ
(while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp3while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jczmzyhsca/MatMul_1/ReadVariableOpº
while/jczmzyhsca/MatMul_1MatMulwhile_placeholder_20while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMul_1°
while/jczmzyhsca/addAddV2!while/jczmzyhsca/MatMul:product:0#while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/addÂ
'while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp2while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jczmzyhsca/BiasAdd/ReadVariableOp½
while/jczmzyhsca/BiasAddBiasAddwhile/jczmzyhsca/add:z:0/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/BiasAdd
 while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jczmzyhsca/split/split_dim
while/jczmzyhsca/splitSplit)while/jczmzyhsca/split/split_dim:output:0!while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jczmzyhsca/split©
while/jczmzyhsca/ReadVariableOpReadVariableOp*while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jczmzyhsca/ReadVariableOp£
while/jczmzyhsca/mulMul'while/jczmzyhsca/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul¦
while/jczmzyhsca/add_1AddV2while/jczmzyhsca/split:output:0while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_1
while/jczmzyhsca/SigmoidSigmoidwhile/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid¯
!while/jczmzyhsca/ReadVariableOp_1ReadVariableOp,while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_1©
while/jczmzyhsca/mul_1Mul)while/jczmzyhsca/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_1¨
while/jczmzyhsca/add_2AddV2while/jczmzyhsca/split:output:1while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_2
while/jczmzyhsca/Sigmoid_1Sigmoidwhile/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_1
while/jczmzyhsca/mul_2Mulwhile/jczmzyhsca/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_2
while/jczmzyhsca/TanhTanhwhile/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh¢
while/jczmzyhsca/mul_3Mulwhile/jczmzyhsca/Sigmoid:y:0while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_3£
while/jczmzyhsca/add_3AddV2while/jczmzyhsca/mul_2:z:0while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_3¯
!while/jczmzyhsca/ReadVariableOp_2ReadVariableOp,while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_2°
while/jczmzyhsca/mul_4Mul)while/jczmzyhsca/ReadVariableOp_2:value:0while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_4¨
while/jczmzyhsca/add_4AddV2while/jczmzyhsca/split:output:3while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_4
while/jczmzyhsca/Sigmoid_2Sigmoidwhile/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_2
while/jczmzyhsca/Tanh_1Tanhwhile/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh_1¦
while/jczmzyhsca/mul_5Mulwhile/jczmzyhsca/Sigmoid_2:y:0while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jczmzyhsca/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jczmzyhsca/mul_5:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jczmzyhsca/add_3:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
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
0while_jczmzyhsca_biasadd_readvariableop_resource2while_jczmzyhsca_biasadd_readvariableop_resource_0"h
1while_jczmzyhsca_matmul_1_readvariableop_resource3while_jczmzyhsca_matmul_1_readvariableop_resource_0"d
/while_jczmzyhsca_matmul_readvariableop_resource1while_jczmzyhsca_matmul_readvariableop_resource_0"Z
*while_jczmzyhsca_readvariableop_1_resource,while_jczmzyhsca_readvariableop_1_resource_0"Z
*while_jczmzyhsca_readvariableop_2_resource,while_jczmzyhsca_readvariableop_2_resource_0"V
(while_jczmzyhsca_readvariableop_resource*while_jczmzyhsca_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jczmzyhsca/BiasAdd/ReadVariableOp'while/jczmzyhsca/BiasAdd/ReadVariableOp2P
&while/jczmzyhsca/MatMul/ReadVariableOp&while/jczmzyhsca/MatMul/ReadVariableOp2T
(while/jczmzyhsca/MatMul_1/ReadVariableOp(while/jczmzyhsca/MatMul_1/ReadVariableOp2B
while/jczmzyhsca/ReadVariableOpwhile/jczmzyhsca/ReadVariableOp2F
!while/jczmzyhsca/ReadVariableOp_1!while/jczmzyhsca/ReadVariableOp_12F
!while/jczmzyhsca/ReadVariableOp_2!while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2158709

inputs%
wdwulgrltk_2158610:	 %
wdwulgrltk_2158612:	 !
wdwulgrltk_2158614:	 
wdwulgrltk_2158616:  
wdwulgrltk_2158618:  
wdwulgrltk_2158620: 
identity¢"wdwulgrltk/StatefulPartitionedCall¢whileD
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
"wdwulgrltk/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0wdwulgrltk_2158610wdwulgrltk_2158612wdwulgrltk_2158614wdwulgrltk_2158616wdwulgrltk_2158618wdwulgrltk_2158620*
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_21585332$
"wdwulgrltk/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0wdwulgrltk_2158610wdwulgrltk_2158612wdwulgrltk_2158614wdwulgrltk_2158616wdwulgrltk_2158618wdwulgrltk_2158620*
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
while_body_2158629*
condR
while_cond_2158628*Q
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
IdentityIdentitystrided_slice_3:output:0#^wdwulgrltk/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"wdwulgrltk/StatefulPartitionedCall"wdwulgrltk/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Üh

G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162347
inputs_0<
)wdwulgrltk_matmul_readvariableop_resource:	 >
+wdwulgrltk_matmul_1_readvariableop_resource:	 9
*wdwulgrltk_biasadd_readvariableop_resource:	0
"wdwulgrltk_readvariableop_resource: 2
$wdwulgrltk_readvariableop_1_resource: 2
$wdwulgrltk_readvariableop_2_resource: 
identity¢!wdwulgrltk/BiasAdd/ReadVariableOp¢ wdwulgrltk/MatMul/ReadVariableOp¢"wdwulgrltk/MatMul_1/ReadVariableOp¢wdwulgrltk/ReadVariableOp¢wdwulgrltk/ReadVariableOp_1¢wdwulgrltk/ReadVariableOp_2¢whileF
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
 wdwulgrltk/MatMul/ReadVariableOpReadVariableOp)wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 wdwulgrltk/MatMul/ReadVariableOp§
wdwulgrltk/MatMulMatMulstrided_slice_2:output:0(wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMulµ
"wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp+wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"wdwulgrltk/MatMul_1/ReadVariableOp£
wdwulgrltk/MatMul_1MatMulzeros:output:0*wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMul_1
wdwulgrltk/addAddV2wdwulgrltk/MatMul:product:0wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/add®
!wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp*wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!wdwulgrltk/BiasAdd/ReadVariableOp¥
wdwulgrltk/BiasAddBiasAddwdwulgrltk/add:z:0)wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/BiasAddz
wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
wdwulgrltk/split/split_dimë
wdwulgrltk/splitSplit#wdwulgrltk/split/split_dim:output:0wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
wdwulgrltk/split
wdwulgrltk/ReadVariableOpReadVariableOp"wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp
wdwulgrltk/mulMul!wdwulgrltk/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul
wdwulgrltk/add_1AddV2wdwulgrltk/split:output:0wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_1{
wdwulgrltk/SigmoidSigmoidwdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid
wdwulgrltk/ReadVariableOp_1ReadVariableOp$wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_1
wdwulgrltk/mul_1Mul#wdwulgrltk/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_1
wdwulgrltk/add_2AddV2wdwulgrltk/split:output:1wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_2
wdwulgrltk/Sigmoid_1Sigmoidwdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_1
wdwulgrltk/mul_2Mulwdwulgrltk/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_2w
wdwulgrltk/TanhTanhwdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh
wdwulgrltk/mul_3Mulwdwulgrltk/Sigmoid:y:0wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_3
wdwulgrltk/add_3AddV2wdwulgrltk/mul_2:z:0wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_3
wdwulgrltk/ReadVariableOp_2ReadVariableOp$wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_2
wdwulgrltk/mul_4Mul#wdwulgrltk/ReadVariableOp_2:value:0wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_4
wdwulgrltk/add_4AddV2wdwulgrltk/split:output:3wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_4
wdwulgrltk/Sigmoid_2Sigmoidwdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_2v
wdwulgrltk/Tanh_1Tanhwdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh_1
wdwulgrltk/mul_5Mulwdwulgrltk/Sigmoid_2:y:0wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)wdwulgrltk_matmul_readvariableop_resource+wdwulgrltk_matmul_1_readvariableop_resource*wdwulgrltk_biasadd_readvariableop_resource"wdwulgrltk_readvariableop_resource$wdwulgrltk_readvariableop_1_resource$wdwulgrltk_readvariableop_2_resource*
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
while_body_2162246*
condR
while_cond_2162245*Q
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
IdentityIdentitystrided_slice_3:output:0"^wdwulgrltk/BiasAdd/ReadVariableOp!^wdwulgrltk/MatMul/ReadVariableOp#^wdwulgrltk/MatMul_1/ReadVariableOp^wdwulgrltk/ReadVariableOp^wdwulgrltk/ReadVariableOp_1^wdwulgrltk/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!wdwulgrltk/BiasAdd/ReadVariableOp!wdwulgrltk/BiasAdd/ReadVariableOp2D
 wdwulgrltk/MatMul/ReadVariableOp wdwulgrltk/MatMul/ReadVariableOp2H
"wdwulgrltk/MatMul_1/ReadVariableOp"wdwulgrltk/MatMul_1/ReadVariableOp26
wdwulgrltk/ReadVariableOpwdwulgrltk/ReadVariableOp2:
wdwulgrltk/ReadVariableOp_1wdwulgrltk/ReadVariableOp_12:
wdwulgrltk/ReadVariableOp_2wdwulgrltk/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
±'
³
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2158533

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

À
,__inference_wdwulgrltk_layer_call_fn_2163039

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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_21583462
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
while_cond_2159627
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2159627___redundant_placeholder05
1while_while_cond_2159627___redundant_placeholder15
1while_while_cond_2159627___redundant_placeholder25
1while_while_cond_2159627___redundant_placeholder35
1while_while_cond_2159627___redundant_placeholder45
1while_while_cond_2159627___redundant_placeholder55
1while_while_cond_2159627___redundant_placeholder6
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
³F
ê
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2157688

inputs%
jczmzyhsca_2157589:	%
jczmzyhsca_2157591:	 !
jczmzyhsca_2157593:	 
jczmzyhsca_2157595:  
jczmzyhsca_2157597:  
jczmzyhsca_2157599: 
identity¢"jczmzyhsca/StatefulPartitionedCall¢whileD
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
"jczmzyhsca/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0jczmzyhsca_2157589jczmzyhsca_2157591jczmzyhsca_2157593jczmzyhsca_2157595jczmzyhsca_2157597jczmzyhsca_2157599*
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_21575882$
"jczmzyhsca/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0jczmzyhsca_2157589jczmzyhsca_2157591jczmzyhsca_2157593jczmzyhsca_2157595jczmzyhsca_2157597jczmzyhsca_2157599*
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
while_body_2157608*
condR
while_cond_2157607*Q
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
IdentityIdentitytranspose_1:y:0#^jczmzyhsca/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"jczmzyhsca/StatefulPartitionedCall"jczmzyhsca/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦h

G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162707

inputs<
)wdwulgrltk_matmul_readvariableop_resource:	 >
+wdwulgrltk_matmul_1_readvariableop_resource:	 9
*wdwulgrltk_biasadd_readvariableop_resource:	0
"wdwulgrltk_readvariableop_resource: 2
$wdwulgrltk_readvariableop_1_resource: 2
$wdwulgrltk_readvariableop_2_resource: 
identity¢!wdwulgrltk/BiasAdd/ReadVariableOp¢ wdwulgrltk/MatMul/ReadVariableOp¢"wdwulgrltk/MatMul_1/ReadVariableOp¢wdwulgrltk/ReadVariableOp¢wdwulgrltk/ReadVariableOp_1¢wdwulgrltk/ReadVariableOp_2¢whileD
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
 wdwulgrltk/MatMul/ReadVariableOpReadVariableOp)wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 wdwulgrltk/MatMul/ReadVariableOp§
wdwulgrltk/MatMulMatMulstrided_slice_2:output:0(wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMulµ
"wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp+wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"wdwulgrltk/MatMul_1/ReadVariableOp£
wdwulgrltk/MatMul_1MatMulzeros:output:0*wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMul_1
wdwulgrltk/addAddV2wdwulgrltk/MatMul:product:0wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/add®
!wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp*wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!wdwulgrltk/BiasAdd/ReadVariableOp¥
wdwulgrltk/BiasAddBiasAddwdwulgrltk/add:z:0)wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/BiasAddz
wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
wdwulgrltk/split/split_dimë
wdwulgrltk/splitSplit#wdwulgrltk/split/split_dim:output:0wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
wdwulgrltk/split
wdwulgrltk/ReadVariableOpReadVariableOp"wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp
wdwulgrltk/mulMul!wdwulgrltk/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul
wdwulgrltk/add_1AddV2wdwulgrltk/split:output:0wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_1{
wdwulgrltk/SigmoidSigmoidwdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid
wdwulgrltk/ReadVariableOp_1ReadVariableOp$wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_1
wdwulgrltk/mul_1Mul#wdwulgrltk/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_1
wdwulgrltk/add_2AddV2wdwulgrltk/split:output:1wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_2
wdwulgrltk/Sigmoid_1Sigmoidwdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_1
wdwulgrltk/mul_2Mulwdwulgrltk/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_2w
wdwulgrltk/TanhTanhwdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh
wdwulgrltk/mul_3Mulwdwulgrltk/Sigmoid:y:0wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_3
wdwulgrltk/add_3AddV2wdwulgrltk/mul_2:z:0wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_3
wdwulgrltk/ReadVariableOp_2ReadVariableOp$wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_2
wdwulgrltk/mul_4Mul#wdwulgrltk/ReadVariableOp_2:value:0wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_4
wdwulgrltk/add_4AddV2wdwulgrltk/split:output:3wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_4
wdwulgrltk/Sigmoid_2Sigmoidwdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_2v
wdwulgrltk/Tanh_1Tanhwdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh_1
wdwulgrltk/mul_5Mulwdwulgrltk/Sigmoid_2:y:0wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)wdwulgrltk_matmul_readvariableop_resource+wdwulgrltk_matmul_1_readvariableop_resource*wdwulgrltk_biasadd_readvariableop_resource"wdwulgrltk_readvariableop_resource$wdwulgrltk_readvariableop_1_resource$wdwulgrltk_readvariableop_2_resource*
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
while_body_2162606*
condR
while_cond_2162605*Q
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
IdentityIdentitystrided_slice_3:output:0"^wdwulgrltk/BiasAdd/ReadVariableOp!^wdwulgrltk/MatMul/ReadVariableOp#^wdwulgrltk/MatMul_1/ReadVariableOp^wdwulgrltk/ReadVariableOp^wdwulgrltk/ReadVariableOp_1^wdwulgrltk/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!wdwulgrltk/BiasAdd/ReadVariableOp!wdwulgrltk/BiasAdd/ReadVariableOp2D
 wdwulgrltk/MatMul/ReadVariableOp wdwulgrltk/MatMul/ReadVariableOp2H
"wdwulgrltk/MatMul_1/ReadVariableOp"wdwulgrltk/MatMul_1/ReadVariableOp26
wdwulgrltk/ReadVariableOpwdwulgrltk/ReadVariableOp2:
wdwulgrltk/ReadVariableOp_1wdwulgrltk/ReadVariableOp_12:
wdwulgrltk/ReadVariableOp_2wdwulgrltk/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_2162605
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2162605___redundant_placeholder05
1while_while_cond_2162605___redundant_placeholder15
1while_while_cond_2162605___redundant_placeholder25
1while_while_cond_2162605___redundant_placeholder35
1while_while_cond_2162605___redundant_placeholder45
1while_while_cond_2162605___redundant_placeholder55
1while_while_cond_2162605___redundant_placeholder6
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
Ó	
ø
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_2162785

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
àY

while_body_2159628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_wdwulgrltk_matmul_readvariableop_resource_0:	 F
3while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 A
2while_wdwulgrltk_biasadd_readvariableop_resource_0:	8
*while_wdwulgrltk_readvariableop_resource_0: :
,while_wdwulgrltk_readvariableop_1_resource_0: :
,while_wdwulgrltk_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_wdwulgrltk_matmul_readvariableop_resource:	 D
1while_wdwulgrltk_matmul_1_readvariableop_resource:	 ?
0while_wdwulgrltk_biasadd_readvariableop_resource:	6
(while_wdwulgrltk_readvariableop_resource: 8
*while_wdwulgrltk_readvariableop_1_resource: 8
*while_wdwulgrltk_readvariableop_2_resource: ¢'while/wdwulgrltk/BiasAdd/ReadVariableOp¢&while/wdwulgrltk/MatMul/ReadVariableOp¢(while/wdwulgrltk/MatMul_1/ReadVariableOp¢while/wdwulgrltk/ReadVariableOp¢!while/wdwulgrltk/ReadVariableOp_1¢!while/wdwulgrltk/ReadVariableOp_2Ã
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
&while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp1while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/wdwulgrltk/MatMul/ReadVariableOpÑ
while/wdwulgrltk/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMulÉ
(while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp3while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/wdwulgrltk/MatMul_1/ReadVariableOpº
while/wdwulgrltk/MatMul_1MatMulwhile_placeholder_20while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMul_1°
while/wdwulgrltk/addAddV2!while/wdwulgrltk/MatMul:product:0#while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/addÂ
'while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp2while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/wdwulgrltk/BiasAdd/ReadVariableOp½
while/wdwulgrltk/BiasAddBiasAddwhile/wdwulgrltk/add:z:0/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/BiasAdd
 while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/wdwulgrltk/split/split_dim
while/wdwulgrltk/splitSplit)while/wdwulgrltk/split/split_dim:output:0!while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/wdwulgrltk/split©
while/wdwulgrltk/ReadVariableOpReadVariableOp*while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/wdwulgrltk/ReadVariableOp£
while/wdwulgrltk/mulMul'while/wdwulgrltk/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul¦
while/wdwulgrltk/add_1AddV2while/wdwulgrltk/split:output:0while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_1
while/wdwulgrltk/SigmoidSigmoidwhile/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid¯
!while/wdwulgrltk/ReadVariableOp_1ReadVariableOp,while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_1©
while/wdwulgrltk/mul_1Mul)while/wdwulgrltk/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_1¨
while/wdwulgrltk/add_2AddV2while/wdwulgrltk/split:output:1while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_2
while/wdwulgrltk/Sigmoid_1Sigmoidwhile/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_1
while/wdwulgrltk/mul_2Mulwhile/wdwulgrltk/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_2
while/wdwulgrltk/TanhTanhwhile/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh¢
while/wdwulgrltk/mul_3Mulwhile/wdwulgrltk/Sigmoid:y:0while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_3£
while/wdwulgrltk/add_3AddV2while/wdwulgrltk/mul_2:z:0while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_3¯
!while/wdwulgrltk/ReadVariableOp_2ReadVariableOp,while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_2°
while/wdwulgrltk/mul_4Mul)while/wdwulgrltk/ReadVariableOp_2:value:0while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_4¨
while/wdwulgrltk/add_4AddV2while/wdwulgrltk/split:output:3while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_4
while/wdwulgrltk/Sigmoid_2Sigmoidwhile/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_2
while/wdwulgrltk/Tanh_1Tanhwhile/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh_1¦
while/wdwulgrltk/mul_5Mulwhile/wdwulgrltk/Sigmoid_2:y:0while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/wdwulgrltk/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/wdwulgrltk/mul_5:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/wdwulgrltk/add_3:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
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
0while_wdwulgrltk_biasadd_readvariableop_resource2while_wdwulgrltk_biasadd_readvariableop_resource_0"h
1while_wdwulgrltk_matmul_1_readvariableop_resource3while_wdwulgrltk_matmul_1_readvariableop_resource_0"d
/while_wdwulgrltk_matmul_readvariableop_resource1while_wdwulgrltk_matmul_readvariableop_resource_0"Z
*while_wdwulgrltk_readvariableop_1_resource,while_wdwulgrltk_readvariableop_1_resource_0"Z
*while_wdwulgrltk_readvariableop_2_resource,while_wdwulgrltk_readvariableop_2_resource_0"V
(while_wdwulgrltk_readvariableop_resource*while_wdwulgrltk_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/wdwulgrltk/BiasAdd/ReadVariableOp'while/wdwulgrltk/BiasAdd/ReadVariableOp2P
&while/wdwulgrltk/MatMul/ReadVariableOp&while/wdwulgrltk/MatMul/ReadVariableOp2T
(while/wdwulgrltk/MatMul_1/ReadVariableOp(while/wdwulgrltk/MatMul_1/ReadVariableOp2B
while/wdwulgrltk/ReadVariableOpwhile/wdwulgrltk/ReadVariableOp2F
!while/wdwulgrltk/ReadVariableOp_1!while/wdwulgrltk/ReadVariableOp_12F
!while/wdwulgrltk/ReadVariableOp_2!while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2163016

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
¸
¼
G__inference_sequential_layer_call_and_return_conditional_losses_2159485

inputs(
vfwtupxpzf_2159062: 
vfwtupxpzf_2159064:%
vhacowjcza_2159262:	%
vhacowjcza_2159264:	 !
vhacowjcza_2159266:	 
vhacowjcza_2159268:  
vhacowjcza_2159270:  
vhacowjcza_2159272: %
zgrdiwrovx_2159455:	 %
zgrdiwrovx_2159457:	 !
zgrdiwrovx_2159459:	 
zgrdiwrovx_2159461:  
zgrdiwrovx_2159463:  
zgrdiwrovx_2159465: $
kekwghyimt_2159479:  
kekwghyimt_2159481:
identity¢"kekwghyimt/StatefulPartitionedCall¢"vfwtupxpzf/StatefulPartitionedCall¢"vhacowjcza/StatefulPartitionedCall¢"zgrdiwrovx/StatefulPartitionedCall¬
"vfwtupxpzf/StatefulPartitionedCallStatefulPartitionedCallinputsvfwtupxpzf_2159062vfwtupxpzf_2159064*
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
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_21590612$
"vfwtupxpzf/StatefulPartitionedCall
ojzbgzevue/PartitionedCallPartitionedCall+vfwtupxpzf/StatefulPartitionedCall:output:0*
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
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_21590802
ojzbgzevue/PartitionedCall
"vhacowjcza/StatefulPartitionedCallStatefulPartitionedCall#ojzbgzevue/PartitionedCall:output:0vhacowjcza_2159262vhacowjcza_2159264vhacowjcza_2159266vhacowjcza_2159268vhacowjcza_2159270vhacowjcza_2159272*
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21592612$
"vhacowjcza/StatefulPartitionedCall¡
"zgrdiwrovx/StatefulPartitionedCallStatefulPartitionedCall+vhacowjcza/StatefulPartitionedCall:output:0zgrdiwrovx_2159455zgrdiwrovx_2159457zgrdiwrovx_2159459zgrdiwrovx_2159461zgrdiwrovx_2159463zgrdiwrovx_2159465*
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21594542$
"zgrdiwrovx/StatefulPartitionedCallÉ
"kekwghyimt/StatefulPartitionedCallStatefulPartitionedCall+zgrdiwrovx/StatefulPartitionedCall:output:0kekwghyimt_2159479kekwghyimt_2159481*
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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_21594782$
"kekwghyimt/StatefulPartitionedCall
IdentityIdentity+kekwghyimt/StatefulPartitionedCall:output:0#^kekwghyimt/StatefulPartitionedCall#^vfwtupxpzf/StatefulPartitionedCall#^vhacowjcza/StatefulPartitionedCall#^zgrdiwrovx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kekwghyimt/StatefulPartitionedCall"kekwghyimt/StatefulPartitionedCall2H
"vfwtupxpzf/StatefulPartitionedCall"vfwtupxpzf/StatefulPartitionedCall2H
"vhacowjcza/StatefulPartitionedCall"vhacowjcza/StatefulPartitionedCall2H
"zgrdiwrovx/StatefulPartitionedCall"zgrdiwrovx/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

,__inference_kekwghyimt_layer_call_fn_2162794

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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_21594782
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
àY

while_body_2161818
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jczmzyhsca_matmul_readvariableop_resource_0:	F
3while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 A
2while_jczmzyhsca_biasadd_readvariableop_resource_0:	8
*while_jczmzyhsca_readvariableop_resource_0: :
,while_jczmzyhsca_readvariableop_1_resource_0: :
,while_jczmzyhsca_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jczmzyhsca_matmul_readvariableop_resource:	D
1while_jczmzyhsca_matmul_1_readvariableop_resource:	 ?
0while_jczmzyhsca_biasadd_readvariableop_resource:	6
(while_jczmzyhsca_readvariableop_resource: 8
*while_jczmzyhsca_readvariableop_1_resource: 8
*while_jczmzyhsca_readvariableop_2_resource: ¢'while/jczmzyhsca/BiasAdd/ReadVariableOp¢&while/jczmzyhsca/MatMul/ReadVariableOp¢(while/jczmzyhsca/MatMul_1/ReadVariableOp¢while/jczmzyhsca/ReadVariableOp¢!while/jczmzyhsca/ReadVariableOp_1¢!while/jczmzyhsca/ReadVariableOp_2Ã
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
&while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp1while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jczmzyhsca/MatMul/ReadVariableOpÑ
while/jczmzyhsca/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMulÉ
(while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp3while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jczmzyhsca/MatMul_1/ReadVariableOpº
while/jczmzyhsca/MatMul_1MatMulwhile_placeholder_20while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMul_1°
while/jczmzyhsca/addAddV2!while/jczmzyhsca/MatMul:product:0#while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/addÂ
'while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp2while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jczmzyhsca/BiasAdd/ReadVariableOp½
while/jczmzyhsca/BiasAddBiasAddwhile/jczmzyhsca/add:z:0/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/BiasAdd
 while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jczmzyhsca/split/split_dim
while/jczmzyhsca/splitSplit)while/jczmzyhsca/split/split_dim:output:0!while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jczmzyhsca/split©
while/jczmzyhsca/ReadVariableOpReadVariableOp*while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jczmzyhsca/ReadVariableOp£
while/jczmzyhsca/mulMul'while/jczmzyhsca/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul¦
while/jczmzyhsca/add_1AddV2while/jczmzyhsca/split:output:0while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_1
while/jczmzyhsca/SigmoidSigmoidwhile/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid¯
!while/jczmzyhsca/ReadVariableOp_1ReadVariableOp,while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_1©
while/jczmzyhsca/mul_1Mul)while/jczmzyhsca/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_1¨
while/jczmzyhsca/add_2AddV2while/jczmzyhsca/split:output:1while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_2
while/jczmzyhsca/Sigmoid_1Sigmoidwhile/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_1
while/jczmzyhsca/mul_2Mulwhile/jczmzyhsca/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_2
while/jczmzyhsca/TanhTanhwhile/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh¢
while/jczmzyhsca/mul_3Mulwhile/jczmzyhsca/Sigmoid:y:0while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_3£
while/jczmzyhsca/add_3AddV2while/jczmzyhsca/mul_2:z:0while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_3¯
!while/jczmzyhsca/ReadVariableOp_2ReadVariableOp,while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_2°
while/jczmzyhsca/mul_4Mul)while/jczmzyhsca/ReadVariableOp_2:value:0while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_4¨
while/jczmzyhsca/add_4AddV2while/jczmzyhsca/split:output:3while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_4
while/jczmzyhsca/Sigmoid_2Sigmoidwhile/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_2
while/jczmzyhsca/Tanh_1Tanhwhile/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh_1¦
while/jczmzyhsca/mul_5Mulwhile/jczmzyhsca/Sigmoid_2:y:0while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jczmzyhsca/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jczmzyhsca/mul_5:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jczmzyhsca/add_3:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
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
0while_jczmzyhsca_biasadd_readvariableop_resource2while_jczmzyhsca_biasadd_readvariableop_resource_0"h
1while_jczmzyhsca_matmul_1_readvariableop_resource3while_jczmzyhsca_matmul_1_readvariableop_resource_0"d
/while_jczmzyhsca_matmul_readvariableop_resource1while_jczmzyhsca_matmul_readvariableop_resource_0"Z
*while_jczmzyhsca_readvariableop_1_resource,while_jczmzyhsca_readvariableop_1_resource_0"Z
*while_jczmzyhsca_readvariableop_2_resource,while_jczmzyhsca_readvariableop_2_resource_0"V
(while_jczmzyhsca_readvariableop_resource*while_jczmzyhsca_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jczmzyhsca/BiasAdd/ReadVariableOp'while/jczmzyhsca/BiasAdd/ReadVariableOp2P
&while/jczmzyhsca/MatMul/ReadVariableOp&while/jczmzyhsca/MatMul/ReadVariableOp2T
(while/jczmzyhsca/MatMul_1/ReadVariableOp(while/jczmzyhsca/MatMul_1/ReadVariableOp2B
while/jczmzyhsca/ReadVariableOpwhile/jczmzyhsca/ReadVariableOp2F
!while/jczmzyhsca/ReadVariableOp_1!while/jczmzyhsca/ReadVariableOp_12F
!while/jczmzyhsca/ReadVariableOp_2!while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_zgrdiwrovx_layer_call_fn_2162741
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21587092
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


,__inference_sequential_layer_call_fn_2159520

aveeivcxur
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
aveeivcxurunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_21594852
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
aveeivcxur


í
while_cond_2159159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2159159___redundant_placeholder05
1while_while_cond_2159159___redundant_placeholder15
1while_while_cond_2159159___redundant_placeholder25
1while_while_cond_2159159___redundant_placeholder35
1while_while_cond_2159159___redundant_placeholder45
1while_while_cond_2159159___redundant_placeholder55
1while_while_cond_2159159___redundant_placeholder6
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
©0
¼
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_2161172

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
G__inference_sequential_layer_call_and_return_conditional_losses_2160167

aveeivcxur(
vfwtupxpzf_2160129: 
vfwtupxpzf_2160131:%
vhacowjcza_2160135:	%
vhacowjcza_2160137:	 !
vhacowjcza_2160139:	 
vhacowjcza_2160141:  
vhacowjcza_2160143:  
vhacowjcza_2160145: %
zgrdiwrovx_2160148:	 %
zgrdiwrovx_2160150:	 !
zgrdiwrovx_2160152:	 
zgrdiwrovx_2160154:  
zgrdiwrovx_2160156:  
zgrdiwrovx_2160158: $
kekwghyimt_2160161:  
kekwghyimt_2160163:
identity¢"kekwghyimt/StatefulPartitionedCall¢"vfwtupxpzf/StatefulPartitionedCall¢"vhacowjcza/StatefulPartitionedCall¢"zgrdiwrovx/StatefulPartitionedCall°
"vfwtupxpzf/StatefulPartitionedCallStatefulPartitionedCall
aveeivcxurvfwtupxpzf_2160129vfwtupxpzf_2160131*
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
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_21590612$
"vfwtupxpzf/StatefulPartitionedCall
ojzbgzevue/PartitionedCallPartitionedCall+vfwtupxpzf/StatefulPartitionedCall:output:0*
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
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_21590802
ojzbgzevue/PartitionedCall
"vhacowjcza/StatefulPartitionedCallStatefulPartitionedCall#ojzbgzevue/PartitionedCall:output:0vhacowjcza_2160135vhacowjcza_2160137vhacowjcza_2160139vhacowjcza_2160141vhacowjcza_2160143vhacowjcza_2160145*
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21592612$
"vhacowjcza/StatefulPartitionedCall¡
"zgrdiwrovx/StatefulPartitionedCallStatefulPartitionedCall+vhacowjcza/StatefulPartitionedCall:output:0zgrdiwrovx_2160148zgrdiwrovx_2160150zgrdiwrovx_2160152zgrdiwrovx_2160154zgrdiwrovx_2160156zgrdiwrovx_2160158*
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21594542$
"zgrdiwrovx/StatefulPartitionedCallÉ
"kekwghyimt/StatefulPartitionedCallStatefulPartitionedCall+zgrdiwrovx/StatefulPartitionedCall:output:0kekwghyimt_2160161kekwghyimt_2160163*
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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_21594782$
"kekwghyimt/StatefulPartitionedCall
IdentityIdentity+kekwghyimt/StatefulPartitionedCall:output:0#^kekwghyimt/StatefulPartitionedCall#^vfwtupxpzf/StatefulPartitionedCall#^vhacowjcza/StatefulPartitionedCall#^zgrdiwrovx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kekwghyimt/StatefulPartitionedCall"kekwghyimt/StatefulPartitionedCall2H
"vfwtupxpzf/StatefulPartitionedCall"vfwtupxpzf/StatefulPartitionedCall2H
"vhacowjcza/StatefulPartitionedCall"vhacowjcza/StatefulPartitionedCall2H
"zgrdiwrovx/StatefulPartitionedCall"zgrdiwrovx/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
aveeivcxur
ë

,__inference_zgrdiwrovx_layer_call_fn_2162724
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21584462
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2159454

inputs<
)wdwulgrltk_matmul_readvariableop_resource:	 >
+wdwulgrltk_matmul_1_readvariableop_resource:	 9
*wdwulgrltk_biasadd_readvariableop_resource:	0
"wdwulgrltk_readvariableop_resource: 2
$wdwulgrltk_readvariableop_1_resource: 2
$wdwulgrltk_readvariableop_2_resource: 
identity¢!wdwulgrltk/BiasAdd/ReadVariableOp¢ wdwulgrltk/MatMul/ReadVariableOp¢"wdwulgrltk/MatMul_1/ReadVariableOp¢wdwulgrltk/ReadVariableOp¢wdwulgrltk/ReadVariableOp_1¢wdwulgrltk/ReadVariableOp_2¢whileD
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
 wdwulgrltk/MatMul/ReadVariableOpReadVariableOp)wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 wdwulgrltk/MatMul/ReadVariableOp§
wdwulgrltk/MatMulMatMulstrided_slice_2:output:0(wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMulµ
"wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp+wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"wdwulgrltk/MatMul_1/ReadVariableOp£
wdwulgrltk/MatMul_1MatMulzeros:output:0*wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMul_1
wdwulgrltk/addAddV2wdwulgrltk/MatMul:product:0wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/add®
!wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp*wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!wdwulgrltk/BiasAdd/ReadVariableOp¥
wdwulgrltk/BiasAddBiasAddwdwulgrltk/add:z:0)wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/BiasAddz
wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
wdwulgrltk/split/split_dimë
wdwulgrltk/splitSplit#wdwulgrltk/split/split_dim:output:0wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
wdwulgrltk/split
wdwulgrltk/ReadVariableOpReadVariableOp"wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp
wdwulgrltk/mulMul!wdwulgrltk/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul
wdwulgrltk/add_1AddV2wdwulgrltk/split:output:0wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_1{
wdwulgrltk/SigmoidSigmoidwdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid
wdwulgrltk/ReadVariableOp_1ReadVariableOp$wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_1
wdwulgrltk/mul_1Mul#wdwulgrltk/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_1
wdwulgrltk/add_2AddV2wdwulgrltk/split:output:1wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_2
wdwulgrltk/Sigmoid_1Sigmoidwdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_1
wdwulgrltk/mul_2Mulwdwulgrltk/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_2w
wdwulgrltk/TanhTanhwdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh
wdwulgrltk/mul_3Mulwdwulgrltk/Sigmoid:y:0wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_3
wdwulgrltk/add_3AddV2wdwulgrltk/mul_2:z:0wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_3
wdwulgrltk/ReadVariableOp_2ReadVariableOp$wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_2
wdwulgrltk/mul_4Mul#wdwulgrltk/ReadVariableOp_2:value:0wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_4
wdwulgrltk/add_4AddV2wdwulgrltk/split:output:3wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_4
wdwulgrltk/Sigmoid_2Sigmoidwdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_2v
wdwulgrltk/Tanh_1Tanhwdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh_1
wdwulgrltk/mul_5Mulwdwulgrltk/Sigmoid_2:y:0wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)wdwulgrltk_matmul_readvariableop_resource+wdwulgrltk_matmul_1_readvariableop_resource*wdwulgrltk_biasadd_readvariableop_resource"wdwulgrltk_readvariableop_resource$wdwulgrltk_readvariableop_1_resource$wdwulgrltk_readvariableop_2_resource*
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
while_body_2159353*
condR
while_cond_2159352*Q
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
IdentityIdentitystrided_slice_3:output:0"^wdwulgrltk/BiasAdd/ReadVariableOp!^wdwulgrltk/MatMul/ReadVariableOp#^wdwulgrltk/MatMul_1/ReadVariableOp^wdwulgrltk/ReadVariableOp^wdwulgrltk/ReadVariableOp_1^wdwulgrltk/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!wdwulgrltk/BiasAdd/ReadVariableOp!wdwulgrltk/BiasAdd/ReadVariableOp2D
 wdwulgrltk/MatMul/ReadVariableOp wdwulgrltk/MatMul/ReadVariableOp2H
"wdwulgrltk/MatMul_1/ReadVariableOp"wdwulgrltk/MatMul_1/ReadVariableOp26
wdwulgrltk/ReadVariableOpwdwulgrltk/ReadVariableOp2:
wdwulgrltk/ReadVariableOp_1wdwulgrltk/ReadVariableOp_12:
wdwulgrltk/ReadVariableOp_2wdwulgrltk/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_2161277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2161277___redundant_placeholder05
1while_while_cond_2161277___redundant_placeholder15
1while_while_cond_2161277___redundant_placeholder25
1while_while_cond_2161277___redundant_placeholder35
1while_while_cond_2161277___redundant_placeholder45
1while_while_cond_2161277___redundant_placeholder55
1while_while_cond_2161277___redundant_placeholder6
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2162972

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
Û

,__inference_vhacowjcza_layer_call_fn_2161987

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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21599432
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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_2159478

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
while_cond_2158365
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2158365___redundant_placeholder05
1while_while_cond_2158365___redundant_placeholder15
1while_while_cond_2158365___redundant_placeholder25
1while_while_cond_2158365___redundant_placeholder35
1while_while_cond_2158365___redundant_placeholder45
1while_while_cond_2158365___redundant_placeholder55
1while_while_cond_2158365___redundant_placeholder6
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2162838

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
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_2160208

aveeivcxur(
vfwtupxpzf_2160170: 
vfwtupxpzf_2160172:%
vhacowjcza_2160176:	%
vhacowjcza_2160178:	 !
vhacowjcza_2160180:	 
vhacowjcza_2160182:  
vhacowjcza_2160184:  
vhacowjcza_2160186: %
zgrdiwrovx_2160189:	 %
zgrdiwrovx_2160191:	 !
zgrdiwrovx_2160193:	 
zgrdiwrovx_2160195:  
zgrdiwrovx_2160197:  
zgrdiwrovx_2160199: $
kekwghyimt_2160202:  
kekwghyimt_2160204:
identity¢"kekwghyimt/StatefulPartitionedCall¢"vfwtupxpzf/StatefulPartitionedCall¢"vhacowjcza/StatefulPartitionedCall¢"zgrdiwrovx/StatefulPartitionedCall°
"vfwtupxpzf/StatefulPartitionedCallStatefulPartitionedCall
aveeivcxurvfwtupxpzf_2160170vfwtupxpzf_2160172*
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
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_21590612$
"vfwtupxpzf/StatefulPartitionedCall
ojzbgzevue/PartitionedCallPartitionedCall+vfwtupxpzf/StatefulPartitionedCall:output:0*
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
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_21590802
ojzbgzevue/PartitionedCall
"vhacowjcza/StatefulPartitionedCallStatefulPartitionedCall#ojzbgzevue/PartitionedCall:output:0vhacowjcza_2160176vhacowjcza_2160178vhacowjcza_2160180vhacowjcza_2160182vhacowjcza_2160184vhacowjcza_2160186*
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21599432$
"vhacowjcza/StatefulPartitionedCall¡
"zgrdiwrovx/StatefulPartitionedCallStatefulPartitionedCall+vhacowjcza/StatefulPartitionedCall:output:0zgrdiwrovx_2160189zgrdiwrovx_2160191zgrdiwrovx_2160193zgrdiwrovx_2160195zgrdiwrovx_2160197zgrdiwrovx_2160199*
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21597292$
"zgrdiwrovx/StatefulPartitionedCallÉ
"kekwghyimt/StatefulPartitionedCallStatefulPartitionedCall+zgrdiwrovx/StatefulPartitionedCall:output:0kekwghyimt_2160202kekwghyimt_2160204*
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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_21594782$
"kekwghyimt/StatefulPartitionedCall
IdentityIdentity+kekwghyimt/StatefulPartitionedCall:output:0#^kekwghyimt/StatefulPartitionedCall#^vfwtupxpzf/StatefulPartitionedCall#^vhacowjcza/StatefulPartitionedCall#^zgrdiwrovx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kekwghyimt/StatefulPartitionedCall"kekwghyimt/StatefulPartitionedCall2H
"vfwtupxpzf/StatefulPartitionedCall"vfwtupxpzf/StatefulPartitionedCall2H
"vhacowjcza/StatefulPartitionedCall"vhacowjcza/StatefulPartitionedCall2H
"zgrdiwrovx/StatefulPartitionedCall"zgrdiwrovx/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
aveeivcxur
Ó

,__inference_zgrdiwrovx_layer_call_fn_2162775

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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21597292
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
	

,__inference_vhacowjcza_layer_call_fn_2161936
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21576882
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
Ó

,__inference_zgrdiwrovx_layer_call_fn_2162758

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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21594542
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
while_cond_2161457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2161457___redundant_placeholder05
1while_while_cond_2161457___redundant_placeholder15
1while_while_cond_2161457___redundant_placeholder25
1while_while_cond_2161457___redundant_placeholder35
1while_while_cond_2161457___redundant_placeholder45
1while_while_cond_2161457___redundant_placeholder55
1while_while_cond_2161457___redundant_placeholder6
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161739

inputs<
)jczmzyhsca_matmul_readvariableop_resource:	>
+jczmzyhsca_matmul_1_readvariableop_resource:	 9
*jczmzyhsca_biasadd_readvariableop_resource:	0
"jczmzyhsca_readvariableop_resource: 2
$jczmzyhsca_readvariableop_1_resource: 2
$jczmzyhsca_readvariableop_2_resource: 
identity¢!jczmzyhsca/BiasAdd/ReadVariableOp¢ jczmzyhsca/MatMul/ReadVariableOp¢"jczmzyhsca/MatMul_1/ReadVariableOp¢jczmzyhsca/ReadVariableOp¢jczmzyhsca/ReadVariableOp_1¢jczmzyhsca/ReadVariableOp_2¢whileD
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
 jczmzyhsca/MatMul/ReadVariableOpReadVariableOp)jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jczmzyhsca/MatMul/ReadVariableOp§
jczmzyhsca/MatMulMatMulstrided_slice_2:output:0(jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMulµ
"jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp+jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jczmzyhsca/MatMul_1/ReadVariableOp£
jczmzyhsca/MatMul_1MatMulzeros:output:0*jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMul_1
jczmzyhsca/addAddV2jczmzyhsca/MatMul:product:0jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/add®
!jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp*jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jczmzyhsca/BiasAdd/ReadVariableOp¥
jczmzyhsca/BiasAddBiasAddjczmzyhsca/add:z:0)jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/BiasAddz
jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jczmzyhsca/split/split_dimë
jczmzyhsca/splitSplit#jczmzyhsca/split/split_dim:output:0jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jczmzyhsca/split
jczmzyhsca/ReadVariableOpReadVariableOp"jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp
jczmzyhsca/mulMul!jczmzyhsca/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul
jczmzyhsca/add_1AddV2jczmzyhsca/split:output:0jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_1{
jczmzyhsca/SigmoidSigmoidjczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid
jczmzyhsca/ReadVariableOp_1ReadVariableOp$jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_1
jczmzyhsca/mul_1Mul#jczmzyhsca/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_1
jczmzyhsca/add_2AddV2jczmzyhsca/split:output:1jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_2
jczmzyhsca/Sigmoid_1Sigmoidjczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_1
jczmzyhsca/mul_2Muljczmzyhsca/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_2w
jczmzyhsca/TanhTanhjczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh
jczmzyhsca/mul_3Muljczmzyhsca/Sigmoid:y:0jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_3
jczmzyhsca/add_3AddV2jczmzyhsca/mul_2:z:0jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_3
jczmzyhsca/ReadVariableOp_2ReadVariableOp$jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_2
jczmzyhsca/mul_4Mul#jczmzyhsca/ReadVariableOp_2:value:0jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_4
jczmzyhsca/add_4AddV2jczmzyhsca/split:output:3jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_4
jczmzyhsca/Sigmoid_2Sigmoidjczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_2v
jczmzyhsca/Tanh_1Tanhjczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh_1
jczmzyhsca/mul_5Muljczmzyhsca/Sigmoid_2:y:0jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jczmzyhsca_matmul_readvariableop_resource+jczmzyhsca_matmul_1_readvariableop_resource*jczmzyhsca_biasadd_readvariableop_resource"jczmzyhsca_readvariableop_resource$jczmzyhsca_readvariableop_1_resource$jczmzyhsca_readvariableop_2_resource*
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
while_body_2161638*
condR
while_cond_2161637*Q
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
IdentityIdentitytranspose_1:y:0"^jczmzyhsca/BiasAdd/ReadVariableOp!^jczmzyhsca/MatMul/ReadVariableOp#^jczmzyhsca/MatMul_1/ReadVariableOp^jczmzyhsca/ReadVariableOp^jczmzyhsca/ReadVariableOp_1^jczmzyhsca/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jczmzyhsca/BiasAdd/ReadVariableOp!jczmzyhsca/BiasAdd/ReadVariableOp2D
 jczmzyhsca/MatMul/ReadVariableOp jczmzyhsca/MatMul/ReadVariableOp2H
"jczmzyhsca/MatMul_1/ReadVariableOp"jczmzyhsca/MatMul_1/ReadVariableOp26
jczmzyhsca/ReadVariableOpjczmzyhsca/ReadVariableOp2:
jczmzyhsca/ReadVariableOp_1jczmzyhsca/ReadVariableOp_12:
jczmzyhsca/ReadVariableOp_2jczmzyhsca/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_sequential_layer_call_fn_2160126

aveeivcxur
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
aveeivcxurunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_21600542
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
aveeivcxur
àY

while_body_2161278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jczmzyhsca_matmul_readvariableop_resource_0:	F
3while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 A
2while_jczmzyhsca_biasadd_readvariableop_resource_0:	8
*while_jczmzyhsca_readvariableop_resource_0: :
,while_jczmzyhsca_readvariableop_1_resource_0: :
,while_jczmzyhsca_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jczmzyhsca_matmul_readvariableop_resource:	D
1while_jczmzyhsca_matmul_1_readvariableop_resource:	 ?
0while_jczmzyhsca_biasadd_readvariableop_resource:	6
(while_jczmzyhsca_readvariableop_resource: 8
*while_jczmzyhsca_readvariableop_1_resource: 8
*while_jczmzyhsca_readvariableop_2_resource: ¢'while/jczmzyhsca/BiasAdd/ReadVariableOp¢&while/jczmzyhsca/MatMul/ReadVariableOp¢(while/jczmzyhsca/MatMul_1/ReadVariableOp¢while/jczmzyhsca/ReadVariableOp¢!while/jczmzyhsca/ReadVariableOp_1¢!while/jczmzyhsca/ReadVariableOp_2Ã
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
&while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp1while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jczmzyhsca/MatMul/ReadVariableOpÑ
while/jczmzyhsca/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMulÉ
(while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp3while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jczmzyhsca/MatMul_1/ReadVariableOpº
while/jczmzyhsca/MatMul_1MatMulwhile_placeholder_20while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMul_1°
while/jczmzyhsca/addAddV2!while/jczmzyhsca/MatMul:product:0#while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/addÂ
'while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp2while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jczmzyhsca/BiasAdd/ReadVariableOp½
while/jczmzyhsca/BiasAddBiasAddwhile/jczmzyhsca/add:z:0/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/BiasAdd
 while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jczmzyhsca/split/split_dim
while/jczmzyhsca/splitSplit)while/jczmzyhsca/split/split_dim:output:0!while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jczmzyhsca/split©
while/jczmzyhsca/ReadVariableOpReadVariableOp*while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jczmzyhsca/ReadVariableOp£
while/jczmzyhsca/mulMul'while/jczmzyhsca/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul¦
while/jczmzyhsca/add_1AddV2while/jczmzyhsca/split:output:0while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_1
while/jczmzyhsca/SigmoidSigmoidwhile/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid¯
!while/jczmzyhsca/ReadVariableOp_1ReadVariableOp,while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_1©
while/jczmzyhsca/mul_1Mul)while/jczmzyhsca/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_1¨
while/jczmzyhsca/add_2AddV2while/jczmzyhsca/split:output:1while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_2
while/jczmzyhsca/Sigmoid_1Sigmoidwhile/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_1
while/jczmzyhsca/mul_2Mulwhile/jczmzyhsca/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_2
while/jczmzyhsca/TanhTanhwhile/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh¢
while/jczmzyhsca/mul_3Mulwhile/jczmzyhsca/Sigmoid:y:0while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_3£
while/jczmzyhsca/add_3AddV2while/jczmzyhsca/mul_2:z:0while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_3¯
!while/jczmzyhsca/ReadVariableOp_2ReadVariableOp,while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_2°
while/jczmzyhsca/mul_4Mul)while/jczmzyhsca/ReadVariableOp_2:value:0while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_4¨
while/jczmzyhsca/add_4AddV2while/jczmzyhsca/split:output:3while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_4
while/jczmzyhsca/Sigmoid_2Sigmoidwhile/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_2
while/jczmzyhsca/Tanh_1Tanhwhile/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh_1¦
while/jczmzyhsca/mul_5Mulwhile/jczmzyhsca/Sigmoid_2:y:0while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jczmzyhsca/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jczmzyhsca/mul_5:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jczmzyhsca/add_3:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
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
0while_jczmzyhsca_biasadd_readvariableop_resource2while_jczmzyhsca_biasadd_readvariableop_resource_0"h
1while_jczmzyhsca_matmul_1_readvariableop_resource3while_jczmzyhsca_matmul_1_readvariableop_resource_0"d
/while_jczmzyhsca_matmul_readvariableop_resource1while_jczmzyhsca_matmul_readvariableop_resource_0"Z
*while_jczmzyhsca_readvariableop_1_resource,while_jczmzyhsca_readvariableop_1_resource_0"Z
*while_jczmzyhsca_readvariableop_2_resource,while_jczmzyhsca_readvariableop_2_resource_0"V
(while_jczmzyhsca_readvariableop_resource*while_jczmzyhsca_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jczmzyhsca/BiasAdd/ReadVariableOp'while/jczmzyhsca/BiasAdd/ReadVariableOp2P
&while/jczmzyhsca/MatMul/ReadVariableOp&while/jczmzyhsca/MatMul/ReadVariableOp2T
(while/jczmzyhsca/MatMul_1/ReadVariableOp(while/jczmzyhsca/MatMul_1/ReadVariableOp2B
while/jczmzyhsca/ReadVariableOpwhile/jczmzyhsca/ReadVariableOp2F
!while/jczmzyhsca/ReadVariableOp_1!while/jczmzyhsca/ReadVariableOp_12F
!while/jczmzyhsca/ReadVariableOp_2!while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2157870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2157870___redundant_placeholder05
1while_while_cond_2157870___redundant_placeholder15
1while_while_cond_2157870___redundant_placeholder25
1while_while_cond_2157870___redundant_placeholder35
1while_while_cond_2157870___redundant_placeholder45
1while_while_cond_2157870___redundant_placeholder55
1while_while_cond_2157870___redundant_placeholder6
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
©0
¼
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_2159061

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
¦h

G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2159729

inputs<
)wdwulgrltk_matmul_readvariableop_resource:	 >
+wdwulgrltk_matmul_1_readvariableop_resource:	 9
*wdwulgrltk_biasadd_readvariableop_resource:	0
"wdwulgrltk_readvariableop_resource: 2
$wdwulgrltk_readvariableop_1_resource: 2
$wdwulgrltk_readvariableop_2_resource: 
identity¢!wdwulgrltk/BiasAdd/ReadVariableOp¢ wdwulgrltk/MatMul/ReadVariableOp¢"wdwulgrltk/MatMul_1/ReadVariableOp¢wdwulgrltk/ReadVariableOp¢wdwulgrltk/ReadVariableOp_1¢wdwulgrltk/ReadVariableOp_2¢whileD
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
 wdwulgrltk/MatMul/ReadVariableOpReadVariableOp)wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 wdwulgrltk/MatMul/ReadVariableOp§
wdwulgrltk/MatMulMatMulstrided_slice_2:output:0(wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMulµ
"wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp+wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"wdwulgrltk/MatMul_1/ReadVariableOp£
wdwulgrltk/MatMul_1MatMulzeros:output:0*wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMul_1
wdwulgrltk/addAddV2wdwulgrltk/MatMul:product:0wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/add®
!wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp*wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!wdwulgrltk/BiasAdd/ReadVariableOp¥
wdwulgrltk/BiasAddBiasAddwdwulgrltk/add:z:0)wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/BiasAddz
wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
wdwulgrltk/split/split_dimë
wdwulgrltk/splitSplit#wdwulgrltk/split/split_dim:output:0wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
wdwulgrltk/split
wdwulgrltk/ReadVariableOpReadVariableOp"wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp
wdwulgrltk/mulMul!wdwulgrltk/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul
wdwulgrltk/add_1AddV2wdwulgrltk/split:output:0wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_1{
wdwulgrltk/SigmoidSigmoidwdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid
wdwulgrltk/ReadVariableOp_1ReadVariableOp$wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_1
wdwulgrltk/mul_1Mul#wdwulgrltk/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_1
wdwulgrltk/add_2AddV2wdwulgrltk/split:output:1wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_2
wdwulgrltk/Sigmoid_1Sigmoidwdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_1
wdwulgrltk/mul_2Mulwdwulgrltk/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_2w
wdwulgrltk/TanhTanhwdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh
wdwulgrltk/mul_3Mulwdwulgrltk/Sigmoid:y:0wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_3
wdwulgrltk/add_3AddV2wdwulgrltk/mul_2:z:0wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_3
wdwulgrltk/ReadVariableOp_2ReadVariableOp$wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_2
wdwulgrltk/mul_4Mul#wdwulgrltk/ReadVariableOp_2:value:0wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_4
wdwulgrltk/add_4AddV2wdwulgrltk/split:output:3wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_4
wdwulgrltk/Sigmoid_2Sigmoidwdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_2v
wdwulgrltk/Tanh_1Tanhwdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh_1
wdwulgrltk/mul_5Mulwdwulgrltk/Sigmoid_2:y:0wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)wdwulgrltk_matmul_readvariableop_resource+wdwulgrltk_matmul_1_readvariableop_resource*wdwulgrltk_biasadd_readvariableop_resource"wdwulgrltk_readvariableop_resource$wdwulgrltk_readvariableop_1_resource$wdwulgrltk_readvariableop_2_resource*
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
while_body_2159628*
condR
while_cond_2159627*Q
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
IdentityIdentitystrided_slice_3:output:0"^wdwulgrltk/BiasAdd/ReadVariableOp!^wdwulgrltk/MatMul/ReadVariableOp#^wdwulgrltk/MatMul_1/ReadVariableOp^wdwulgrltk/ReadVariableOp^wdwulgrltk/ReadVariableOp_1^wdwulgrltk/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!wdwulgrltk/BiasAdd/ReadVariableOp!wdwulgrltk/BiasAdd/ReadVariableOp2D
 wdwulgrltk/MatMul/ReadVariableOp wdwulgrltk/MatMul/ReadVariableOp2H
"wdwulgrltk/MatMul_1/ReadVariableOp"wdwulgrltk/MatMul_1/ReadVariableOp26
wdwulgrltk/ReadVariableOpwdwulgrltk/ReadVariableOp2:
wdwulgrltk/ReadVariableOp_1wdwulgrltk/ReadVariableOp_12:
wdwulgrltk/ReadVariableOp_2wdwulgrltk/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


zgrdiwrovx_while_cond_21605492
.zgrdiwrovx_while_zgrdiwrovx_while_loop_counter8
4zgrdiwrovx_while_zgrdiwrovx_while_maximum_iterations 
zgrdiwrovx_while_placeholder"
zgrdiwrovx_while_placeholder_1"
zgrdiwrovx_while_placeholder_2"
zgrdiwrovx_while_placeholder_34
0zgrdiwrovx_while_less_zgrdiwrovx_strided_slice_1K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160549___redundant_placeholder0K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160549___redundant_placeholder1K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160549___redundant_placeholder2K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160549___redundant_placeholder3K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160549___redundant_placeholder4K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160549___redundant_placeholder5K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160549___redundant_placeholder6
zgrdiwrovx_while_identity
§
zgrdiwrovx/while/LessLesszgrdiwrovx_while_placeholder0zgrdiwrovx_while_less_zgrdiwrovx_strided_slice_1*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Less~
zgrdiwrovx/while/IdentityIdentityzgrdiwrovx/while/Less:z:0*
T0
*
_output_shapes
: 2
zgrdiwrovx/while/Identity"?
zgrdiwrovx_while_identity"zgrdiwrovx/while/Identity:output:0*(
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
while_body_2162606
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_wdwulgrltk_matmul_readvariableop_resource_0:	 F
3while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 A
2while_wdwulgrltk_biasadd_readvariableop_resource_0:	8
*while_wdwulgrltk_readvariableop_resource_0: :
,while_wdwulgrltk_readvariableop_1_resource_0: :
,while_wdwulgrltk_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_wdwulgrltk_matmul_readvariableop_resource:	 D
1while_wdwulgrltk_matmul_1_readvariableop_resource:	 ?
0while_wdwulgrltk_biasadd_readvariableop_resource:	6
(while_wdwulgrltk_readvariableop_resource: 8
*while_wdwulgrltk_readvariableop_1_resource: 8
*while_wdwulgrltk_readvariableop_2_resource: ¢'while/wdwulgrltk/BiasAdd/ReadVariableOp¢&while/wdwulgrltk/MatMul/ReadVariableOp¢(while/wdwulgrltk/MatMul_1/ReadVariableOp¢while/wdwulgrltk/ReadVariableOp¢!while/wdwulgrltk/ReadVariableOp_1¢!while/wdwulgrltk/ReadVariableOp_2Ã
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
&while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp1while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/wdwulgrltk/MatMul/ReadVariableOpÑ
while/wdwulgrltk/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMulÉ
(while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp3while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/wdwulgrltk/MatMul_1/ReadVariableOpº
while/wdwulgrltk/MatMul_1MatMulwhile_placeholder_20while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMul_1°
while/wdwulgrltk/addAddV2!while/wdwulgrltk/MatMul:product:0#while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/addÂ
'while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp2while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/wdwulgrltk/BiasAdd/ReadVariableOp½
while/wdwulgrltk/BiasAddBiasAddwhile/wdwulgrltk/add:z:0/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/BiasAdd
 while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/wdwulgrltk/split/split_dim
while/wdwulgrltk/splitSplit)while/wdwulgrltk/split/split_dim:output:0!while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/wdwulgrltk/split©
while/wdwulgrltk/ReadVariableOpReadVariableOp*while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/wdwulgrltk/ReadVariableOp£
while/wdwulgrltk/mulMul'while/wdwulgrltk/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul¦
while/wdwulgrltk/add_1AddV2while/wdwulgrltk/split:output:0while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_1
while/wdwulgrltk/SigmoidSigmoidwhile/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid¯
!while/wdwulgrltk/ReadVariableOp_1ReadVariableOp,while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_1©
while/wdwulgrltk/mul_1Mul)while/wdwulgrltk/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_1¨
while/wdwulgrltk/add_2AddV2while/wdwulgrltk/split:output:1while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_2
while/wdwulgrltk/Sigmoid_1Sigmoidwhile/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_1
while/wdwulgrltk/mul_2Mulwhile/wdwulgrltk/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_2
while/wdwulgrltk/TanhTanhwhile/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh¢
while/wdwulgrltk/mul_3Mulwhile/wdwulgrltk/Sigmoid:y:0while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_3£
while/wdwulgrltk/add_3AddV2while/wdwulgrltk/mul_2:z:0while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_3¯
!while/wdwulgrltk/ReadVariableOp_2ReadVariableOp,while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_2°
while/wdwulgrltk/mul_4Mul)while/wdwulgrltk/ReadVariableOp_2:value:0while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_4¨
while/wdwulgrltk/add_4AddV2while/wdwulgrltk/split:output:3while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_4
while/wdwulgrltk/Sigmoid_2Sigmoidwhile/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_2
while/wdwulgrltk/Tanh_1Tanhwhile/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh_1¦
while/wdwulgrltk/mul_5Mulwhile/wdwulgrltk/Sigmoid_2:y:0while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/wdwulgrltk/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/wdwulgrltk/mul_5:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/wdwulgrltk/add_3:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
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
0while_wdwulgrltk_biasadd_readvariableop_resource2while_wdwulgrltk_biasadd_readvariableop_resource_0"h
1while_wdwulgrltk_matmul_1_readvariableop_resource3while_wdwulgrltk_matmul_1_readvariableop_resource_0"d
/while_wdwulgrltk_matmul_readvariableop_resource1while_wdwulgrltk_matmul_readvariableop_resource_0"Z
*while_wdwulgrltk_readvariableop_1_resource,while_wdwulgrltk_readvariableop_1_resource_0"Z
*while_wdwulgrltk_readvariableop_2_resource,while_wdwulgrltk_readvariableop_2_resource_0"V
(while_wdwulgrltk_readvariableop_resource*while_wdwulgrltk_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/wdwulgrltk/BiasAdd/ReadVariableOp'while/wdwulgrltk/BiasAdd/ReadVariableOp2P
&while/wdwulgrltk/MatMul/ReadVariableOp&while/wdwulgrltk/MatMul/ReadVariableOp2T
(while/wdwulgrltk/MatMul_1/ReadVariableOp(while/wdwulgrltk/MatMul_1/ReadVariableOp2B
while/wdwulgrltk/ReadVariableOpwhile/wdwulgrltk/ReadVariableOp2F
!while/wdwulgrltk/ReadVariableOp_1!while/wdwulgrltk/ReadVariableOp_12F
!while/wdwulgrltk/ReadVariableOp_2!while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2161637
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2161637___redundant_placeholder05
1while_while_cond_2161637___redundant_placeholder15
1while_while_cond_2161637___redundant_placeholder25
1while_while_cond_2161637___redundant_placeholder35
1while_while_cond_2161637___redundant_placeholder45
1while_while_cond_2161637___redundant_placeholder55
1while_while_cond_2161637___redundant_placeholder6
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


,__inference_sequential_layer_call_fn_2161135

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
G__inference_sequential_layer_call_and_return_conditional_losses_21600542
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
Û

,__inference_vhacowjcza_layer_call_fn_2161970

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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21592612
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
ì

%__inference_signature_wrapper_2160253

aveeivcxur
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
aveeivcxurunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_21575012
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
aveeivcxur
¦h

G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162527

inputs<
)wdwulgrltk_matmul_readvariableop_resource:	 >
+wdwulgrltk_matmul_1_readvariableop_resource:	 9
*wdwulgrltk_biasadd_readvariableop_resource:	0
"wdwulgrltk_readvariableop_resource: 2
$wdwulgrltk_readvariableop_1_resource: 2
$wdwulgrltk_readvariableop_2_resource: 
identity¢!wdwulgrltk/BiasAdd/ReadVariableOp¢ wdwulgrltk/MatMul/ReadVariableOp¢"wdwulgrltk/MatMul_1/ReadVariableOp¢wdwulgrltk/ReadVariableOp¢wdwulgrltk/ReadVariableOp_1¢wdwulgrltk/ReadVariableOp_2¢whileD
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
 wdwulgrltk/MatMul/ReadVariableOpReadVariableOp)wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 wdwulgrltk/MatMul/ReadVariableOp§
wdwulgrltk/MatMulMatMulstrided_slice_2:output:0(wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMulµ
"wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp+wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"wdwulgrltk/MatMul_1/ReadVariableOp£
wdwulgrltk/MatMul_1MatMulzeros:output:0*wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMul_1
wdwulgrltk/addAddV2wdwulgrltk/MatMul:product:0wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/add®
!wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp*wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!wdwulgrltk/BiasAdd/ReadVariableOp¥
wdwulgrltk/BiasAddBiasAddwdwulgrltk/add:z:0)wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/BiasAddz
wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
wdwulgrltk/split/split_dimë
wdwulgrltk/splitSplit#wdwulgrltk/split/split_dim:output:0wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
wdwulgrltk/split
wdwulgrltk/ReadVariableOpReadVariableOp"wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp
wdwulgrltk/mulMul!wdwulgrltk/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul
wdwulgrltk/add_1AddV2wdwulgrltk/split:output:0wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_1{
wdwulgrltk/SigmoidSigmoidwdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid
wdwulgrltk/ReadVariableOp_1ReadVariableOp$wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_1
wdwulgrltk/mul_1Mul#wdwulgrltk/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_1
wdwulgrltk/add_2AddV2wdwulgrltk/split:output:1wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_2
wdwulgrltk/Sigmoid_1Sigmoidwdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_1
wdwulgrltk/mul_2Mulwdwulgrltk/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_2w
wdwulgrltk/TanhTanhwdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh
wdwulgrltk/mul_3Mulwdwulgrltk/Sigmoid:y:0wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_3
wdwulgrltk/add_3AddV2wdwulgrltk/mul_2:z:0wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_3
wdwulgrltk/ReadVariableOp_2ReadVariableOp$wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_2
wdwulgrltk/mul_4Mul#wdwulgrltk/ReadVariableOp_2:value:0wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_4
wdwulgrltk/add_4AddV2wdwulgrltk/split:output:3wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_4
wdwulgrltk/Sigmoid_2Sigmoidwdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_2v
wdwulgrltk/Tanh_1Tanhwdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh_1
wdwulgrltk/mul_5Mulwdwulgrltk/Sigmoid_2:y:0wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)wdwulgrltk_matmul_readvariableop_resource+wdwulgrltk_matmul_1_readvariableop_resource*wdwulgrltk_biasadd_readvariableop_resource"wdwulgrltk_readvariableop_resource$wdwulgrltk_readvariableop_1_resource$wdwulgrltk_readvariableop_2_resource*
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
while_body_2162426*
condR
while_cond_2162425*Q
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
IdentityIdentitystrided_slice_3:output:0"^wdwulgrltk/BiasAdd/ReadVariableOp!^wdwulgrltk/MatMul/ReadVariableOp#^wdwulgrltk/MatMul_1/ReadVariableOp^wdwulgrltk/ReadVariableOp^wdwulgrltk/ReadVariableOp_1^wdwulgrltk/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!wdwulgrltk/BiasAdd/ReadVariableOp!wdwulgrltk/BiasAdd/ReadVariableOp2D
 wdwulgrltk/MatMul/ReadVariableOp wdwulgrltk/MatMul/ReadVariableOp2H
"wdwulgrltk/MatMul_1/ReadVariableOp"wdwulgrltk/MatMul_1/ReadVariableOp26
wdwulgrltk/ReadVariableOpwdwulgrltk/ReadVariableOp2:
wdwulgrltk/ReadVariableOp_1wdwulgrltk/ReadVariableOp_12:
wdwulgrltk/ReadVariableOp_2wdwulgrltk/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


í
while_cond_2161817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2161817___redundant_placeholder05
1while_while_cond_2161817___redundant_placeholder15
1while_while_cond_2161817___redundant_placeholder25
1while_while_cond_2161817___redundant_placeholder35
1while_while_cond_2161817___redundant_placeholder45
1while_while_cond_2161817___redundant_placeholder55
1while_while_cond_2161817___redundant_placeholder6
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
¯F
ê
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2158446

inputs%
wdwulgrltk_2158347:	 %
wdwulgrltk_2158349:	 !
wdwulgrltk_2158351:	 
wdwulgrltk_2158353:  
wdwulgrltk_2158355:  
wdwulgrltk_2158357: 
identity¢"wdwulgrltk/StatefulPartitionedCall¢whileD
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
"wdwulgrltk/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0wdwulgrltk_2158347wdwulgrltk_2158349wdwulgrltk_2158351wdwulgrltk_2158353wdwulgrltk_2158355wdwulgrltk_2158357*
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_21583462$
"wdwulgrltk/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0wdwulgrltk_2158347wdwulgrltk_2158349wdwulgrltk_2158351wdwulgrltk_2158353wdwulgrltk_2158355wdwulgrltk_2158357*
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
while_body_2158366*
condR
while_cond_2158365*Q
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
IdentityIdentitystrided_slice_3:output:0#^wdwulgrltk/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"wdwulgrltk/StatefulPartitionedCall"wdwulgrltk/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±'
³
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2157775

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
³
×
"__inference__wrapped_model_2157501

aveeivcxurW
Asequential_vfwtupxpzf_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_vfwtupxpzf_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_vhacowjcza_jczmzyhsca_matmul_readvariableop_resource:	T
Asequential_vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource:	 O
@sequential_vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource:	F
8sequential_vhacowjcza_jczmzyhsca_readvariableop_resource: H
:sequential_vhacowjcza_jczmzyhsca_readvariableop_1_resource: H
:sequential_vhacowjcza_jczmzyhsca_readvariableop_2_resource: R
?sequential_zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource:	 T
Asequential_zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource:	 O
@sequential_zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource:	F
8sequential_zgrdiwrovx_wdwulgrltk_readvariableop_resource: H
:sequential_zgrdiwrovx_wdwulgrltk_readvariableop_1_resource: H
:sequential_zgrdiwrovx_wdwulgrltk_readvariableop_2_resource: F
4sequential_kekwghyimt_matmul_readvariableop_resource: C
5sequential_kekwghyimt_biasadd_readvariableop_resource:
identity¢,sequential/kekwghyimt/BiasAdd/ReadVariableOp¢+sequential/kekwghyimt/MatMul/ReadVariableOp¢8sequential/vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7sequential/vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp¢6sequential/vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp¢8sequential/vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp¢/sequential/vhacowjcza/jczmzyhsca/ReadVariableOp¢1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_1¢1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_2¢sequential/vhacowjcza/while¢7sequential/zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp¢6sequential/zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp¢8sequential/zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp¢/sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp¢1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_1¢1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_2¢sequential/zgrdiwrovx/while¥
+sequential/vfwtupxpzf/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/vfwtupxpzf/conv1d/ExpandDims/dimà
'sequential/vfwtupxpzf/conv1d/ExpandDims
ExpandDims
aveeivcxur4sequential/vfwtupxpzf/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/vfwtupxpzf/conv1d/ExpandDimsú
8sequential/vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_vfwtupxpzf_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/vfwtupxpzf/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/vfwtupxpzf/conv1d/ExpandDims_1/dim
)sequential/vfwtupxpzf/conv1d/ExpandDims_1
ExpandDims@sequential/vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/vfwtupxpzf/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/vfwtupxpzf/conv1d/ExpandDims_1¨
"sequential/vfwtupxpzf/conv1d/ShapeShape0sequential/vfwtupxpzf/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/vfwtupxpzf/conv1d/Shape®
0sequential/vfwtupxpzf/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/vfwtupxpzf/conv1d/strided_slice/stack»
2sequential/vfwtupxpzf/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/vfwtupxpzf/conv1d/strided_slice/stack_1²
2sequential/vfwtupxpzf/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/vfwtupxpzf/conv1d/strided_slice/stack_2
*sequential/vfwtupxpzf/conv1d/strided_sliceStridedSlice+sequential/vfwtupxpzf/conv1d/Shape:output:09sequential/vfwtupxpzf/conv1d/strided_slice/stack:output:0;sequential/vfwtupxpzf/conv1d/strided_slice/stack_1:output:0;sequential/vfwtupxpzf/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/vfwtupxpzf/conv1d/strided_slice±
*sequential/vfwtupxpzf/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/vfwtupxpzf/conv1d/Reshape/shapeø
$sequential/vfwtupxpzf/conv1d/ReshapeReshape0sequential/vfwtupxpzf/conv1d/ExpandDims:output:03sequential/vfwtupxpzf/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/vfwtupxpzf/conv1d/Reshape
#sequential/vfwtupxpzf/conv1d/Conv2DConv2D-sequential/vfwtupxpzf/conv1d/Reshape:output:02sequential/vfwtupxpzf/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/vfwtupxpzf/conv1d/Conv2D±
,sequential/vfwtupxpzf/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/vfwtupxpzf/conv1d/concat/values_1
(sequential/vfwtupxpzf/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/vfwtupxpzf/conv1d/concat/axis£
#sequential/vfwtupxpzf/conv1d/concatConcatV23sequential/vfwtupxpzf/conv1d/strided_slice:output:05sequential/vfwtupxpzf/conv1d/concat/values_1:output:01sequential/vfwtupxpzf/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/vfwtupxpzf/conv1d/concatõ
&sequential/vfwtupxpzf/conv1d/Reshape_1Reshape,sequential/vfwtupxpzf/conv1d/Conv2D:output:0,sequential/vfwtupxpzf/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/vfwtupxpzf/conv1d/Reshape_1â
$sequential/vfwtupxpzf/conv1d/SqueezeSqueeze/sequential/vfwtupxpzf/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/vfwtupxpzf/conv1d/Squeeze½
.sequential/vfwtupxpzf/squeeze_batch_dims/ShapeShape-sequential/vfwtupxpzf/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/vfwtupxpzf/squeeze_batch_dims/ShapeÆ
<sequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stackÓ
>sequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/vfwtupxpzf/squeeze_batch_dims/strided_sliceStridedSlice7sequential/vfwtupxpzf/squeeze_batch_dims/Shape:output:0Esequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/vfwtupxpzf/squeeze_batch_dims/strided_sliceÅ
6sequential/vfwtupxpzf/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/vfwtupxpzf/squeeze_batch_dims/Reshape/shape
0sequential/vfwtupxpzf/squeeze_batch_dims/ReshapeReshape-sequential/vfwtupxpzf/conv1d/Squeeze:output:0?sequential/vfwtupxpzf/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/vfwtupxpzf/squeeze_batch_dims/Reshape
?sequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_vfwtupxpzf_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/vfwtupxpzf/squeeze_batch_dims/BiasAddBiasAdd9sequential/vfwtupxpzf/squeeze_batch_dims/Reshape:output:0Gsequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/vfwtupxpzf/squeeze_batch_dims/BiasAddÅ
8sequential/vfwtupxpzf/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/vfwtupxpzf/squeeze_batch_dims/concat/values_1·
4sequential/vfwtupxpzf/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/vfwtupxpzf/squeeze_batch_dims/concat/axisß
/sequential/vfwtupxpzf/squeeze_batch_dims/concatConcatV2?sequential/vfwtupxpzf/squeeze_batch_dims/strided_slice:output:0Asequential/vfwtupxpzf/squeeze_batch_dims/concat/values_1:output:0=sequential/vfwtupxpzf/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/vfwtupxpzf/squeeze_batch_dims/concat¢
2sequential/vfwtupxpzf/squeeze_batch_dims/Reshape_1Reshape9sequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd:output:08sequential/vfwtupxpzf/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/vfwtupxpzf/squeeze_batch_dims/Reshape_1¥
sequential/ojzbgzevue/ShapeShape;sequential/vfwtupxpzf/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/ojzbgzevue/Shape 
)sequential/ojzbgzevue/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/ojzbgzevue/strided_slice/stack¤
+sequential/ojzbgzevue/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ojzbgzevue/strided_slice/stack_1¤
+sequential/ojzbgzevue/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ojzbgzevue/strided_slice/stack_2æ
#sequential/ojzbgzevue/strided_sliceStridedSlice$sequential/ojzbgzevue/Shape:output:02sequential/ojzbgzevue/strided_slice/stack:output:04sequential/ojzbgzevue/strided_slice/stack_1:output:04sequential/ojzbgzevue/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/ojzbgzevue/strided_slice
%sequential/ojzbgzevue/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ojzbgzevue/Reshape/shape/1
%sequential/ojzbgzevue/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ojzbgzevue/Reshape/shape/2
#sequential/ojzbgzevue/Reshape/shapePack,sequential/ojzbgzevue/strided_slice:output:0.sequential/ojzbgzevue/Reshape/shape/1:output:0.sequential/ojzbgzevue/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/ojzbgzevue/Reshape/shapeê
sequential/ojzbgzevue/ReshapeReshape;sequential/vfwtupxpzf/squeeze_batch_dims/Reshape_1:output:0,sequential/ojzbgzevue/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/ojzbgzevue/Reshape
sequential/vhacowjcza/ShapeShape&sequential/ojzbgzevue/Reshape:output:0*
T0*
_output_shapes
:2
sequential/vhacowjcza/Shape 
)sequential/vhacowjcza/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/vhacowjcza/strided_slice/stack¤
+sequential/vhacowjcza/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/vhacowjcza/strided_slice/stack_1¤
+sequential/vhacowjcza/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/vhacowjcza/strided_slice/stack_2æ
#sequential/vhacowjcza/strided_sliceStridedSlice$sequential/vhacowjcza/Shape:output:02sequential/vhacowjcza/strided_slice/stack:output:04sequential/vhacowjcza/strided_slice/stack_1:output:04sequential/vhacowjcza/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/vhacowjcza/strided_slice
!sequential/vhacowjcza/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/vhacowjcza/zeros/mul/yÄ
sequential/vhacowjcza/zeros/mulMul,sequential/vhacowjcza/strided_slice:output:0*sequential/vhacowjcza/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/vhacowjcza/zeros/mul
"sequential/vhacowjcza/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/vhacowjcza/zeros/Less/y¿
 sequential/vhacowjcza/zeros/LessLess#sequential/vhacowjcza/zeros/mul:z:0+sequential/vhacowjcza/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/vhacowjcza/zeros/Less
$sequential/vhacowjcza/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/vhacowjcza/zeros/packed/1Û
"sequential/vhacowjcza/zeros/packedPack,sequential/vhacowjcza/strided_slice:output:0-sequential/vhacowjcza/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/vhacowjcza/zeros/packed
!sequential/vhacowjcza/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/vhacowjcza/zeros/ConstÍ
sequential/vhacowjcza/zerosFill+sequential/vhacowjcza/zeros/packed:output:0*sequential/vhacowjcza/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/vhacowjcza/zeros
#sequential/vhacowjcza/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/vhacowjcza/zeros_1/mul/yÊ
!sequential/vhacowjcza/zeros_1/mulMul,sequential/vhacowjcza/strided_slice:output:0,sequential/vhacowjcza/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/vhacowjcza/zeros_1/mul
$sequential/vhacowjcza/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/vhacowjcza/zeros_1/Less/yÇ
"sequential/vhacowjcza/zeros_1/LessLess%sequential/vhacowjcza/zeros_1/mul:z:0-sequential/vhacowjcza/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/vhacowjcza/zeros_1/Less
&sequential/vhacowjcza/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/vhacowjcza/zeros_1/packed/1á
$sequential/vhacowjcza/zeros_1/packedPack,sequential/vhacowjcza/strided_slice:output:0/sequential/vhacowjcza/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/vhacowjcza/zeros_1/packed
#sequential/vhacowjcza/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/vhacowjcza/zeros_1/ConstÕ
sequential/vhacowjcza/zeros_1Fill-sequential/vhacowjcza/zeros_1/packed:output:0,sequential/vhacowjcza/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/vhacowjcza/zeros_1¡
$sequential/vhacowjcza/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/vhacowjcza/transpose/permÜ
sequential/vhacowjcza/transpose	Transpose&sequential/ojzbgzevue/Reshape:output:0-sequential/vhacowjcza/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/vhacowjcza/transpose
sequential/vhacowjcza/Shape_1Shape#sequential/vhacowjcza/transpose:y:0*
T0*
_output_shapes
:2
sequential/vhacowjcza/Shape_1¤
+sequential/vhacowjcza/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/vhacowjcza/strided_slice_1/stack¨
-sequential/vhacowjcza/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vhacowjcza/strided_slice_1/stack_1¨
-sequential/vhacowjcza/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vhacowjcza/strided_slice_1/stack_2ò
%sequential/vhacowjcza/strided_slice_1StridedSlice&sequential/vhacowjcza/Shape_1:output:04sequential/vhacowjcza/strided_slice_1/stack:output:06sequential/vhacowjcza/strided_slice_1/stack_1:output:06sequential/vhacowjcza/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/vhacowjcza/strided_slice_1±
1sequential/vhacowjcza/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/vhacowjcza/TensorArrayV2/element_shape
#sequential/vhacowjcza/TensorArrayV2TensorListReserve:sequential/vhacowjcza/TensorArrayV2/element_shape:output:0.sequential/vhacowjcza/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/vhacowjcza/TensorArrayV2ë
Ksequential/vhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/vhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/vhacowjcza/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/vhacowjcza/transpose:y:0Tsequential/vhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/vhacowjcza/TensorArrayUnstack/TensorListFromTensor¤
+sequential/vhacowjcza/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/vhacowjcza/strided_slice_2/stack¨
-sequential/vhacowjcza/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vhacowjcza/strided_slice_2/stack_1¨
-sequential/vhacowjcza/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vhacowjcza/strided_slice_2/stack_2
%sequential/vhacowjcza/strided_slice_2StridedSlice#sequential/vhacowjcza/transpose:y:04sequential/vhacowjcza/strided_slice_2/stack:output:06sequential/vhacowjcza/strided_slice_2/stack_1:output:06sequential/vhacowjcza/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/vhacowjcza/strided_slice_2ñ
6sequential/vhacowjcza/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp?sequential_vhacowjcza_jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/vhacowjcza/jczmzyhsca/MatMul/ReadVariableOpÿ
'sequential/vhacowjcza/jczmzyhsca/MatMulMatMul.sequential/vhacowjcza/strided_slice_2:output:0>sequential/vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/vhacowjcza/jczmzyhsca/MatMul÷
8sequential/vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOpAsequential_vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOpû
)sequential/vhacowjcza/jczmzyhsca/MatMul_1MatMul$sequential/vhacowjcza/zeros:output:0@sequential/vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/vhacowjcza/jczmzyhsca/MatMul_1ð
$sequential/vhacowjcza/jczmzyhsca/addAddV21sequential/vhacowjcza/jczmzyhsca/MatMul:product:03sequential/vhacowjcza/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/vhacowjcza/jczmzyhsca/addð
7sequential/vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp@sequential_vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOpý
(sequential/vhacowjcza/jczmzyhsca/BiasAddBiasAdd(sequential/vhacowjcza/jczmzyhsca/add:z:0?sequential/vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/vhacowjcza/jczmzyhsca/BiasAdd¦
0sequential/vhacowjcza/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/vhacowjcza/jczmzyhsca/split/split_dimÃ
&sequential/vhacowjcza/jczmzyhsca/splitSplit9sequential/vhacowjcza/jczmzyhsca/split/split_dim:output:01sequential/vhacowjcza/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/vhacowjcza/jczmzyhsca/split×
/sequential/vhacowjcza/jczmzyhsca/ReadVariableOpReadVariableOp8sequential_vhacowjcza_jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/vhacowjcza/jczmzyhsca/ReadVariableOpæ
$sequential/vhacowjcza/jczmzyhsca/mulMul7sequential/vhacowjcza/jczmzyhsca/ReadVariableOp:value:0&sequential/vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/vhacowjcza/jczmzyhsca/mulæ
&sequential/vhacowjcza/jczmzyhsca/add_1AddV2/sequential/vhacowjcza/jczmzyhsca/split:output:0(sequential/vhacowjcza/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/add_1½
(sequential/vhacowjcza/jczmzyhsca/SigmoidSigmoid*sequential/vhacowjcza/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/vhacowjcza/jczmzyhsca/SigmoidÝ
1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_1ReadVariableOp:sequential_vhacowjcza_jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_1ì
&sequential/vhacowjcza/jczmzyhsca/mul_1Mul9sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_1:value:0&sequential/vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/mul_1è
&sequential/vhacowjcza/jczmzyhsca/add_2AddV2/sequential/vhacowjcza/jczmzyhsca/split:output:1*sequential/vhacowjcza/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/add_2Á
*sequential/vhacowjcza/jczmzyhsca/Sigmoid_1Sigmoid*sequential/vhacowjcza/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/vhacowjcza/jczmzyhsca/Sigmoid_1á
&sequential/vhacowjcza/jczmzyhsca/mul_2Mul.sequential/vhacowjcza/jczmzyhsca/Sigmoid_1:y:0&sequential/vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/mul_2¹
%sequential/vhacowjcza/jczmzyhsca/TanhTanh/sequential/vhacowjcza/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/vhacowjcza/jczmzyhsca/Tanhâ
&sequential/vhacowjcza/jczmzyhsca/mul_3Mul,sequential/vhacowjcza/jczmzyhsca/Sigmoid:y:0)sequential/vhacowjcza/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/mul_3ã
&sequential/vhacowjcza/jczmzyhsca/add_3AddV2*sequential/vhacowjcza/jczmzyhsca/mul_2:z:0*sequential/vhacowjcza/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/add_3Ý
1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_2ReadVariableOp:sequential_vhacowjcza_jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_2ð
&sequential/vhacowjcza/jczmzyhsca/mul_4Mul9sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_2:value:0*sequential/vhacowjcza/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/mul_4è
&sequential/vhacowjcza/jczmzyhsca/add_4AddV2/sequential/vhacowjcza/jczmzyhsca/split:output:3*sequential/vhacowjcza/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/add_4Á
*sequential/vhacowjcza/jczmzyhsca/Sigmoid_2Sigmoid*sequential/vhacowjcza/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/vhacowjcza/jczmzyhsca/Sigmoid_2¸
'sequential/vhacowjcza/jczmzyhsca/Tanh_1Tanh*sequential/vhacowjcza/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/vhacowjcza/jczmzyhsca/Tanh_1æ
&sequential/vhacowjcza/jczmzyhsca/mul_5Mul.sequential/vhacowjcza/jczmzyhsca/Sigmoid_2:y:0+sequential/vhacowjcza/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/jczmzyhsca/mul_5»
3sequential/vhacowjcza/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/vhacowjcza/TensorArrayV2_1/element_shape
%sequential/vhacowjcza/TensorArrayV2_1TensorListReserve<sequential/vhacowjcza/TensorArrayV2_1/element_shape:output:0.sequential/vhacowjcza/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/vhacowjcza/TensorArrayV2_1z
sequential/vhacowjcza/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/vhacowjcza/time«
.sequential/vhacowjcza/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/vhacowjcza/while/maximum_iterations
(sequential/vhacowjcza/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/vhacowjcza/while/loop_counterø	
sequential/vhacowjcza/whileWhile1sequential/vhacowjcza/while/loop_counter:output:07sequential/vhacowjcza/while/maximum_iterations:output:0#sequential/vhacowjcza/time:output:0.sequential/vhacowjcza/TensorArrayV2_1:handle:0$sequential/vhacowjcza/zeros:output:0&sequential/vhacowjcza/zeros_1:output:0.sequential/vhacowjcza/strided_slice_1:output:0Msequential/vhacowjcza/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_vhacowjcza_jczmzyhsca_matmul_readvariableop_resourceAsequential_vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource@sequential_vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource8sequential_vhacowjcza_jczmzyhsca_readvariableop_resource:sequential_vhacowjcza_jczmzyhsca_readvariableop_1_resource:sequential_vhacowjcza_jczmzyhsca_readvariableop_2_resource*
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
(sequential_vhacowjcza_while_body_2157218*4
cond,R*
(sequential_vhacowjcza_while_cond_2157217*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/vhacowjcza/whileá
Fsequential/vhacowjcza/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/vhacowjcza/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/vhacowjcza/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/vhacowjcza/while:output:3Osequential/vhacowjcza/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/vhacowjcza/TensorArrayV2Stack/TensorListStack­
+sequential/vhacowjcza/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/vhacowjcza/strided_slice_3/stack¨
-sequential/vhacowjcza/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/vhacowjcza/strided_slice_3/stack_1¨
-sequential/vhacowjcza/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/vhacowjcza/strided_slice_3/stack_2
%sequential/vhacowjcza/strided_slice_3StridedSliceAsequential/vhacowjcza/TensorArrayV2Stack/TensorListStack:tensor:04sequential/vhacowjcza/strided_slice_3/stack:output:06sequential/vhacowjcza/strided_slice_3/stack_1:output:06sequential/vhacowjcza/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/vhacowjcza/strided_slice_3¥
&sequential/vhacowjcza/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/vhacowjcza/transpose_1/permý
!sequential/vhacowjcza/transpose_1	TransposeAsequential/vhacowjcza/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/vhacowjcza/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/vhacowjcza/transpose_1
sequential/zgrdiwrovx/ShapeShape%sequential/vhacowjcza/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/zgrdiwrovx/Shape 
)sequential/zgrdiwrovx/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/zgrdiwrovx/strided_slice/stack¤
+sequential/zgrdiwrovx/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/zgrdiwrovx/strided_slice/stack_1¤
+sequential/zgrdiwrovx/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/zgrdiwrovx/strided_slice/stack_2æ
#sequential/zgrdiwrovx/strided_sliceStridedSlice$sequential/zgrdiwrovx/Shape:output:02sequential/zgrdiwrovx/strided_slice/stack:output:04sequential/zgrdiwrovx/strided_slice/stack_1:output:04sequential/zgrdiwrovx/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/zgrdiwrovx/strided_slice
!sequential/zgrdiwrovx/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/zgrdiwrovx/zeros/mul/yÄ
sequential/zgrdiwrovx/zeros/mulMul,sequential/zgrdiwrovx/strided_slice:output:0*sequential/zgrdiwrovx/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/zgrdiwrovx/zeros/mul
"sequential/zgrdiwrovx/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/zgrdiwrovx/zeros/Less/y¿
 sequential/zgrdiwrovx/zeros/LessLess#sequential/zgrdiwrovx/zeros/mul:z:0+sequential/zgrdiwrovx/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/zgrdiwrovx/zeros/Less
$sequential/zgrdiwrovx/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/zgrdiwrovx/zeros/packed/1Û
"sequential/zgrdiwrovx/zeros/packedPack,sequential/zgrdiwrovx/strided_slice:output:0-sequential/zgrdiwrovx/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/zgrdiwrovx/zeros/packed
!sequential/zgrdiwrovx/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/zgrdiwrovx/zeros/ConstÍ
sequential/zgrdiwrovx/zerosFill+sequential/zgrdiwrovx/zeros/packed:output:0*sequential/zgrdiwrovx/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/zgrdiwrovx/zeros
#sequential/zgrdiwrovx/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/zgrdiwrovx/zeros_1/mul/yÊ
!sequential/zgrdiwrovx/zeros_1/mulMul,sequential/zgrdiwrovx/strided_slice:output:0,sequential/zgrdiwrovx/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/zgrdiwrovx/zeros_1/mul
$sequential/zgrdiwrovx/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/zgrdiwrovx/zeros_1/Less/yÇ
"sequential/zgrdiwrovx/zeros_1/LessLess%sequential/zgrdiwrovx/zeros_1/mul:z:0-sequential/zgrdiwrovx/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/zgrdiwrovx/zeros_1/Less
&sequential/zgrdiwrovx/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/zgrdiwrovx/zeros_1/packed/1á
$sequential/zgrdiwrovx/zeros_1/packedPack,sequential/zgrdiwrovx/strided_slice:output:0/sequential/zgrdiwrovx/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/zgrdiwrovx/zeros_1/packed
#sequential/zgrdiwrovx/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/zgrdiwrovx/zeros_1/ConstÕ
sequential/zgrdiwrovx/zeros_1Fill-sequential/zgrdiwrovx/zeros_1/packed:output:0,sequential/zgrdiwrovx/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/zgrdiwrovx/zeros_1¡
$sequential/zgrdiwrovx/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/zgrdiwrovx/transpose/permÛ
sequential/zgrdiwrovx/transpose	Transpose%sequential/vhacowjcza/transpose_1:y:0-sequential/zgrdiwrovx/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/zgrdiwrovx/transpose
sequential/zgrdiwrovx/Shape_1Shape#sequential/zgrdiwrovx/transpose:y:0*
T0*
_output_shapes
:2
sequential/zgrdiwrovx/Shape_1¤
+sequential/zgrdiwrovx/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/zgrdiwrovx/strided_slice_1/stack¨
-sequential/zgrdiwrovx/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zgrdiwrovx/strided_slice_1/stack_1¨
-sequential/zgrdiwrovx/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zgrdiwrovx/strided_slice_1/stack_2ò
%sequential/zgrdiwrovx/strided_slice_1StridedSlice&sequential/zgrdiwrovx/Shape_1:output:04sequential/zgrdiwrovx/strided_slice_1/stack:output:06sequential/zgrdiwrovx/strided_slice_1/stack_1:output:06sequential/zgrdiwrovx/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/zgrdiwrovx/strided_slice_1±
1sequential/zgrdiwrovx/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/zgrdiwrovx/TensorArrayV2/element_shape
#sequential/zgrdiwrovx/TensorArrayV2TensorListReserve:sequential/zgrdiwrovx/TensorArrayV2/element_shape:output:0.sequential/zgrdiwrovx/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/zgrdiwrovx/TensorArrayV2ë
Ksequential/zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/zgrdiwrovx/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/zgrdiwrovx/transpose:y:0Tsequential/zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor¤
+sequential/zgrdiwrovx/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/zgrdiwrovx/strided_slice_2/stack¨
-sequential/zgrdiwrovx/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zgrdiwrovx/strided_slice_2/stack_1¨
-sequential/zgrdiwrovx/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zgrdiwrovx/strided_slice_2/stack_2
%sequential/zgrdiwrovx/strided_slice_2StridedSlice#sequential/zgrdiwrovx/transpose:y:04sequential/zgrdiwrovx/strided_slice_2/stack:output:06sequential/zgrdiwrovx/strided_slice_2/stack_1:output:06sequential/zgrdiwrovx/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/zgrdiwrovx/strided_slice_2ñ
6sequential/zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp?sequential_zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOpÿ
'sequential/zgrdiwrovx/wdwulgrltk/MatMulMatMul.sequential/zgrdiwrovx/strided_slice_2:output:0>sequential/zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/zgrdiwrovx/wdwulgrltk/MatMul÷
8sequential/zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOpAsequential_zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOpû
)sequential/zgrdiwrovx/wdwulgrltk/MatMul_1MatMul$sequential/zgrdiwrovx/zeros:output:0@sequential/zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/zgrdiwrovx/wdwulgrltk/MatMul_1ð
$sequential/zgrdiwrovx/wdwulgrltk/addAddV21sequential/zgrdiwrovx/wdwulgrltk/MatMul:product:03sequential/zgrdiwrovx/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/zgrdiwrovx/wdwulgrltk/addð
7sequential/zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp@sequential_zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOpý
(sequential/zgrdiwrovx/wdwulgrltk/BiasAddBiasAdd(sequential/zgrdiwrovx/wdwulgrltk/add:z:0?sequential/zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/zgrdiwrovx/wdwulgrltk/BiasAdd¦
0sequential/zgrdiwrovx/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/zgrdiwrovx/wdwulgrltk/split/split_dimÃ
&sequential/zgrdiwrovx/wdwulgrltk/splitSplit9sequential/zgrdiwrovx/wdwulgrltk/split/split_dim:output:01sequential/zgrdiwrovx/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/zgrdiwrovx/wdwulgrltk/split×
/sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOpReadVariableOp8sequential_zgrdiwrovx_wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOpæ
$sequential/zgrdiwrovx/wdwulgrltk/mulMul7sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp:value:0&sequential/zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/zgrdiwrovx/wdwulgrltk/mulæ
&sequential/zgrdiwrovx/wdwulgrltk/add_1AddV2/sequential/zgrdiwrovx/wdwulgrltk/split:output:0(sequential/zgrdiwrovx/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/add_1½
(sequential/zgrdiwrovx/wdwulgrltk/SigmoidSigmoid*sequential/zgrdiwrovx/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/zgrdiwrovx/wdwulgrltk/SigmoidÝ
1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_1ReadVariableOp:sequential_zgrdiwrovx_wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_1ì
&sequential/zgrdiwrovx/wdwulgrltk/mul_1Mul9sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_1:value:0&sequential/zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/mul_1è
&sequential/zgrdiwrovx/wdwulgrltk/add_2AddV2/sequential/zgrdiwrovx/wdwulgrltk/split:output:1*sequential/zgrdiwrovx/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/add_2Á
*sequential/zgrdiwrovx/wdwulgrltk/Sigmoid_1Sigmoid*sequential/zgrdiwrovx/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/zgrdiwrovx/wdwulgrltk/Sigmoid_1á
&sequential/zgrdiwrovx/wdwulgrltk/mul_2Mul.sequential/zgrdiwrovx/wdwulgrltk/Sigmoid_1:y:0&sequential/zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/mul_2¹
%sequential/zgrdiwrovx/wdwulgrltk/TanhTanh/sequential/zgrdiwrovx/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/zgrdiwrovx/wdwulgrltk/Tanhâ
&sequential/zgrdiwrovx/wdwulgrltk/mul_3Mul,sequential/zgrdiwrovx/wdwulgrltk/Sigmoid:y:0)sequential/zgrdiwrovx/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/mul_3ã
&sequential/zgrdiwrovx/wdwulgrltk/add_3AddV2*sequential/zgrdiwrovx/wdwulgrltk/mul_2:z:0*sequential/zgrdiwrovx/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/add_3Ý
1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_2ReadVariableOp:sequential_zgrdiwrovx_wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_2ð
&sequential/zgrdiwrovx/wdwulgrltk/mul_4Mul9sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_2:value:0*sequential/zgrdiwrovx/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/mul_4è
&sequential/zgrdiwrovx/wdwulgrltk/add_4AddV2/sequential/zgrdiwrovx/wdwulgrltk/split:output:3*sequential/zgrdiwrovx/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/add_4Á
*sequential/zgrdiwrovx/wdwulgrltk/Sigmoid_2Sigmoid*sequential/zgrdiwrovx/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/zgrdiwrovx/wdwulgrltk/Sigmoid_2¸
'sequential/zgrdiwrovx/wdwulgrltk/Tanh_1Tanh*sequential/zgrdiwrovx/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/zgrdiwrovx/wdwulgrltk/Tanh_1æ
&sequential/zgrdiwrovx/wdwulgrltk/mul_5Mul.sequential/zgrdiwrovx/wdwulgrltk/Sigmoid_2:y:0+sequential/zgrdiwrovx/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/wdwulgrltk/mul_5»
3sequential/zgrdiwrovx/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/zgrdiwrovx/TensorArrayV2_1/element_shape
%sequential/zgrdiwrovx/TensorArrayV2_1TensorListReserve<sequential/zgrdiwrovx/TensorArrayV2_1/element_shape:output:0.sequential/zgrdiwrovx/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/zgrdiwrovx/TensorArrayV2_1z
sequential/zgrdiwrovx/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/zgrdiwrovx/time«
.sequential/zgrdiwrovx/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/zgrdiwrovx/while/maximum_iterations
(sequential/zgrdiwrovx/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/zgrdiwrovx/while/loop_counterø	
sequential/zgrdiwrovx/whileWhile1sequential/zgrdiwrovx/while/loop_counter:output:07sequential/zgrdiwrovx/while/maximum_iterations:output:0#sequential/zgrdiwrovx/time:output:0.sequential/zgrdiwrovx/TensorArrayV2_1:handle:0$sequential/zgrdiwrovx/zeros:output:0&sequential/zgrdiwrovx/zeros_1:output:0.sequential/zgrdiwrovx/strided_slice_1:output:0Msequential/zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resourceAsequential_zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource@sequential_zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource8sequential_zgrdiwrovx_wdwulgrltk_readvariableop_resource:sequential_zgrdiwrovx_wdwulgrltk_readvariableop_1_resource:sequential_zgrdiwrovx_wdwulgrltk_readvariableop_2_resource*
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
(sequential_zgrdiwrovx_while_body_2157394*4
cond,R*
(sequential_zgrdiwrovx_while_cond_2157393*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/zgrdiwrovx/whileá
Fsequential/zgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/zgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/zgrdiwrovx/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/zgrdiwrovx/while:output:3Osequential/zgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/zgrdiwrovx/TensorArrayV2Stack/TensorListStack­
+sequential/zgrdiwrovx/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/zgrdiwrovx/strided_slice_3/stack¨
-sequential/zgrdiwrovx/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/zgrdiwrovx/strided_slice_3/stack_1¨
-sequential/zgrdiwrovx/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zgrdiwrovx/strided_slice_3/stack_2
%sequential/zgrdiwrovx/strided_slice_3StridedSliceAsequential/zgrdiwrovx/TensorArrayV2Stack/TensorListStack:tensor:04sequential/zgrdiwrovx/strided_slice_3/stack:output:06sequential/zgrdiwrovx/strided_slice_3/stack_1:output:06sequential/zgrdiwrovx/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/zgrdiwrovx/strided_slice_3¥
&sequential/zgrdiwrovx/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/zgrdiwrovx/transpose_1/permý
!sequential/zgrdiwrovx/transpose_1	TransposeAsequential/zgrdiwrovx/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/zgrdiwrovx/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/zgrdiwrovx/transpose_1Ï
+sequential/kekwghyimt/MatMul/ReadVariableOpReadVariableOp4sequential_kekwghyimt_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/kekwghyimt/MatMul/ReadVariableOpÝ
sequential/kekwghyimt/MatMulMatMul.sequential/zgrdiwrovx/strided_slice_3:output:03sequential/kekwghyimt/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/kekwghyimt/MatMulÎ
,sequential/kekwghyimt/BiasAdd/ReadVariableOpReadVariableOp5sequential_kekwghyimt_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/kekwghyimt/BiasAdd/ReadVariableOpÙ
sequential/kekwghyimt/BiasAddBiasAdd&sequential/kekwghyimt/MatMul:product:04sequential/kekwghyimt/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/kekwghyimt/BiasAdd 
IdentityIdentity&sequential/kekwghyimt/BiasAdd:output:0-^sequential/kekwghyimt/BiasAdd/ReadVariableOp,^sequential/kekwghyimt/MatMul/ReadVariableOp9^sequential/vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp@^sequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp8^sequential/vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp7^sequential/vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp9^sequential/vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp0^sequential/vhacowjcza/jczmzyhsca/ReadVariableOp2^sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_12^sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_2^sequential/vhacowjcza/while8^sequential/zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp7^sequential/zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp9^sequential/zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp0^sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp2^sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_12^sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_2^sequential/zgrdiwrovx/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,sequential/kekwghyimt/BiasAdd/ReadVariableOp,sequential/kekwghyimt/BiasAdd/ReadVariableOp2Z
+sequential/kekwghyimt/MatMul/ReadVariableOp+sequential/kekwghyimt/MatMul/ReadVariableOp2t
8sequential/vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp8sequential/vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7sequential/vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp7sequential/vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp2p
6sequential/vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp6sequential/vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp2t
8sequential/vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp8sequential/vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp2b
/sequential/vhacowjcza/jczmzyhsca/ReadVariableOp/sequential/vhacowjcza/jczmzyhsca/ReadVariableOp2f
1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_11sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_12f
1sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_21sequential/vhacowjcza/jczmzyhsca/ReadVariableOp_22:
sequential/vhacowjcza/whilesequential/vhacowjcza/while2r
7sequential/zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp7sequential/zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp2p
6sequential/zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp6sequential/zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp2t
8sequential/zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp8sequential/zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp2b
/sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp/sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp2f
1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_11sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_12f
1sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_21sequential/zgrdiwrovx/wdwulgrltk/ReadVariableOp_22:
sequential/zgrdiwrovx/whilesequential/zgrdiwrovx/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
aveeivcxur


í
while_cond_2159841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2159841___redundant_placeholder05
1while_while_cond_2159841___redundant_placeholder15
1while_while_cond_2159841___redundant_placeholder25
1while_while_cond_2159841___redundant_placeholder35
1while_while_cond_2159841___redundant_placeholder45
1while_while_cond_2159841___redundant_placeholder55
1while_while_cond_2159841___redundant_placeholder6
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161919

inputs<
)jczmzyhsca_matmul_readvariableop_resource:	>
+jczmzyhsca_matmul_1_readvariableop_resource:	 9
*jczmzyhsca_biasadd_readvariableop_resource:	0
"jczmzyhsca_readvariableop_resource: 2
$jczmzyhsca_readvariableop_1_resource: 2
$jczmzyhsca_readvariableop_2_resource: 
identity¢!jczmzyhsca/BiasAdd/ReadVariableOp¢ jczmzyhsca/MatMul/ReadVariableOp¢"jczmzyhsca/MatMul_1/ReadVariableOp¢jczmzyhsca/ReadVariableOp¢jczmzyhsca/ReadVariableOp_1¢jczmzyhsca/ReadVariableOp_2¢whileD
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
 jczmzyhsca/MatMul/ReadVariableOpReadVariableOp)jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jczmzyhsca/MatMul/ReadVariableOp§
jczmzyhsca/MatMulMatMulstrided_slice_2:output:0(jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMulµ
"jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp+jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jczmzyhsca/MatMul_1/ReadVariableOp£
jczmzyhsca/MatMul_1MatMulzeros:output:0*jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMul_1
jczmzyhsca/addAddV2jczmzyhsca/MatMul:product:0jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/add®
!jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp*jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jczmzyhsca/BiasAdd/ReadVariableOp¥
jczmzyhsca/BiasAddBiasAddjczmzyhsca/add:z:0)jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/BiasAddz
jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jczmzyhsca/split/split_dimë
jczmzyhsca/splitSplit#jczmzyhsca/split/split_dim:output:0jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jczmzyhsca/split
jczmzyhsca/ReadVariableOpReadVariableOp"jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp
jczmzyhsca/mulMul!jczmzyhsca/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul
jczmzyhsca/add_1AddV2jczmzyhsca/split:output:0jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_1{
jczmzyhsca/SigmoidSigmoidjczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid
jczmzyhsca/ReadVariableOp_1ReadVariableOp$jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_1
jczmzyhsca/mul_1Mul#jczmzyhsca/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_1
jczmzyhsca/add_2AddV2jczmzyhsca/split:output:1jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_2
jczmzyhsca/Sigmoid_1Sigmoidjczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_1
jczmzyhsca/mul_2Muljczmzyhsca/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_2w
jczmzyhsca/TanhTanhjczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh
jczmzyhsca/mul_3Muljczmzyhsca/Sigmoid:y:0jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_3
jczmzyhsca/add_3AddV2jczmzyhsca/mul_2:z:0jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_3
jczmzyhsca/ReadVariableOp_2ReadVariableOp$jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_2
jczmzyhsca/mul_4Mul#jczmzyhsca/ReadVariableOp_2:value:0jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_4
jczmzyhsca/add_4AddV2jczmzyhsca/split:output:3jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_4
jczmzyhsca/Sigmoid_2Sigmoidjczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_2v
jczmzyhsca/Tanh_1Tanhjczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh_1
jczmzyhsca/mul_5Muljczmzyhsca/Sigmoid_2:y:0jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jczmzyhsca_matmul_readvariableop_resource+jczmzyhsca_matmul_1_readvariableop_resource*jczmzyhsca_biasadd_readvariableop_resource"jczmzyhsca_readvariableop_resource$jczmzyhsca_readvariableop_1_resource$jczmzyhsca_readvariableop_2_resource*
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
while_body_2161818*
condR
while_cond_2161817*Q
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
IdentityIdentitytranspose_1:y:0"^jczmzyhsca/BiasAdd/ReadVariableOp!^jczmzyhsca/MatMul/ReadVariableOp#^jczmzyhsca/MatMul_1/ReadVariableOp^jczmzyhsca/ReadVariableOp^jczmzyhsca/ReadVariableOp_1^jczmzyhsca/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jczmzyhsca/BiasAdd/ReadVariableOp!jczmzyhsca/BiasAdd/ReadVariableOp2D
 jczmzyhsca/MatMul/ReadVariableOp jczmzyhsca/MatMul/ReadVariableOp2H
"jczmzyhsca/MatMul_1/ReadVariableOp"jczmzyhsca/MatMul_1/ReadVariableOp26
jczmzyhsca/ReadVariableOpjczmzyhsca/ReadVariableOp2:
jczmzyhsca/ReadVariableOp_1jczmzyhsca/ReadVariableOp_12:
jczmzyhsca/ReadVariableOp_2jczmzyhsca/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_vfwtupxpzf_layer_call_fn_2161181

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
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_21590612
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
¯

#__inference__traced_restore_2163329
file_prefix8
"assignvariableop_vfwtupxpzf_kernel:0
"assignvariableop_1_vfwtupxpzf_bias:6
$assignvariableop_2_kekwghyimt_kernel: 0
"assignvariableop_3_kekwghyimt_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_vhacowjcza_jczmzyhsca_kernel:	M
:assignvariableop_10_vhacowjcza_jczmzyhsca_recurrent_kernel:	 =
.assignvariableop_11_vhacowjcza_jczmzyhsca_bias:	S
Eassignvariableop_12_vhacowjcza_jczmzyhsca_input_gate_peephole_weights: T
Fassignvariableop_13_vhacowjcza_jczmzyhsca_forget_gate_peephole_weights: T
Fassignvariableop_14_vhacowjcza_jczmzyhsca_output_gate_peephole_weights: C
0assignvariableop_15_zgrdiwrovx_wdwulgrltk_kernel:	 M
:assignvariableop_16_zgrdiwrovx_wdwulgrltk_recurrent_kernel:	 =
.assignvariableop_17_zgrdiwrovx_wdwulgrltk_bias:	S
Eassignvariableop_18_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weights: T
Fassignvariableop_19_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weights: T
Fassignvariableop_20_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_vfwtupxpzf_kernel_rms:=
/assignvariableop_24_rmsprop_vfwtupxpzf_bias_rms:C
1assignvariableop_25_rmsprop_kekwghyimt_kernel_rms: =
/assignvariableop_26_rmsprop_kekwghyimt_bias_rms:O
<assignvariableop_27_rmsprop_vhacowjcza_jczmzyhsca_kernel_rms:	Y
Fassignvariableop_28_rmsprop_vhacowjcza_jczmzyhsca_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_vhacowjcza_jczmzyhsca_bias_rms:	_
Qassignvariableop_30_rmsprop_vhacowjcza_jczmzyhsca_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_vhacowjcza_jczmzyhsca_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_vhacowjcza_jczmzyhsca_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_zgrdiwrovx_wdwulgrltk_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_zgrdiwrovx_wdwulgrltk_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_zgrdiwrovx_wdwulgrltk_bias_rms:	_
Qassignvariableop_36_rmsprop_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_vfwtupxpzf_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_vfwtupxpzf_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_kekwghyimt_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_kekwghyimt_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_vhacowjcza_jczmzyhsca_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_vhacowjcza_jczmzyhsca_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_vhacowjcza_jczmzyhsca_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_vhacowjcza_jczmzyhsca_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_vhacowjcza_jczmzyhsca_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_vhacowjcza_jczmzyhsca_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_zgrdiwrovx_wdwulgrltk_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_zgrdiwrovx_wdwulgrltk_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_zgrdiwrovx_wdwulgrltk_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_vfwtupxpzf_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_vfwtupxpzf_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_kekwghyimt_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_kekwghyimt_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_vhacowjcza_jczmzyhsca_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_vhacowjcza_jczmzyhsca_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_vhacowjcza_jczmzyhsca_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_vhacowjcza_jczmzyhsca_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_vhacowjcza_jczmzyhsca_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_vhacowjcza_jczmzyhsca_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_zgrdiwrovx_wdwulgrltk_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_zgrdiwrovx_wdwulgrltk_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_zgrdiwrovx_wdwulgrltk_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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
àY

while_body_2162246
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_wdwulgrltk_matmul_readvariableop_resource_0:	 F
3while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 A
2while_wdwulgrltk_biasadd_readvariableop_resource_0:	8
*while_wdwulgrltk_readvariableop_resource_0: :
,while_wdwulgrltk_readvariableop_1_resource_0: :
,while_wdwulgrltk_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_wdwulgrltk_matmul_readvariableop_resource:	 D
1while_wdwulgrltk_matmul_1_readvariableop_resource:	 ?
0while_wdwulgrltk_biasadd_readvariableop_resource:	6
(while_wdwulgrltk_readvariableop_resource: 8
*while_wdwulgrltk_readvariableop_1_resource: 8
*while_wdwulgrltk_readvariableop_2_resource: ¢'while/wdwulgrltk/BiasAdd/ReadVariableOp¢&while/wdwulgrltk/MatMul/ReadVariableOp¢(while/wdwulgrltk/MatMul_1/ReadVariableOp¢while/wdwulgrltk/ReadVariableOp¢!while/wdwulgrltk/ReadVariableOp_1¢!while/wdwulgrltk/ReadVariableOp_2Ã
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
&while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp1while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/wdwulgrltk/MatMul/ReadVariableOpÑ
while/wdwulgrltk/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMulÉ
(while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp3while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/wdwulgrltk/MatMul_1/ReadVariableOpº
while/wdwulgrltk/MatMul_1MatMulwhile_placeholder_20while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMul_1°
while/wdwulgrltk/addAddV2!while/wdwulgrltk/MatMul:product:0#while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/addÂ
'while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp2while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/wdwulgrltk/BiasAdd/ReadVariableOp½
while/wdwulgrltk/BiasAddBiasAddwhile/wdwulgrltk/add:z:0/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/BiasAdd
 while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/wdwulgrltk/split/split_dim
while/wdwulgrltk/splitSplit)while/wdwulgrltk/split/split_dim:output:0!while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/wdwulgrltk/split©
while/wdwulgrltk/ReadVariableOpReadVariableOp*while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/wdwulgrltk/ReadVariableOp£
while/wdwulgrltk/mulMul'while/wdwulgrltk/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul¦
while/wdwulgrltk/add_1AddV2while/wdwulgrltk/split:output:0while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_1
while/wdwulgrltk/SigmoidSigmoidwhile/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid¯
!while/wdwulgrltk/ReadVariableOp_1ReadVariableOp,while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_1©
while/wdwulgrltk/mul_1Mul)while/wdwulgrltk/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_1¨
while/wdwulgrltk/add_2AddV2while/wdwulgrltk/split:output:1while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_2
while/wdwulgrltk/Sigmoid_1Sigmoidwhile/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_1
while/wdwulgrltk/mul_2Mulwhile/wdwulgrltk/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_2
while/wdwulgrltk/TanhTanhwhile/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh¢
while/wdwulgrltk/mul_3Mulwhile/wdwulgrltk/Sigmoid:y:0while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_3£
while/wdwulgrltk/add_3AddV2while/wdwulgrltk/mul_2:z:0while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_3¯
!while/wdwulgrltk/ReadVariableOp_2ReadVariableOp,while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_2°
while/wdwulgrltk/mul_4Mul)while/wdwulgrltk/ReadVariableOp_2:value:0while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_4¨
while/wdwulgrltk/add_4AddV2while/wdwulgrltk/split:output:3while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_4
while/wdwulgrltk/Sigmoid_2Sigmoidwhile/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_2
while/wdwulgrltk/Tanh_1Tanhwhile/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh_1¦
while/wdwulgrltk/mul_5Mulwhile/wdwulgrltk/Sigmoid_2:y:0while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/wdwulgrltk/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/wdwulgrltk/mul_5:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/wdwulgrltk/add_3:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
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
0while_wdwulgrltk_biasadd_readvariableop_resource2while_wdwulgrltk_biasadd_readvariableop_resource_0"h
1while_wdwulgrltk_matmul_1_readvariableop_resource3while_wdwulgrltk_matmul_1_readvariableop_resource_0"d
/while_wdwulgrltk_matmul_readvariableop_resource1while_wdwulgrltk_matmul_readvariableop_resource_0"Z
*while_wdwulgrltk_readvariableop_1_resource,while_wdwulgrltk_readvariableop_1_resource_0"Z
*while_wdwulgrltk_readvariableop_2_resource,while_wdwulgrltk_readvariableop_2_resource_0"V
(while_wdwulgrltk_readvariableop_resource*while_wdwulgrltk_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/wdwulgrltk/BiasAdd/ReadVariableOp'while/wdwulgrltk/BiasAdd/ReadVariableOp2P
&while/wdwulgrltk/MatMul/ReadVariableOp&while/wdwulgrltk/MatMul/ReadVariableOp2T
(while/wdwulgrltk/MatMul_1/ReadVariableOp(while/wdwulgrltk/MatMul_1/ReadVariableOp2B
while/wdwulgrltk/ReadVariableOpwhile/wdwulgrltk/ReadVariableOp2F
!while/wdwulgrltk/ReadVariableOp_1!while/wdwulgrltk/ReadVariableOp_12F
!while/wdwulgrltk/ReadVariableOp_2!while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2159352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2159352___redundant_placeholder05
1while_while_cond_2159352___redundant_placeholder15
1while_while_cond_2159352___redundant_placeholder25
1while_while_cond_2159352___redundant_placeholder35
1while_while_cond_2159352___redundant_placeholder45
1while_while_cond_2159352___redundant_placeholder55
1while_while_cond_2159352___redundant_placeholder6
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
while_body_2158629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_wdwulgrltk_2158653_0:	 -
while_wdwulgrltk_2158655_0:	 )
while_wdwulgrltk_2158657_0:	(
while_wdwulgrltk_2158659_0: (
while_wdwulgrltk_2158661_0: (
while_wdwulgrltk_2158663_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_wdwulgrltk_2158653:	 +
while_wdwulgrltk_2158655:	 '
while_wdwulgrltk_2158657:	&
while_wdwulgrltk_2158659: &
while_wdwulgrltk_2158661: &
while_wdwulgrltk_2158663: ¢(while/wdwulgrltk/StatefulPartitionedCallÃ
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
(while/wdwulgrltk/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_wdwulgrltk_2158653_0while_wdwulgrltk_2158655_0while_wdwulgrltk_2158657_0while_wdwulgrltk_2158659_0while_wdwulgrltk_2158661_0while_wdwulgrltk_2158663_0*
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_21585332*
(while/wdwulgrltk/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/wdwulgrltk/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/wdwulgrltk/StatefulPartitionedCall:output:1)^while/wdwulgrltk/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/wdwulgrltk/StatefulPartitionedCall:output:2)^while/wdwulgrltk/StatefulPartitionedCall*
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
while_wdwulgrltk_2158653while_wdwulgrltk_2158653_0"6
while_wdwulgrltk_2158655while_wdwulgrltk_2158655_0"6
while_wdwulgrltk_2158657while_wdwulgrltk_2158657_0"6
while_wdwulgrltk_2158659while_wdwulgrltk_2158659_0"6
while_wdwulgrltk_2158661while_wdwulgrltk_2158661_0"6
while_wdwulgrltk_2158663while_wdwulgrltk_2158663_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/wdwulgrltk/StatefulPartitionedCall(while/wdwulgrltk/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Ü

(sequential_vhacowjcza_while_body_2157218H
Dsequential_vhacowjcza_while_sequential_vhacowjcza_while_loop_counterN
Jsequential_vhacowjcza_while_sequential_vhacowjcza_while_maximum_iterations+
'sequential_vhacowjcza_while_placeholder-
)sequential_vhacowjcza_while_placeholder_1-
)sequential_vhacowjcza_while_placeholder_2-
)sequential_vhacowjcza_while_placeholder_3G
Csequential_vhacowjcza_while_sequential_vhacowjcza_strided_slice_1_0
sequential_vhacowjcza_while_tensorarrayv2read_tensorlistgetitem_sequential_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0:	\
Isequential_vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 W
Hsequential_vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0:	N
@sequential_vhacowjcza_while_jczmzyhsca_readvariableop_resource_0: P
Bsequential_vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0: P
Bsequential_vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0: (
$sequential_vhacowjcza_while_identity*
&sequential_vhacowjcza_while_identity_1*
&sequential_vhacowjcza_while_identity_2*
&sequential_vhacowjcza_while_identity_3*
&sequential_vhacowjcza_while_identity_4*
&sequential_vhacowjcza_while_identity_5E
Asequential_vhacowjcza_while_sequential_vhacowjcza_strided_slice_1
}sequential_vhacowjcza_while_tensorarrayv2read_tensorlistgetitem_sequential_vhacowjcza_tensorarrayunstack_tensorlistfromtensorX
Esequential_vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource:	Z
Gsequential_vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource:	 U
Fsequential_vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource:	L
>sequential_vhacowjcza_while_jczmzyhsca_readvariableop_resource: N
@sequential_vhacowjcza_while_jczmzyhsca_readvariableop_1_resource: N
@sequential_vhacowjcza_while_jczmzyhsca_readvariableop_2_resource: ¢=sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp¢<sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp¢>sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp¢5sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp¢7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_1¢7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2ï
Msequential/vhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/vhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/vhacowjcza/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_vhacowjcza_while_tensorarrayv2read_tensorlistgetitem_sequential_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0'sequential_vhacowjcza_while_placeholderVsequential/vhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/vhacowjcza/while/TensorArrayV2Read/TensorListGetItem
<sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOpGsequential_vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp©
-sequential/vhacowjcza/while/jczmzyhsca/MatMulMatMulFsequential/vhacowjcza/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/vhacowjcza/while/jczmzyhsca/MatMul
>sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOpIsequential_vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp
/sequential/vhacowjcza/while/jczmzyhsca/MatMul_1MatMul)sequential_vhacowjcza_while_placeholder_2Fsequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/vhacowjcza/while/jczmzyhsca/MatMul_1
*sequential/vhacowjcza/while/jczmzyhsca/addAddV27sequential/vhacowjcza/while/jczmzyhsca/MatMul:product:09sequential/vhacowjcza/while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/vhacowjcza/while/jczmzyhsca/add
=sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOpHsequential_vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp
.sequential/vhacowjcza/while/jczmzyhsca/BiasAddBiasAdd.sequential/vhacowjcza/while/jczmzyhsca/add:z:0Esequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/vhacowjcza/while/jczmzyhsca/BiasAdd²
6sequential/vhacowjcza/while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/vhacowjcza/while/jczmzyhsca/split/split_dimÛ
,sequential/vhacowjcza/while/jczmzyhsca/splitSplit?sequential/vhacowjcza/while/jczmzyhsca/split/split_dim:output:07sequential/vhacowjcza/while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/vhacowjcza/while/jczmzyhsca/splitë
5sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOpReadVariableOp@sequential_vhacowjcza_while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOpû
*sequential/vhacowjcza/while/jczmzyhsca/mulMul=sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp:value:0)sequential_vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/vhacowjcza/while/jczmzyhsca/mulþ
,sequential/vhacowjcza/while/jczmzyhsca/add_1AddV25sequential/vhacowjcza/while/jczmzyhsca/split:output:0.sequential/vhacowjcza/while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/add_1Ï
.sequential/vhacowjcza/while/jczmzyhsca/SigmoidSigmoid0sequential/vhacowjcza/while/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/vhacowjcza/while/jczmzyhsca/Sigmoidñ
7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_1ReadVariableOpBsequential_vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_1
,sequential/vhacowjcza/while/jczmzyhsca/mul_1Mul?sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_1:value:0)sequential_vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/mul_1
,sequential/vhacowjcza/while/jczmzyhsca/add_2AddV25sequential/vhacowjcza/while/jczmzyhsca/split:output:10sequential/vhacowjcza/while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/add_2Ó
0sequential/vhacowjcza/while/jczmzyhsca/Sigmoid_1Sigmoid0sequential/vhacowjcza/while/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/vhacowjcza/while/jczmzyhsca/Sigmoid_1ö
,sequential/vhacowjcza/while/jczmzyhsca/mul_2Mul4sequential/vhacowjcza/while/jczmzyhsca/Sigmoid_1:y:0)sequential_vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/mul_2Ë
+sequential/vhacowjcza/while/jczmzyhsca/TanhTanh5sequential/vhacowjcza/while/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/vhacowjcza/while/jczmzyhsca/Tanhú
,sequential/vhacowjcza/while/jczmzyhsca/mul_3Mul2sequential/vhacowjcza/while/jczmzyhsca/Sigmoid:y:0/sequential/vhacowjcza/while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/mul_3û
,sequential/vhacowjcza/while/jczmzyhsca/add_3AddV20sequential/vhacowjcza/while/jczmzyhsca/mul_2:z:00sequential/vhacowjcza/while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/add_3ñ
7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2ReadVariableOpBsequential_vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2
,sequential/vhacowjcza/while/jczmzyhsca/mul_4Mul?sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2:value:00sequential/vhacowjcza/while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/mul_4
,sequential/vhacowjcza/while/jczmzyhsca/add_4AddV25sequential/vhacowjcza/while/jczmzyhsca/split:output:30sequential/vhacowjcza/while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/add_4Ó
0sequential/vhacowjcza/while/jczmzyhsca/Sigmoid_2Sigmoid0sequential/vhacowjcza/while/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/vhacowjcza/while/jczmzyhsca/Sigmoid_2Ê
-sequential/vhacowjcza/while/jczmzyhsca/Tanh_1Tanh0sequential/vhacowjcza/while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/vhacowjcza/while/jczmzyhsca/Tanh_1þ
,sequential/vhacowjcza/while/jczmzyhsca/mul_5Mul4sequential/vhacowjcza/while/jczmzyhsca/Sigmoid_2:y:01sequential/vhacowjcza/while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/vhacowjcza/while/jczmzyhsca/mul_5Ì
@sequential/vhacowjcza/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_vhacowjcza_while_placeholder_1'sequential_vhacowjcza_while_placeholder0sequential/vhacowjcza/while/jczmzyhsca/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/vhacowjcza/while/TensorArrayV2Write/TensorListSetItem
!sequential/vhacowjcza/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/vhacowjcza/while/add/yÁ
sequential/vhacowjcza/while/addAddV2'sequential_vhacowjcza_while_placeholder*sequential/vhacowjcza/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/vhacowjcza/while/add
#sequential/vhacowjcza/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/vhacowjcza/while/add_1/yä
!sequential/vhacowjcza/while/add_1AddV2Dsequential_vhacowjcza_while_sequential_vhacowjcza_while_loop_counter,sequential/vhacowjcza/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/vhacowjcza/while/add_1
$sequential/vhacowjcza/while/IdentityIdentity%sequential/vhacowjcza/while/add_1:z:0>^sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp=^sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp?^sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp6^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp8^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_18^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/vhacowjcza/while/Identityµ
&sequential/vhacowjcza/while/Identity_1IdentityJsequential_vhacowjcza_while_sequential_vhacowjcza_while_maximum_iterations>^sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp=^sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp?^sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp6^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp8^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_18^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/vhacowjcza/while/Identity_1
&sequential/vhacowjcza/while/Identity_2Identity#sequential/vhacowjcza/while/add:z:0>^sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp=^sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp?^sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp6^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp8^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_18^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/vhacowjcza/while/Identity_2»
&sequential/vhacowjcza/while/Identity_3IdentityPsequential/vhacowjcza/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp=^sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp?^sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp6^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp8^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_18^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/vhacowjcza/while/Identity_3¬
&sequential/vhacowjcza/while/Identity_4Identity0sequential/vhacowjcza/while/jczmzyhsca/mul_5:z:0>^sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp=^sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp?^sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp6^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp8^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_18^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/while/Identity_4¬
&sequential/vhacowjcza/while/Identity_5Identity0sequential/vhacowjcza/while/jczmzyhsca/add_3:z:0>^sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp=^sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp?^sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp6^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp8^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_18^sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/vhacowjcza/while/Identity_5"U
$sequential_vhacowjcza_while_identity-sequential/vhacowjcza/while/Identity:output:0"Y
&sequential_vhacowjcza_while_identity_1/sequential/vhacowjcza/while/Identity_1:output:0"Y
&sequential_vhacowjcza_while_identity_2/sequential/vhacowjcza/while/Identity_2:output:0"Y
&sequential_vhacowjcza_while_identity_3/sequential/vhacowjcza/while/Identity_3:output:0"Y
&sequential_vhacowjcza_while_identity_4/sequential/vhacowjcza/while/Identity_4:output:0"Y
&sequential_vhacowjcza_while_identity_5/sequential/vhacowjcza/while/Identity_5:output:0"
Fsequential_vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resourceHsequential_vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0"
Gsequential_vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resourceIsequential_vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0"
Esequential_vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resourceGsequential_vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0"
@sequential_vhacowjcza_while_jczmzyhsca_readvariableop_1_resourceBsequential_vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0"
@sequential_vhacowjcza_while_jczmzyhsca_readvariableop_2_resourceBsequential_vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0"
>sequential_vhacowjcza_while_jczmzyhsca_readvariableop_resource@sequential_vhacowjcza_while_jczmzyhsca_readvariableop_resource_0"
Asequential_vhacowjcza_while_sequential_vhacowjcza_strided_slice_1Csequential_vhacowjcza_while_sequential_vhacowjcza_strided_slice_1_0"
}sequential_vhacowjcza_while_tensorarrayv2read_tensorlistgetitem_sequential_vhacowjcza_tensorarrayunstack_tensorlistfromtensorsequential_vhacowjcza_while_tensorarrayv2read_tensorlistgetitem_sequential_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp=sequential/vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2|
<sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp<sequential/vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp2
>sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp>sequential/vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp2n
5sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp5sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp2r
7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_17sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_12r
7sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_27sequential/vhacowjcza/while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
ç)
Ò
while_body_2157871
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_jczmzyhsca_2157895_0:	-
while_jczmzyhsca_2157897_0:	 )
while_jczmzyhsca_2157899_0:	(
while_jczmzyhsca_2157901_0: (
while_jczmzyhsca_2157903_0: (
while_jczmzyhsca_2157905_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_jczmzyhsca_2157895:	+
while_jczmzyhsca_2157897:	 '
while_jczmzyhsca_2157899:	&
while_jczmzyhsca_2157901: &
while_jczmzyhsca_2157903: &
while_jczmzyhsca_2157905: ¢(while/jczmzyhsca/StatefulPartitionedCallÃ
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
(while/jczmzyhsca/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_jczmzyhsca_2157895_0while_jczmzyhsca_2157897_0while_jczmzyhsca_2157899_0while_jczmzyhsca_2157901_0while_jczmzyhsca_2157903_0while_jczmzyhsca_2157905_0*
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_21577752*
(while/jczmzyhsca/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/jczmzyhsca/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/jczmzyhsca/StatefulPartitionedCall:output:1)^while/jczmzyhsca/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/jczmzyhsca/StatefulPartitionedCall:output:2)^while/jczmzyhsca/StatefulPartitionedCall*
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
while_jczmzyhsca_2157895while_jczmzyhsca_2157895_0"6
while_jczmzyhsca_2157897while_jczmzyhsca_2157897_0"6
while_jczmzyhsca_2157899while_jczmzyhsca_2157899_0"6
while_jczmzyhsca_2157901while_jczmzyhsca_2157901_0"6
while_jczmzyhsca_2157903while_jczmzyhsca_2157903_0"6
while_jczmzyhsca_2157905while_jczmzyhsca_2157905_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/jczmzyhsca/StatefulPartitionedCall(while/jczmzyhsca/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
vhacowjcza_while_cond_21607772
.vhacowjcza_while_vhacowjcza_while_loop_counter8
4vhacowjcza_while_vhacowjcza_while_maximum_iterations 
vhacowjcza_while_placeholder"
vhacowjcza_while_placeholder_1"
vhacowjcza_while_placeholder_2"
vhacowjcza_while_placeholder_34
0vhacowjcza_while_less_vhacowjcza_strided_slice_1K
Gvhacowjcza_while_vhacowjcza_while_cond_2160777___redundant_placeholder0K
Gvhacowjcza_while_vhacowjcza_while_cond_2160777___redundant_placeholder1K
Gvhacowjcza_while_vhacowjcza_while_cond_2160777___redundant_placeholder2K
Gvhacowjcza_while_vhacowjcza_while_cond_2160777___redundant_placeholder3K
Gvhacowjcza_while_vhacowjcza_while_cond_2160777___redundant_placeholder4K
Gvhacowjcza_while_vhacowjcza_while_cond_2160777___redundant_placeholder5K
Gvhacowjcza_while_vhacowjcza_while_cond_2160777___redundant_placeholder6
vhacowjcza_while_identity
§
vhacowjcza/while/LessLessvhacowjcza_while_placeholder0vhacowjcza_while_less_vhacowjcza_strided_slice_1*
T0*
_output_shapes
: 2
vhacowjcza/while/Less~
vhacowjcza/while/IdentityIdentityvhacowjcza/while/Less:z:0*
T0
*
_output_shapes
: 2
vhacowjcza/while/Identity"?
vhacowjcza_while_identity"vhacowjcza/while/Identity:output:0*(
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
vhacowjcza_while_cond_21603732
.vhacowjcza_while_vhacowjcza_while_loop_counter8
4vhacowjcza_while_vhacowjcza_while_maximum_iterations 
vhacowjcza_while_placeholder"
vhacowjcza_while_placeholder_1"
vhacowjcza_while_placeholder_2"
vhacowjcza_while_placeholder_34
0vhacowjcza_while_less_vhacowjcza_strided_slice_1K
Gvhacowjcza_while_vhacowjcza_while_cond_2160373___redundant_placeholder0K
Gvhacowjcza_while_vhacowjcza_while_cond_2160373___redundant_placeholder1K
Gvhacowjcza_while_vhacowjcza_while_cond_2160373___redundant_placeholder2K
Gvhacowjcza_while_vhacowjcza_while_cond_2160373___redundant_placeholder3K
Gvhacowjcza_while_vhacowjcza_while_cond_2160373___redundant_placeholder4K
Gvhacowjcza_while_vhacowjcza_while_cond_2160373___redundant_placeholder5K
Gvhacowjcza_while_vhacowjcza_while_cond_2160373___redundant_placeholder6
vhacowjcza_while_identity
§
vhacowjcza/while/LessLessvhacowjcza_while_placeholder0vhacowjcza_while_less_vhacowjcza_strided_slice_1*
T0*
_output_shapes
: 2
vhacowjcza/while/Less~
vhacowjcza/while/IdentityIdentityvhacowjcza/while/Less:z:0*
T0
*
_output_shapes
: 2
vhacowjcza/while/Identity"?
vhacowjcza_while_identity"vhacowjcza/while/Identity:output:0*(
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
while_body_2162066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_wdwulgrltk_matmul_readvariableop_resource_0:	 F
3while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 A
2while_wdwulgrltk_biasadd_readvariableop_resource_0:	8
*while_wdwulgrltk_readvariableop_resource_0: :
,while_wdwulgrltk_readvariableop_1_resource_0: :
,while_wdwulgrltk_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_wdwulgrltk_matmul_readvariableop_resource:	 D
1while_wdwulgrltk_matmul_1_readvariableop_resource:	 ?
0while_wdwulgrltk_biasadd_readvariableop_resource:	6
(while_wdwulgrltk_readvariableop_resource: 8
*while_wdwulgrltk_readvariableop_1_resource: 8
*while_wdwulgrltk_readvariableop_2_resource: ¢'while/wdwulgrltk/BiasAdd/ReadVariableOp¢&while/wdwulgrltk/MatMul/ReadVariableOp¢(while/wdwulgrltk/MatMul_1/ReadVariableOp¢while/wdwulgrltk/ReadVariableOp¢!while/wdwulgrltk/ReadVariableOp_1¢!while/wdwulgrltk/ReadVariableOp_2Ã
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
&while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp1while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/wdwulgrltk/MatMul/ReadVariableOpÑ
while/wdwulgrltk/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMulÉ
(while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp3while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/wdwulgrltk/MatMul_1/ReadVariableOpº
while/wdwulgrltk/MatMul_1MatMulwhile_placeholder_20while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMul_1°
while/wdwulgrltk/addAddV2!while/wdwulgrltk/MatMul:product:0#while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/addÂ
'while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp2while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/wdwulgrltk/BiasAdd/ReadVariableOp½
while/wdwulgrltk/BiasAddBiasAddwhile/wdwulgrltk/add:z:0/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/BiasAdd
 while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/wdwulgrltk/split/split_dim
while/wdwulgrltk/splitSplit)while/wdwulgrltk/split/split_dim:output:0!while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/wdwulgrltk/split©
while/wdwulgrltk/ReadVariableOpReadVariableOp*while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/wdwulgrltk/ReadVariableOp£
while/wdwulgrltk/mulMul'while/wdwulgrltk/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul¦
while/wdwulgrltk/add_1AddV2while/wdwulgrltk/split:output:0while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_1
while/wdwulgrltk/SigmoidSigmoidwhile/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid¯
!while/wdwulgrltk/ReadVariableOp_1ReadVariableOp,while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_1©
while/wdwulgrltk/mul_1Mul)while/wdwulgrltk/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_1¨
while/wdwulgrltk/add_2AddV2while/wdwulgrltk/split:output:1while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_2
while/wdwulgrltk/Sigmoid_1Sigmoidwhile/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_1
while/wdwulgrltk/mul_2Mulwhile/wdwulgrltk/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_2
while/wdwulgrltk/TanhTanhwhile/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh¢
while/wdwulgrltk/mul_3Mulwhile/wdwulgrltk/Sigmoid:y:0while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_3£
while/wdwulgrltk/add_3AddV2while/wdwulgrltk/mul_2:z:0while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_3¯
!while/wdwulgrltk/ReadVariableOp_2ReadVariableOp,while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_2°
while/wdwulgrltk/mul_4Mul)while/wdwulgrltk/ReadVariableOp_2:value:0while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_4¨
while/wdwulgrltk/add_4AddV2while/wdwulgrltk/split:output:3while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_4
while/wdwulgrltk/Sigmoid_2Sigmoidwhile/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_2
while/wdwulgrltk/Tanh_1Tanhwhile/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh_1¦
while/wdwulgrltk/mul_5Mulwhile/wdwulgrltk/Sigmoid_2:y:0while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/wdwulgrltk/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/wdwulgrltk/mul_5:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/wdwulgrltk/add_3:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
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
0while_wdwulgrltk_biasadd_readvariableop_resource2while_wdwulgrltk_biasadd_readvariableop_resource_0"h
1while_wdwulgrltk_matmul_1_readvariableop_resource3while_wdwulgrltk_matmul_1_readvariableop_resource_0"d
/while_wdwulgrltk_matmul_readvariableop_resource1while_wdwulgrltk_matmul_readvariableop_resource_0"Z
*while_wdwulgrltk_readvariableop_1_resource,while_wdwulgrltk_readvariableop_1_resource_0"Z
*while_wdwulgrltk_readvariableop_2_resource,while_wdwulgrltk_readvariableop_2_resource_0"V
(while_wdwulgrltk_readvariableop_resource*while_wdwulgrltk_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/wdwulgrltk/BiasAdd/ReadVariableOp'while/wdwulgrltk/BiasAdd/ReadVariableOp2P
&while/wdwulgrltk/MatMul/ReadVariableOp&while/wdwulgrltk/MatMul/ReadVariableOp2T
(while/wdwulgrltk/MatMul_1/ReadVariableOp(while/wdwulgrltk/MatMul_1/ReadVariableOp2B
while/wdwulgrltk/ReadVariableOpwhile/wdwulgrltk/ReadVariableOp2F
!while/wdwulgrltk/ReadVariableOp_1!while/wdwulgrltk/ReadVariableOp_12F
!while/wdwulgrltk/ReadVariableOp_2!while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_sequential_layer_call_and_return_conditional_losses_2160054

inputs(
vfwtupxpzf_2160016: 
vfwtupxpzf_2160018:%
vhacowjcza_2160022:	%
vhacowjcza_2160024:	 !
vhacowjcza_2160026:	 
vhacowjcza_2160028:  
vhacowjcza_2160030:  
vhacowjcza_2160032: %
zgrdiwrovx_2160035:	 %
zgrdiwrovx_2160037:	 !
zgrdiwrovx_2160039:	 
zgrdiwrovx_2160041:  
zgrdiwrovx_2160043:  
zgrdiwrovx_2160045: $
kekwghyimt_2160048:  
kekwghyimt_2160050:
identity¢"kekwghyimt/StatefulPartitionedCall¢"vfwtupxpzf/StatefulPartitionedCall¢"vhacowjcza/StatefulPartitionedCall¢"zgrdiwrovx/StatefulPartitionedCall¬
"vfwtupxpzf/StatefulPartitionedCallStatefulPartitionedCallinputsvfwtupxpzf_2160016vfwtupxpzf_2160018*
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
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_21590612$
"vfwtupxpzf/StatefulPartitionedCall
ojzbgzevue/PartitionedCallPartitionedCall+vfwtupxpzf/StatefulPartitionedCall:output:0*
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
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_21590802
ojzbgzevue/PartitionedCall
"vhacowjcza/StatefulPartitionedCallStatefulPartitionedCall#ojzbgzevue/PartitionedCall:output:0vhacowjcza_2160022vhacowjcza_2160024vhacowjcza_2160026vhacowjcza_2160028vhacowjcza_2160030vhacowjcza_2160032*
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21599432$
"vhacowjcza/StatefulPartitionedCall¡
"zgrdiwrovx/StatefulPartitionedCallStatefulPartitionedCall+vhacowjcza/StatefulPartitionedCall:output:0zgrdiwrovx_2160035zgrdiwrovx_2160037zgrdiwrovx_2160039zgrdiwrovx_2160041zgrdiwrovx_2160043zgrdiwrovx_2160045*
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_21597292$
"zgrdiwrovx/StatefulPartitionedCallÉ
"kekwghyimt/StatefulPartitionedCallStatefulPartitionedCall+zgrdiwrovx/StatefulPartitionedCall:output:0kekwghyimt_2160048kekwghyimt_2160050*
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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_21594782$
"kekwghyimt/StatefulPartitionedCall
IdentityIdentity+kekwghyimt/StatefulPartitionedCall:output:0#^kekwghyimt/StatefulPartitionedCall#^vfwtupxpzf/StatefulPartitionedCall#^vhacowjcza/StatefulPartitionedCall#^zgrdiwrovx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"kekwghyimt/StatefulPartitionedCall"kekwghyimt/StatefulPartitionedCall2H
"vfwtupxpzf/StatefulPartitionedCall"vfwtupxpzf/StatefulPartitionedCall2H
"vhacowjcza/StatefulPartitionedCall"vhacowjcza/StatefulPartitionedCall2H
"zgrdiwrovx/StatefulPartitionedCall"zgrdiwrovx/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_2162425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2162425___redundant_placeholder05
1while_while_cond_2162425___redundant_placeholder15
1while_while_cond_2162425___redundant_placeholder25
1while_while_cond_2162425___redundant_placeholder35
1while_while_cond_2162425___redundant_placeholder45
1while_while_cond_2162425___redundant_placeholder55
1while_while_cond_2162425___redundant_placeholder6
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2159943

inputs<
)jczmzyhsca_matmul_readvariableop_resource:	>
+jczmzyhsca_matmul_1_readvariableop_resource:	 9
*jczmzyhsca_biasadd_readvariableop_resource:	0
"jczmzyhsca_readvariableop_resource: 2
$jczmzyhsca_readvariableop_1_resource: 2
$jczmzyhsca_readvariableop_2_resource: 
identity¢!jczmzyhsca/BiasAdd/ReadVariableOp¢ jczmzyhsca/MatMul/ReadVariableOp¢"jczmzyhsca/MatMul_1/ReadVariableOp¢jczmzyhsca/ReadVariableOp¢jczmzyhsca/ReadVariableOp_1¢jczmzyhsca/ReadVariableOp_2¢whileD
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
 jczmzyhsca/MatMul/ReadVariableOpReadVariableOp)jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jczmzyhsca/MatMul/ReadVariableOp§
jczmzyhsca/MatMulMatMulstrided_slice_2:output:0(jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMulµ
"jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp+jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jczmzyhsca/MatMul_1/ReadVariableOp£
jczmzyhsca/MatMul_1MatMulzeros:output:0*jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMul_1
jczmzyhsca/addAddV2jczmzyhsca/MatMul:product:0jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/add®
!jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp*jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jczmzyhsca/BiasAdd/ReadVariableOp¥
jczmzyhsca/BiasAddBiasAddjczmzyhsca/add:z:0)jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/BiasAddz
jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jczmzyhsca/split/split_dimë
jczmzyhsca/splitSplit#jczmzyhsca/split/split_dim:output:0jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jczmzyhsca/split
jczmzyhsca/ReadVariableOpReadVariableOp"jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp
jczmzyhsca/mulMul!jczmzyhsca/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul
jczmzyhsca/add_1AddV2jczmzyhsca/split:output:0jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_1{
jczmzyhsca/SigmoidSigmoidjczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid
jczmzyhsca/ReadVariableOp_1ReadVariableOp$jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_1
jczmzyhsca/mul_1Mul#jczmzyhsca/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_1
jczmzyhsca/add_2AddV2jczmzyhsca/split:output:1jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_2
jczmzyhsca/Sigmoid_1Sigmoidjczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_1
jczmzyhsca/mul_2Muljczmzyhsca/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_2w
jczmzyhsca/TanhTanhjczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh
jczmzyhsca/mul_3Muljczmzyhsca/Sigmoid:y:0jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_3
jczmzyhsca/add_3AddV2jczmzyhsca/mul_2:z:0jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_3
jczmzyhsca/ReadVariableOp_2ReadVariableOp$jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_2
jczmzyhsca/mul_4Mul#jczmzyhsca/ReadVariableOp_2:value:0jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_4
jczmzyhsca/add_4AddV2jczmzyhsca/split:output:3jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_4
jczmzyhsca/Sigmoid_2Sigmoidjczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_2v
jczmzyhsca/Tanh_1Tanhjczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh_1
jczmzyhsca/mul_5Muljczmzyhsca/Sigmoid_2:y:0jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jczmzyhsca_matmul_readvariableop_resource+jczmzyhsca_matmul_1_readvariableop_resource*jczmzyhsca_biasadd_readvariableop_resource"jczmzyhsca_readvariableop_resource$jczmzyhsca_readvariableop_1_resource$jczmzyhsca_readvariableop_2_resource*
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
while_body_2159842*
condR
while_cond_2159841*Q
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
IdentityIdentitytranspose_1:y:0"^jczmzyhsca/BiasAdd/ReadVariableOp!^jczmzyhsca/MatMul/ReadVariableOp#^jczmzyhsca/MatMul_1/ReadVariableOp^jczmzyhsca/ReadVariableOp^jczmzyhsca/ReadVariableOp_1^jczmzyhsca/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jczmzyhsca/BiasAdd/ReadVariableOp!jczmzyhsca/BiasAdd/ReadVariableOp2D
 jczmzyhsca/MatMul/ReadVariableOp jczmzyhsca/MatMul/ReadVariableOp2H
"jczmzyhsca/MatMul_1/ReadVariableOp"jczmzyhsca/MatMul_1/ReadVariableOp26
jczmzyhsca/ReadVariableOpjczmzyhsca/ReadVariableOp2:
jczmzyhsca/ReadVariableOp_1jczmzyhsca/ReadVariableOp_12:
jczmzyhsca/ReadVariableOp_2jczmzyhsca/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

(sequential_zgrdiwrovx_while_body_2157394H
Dsequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_loop_counterN
Jsequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_maximum_iterations+
'sequential_zgrdiwrovx_while_placeholder-
)sequential_zgrdiwrovx_while_placeholder_1-
)sequential_zgrdiwrovx_while_placeholder_2-
)sequential_zgrdiwrovx_while_placeholder_3G
Csequential_zgrdiwrovx_while_sequential_zgrdiwrovx_strided_slice_1_0
sequential_zgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_sequential_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0:	 \
Isequential_zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 W
Hsequential_zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0:	N
@sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0: P
Bsequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0: P
Bsequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0: (
$sequential_zgrdiwrovx_while_identity*
&sequential_zgrdiwrovx_while_identity_1*
&sequential_zgrdiwrovx_while_identity_2*
&sequential_zgrdiwrovx_while_identity_3*
&sequential_zgrdiwrovx_while_identity_4*
&sequential_zgrdiwrovx_while_identity_5E
Asequential_zgrdiwrovx_while_sequential_zgrdiwrovx_strided_slice_1
}sequential_zgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_sequential_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensorX
Esequential_zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource:	 Z
Gsequential_zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource:	 U
Fsequential_zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource:	L
>sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_resource: N
@sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource: N
@sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource: ¢=sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp¢<sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp¢>sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp¢5sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp¢7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1¢7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2ï
Msequential/zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_zgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_sequential_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0'sequential_zgrdiwrovx_while_placeholderVsequential/zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem
<sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOpGsequential_zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp©
-sequential/zgrdiwrovx/while/wdwulgrltk/MatMulMatMulFsequential/zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/zgrdiwrovx/while/wdwulgrltk/MatMul
>sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOpIsequential_zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp
/sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1MatMul)sequential_zgrdiwrovx_while_placeholder_2Fsequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1
*sequential/zgrdiwrovx/while/wdwulgrltk/addAddV27sequential/zgrdiwrovx/while/wdwulgrltk/MatMul:product:09sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/zgrdiwrovx/while/wdwulgrltk/add
=sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOpHsequential_zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp
.sequential/zgrdiwrovx/while/wdwulgrltk/BiasAddBiasAdd.sequential/zgrdiwrovx/while/wdwulgrltk/add:z:0Esequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd²
6sequential/zgrdiwrovx/while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/zgrdiwrovx/while/wdwulgrltk/split/split_dimÛ
,sequential/zgrdiwrovx/while/wdwulgrltk/splitSplit?sequential/zgrdiwrovx/while/wdwulgrltk/split/split_dim:output:07sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/zgrdiwrovx/while/wdwulgrltk/splitë
5sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOpReadVariableOp@sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOpû
*sequential/zgrdiwrovx/while/wdwulgrltk/mulMul=sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp:value:0)sequential_zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/zgrdiwrovx/while/wdwulgrltk/mulþ
,sequential/zgrdiwrovx/while/wdwulgrltk/add_1AddV25sequential/zgrdiwrovx/while/wdwulgrltk/split:output:0.sequential/zgrdiwrovx/while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/add_1Ï
.sequential/zgrdiwrovx/while/wdwulgrltk/SigmoidSigmoid0sequential/zgrdiwrovx/while/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoidñ
7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1ReadVariableOpBsequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_1Mul?sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1:value:0)sequential_zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_1
,sequential/zgrdiwrovx/while/wdwulgrltk/add_2AddV25sequential/zgrdiwrovx/while/wdwulgrltk/split:output:10sequential/zgrdiwrovx/while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/add_2Ó
0sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoid_1Sigmoid0sequential/zgrdiwrovx/while/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoid_1ö
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_2Mul4sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoid_1:y:0)sequential_zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_2Ë
+sequential/zgrdiwrovx/while/wdwulgrltk/TanhTanh5sequential/zgrdiwrovx/while/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/zgrdiwrovx/while/wdwulgrltk/Tanhú
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_3Mul2sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoid:y:0/sequential/zgrdiwrovx/while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_3û
,sequential/zgrdiwrovx/while/wdwulgrltk/add_3AddV20sequential/zgrdiwrovx/while/wdwulgrltk/mul_2:z:00sequential/zgrdiwrovx/while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/add_3ñ
7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2ReadVariableOpBsequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_4Mul?sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2:value:00sequential/zgrdiwrovx/while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_4
,sequential/zgrdiwrovx/while/wdwulgrltk/add_4AddV25sequential/zgrdiwrovx/while/wdwulgrltk/split:output:30sequential/zgrdiwrovx/while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/add_4Ó
0sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoid_2Sigmoid0sequential/zgrdiwrovx/while/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoid_2Ê
-sequential/zgrdiwrovx/while/wdwulgrltk/Tanh_1Tanh0sequential/zgrdiwrovx/while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/zgrdiwrovx/while/wdwulgrltk/Tanh_1þ
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_5Mul4sequential/zgrdiwrovx/while/wdwulgrltk/Sigmoid_2:y:01sequential/zgrdiwrovx/while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zgrdiwrovx/while/wdwulgrltk/mul_5Ì
@sequential/zgrdiwrovx/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_zgrdiwrovx_while_placeholder_1'sequential_zgrdiwrovx_while_placeholder0sequential/zgrdiwrovx/while/wdwulgrltk/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/zgrdiwrovx/while/TensorArrayV2Write/TensorListSetItem
!sequential/zgrdiwrovx/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/zgrdiwrovx/while/add/yÁ
sequential/zgrdiwrovx/while/addAddV2'sequential_zgrdiwrovx_while_placeholder*sequential/zgrdiwrovx/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/zgrdiwrovx/while/add
#sequential/zgrdiwrovx/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/zgrdiwrovx/while/add_1/yä
!sequential/zgrdiwrovx/while/add_1AddV2Dsequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_loop_counter,sequential/zgrdiwrovx/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/zgrdiwrovx/while/add_1
$sequential/zgrdiwrovx/while/IdentityIdentity%sequential/zgrdiwrovx/while/add_1:z:0>^sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp=^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp?^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp6^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp8^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_18^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/zgrdiwrovx/while/Identityµ
&sequential/zgrdiwrovx/while/Identity_1IdentityJsequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_maximum_iterations>^sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp=^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp?^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp6^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp8^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_18^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/zgrdiwrovx/while/Identity_1
&sequential/zgrdiwrovx/while/Identity_2Identity#sequential/zgrdiwrovx/while/add:z:0>^sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp=^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp?^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp6^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp8^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_18^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/zgrdiwrovx/while/Identity_2»
&sequential/zgrdiwrovx/while/Identity_3IdentityPsequential/zgrdiwrovx/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp=^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp?^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp6^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp8^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_18^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/zgrdiwrovx/while/Identity_3¬
&sequential/zgrdiwrovx/while/Identity_4Identity0sequential/zgrdiwrovx/while/wdwulgrltk/mul_5:z:0>^sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp=^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp?^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp6^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp8^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_18^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/while/Identity_4¬
&sequential/zgrdiwrovx/while/Identity_5Identity0sequential/zgrdiwrovx/while/wdwulgrltk/add_3:z:0>^sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp=^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp?^sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp6^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp8^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_18^sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zgrdiwrovx/while/Identity_5"U
$sequential_zgrdiwrovx_while_identity-sequential/zgrdiwrovx/while/Identity:output:0"Y
&sequential_zgrdiwrovx_while_identity_1/sequential/zgrdiwrovx/while/Identity_1:output:0"Y
&sequential_zgrdiwrovx_while_identity_2/sequential/zgrdiwrovx/while/Identity_2:output:0"Y
&sequential_zgrdiwrovx_while_identity_3/sequential/zgrdiwrovx/while/Identity_3:output:0"Y
&sequential_zgrdiwrovx_while_identity_4/sequential/zgrdiwrovx/while/Identity_4:output:0"Y
&sequential_zgrdiwrovx_while_identity_5/sequential/zgrdiwrovx/while/Identity_5:output:0"
Asequential_zgrdiwrovx_while_sequential_zgrdiwrovx_strided_slice_1Csequential_zgrdiwrovx_while_sequential_zgrdiwrovx_strided_slice_1_0"
}sequential_zgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_sequential_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensorsequential_zgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_sequential_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0"
Fsequential_zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resourceHsequential_zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0"
Gsequential_zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resourceIsequential_zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0"
Esequential_zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resourceGsequential_zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0"
@sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resourceBsequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0"
@sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resourceBsequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0"
>sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_resource@sequential_zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp=sequential/zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2|
<sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp<sequential/zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp2
>sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp>sequential/zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp2n
5sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp5sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp2r
7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_17sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_12r
7sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_27sequential/zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2158346

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
àY

while_body_2161638
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jczmzyhsca_matmul_readvariableop_resource_0:	F
3while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 A
2while_jczmzyhsca_biasadd_readvariableop_resource_0:	8
*while_jczmzyhsca_readvariableop_resource_0: :
,while_jczmzyhsca_readvariableop_1_resource_0: :
,while_jczmzyhsca_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jczmzyhsca_matmul_readvariableop_resource:	D
1while_jczmzyhsca_matmul_1_readvariableop_resource:	 ?
0while_jczmzyhsca_biasadd_readvariableop_resource:	6
(while_jczmzyhsca_readvariableop_resource: 8
*while_jczmzyhsca_readvariableop_1_resource: 8
*while_jczmzyhsca_readvariableop_2_resource: ¢'while/jczmzyhsca/BiasAdd/ReadVariableOp¢&while/jczmzyhsca/MatMul/ReadVariableOp¢(while/jczmzyhsca/MatMul_1/ReadVariableOp¢while/jczmzyhsca/ReadVariableOp¢!while/jczmzyhsca/ReadVariableOp_1¢!while/jczmzyhsca/ReadVariableOp_2Ã
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
&while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp1while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jczmzyhsca/MatMul/ReadVariableOpÑ
while/jczmzyhsca/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMulÉ
(while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp3while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jczmzyhsca/MatMul_1/ReadVariableOpº
while/jczmzyhsca/MatMul_1MatMulwhile_placeholder_20while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMul_1°
while/jczmzyhsca/addAddV2!while/jczmzyhsca/MatMul:product:0#while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/addÂ
'while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp2while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jczmzyhsca/BiasAdd/ReadVariableOp½
while/jczmzyhsca/BiasAddBiasAddwhile/jczmzyhsca/add:z:0/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/BiasAdd
 while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jczmzyhsca/split/split_dim
while/jczmzyhsca/splitSplit)while/jczmzyhsca/split/split_dim:output:0!while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jczmzyhsca/split©
while/jczmzyhsca/ReadVariableOpReadVariableOp*while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jczmzyhsca/ReadVariableOp£
while/jczmzyhsca/mulMul'while/jczmzyhsca/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul¦
while/jczmzyhsca/add_1AddV2while/jczmzyhsca/split:output:0while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_1
while/jczmzyhsca/SigmoidSigmoidwhile/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid¯
!while/jczmzyhsca/ReadVariableOp_1ReadVariableOp,while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_1©
while/jczmzyhsca/mul_1Mul)while/jczmzyhsca/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_1¨
while/jczmzyhsca/add_2AddV2while/jczmzyhsca/split:output:1while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_2
while/jczmzyhsca/Sigmoid_1Sigmoidwhile/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_1
while/jczmzyhsca/mul_2Mulwhile/jczmzyhsca/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_2
while/jczmzyhsca/TanhTanhwhile/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh¢
while/jczmzyhsca/mul_3Mulwhile/jczmzyhsca/Sigmoid:y:0while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_3£
while/jczmzyhsca/add_3AddV2while/jczmzyhsca/mul_2:z:0while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_3¯
!while/jczmzyhsca/ReadVariableOp_2ReadVariableOp,while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_2°
while/jczmzyhsca/mul_4Mul)while/jczmzyhsca/ReadVariableOp_2:value:0while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_4¨
while/jczmzyhsca/add_4AddV2while/jczmzyhsca/split:output:3while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_4
while/jczmzyhsca/Sigmoid_2Sigmoidwhile/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_2
while/jczmzyhsca/Tanh_1Tanhwhile/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh_1¦
while/jczmzyhsca/mul_5Mulwhile/jczmzyhsca/Sigmoid_2:y:0while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jczmzyhsca/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jczmzyhsca/mul_5:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jczmzyhsca/add_3:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
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
0while_jczmzyhsca_biasadd_readvariableop_resource2while_jczmzyhsca_biasadd_readvariableop_resource_0"h
1while_jczmzyhsca_matmul_1_readvariableop_resource3while_jczmzyhsca_matmul_1_readvariableop_resource_0"d
/while_jczmzyhsca_matmul_readvariableop_resource1while_jczmzyhsca_matmul_readvariableop_resource_0"Z
*while_jczmzyhsca_readvariableop_1_resource,while_jczmzyhsca_readvariableop_1_resource_0"Z
*while_jczmzyhsca_readvariableop_2_resource,while_jczmzyhsca_readvariableop_2_resource_0"V
(while_jczmzyhsca_readvariableop_resource*while_jczmzyhsca_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jczmzyhsca/BiasAdd/ReadVariableOp'while/jczmzyhsca/BiasAdd/ReadVariableOp2P
&while/jczmzyhsca/MatMul/ReadVariableOp&while/jczmzyhsca/MatMul/ReadVariableOp2T
(while/jczmzyhsca/MatMul_1/ReadVariableOp(while/jczmzyhsca/MatMul_1/ReadVariableOp2B
while/jczmzyhsca/ReadVariableOpwhile/jczmzyhsca/ReadVariableOp2F
!while/jczmzyhsca/ReadVariableOp_1!while/jczmzyhsca/ReadVariableOp_12F
!while/jczmzyhsca/ReadVariableOp_2!while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2158628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2158628___redundant_placeholder05
1while_while_cond_2158628___redundant_placeholder15
1while_while_cond_2158628___redundant_placeholder25
1while_while_cond_2158628___redundant_placeholder35
1while_while_cond_2158628___redundant_placeholder45
1while_while_cond_2158628___redundant_placeholder55
1while_while_cond_2158628___redundant_placeholder6
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
,__inference_wdwulgrltk_layer_call_fn_2163062

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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_21585332
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
while_body_2157608
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_jczmzyhsca_2157632_0:	-
while_jczmzyhsca_2157634_0:	 )
while_jczmzyhsca_2157636_0:	(
while_jczmzyhsca_2157638_0: (
while_jczmzyhsca_2157640_0: (
while_jczmzyhsca_2157642_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_jczmzyhsca_2157632:	+
while_jczmzyhsca_2157634:	 '
while_jczmzyhsca_2157636:	&
while_jczmzyhsca_2157638: &
while_jczmzyhsca_2157640: &
while_jczmzyhsca_2157642: ¢(while/jczmzyhsca/StatefulPartitionedCallÃ
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
(while/jczmzyhsca/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_jczmzyhsca_2157632_0while_jczmzyhsca_2157634_0while_jczmzyhsca_2157636_0while_jczmzyhsca_2157638_0while_jczmzyhsca_2157640_0while_jczmzyhsca_2157642_0*
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_21575882*
(while/jczmzyhsca/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/jczmzyhsca/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/jczmzyhsca/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/jczmzyhsca/StatefulPartitionedCall:output:1)^while/jczmzyhsca/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/jczmzyhsca/StatefulPartitionedCall:output:2)^while/jczmzyhsca/StatefulPartitionedCall*
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
while_jczmzyhsca_2157632while_jczmzyhsca_2157632_0"6
while_jczmzyhsca_2157634while_jczmzyhsca_2157634_0"6
while_jczmzyhsca_2157636while_jczmzyhsca_2157636_0"6
while_jczmzyhsca_2157638while_jczmzyhsca_2157638_0"6
while_jczmzyhsca_2157640while_jczmzyhsca_2157640_0"6
while_jczmzyhsca_2157642while_jczmzyhsca_2157642_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/jczmzyhsca/StatefulPartitionedCall(while/jczmzyhsca/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161379
inputs_0<
)jczmzyhsca_matmul_readvariableop_resource:	>
+jczmzyhsca_matmul_1_readvariableop_resource:	 9
*jczmzyhsca_biasadd_readvariableop_resource:	0
"jczmzyhsca_readvariableop_resource: 2
$jczmzyhsca_readvariableop_1_resource: 2
$jczmzyhsca_readvariableop_2_resource: 
identity¢!jczmzyhsca/BiasAdd/ReadVariableOp¢ jczmzyhsca/MatMul/ReadVariableOp¢"jczmzyhsca/MatMul_1/ReadVariableOp¢jczmzyhsca/ReadVariableOp¢jczmzyhsca/ReadVariableOp_1¢jczmzyhsca/ReadVariableOp_2¢whileF
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
 jczmzyhsca/MatMul/ReadVariableOpReadVariableOp)jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jczmzyhsca/MatMul/ReadVariableOp§
jczmzyhsca/MatMulMatMulstrided_slice_2:output:0(jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMulµ
"jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp+jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jczmzyhsca/MatMul_1/ReadVariableOp£
jczmzyhsca/MatMul_1MatMulzeros:output:0*jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMul_1
jczmzyhsca/addAddV2jczmzyhsca/MatMul:product:0jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/add®
!jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp*jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jczmzyhsca/BiasAdd/ReadVariableOp¥
jczmzyhsca/BiasAddBiasAddjczmzyhsca/add:z:0)jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/BiasAddz
jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jczmzyhsca/split/split_dimë
jczmzyhsca/splitSplit#jczmzyhsca/split/split_dim:output:0jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jczmzyhsca/split
jczmzyhsca/ReadVariableOpReadVariableOp"jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp
jczmzyhsca/mulMul!jczmzyhsca/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul
jczmzyhsca/add_1AddV2jczmzyhsca/split:output:0jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_1{
jczmzyhsca/SigmoidSigmoidjczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid
jczmzyhsca/ReadVariableOp_1ReadVariableOp$jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_1
jczmzyhsca/mul_1Mul#jczmzyhsca/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_1
jczmzyhsca/add_2AddV2jczmzyhsca/split:output:1jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_2
jczmzyhsca/Sigmoid_1Sigmoidjczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_1
jczmzyhsca/mul_2Muljczmzyhsca/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_2w
jczmzyhsca/TanhTanhjczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh
jczmzyhsca/mul_3Muljczmzyhsca/Sigmoid:y:0jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_3
jczmzyhsca/add_3AddV2jczmzyhsca/mul_2:z:0jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_3
jczmzyhsca/ReadVariableOp_2ReadVariableOp$jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_2
jczmzyhsca/mul_4Mul#jczmzyhsca/ReadVariableOp_2:value:0jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_4
jczmzyhsca/add_4AddV2jczmzyhsca/split:output:3jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_4
jczmzyhsca/Sigmoid_2Sigmoidjczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_2v
jczmzyhsca/Tanh_1Tanhjczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh_1
jczmzyhsca/mul_5Muljczmzyhsca/Sigmoid_2:y:0jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jczmzyhsca_matmul_readvariableop_resource+jczmzyhsca_matmul_1_readvariableop_resource*jczmzyhsca_biasadd_readvariableop_resource"jczmzyhsca_readvariableop_resource$jczmzyhsca_readvariableop_1_resource$jczmzyhsca_readvariableop_2_resource*
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
while_body_2161278*
condR
while_cond_2161277*Q
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
IdentityIdentitytranspose_1:y:0"^jczmzyhsca/BiasAdd/ReadVariableOp!^jczmzyhsca/MatMul/ReadVariableOp#^jczmzyhsca/MatMul_1/ReadVariableOp^jczmzyhsca/ReadVariableOp^jczmzyhsca/ReadVariableOp_1^jczmzyhsca/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jczmzyhsca/BiasAdd/ReadVariableOp!jczmzyhsca/BiasAdd/ReadVariableOp2D
 jczmzyhsca/MatMul/ReadVariableOp jczmzyhsca/MatMul/ReadVariableOp2H
"jczmzyhsca/MatMul_1/ReadVariableOp"jczmzyhsca/MatMul_1/ReadVariableOp26
jczmzyhsca/ReadVariableOpjczmzyhsca/ReadVariableOp2:
jczmzyhsca/ReadVariableOp_1jczmzyhsca/ReadVariableOp_12:
jczmzyhsca/ReadVariableOp_2jczmzyhsca/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Z
Ë
 __inference__traced_save_2163202
file_prefix0
,savev2_vfwtupxpzf_kernel_read_readvariableop.
*savev2_vfwtupxpzf_bias_read_readvariableop0
,savev2_kekwghyimt_kernel_read_readvariableop.
*savev2_kekwghyimt_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_vhacowjcza_jczmzyhsca_kernel_read_readvariableopE
Asavev2_vhacowjcza_jczmzyhsca_recurrent_kernel_read_readvariableop9
5savev2_vhacowjcza_jczmzyhsca_bias_read_readvariableopP
Lsavev2_vhacowjcza_jczmzyhsca_input_gate_peephole_weights_read_readvariableopQ
Msavev2_vhacowjcza_jczmzyhsca_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_vhacowjcza_jczmzyhsca_output_gate_peephole_weights_read_readvariableop;
7savev2_zgrdiwrovx_wdwulgrltk_kernel_read_readvariableopE
Asavev2_zgrdiwrovx_wdwulgrltk_recurrent_kernel_read_readvariableop9
5savev2_zgrdiwrovx_wdwulgrltk_bias_read_readvariableopP
Lsavev2_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weights_read_readvariableopQ
Msavev2_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_vfwtupxpzf_kernel_rms_read_readvariableop:
6savev2_rmsprop_vfwtupxpzf_bias_rms_read_readvariableop<
8savev2_rmsprop_kekwghyimt_kernel_rms_read_readvariableop:
6savev2_rmsprop_kekwghyimt_bias_rms_read_readvariableopG
Csavev2_rmsprop_vhacowjcza_jczmzyhsca_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_vhacowjcza_jczmzyhsca_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_vhacowjcza_jczmzyhsca_bias_rms_read_readvariableop\
Xsavev2_rmsprop_vhacowjcza_jczmzyhsca_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_vhacowjcza_jczmzyhsca_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_vhacowjcza_jczmzyhsca_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_zgrdiwrovx_wdwulgrltk_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_zgrdiwrovx_wdwulgrltk_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_zgrdiwrovx_wdwulgrltk_bias_rms_read_readvariableop\
Xsavev2_rmsprop_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_vfwtupxpzf_kernel_read_readvariableop*savev2_vfwtupxpzf_bias_read_readvariableop,savev2_kekwghyimt_kernel_read_readvariableop*savev2_kekwghyimt_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_vhacowjcza_jczmzyhsca_kernel_read_readvariableopAsavev2_vhacowjcza_jczmzyhsca_recurrent_kernel_read_readvariableop5savev2_vhacowjcza_jczmzyhsca_bias_read_readvariableopLsavev2_vhacowjcza_jczmzyhsca_input_gate_peephole_weights_read_readvariableopMsavev2_vhacowjcza_jczmzyhsca_forget_gate_peephole_weights_read_readvariableopMsavev2_vhacowjcza_jczmzyhsca_output_gate_peephole_weights_read_readvariableop7savev2_zgrdiwrovx_wdwulgrltk_kernel_read_readvariableopAsavev2_zgrdiwrovx_wdwulgrltk_recurrent_kernel_read_readvariableop5savev2_zgrdiwrovx_wdwulgrltk_bias_read_readvariableopLsavev2_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weights_read_readvariableopMsavev2_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weights_read_readvariableopMsavev2_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_vfwtupxpzf_kernel_rms_read_readvariableop6savev2_rmsprop_vfwtupxpzf_bias_rms_read_readvariableop8savev2_rmsprop_kekwghyimt_kernel_rms_read_readvariableop6savev2_rmsprop_kekwghyimt_bias_rms_read_readvariableopCsavev2_rmsprop_vhacowjcza_jczmzyhsca_kernel_rms_read_readvariableopMsavev2_rmsprop_vhacowjcza_jczmzyhsca_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_vhacowjcza_jczmzyhsca_bias_rms_read_readvariableopXsavev2_rmsprop_vhacowjcza_jczmzyhsca_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_vhacowjcza_jczmzyhsca_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_vhacowjcza_jczmzyhsca_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_zgrdiwrovx_wdwulgrltk_kernel_rms_read_readvariableopMsavev2_rmsprop_zgrdiwrovx_wdwulgrltk_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_zgrdiwrovx_wdwulgrltk_bias_rms_read_readvariableopXsavev2_rmsprop_zgrdiwrovx_wdwulgrltk_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_zgrdiwrovx_wdwulgrltk_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_zgrdiwrovx_wdwulgrltk_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
¥
©	
(sequential_zgrdiwrovx_while_cond_2157393H
Dsequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_loop_counterN
Jsequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_maximum_iterations+
'sequential_zgrdiwrovx_while_placeholder-
)sequential_zgrdiwrovx_while_placeholder_1-
)sequential_zgrdiwrovx_while_placeholder_2-
)sequential_zgrdiwrovx_while_placeholder_3J
Fsequential_zgrdiwrovx_while_less_sequential_zgrdiwrovx_strided_slice_1a
]sequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_cond_2157393___redundant_placeholder0a
]sequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_cond_2157393___redundant_placeholder1a
]sequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_cond_2157393___redundant_placeholder2a
]sequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_cond_2157393___redundant_placeholder3a
]sequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_cond_2157393___redundant_placeholder4a
]sequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_cond_2157393___redundant_placeholder5a
]sequential_zgrdiwrovx_while_sequential_zgrdiwrovx_while_cond_2157393___redundant_placeholder6(
$sequential_zgrdiwrovx_while_identity
Þ
 sequential/zgrdiwrovx/while/LessLess'sequential_zgrdiwrovx_while_placeholderFsequential_zgrdiwrovx_while_less_sequential_zgrdiwrovx_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/zgrdiwrovx/while/Less
$sequential/zgrdiwrovx/while/IdentityIdentity$sequential/zgrdiwrovx/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/zgrdiwrovx/while/Identity"U
$sequential_zgrdiwrovx_while_identity-sequential/zgrdiwrovx/while/Identity:output:0*(
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
,__inference_vhacowjcza_layer_call_fn_2161953
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_21579512
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

À
,__inference_jczmzyhsca_layer_call_fn_2162905

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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_21575882
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

c
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_2159080

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
while_body_2159842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jczmzyhsca_matmul_readvariableop_resource_0:	F
3while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 A
2while_jczmzyhsca_biasadd_readvariableop_resource_0:	8
*while_jczmzyhsca_readvariableop_resource_0: :
,while_jczmzyhsca_readvariableop_1_resource_0: :
,while_jczmzyhsca_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jczmzyhsca_matmul_readvariableop_resource:	D
1while_jczmzyhsca_matmul_1_readvariableop_resource:	 ?
0while_jczmzyhsca_biasadd_readvariableop_resource:	6
(while_jczmzyhsca_readvariableop_resource: 8
*while_jczmzyhsca_readvariableop_1_resource: 8
*while_jczmzyhsca_readvariableop_2_resource: ¢'while/jczmzyhsca/BiasAdd/ReadVariableOp¢&while/jczmzyhsca/MatMul/ReadVariableOp¢(while/jczmzyhsca/MatMul_1/ReadVariableOp¢while/jczmzyhsca/ReadVariableOp¢!while/jczmzyhsca/ReadVariableOp_1¢!while/jczmzyhsca/ReadVariableOp_2Ã
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
&while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp1while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jczmzyhsca/MatMul/ReadVariableOpÑ
while/jczmzyhsca/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMulÉ
(while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp3while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jczmzyhsca/MatMul_1/ReadVariableOpº
while/jczmzyhsca/MatMul_1MatMulwhile_placeholder_20while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMul_1°
while/jczmzyhsca/addAddV2!while/jczmzyhsca/MatMul:product:0#while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/addÂ
'while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp2while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jczmzyhsca/BiasAdd/ReadVariableOp½
while/jczmzyhsca/BiasAddBiasAddwhile/jczmzyhsca/add:z:0/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/BiasAdd
 while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jczmzyhsca/split/split_dim
while/jczmzyhsca/splitSplit)while/jczmzyhsca/split/split_dim:output:0!while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jczmzyhsca/split©
while/jczmzyhsca/ReadVariableOpReadVariableOp*while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jczmzyhsca/ReadVariableOp£
while/jczmzyhsca/mulMul'while/jczmzyhsca/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul¦
while/jczmzyhsca/add_1AddV2while/jczmzyhsca/split:output:0while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_1
while/jczmzyhsca/SigmoidSigmoidwhile/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid¯
!while/jczmzyhsca/ReadVariableOp_1ReadVariableOp,while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_1©
while/jczmzyhsca/mul_1Mul)while/jczmzyhsca/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_1¨
while/jczmzyhsca/add_2AddV2while/jczmzyhsca/split:output:1while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_2
while/jczmzyhsca/Sigmoid_1Sigmoidwhile/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_1
while/jczmzyhsca/mul_2Mulwhile/jczmzyhsca/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_2
while/jczmzyhsca/TanhTanhwhile/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh¢
while/jczmzyhsca/mul_3Mulwhile/jczmzyhsca/Sigmoid:y:0while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_3£
while/jczmzyhsca/add_3AddV2while/jczmzyhsca/mul_2:z:0while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_3¯
!while/jczmzyhsca/ReadVariableOp_2ReadVariableOp,while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_2°
while/jczmzyhsca/mul_4Mul)while/jczmzyhsca/ReadVariableOp_2:value:0while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_4¨
while/jczmzyhsca/add_4AddV2while/jczmzyhsca/split:output:3while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_4
while/jczmzyhsca/Sigmoid_2Sigmoidwhile/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_2
while/jczmzyhsca/Tanh_1Tanhwhile/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh_1¦
while/jczmzyhsca/mul_5Mulwhile/jczmzyhsca/Sigmoid_2:y:0while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jczmzyhsca/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jczmzyhsca/mul_5:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jczmzyhsca/add_3:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
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
0while_jczmzyhsca_biasadd_readvariableop_resource2while_jczmzyhsca_biasadd_readvariableop_resource_0"h
1while_jczmzyhsca_matmul_1_readvariableop_resource3while_jczmzyhsca_matmul_1_readvariableop_resource_0"d
/while_jczmzyhsca_matmul_readvariableop_resource1while_jczmzyhsca_matmul_readvariableop_resource_0"Z
*while_jczmzyhsca_readvariableop_1_resource,while_jczmzyhsca_readvariableop_1_resource_0"Z
*while_jczmzyhsca_readvariableop_2_resource,while_jczmzyhsca_readvariableop_2_resource_0"V
(while_jczmzyhsca_readvariableop_resource*while_jczmzyhsca_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jczmzyhsca/BiasAdd/ReadVariableOp'while/jczmzyhsca/BiasAdd/ReadVariableOp2P
&while/jczmzyhsca/MatMul/ReadVariableOp&while/jczmzyhsca/MatMul/ReadVariableOp2T
(while/jczmzyhsca/MatMul_1/ReadVariableOp(while/jczmzyhsca/MatMul_1/ReadVariableOp2B
while/jczmzyhsca/ReadVariableOpwhile/jczmzyhsca/ReadVariableOp2F
!while/jczmzyhsca/ReadVariableOp_1!while/jczmzyhsca/ReadVariableOp_12F
!while/jczmzyhsca/ReadVariableOp_2!while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_2161194

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
àh

G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161559
inputs_0<
)jczmzyhsca_matmul_readvariableop_resource:	>
+jczmzyhsca_matmul_1_readvariableop_resource:	 9
*jczmzyhsca_biasadd_readvariableop_resource:	0
"jczmzyhsca_readvariableop_resource: 2
$jczmzyhsca_readvariableop_1_resource: 2
$jczmzyhsca_readvariableop_2_resource: 
identity¢!jczmzyhsca/BiasAdd/ReadVariableOp¢ jczmzyhsca/MatMul/ReadVariableOp¢"jczmzyhsca/MatMul_1/ReadVariableOp¢jczmzyhsca/ReadVariableOp¢jczmzyhsca/ReadVariableOp_1¢jczmzyhsca/ReadVariableOp_2¢whileF
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
 jczmzyhsca/MatMul/ReadVariableOpReadVariableOp)jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jczmzyhsca/MatMul/ReadVariableOp§
jczmzyhsca/MatMulMatMulstrided_slice_2:output:0(jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMulµ
"jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp+jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jczmzyhsca/MatMul_1/ReadVariableOp£
jczmzyhsca/MatMul_1MatMulzeros:output:0*jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMul_1
jczmzyhsca/addAddV2jczmzyhsca/MatMul:product:0jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/add®
!jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp*jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jczmzyhsca/BiasAdd/ReadVariableOp¥
jczmzyhsca/BiasAddBiasAddjczmzyhsca/add:z:0)jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/BiasAddz
jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jczmzyhsca/split/split_dimë
jczmzyhsca/splitSplit#jczmzyhsca/split/split_dim:output:0jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jczmzyhsca/split
jczmzyhsca/ReadVariableOpReadVariableOp"jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp
jczmzyhsca/mulMul!jczmzyhsca/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul
jczmzyhsca/add_1AddV2jczmzyhsca/split:output:0jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_1{
jczmzyhsca/SigmoidSigmoidjczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid
jczmzyhsca/ReadVariableOp_1ReadVariableOp$jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_1
jczmzyhsca/mul_1Mul#jczmzyhsca/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_1
jczmzyhsca/add_2AddV2jczmzyhsca/split:output:1jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_2
jczmzyhsca/Sigmoid_1Sigmoidjczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_1
jczmzyhsca/mul_2Muljczmzyhsca/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_2w
jczmzyhsca/TanhTanhjczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh
jczmzyhsca/mul_3Muljczmzyhsca/Sigmoid:y:0jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_3
jczmzyhsca/add_3AddV2jczmzyhsca/mul_2:z:0jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_3
jczmzyhsca/ReadVariableOp_2ReadVariableOp$jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_2
jczmzyhsca/mul_4Mul#jczmzyhsca/ReadVariableOp_2:value:0jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_4
jczmzyhsca/add_4AddV2jczmzyhsca/split:output:3jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_4
jczmzyhsca/Sigmoid_2Sigmoidjczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_2v
jczmzyhsca/Tanh_1Tanhjczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh_1
jczmzyhsca/mul_5Muljczmzyhsca/Sigmoid_2:y:0jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jczmzyhsca_matmul_readvariableop_resource+jczmzyhsca_matmul_1_readvariableop_resource*jczmzyhsca_biasadd_readvariableop_resource"jczmzyhsca_readvariableop_resource$jczmzyhsca_readvariableop_1_resource$jczmzyhsca_readvariableop_2_resource*
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
while_body_2161458*
condR
while_cond_2161457*Q
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
IdentityIdentitytranspose_1:y:0"^jczmzyhsca/BiasAdd/ReadVariableOp!^jczmzyhsca/MatMul/ReadVariableOp#^jczmzyhsca/MatMul_1/ReadVariableOp^jczmzyhsca/ReadVariableOp^jczmzyhsca/ReadVariableOp_1^jczmzyhsca/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jczmzyhsca/BiasAdd/ReadVariableOp!jczmzyhsca/BiasAdd/ReadVariableOp2D
 jczmzyhsca/MatMul/ReadVariableOp jczmzyhsca/MatMul/ReadVariableOp2H
"jczmzyhsca/MatMul_1/ReadVariableOp"jczmzyhsca/MatMul_1/ReadVariableOp26
jczmzyhsca/ReadVariableOpjczmzyhsca/ReadVariableOp2:
jczmzyhsca/ReadVariableOp_1jczmzyhsca/ReadVariableOp_12:
jczmzyhsca/ReadVariableOp_2jczmzyhsca/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
±'
³
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2157588

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
,__inference_jczmzyhsca_layer_call_fn_2162928

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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_21577752
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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_2161061

inputsL
6vfwtupxpzf_conv1d_expanddims_1_readvariableop_resource:K
=vfwtupxpzf_squeeze_batch_dims_biasadd_readvariableop_resource:G
4vhacowjcza_jczmzyhsca_matmul_readvariableop_resource:	I
6vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource:	 D
5vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource:	;
-vhacowjcza_jczmzyhsca_readvariableop_resource: =
/vhacowjcza_jczmzyhsca_readvariableop_1_resource: =
/vhacowjcza_jczmzyhsca_readvariableop_2_resource: G
4zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource:	 I
6zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource:	 D
5zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource:	;
-zgrdiwrovx_wdwulgrltk_readvariableop_resource: =
/zgrdiwrovx_wdwulgrltk_readvariableop_1_resource: =
/zgrdiwrovx_wdwulgrltk_readvariableop_2_resource: ;
)kekwghyimt_matmul_readvariableop_resource: 8
*kekwghyimt_biasadd_readvariableop_resource:
identity¢!kekwghyimt/BiasAdd/ReadVariableOp¢ kekwghyimt/MatMul/ReadVariableOp¢-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp¢4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp¢+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp¢-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp¢$vhacowjcza/jczmzyhsca/ReadVariableOp¢&vhacowjcza/jczmzyhsca/ReadVariableOp_1¢&vhacowjcza/jczmzyhsca/ReadVariableOp_2¢vhacowjcza/while¢,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp¢+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp¢-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp¢$zgrdiwrovx/wdwulgrltk/ReadVariableOp¢&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1¢&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2¢zgrdiwrovx/while
 vfwtupxpzf/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 vfwtupxpzf/conv1d/ExpandDims/dim»
vfwtupxpzf/conv1d/ExpandDims
ExpandDimsinputs)vfwtupxpzf/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/ExpandDimsÙ
-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6vfwtupxpzf_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp
"vfwtupxpzf/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"vfwtupxpzf/conv1d/ExpandDims_1/dimã
vfwtupxpzf/conv1d/ExpandDims_1
ExpandDims5vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp:value:0+vfwtupxpzf/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
vfwtupxpzf/conv1d/ExpandDims_1
vfwtupxpzf/conv1d/ShapeShape%vfwtupxpzf/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
vfwtupxpzf/conv1d/Shape
%vfwtupxpzf/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%vfwtupxpzf/conv1d/strided_slice/stack¥
'vfwtupxpzf/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'vfwtupxpzf/conv1d/strided_slice/stack_1
'vfwtupxpzf/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'vfwtupxpzf/conv1d/strided_slice/stack_2Ì
vfwtupxpzf/conv1d/strided_sliceStridedSlice vfwtupxpzf/conv1d/Shape:output:0.vfwtupxpzf/conv1d/strided_slice/stack:output:00vfwtupxpzf/conv1d/strided_slice/stack_1:output:00vfwtupxpzf/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
vfwtupxpzf/conv1d/strided_slice
vfwtupxpzf/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
vfwtupxpzf/conv1d/Reshape/shapeÌ
vfwtupxpzf/conv1d/ReshapeReshape%vfwtupxpzf/conv1d/ExpandDims:output:0(vfwtupxpzf/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/Reshapeî
vfwtupxpzf/conv1d/Conv2DConv2D"vfwtupxpzf/conv1d/Reshape:output:0'vfwtupxpzf/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
vfwtupxpzf/conv1d/Conv2D
!vfwtupxpzf/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!vfwtupxpzf/conv1d/concat/values_1
vfwtupxpzf/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/concat/axisì
vfwtupxpzf/conv1d/concatConcatV2(vfwtupxpzf/conv1d/strided_slice:output:0*vfwtupxpzf/conv1d/concat/values_1:output:0&vfwtupxpzf/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
vfwtupxpzf/conv1d/concatÉ
vfwtupxpzf/conv1d/Reshape_1Reshape!vfwtupxpzf/conv1d/Conv2D:output:0!vfwtupxpzf/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/Reshape_1Á
vfwtupxpzf/conv1d/SqueezeSqueeze$vfwtupxpzf/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/Squeeze
#vfwtupxpzf/squeeze_batch_dims/ShapeShape"vfwtupxpzf/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#vfwtupxpzf/squeeze_batch_dims/Shape°
1vfwtupxpzf/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1vfwtupxpzf/squeeze_batch_dims/strided_slice/stack½
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1´
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2
+vfwtupxpzf/squeeze_batch_dims/strided_sliceStridedSlice,vfwtupxpzf/squeeze_batch_dims/Shape:output:0:vfwtupxpzf/squeeze_batch_dims/strided_slice/stack:output:0<vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1:output:0<vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+vfwtupxpzf/squeeze_batch_dims/strided_slice¯
+vfwtupxpzf/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+vfwtupxpzf/squeeze_batch_dims/Reshape/shapeé
%vfwtupxpzf/squeeze_batch_dims/ReshapeReshape"vfwtupxpzf/conv1d/Squeeze:output:04vfwtupxpzf/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%vfwtupxpzf/squeeze_batch_dims/Reshapeæ
4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=vfwtupxpzf_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%vfwtupxpzf/squeeze_batch_dims/BiasAddBiasAdd.vfwtupxpzf/squeeze_batch_dims/Reshape:output:0<vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%vfwtupxpzf/squeeze_batch_dims/BiasAdd¯
-vfwtupxpzf/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-vfwtupxpzf/squeeze_batch_dims/concat/values_1¡
)vfwtupxpzf/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)vfwtupxpzf/squeeze_batch_dims/concat/axis¨
$vfwtupxpzf/squeeze_batch_dims/concatConcatV24vfwtupxpzf/squeeze_batch_dims/strided_slice:output:06vfwtupxpzf/squeeze_batch_dims/concat/values_1:output:02vfwtupxpzf/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$vfwtupxpzf/squeeze_batch_dims/concatö
'vfwtupxpzf/squeeze_batch_dims/Reshape_1Reshape.vfwtupxpzf/squeeze_batch_dims/BiasAdd:output:0-vfwtupxpzf/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'vfwtupxpzf/squeeze_batch_dims/Reshape_1
ojzbgzevue/ShapeShape0vfwtupxpzf/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ojzbgzevue/Shape
ojzbgzevue/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ojzbgzevue/strided_slice/stack
 ojzbgzevue/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojzbgzevue/strided_slice/stack_1
 ojzbgzevue/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojzbgzevue/strided_slice/stack_2¤
ojzbgzevue/strided_sliceStridedSliceojzbgzevue/Shape:output:0'ojzbgzevue/strided_slice/stack:output:0)ojzbgzevue/strided_slice/stack_1:output:0)ojzbgzevue/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ojzbgzevue/strided_slicez
ojzbgzevue/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ojzbgzevue/Reshape/shape/1z
ojzbgzevue/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ojzbgzevue/Reshape/shape/2×
ojzbgzevue/Reshape/shapePack!ojzbgzevue/strided_slice:output:0#ojzbgzevue/Reshape/shape/1:output:0#ojzbgzevue/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ojzbgzevue/Reshape/shape¾
ojzbgzevue/ReshapeReshape0vfwtupxpzf/squeeze_batch_dims/Reshape_1:output:0!ojzbgzevue/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ojzbgzevue/Reshapeo
vhacowjcza/ShapeShapeojzbgzevue/Reshape:output:0*
T0*
_output_shapes
:2
vhacowjcza/Shape
vhacowjcza/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
vhacowjcza/strided_slice/stack
 vhacowjcza/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 vhacowjcza/strided_slice/stack_1
 vhacowjcza/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 vhacowjcza/strided_slice/stack_2¤
vhacowjcza/strided_sliceStridedSlicevhacowjcza/Shape:output:0'vhacowjcza/strided_slice/stack:output:0)vhacowjcza/strided_slice/stack_1:output:0)vhacowjcza/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vhacowjcza/strided_slicer
vhacowjcza/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros/mul/y
vhacowjcza/zeros/mulMul!vhacowjcza/strided_slice:output:0vhacowjcza/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros/mulu
vhacowjcza/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vhacowjcza/zeros/Less/y
vhacowjcza/zeros/LessLessvhacowjcza/zeros/mul:z:0 vhacowjcza/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros/Lessx
vhacowjcza/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros/packed/1¯
vhacowjcza/zeros/packedPack!vhacowjcza/strided_slice:output:0"vhacowjcza/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
vhacowjcza/zeros/packedu
vhacowjcza/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vhacowjcza/zeros/Const¡
vhacowjcza/zerosFill vhacowjcza/zeros/packed:output:0vhacowjcza/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/zerosv
vhacowjcza/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros_1/mul/y
vhacowjcza/zeros_1/mulMul!vhacowjcza/strided_slice:output:0!vhacowjcza/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros_1/muly
vhacowjcza/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vhacowjcza/zeros_1/Less/y
vhacowjcza/zeros_1/LessLessvhacowjcza/zeros_1/mul:z:0"vhacowjcza/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros_1/Less|
vhacowjcza/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros_1/packed/1µ
vhacowjcza/zeros_1/packedPack!vhacowjcza/strided_slice:output:0$vhacowjcza/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
vhacowjcza/zeros_1/packedy
vhacowjcza/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vhacowjcza/zeros_1/Const©
vhacowjcza/zeros_1Fill"vhacowjcza/zeros_1/packed:output:0!vhacowjcza/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/zeros_1
vhacowjcza/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vhacowjcza/transpose/perm°
vhacowjcza/transpose	Transposeojzbgzevue/Reshape:output:0"vhacowjcza/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/transposep
vhacowjcza/Shape_1Shapevhacowjcza/transpose:y:0*
T0*
_output_shapes
:2
vhacowjcza/Shape_1
 vhacowjcza/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vhacowjcza/strided_slice_1/stack
"vhacowjcza/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_1/stack_1
"vhacowjcza/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_1/stack_2°
vhacowjcza/strided_slice_1StridedSlicevhacowjcza/Shape_1:output:0)vhacowjcza/strided_slice_1/stack:output:0+vhacowjcza/strided_slice_1/stack_1:output:0+vhacowjcza/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vhacowjcza/strided_slice_1
&vhacowjcza/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&vhacowjcza/TensorArrayV2/element_shapeÞ
vhacowjcza/TensorArrayV2TensorListReserve/vhacowjcza/TensorArrayV2/element_shape:output:0#vhacowjcza/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vhacowjcza/TensorArrayV2Õ
@vhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@vhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2vhacowjcza/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorvhacowjcza/transpose:y:0Ivhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2vhacowjcza/TensorArrayUnstack/TensorListFromTensor
 vhacowjcza/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vhacowjcza/strided_slice_2/stack
"vhacowjcza/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_2/stack_1
"vhacowjcza/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_2/stack_2¾
vhacowjcza/strided_slice_2StridedSlicevhacowjcza/transpose:y:0)vhacowjcza/strided_slice_2/stack:output:0+vhacowjcza/strided_slice_2/stack_1:output:0+vhacowjcza/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
vhacowjcza/strided_slice_2Ð
+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp4vhacowjcza_jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOpÓ
vhacowjcza/jczmzyhsca/MatMulMatMul#vhacowjcza/strided_slice_2:output:03vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/jczmzyhsca/MatMulÖ
-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp6vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOpÏ
vhacowjcza/jczmzyhsca/MatMul_1MatMulvhacowjcza/zeros:output:05vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
vhacowjcza/jczmzyhsca/MatMul_1Ä
vhacowjcza/jczmzyhsca/addAddV2&vhacowjcza/jczmzyhsca/MatMul:product:0(vhacowjcza/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/jczmzyhsca/addÏ
,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp5vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOpÑ
vhacowjcza/jczmzyhsca/BiasAddBiasAddvhacowjcza/jczmzyhsca/add:z:04vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/jczmzyhsca/BiasAdd
%vhacowjcza/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%vhacowjcza/jczmzyhsca/split/split_dim
vhacowjcza/jczmzyhsca/splitSplit.vhacowjcza/jczmzyhsca/split/split_dim:output:0&vhacowjcza/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vhacowjcza/jczmzyhsca/split¶
$vhacowjcza/jczmzyhsca/ReadVariableOpReadVariableOp-vhacowjcza_jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02&
$vhacowjcza/jczmzyhsca/ReadVariableOpº
vhacowjcza/jczmzyhsca/mulMul,vhacowjcza/jczmzyhsca/ReadVariableOp:value:0vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mulº
vhacowjcza/jczmzyhsca/add_1AddV2$vhacowjcza/jczmzyhsca/split:output:0vhacowjcza/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_1
vhacowjcza/jczmzyhsca/SigmoidSigmoidvhacowjcza/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/Sigmoid¼
&vhacowjcza/jczmzyhsca/ReadVariableOp_1ReadVariableOp/vhacowjcza_jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&vhacowjcza/jczmzyhsca/ReadVariableOp_1À
vhacowjcza/jczmzyhsca/mul_1Mul.vhacowjcza/jczmzyhsca/ReadVariableOp_1:value:0vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_1¼
vhacowjcza/jczmzyhsca/add_2AddV2$vhacowjcza/jczmzyhsca/split:output:1vhacowjcza/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_2 
vhacowjcza/jczmzyhsca/Sigmoid_1Sigmoidvhacowjcza/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vhacowjcza/jczmzyhsca/Sigmoid_1µ
vhacowjcza/jczmzyhsca/mul_2Mul#vhacowjcza/jczmzyhsca/Sigmoid_1:y:0vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_2
vhacowjcza/jczmzyhsca/TanhTanh$vhacowjcza/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/Tanh¶
vhacowjcza/jczmzyhsca/mul_3Mul!vhacowjcza/jczmzyhsca/Sigmoid:y:0vhacowjcza/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_3·
vhacowjcza/jczmzyhsca/add_3AddV2vhacowjcza/jczmzyhsca/mul_2:z:0vhacowjcza/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_3¼
&vhacowjcza/jczmzyhsca/ReadVariableOp_2ReadVariableOp/vhacowjcza_jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&vhacowjcza/jczmzyhsca/ReadVariableOp_2Ä
vhacowjcza/jczmzyhsca/mul_4Mul.vhacowjcza/jczmzyhsca/ReadVariableOp_2:value:0vhacowjcza/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_4¼
vhacowjcza/jczmzyhsca/add_4AddV2$vhacowjcza/jczmzyhsca/split:output:3vhacowjcza/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_4 
vhacowjcza/jczmzyhsca/Sigmoid_2Sigmoidvhacowjcza/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vhacowjcza/jczmzyhsca/Sigmoid_2
vhacowjcza/jczmzyhsca/Tanh_1Tanhvhacowjcza/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/Tanh_1º
vhacowjcza/jczmzyhsca/mul_5Mul#vhacowjcza/jczmzyhsca/Sigmoid_2:y:0 vhacowjcza/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_5¥
(vhacowjcza/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(vhacowjcza/TensorArrayV2_1/element_shapeä
vhacowjcza/TensorArrayV2_1TensorListReserve1vhacowjcza/TensorArrayV2_1/element_shape:output:0#vhacowjcza/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vhacowjcza/TensorArrayV2_1d
vhacowjcza/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/time
#vhacowjcza/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#vhacowjcza/while/maximum_iterations
vhacowjcza/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/while/loop_counter²
vhacowjcza/whileWhile&vhacowjcza/while/loop_counter:output:0,vhacowjcza/while/maximum_iterations:output:0vhacowjcza/time:output:0#vhacowjcza/TensorArrayV2_1:handle:0vhacowjcza/zeros:output:0vhacowjcza/zeros_1:output:0#vhacowjcza/strided_slice_1:output:0Bvhacowjcza/TensorArrayUnstack/TensorListFromTensor:output_handle:04vhacowjcza_jczmzyhsca_matmul_readvariableop_resource6vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource5vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource-vhacowjcza_jczmzyhsca_readvariableop_resource/vhacowjcza_jczmzyhsca_readvariableop_1_resource/vhacowjcza_jczmzyhsca_readvariableop_2_resource*
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
vhacowjcza_while_body_2160778*)
cond!R
vhacowjcza_while_cond_2160777*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
vhacowjcza/whileË
;vhacowjcza/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;vhacowjcza/TensorArrayV2Stack/TensorListStack/element_shape
-vhacowjcza/TensorArrayV2Stack/TensorListStackTensorListStackvhacowjcza/while:output:3Dvhacowjcza/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-vhacowjcza/TensorArrayV2Stack/TensorListStack
 vhacowjcza/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 vhacowjcza/strided_slice_3/stack
"vhacowjcza/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"vhacowjcza/strided_slice_3/stack_1
"vhacowjcza/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_3/stack_2Ü
vhacowjcza/strided_slice_3StridedSlice6vhacowjcza/TensorArrayV2Stack/TensorListStack:tensor:0)vhacowjcza/strided_slice_3/stack:output:0+vhacowjcza/strided_slice_3/stack_1:output:0+vhacowjcza/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
vhacowjcza/strided_slice_3
vhacowjcza/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vhacowjcza/transpose_1/permÑ
vhacowjcza/transpose_1	Transpose6vhacowjcza/TensorArrayV2Stack/TensorListStack:tensor:0$vhacowjcza/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/transpose_1n
zgrdiwrovx/ShapeShapevhacowjcza/transpose_1:y:0*
T0*
_output_shapes
:2
zgrdiwrovx/Shape
zgrdiwrovx/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
zgrdiwrovx/strided_slice/stack
 zgrdiwrovx/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 zgrdiwrovx/strided_slice/stack_1
 zgrdiwrovx/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 zgrdiwrovx/strided_slice/stack_2¤
zgrdiwrovx/strided_sliceStridedSlicezgrdiwrovx/Shape:output:0'zgrdiwrovx/strided_slice/stack:output:0)zgrdiwrovx/strided_slice/stack_1:output:0)zgrdiwrovx/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zgrdiwrovx/strided_slicer
zgrdiwrovx/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros/mul/y
zgrdiwrovx/zeros/mulMul!zgrdiwrovx/strided_slice:output:0zgrdiwrovx/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros/mulu
zgrdiwrovx/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zgrdiwrovx/zeros/Less/y
zgrdiwrovx/zeros/LessLesszgrdiwrovx/zeros/mul:z:0 zgrdiwrovx/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros/Lessx
zgrdiwrovx/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros/packed/1¯
zgrdiwrovx/zeros/packedPack!zgrdiwrovx/strided_slice:output:0"zgrdiwrovx/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zgrdiwrovx/zeros/packedu
zgrdiwrovx/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zgrdiwrovx/zeros/Const¡
zgrdiwrovx/zerosFill zgrdiwrovx/zeros/packed:output:0zgrdiwrovx/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/zerosv
zgrdiwrovx/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros_1/mul/y
zgrdiwrovx/zeros_1/mulMul!zgrdiwrovx/strided_slice:output:0!zgrdiwrovx/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros_1/muly
zgrdiwrovx/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zgrdiwrovx/zeros_1/Less/y
zgrdiwrovx/zeros_1/LessLesszgrdiwrovx/zeros_1/mul:z:0"zgrdiwrovx/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros_1/Less|
zgrdiwrovx/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros_1/packed/1µ
zgrdiwrovx/zeros_1/packedPack!zgrdiwrovx/strided_slice:output:0$zgrdiwrovx/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zgrdiwrovx/zeros_1/packedy
zgrdiwrovx/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zgrdiwrovx/zeros_1/Const©
zgrdiwrovx/zeros_1Fill"zgrdiwrovx/zeros_1/packed:output:0!zgrdiwrovx/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/zeros_1
zgrdiwrovx/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zgrdiwrovx/transpose/perm¯
zgrdiwrovx/transpose	Transposevhacowjcza/transpose_1:y:0"zgrdiwrovx/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/transposep
zgrdiwrovx/Shape_1Shapezgrdiwrovx/transpose:y:0*
T0*
_output_shapes
:2
zgrdiwrovx/Shape_1
 zgrdiwrovx/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zgrdiwrovx/strided_slice_1/stack
"zgrdiwrovx/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_1/stack_1
"zgrdiwrovx/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_1/stack_2°
zgrdiwrovx/strided_slice_1StridedSlicezgrdiwrovx/Shape_1:output:0)zgrdiwrovx/strided_slice_1/stack:output:0+zgrdiwrovx/strided_slice_1/stack_1:output:0+zgrdiwrovx/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zgrdiwrovx/strided_slice_1
&zgrdiwrovx/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&zgrdiwrovx/TensorArrayV2/element_shapeÞ
zgrdiwrovx/TensorArrayV2TensorListReserve/zgrdiwrovx/TensorArrayV2/element_shape:output:0#zgrdiwrovx/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zgrdiwrovx/TensorArrayV2Õ
@zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2zgrdiwrovx/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorzgrdiwrovx/transpose:y:0Izgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor
 zgrdiwrovx/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zgrdiwrovx/strided_slice_2/stack
"zgrdiwrovx/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_2/stack_1
"zgrdiwrovx/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_2/stack_2¾
zgrdiwrovx/strided_slice_2StridedSlicezgrdiwrovx/transpose:y:0)zgrdiwrovx/strided_slice_2/stack:output:0+zgrdiwrovx/strided_slice_2/stack_1:output:0+zgrdiwrovx/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
zgrdiwrovx/strided_slice_2Ð
+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp4zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOpÓ
zgrdiwrovx/wdwulgrltk/MatMulMatMul#zgrdiwrovx/strided_slice_2:output:03zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zgrdiwrovx/wdwulgrltk/MatMulÖ
-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp6zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOpÏ
zgrdiwrovx/wdwulgrltk/MatMul_1MatMulzgrdiwrovx/zeros:output:05zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
zgrdiwrovx/wdwulgrltk/MatMul_1Ä
zgrdiwrovx/wdwulgrltk/addAddV2&zgrdiwrovx/wdwulgrltk/MatMul:product:0(zgrdiwrovx/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zgrdiwrovx/wdwulgrltk/addÏ
,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp5zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOpÑ
zgrdiwrovx/wdwulgrltk/BiasAddBiasAddzgrdiwrovx/wdwulgrltk/add:z:04zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zgrdiwrovx/wdwulgrltk/BiasAdd
%zgrdiwrovx/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%zgrdiwrovx/wdwulgrltk/split/split_dim
zgrdiwrovx/wdwulgrltk/splitSplit.zgrdiwrovx/wdwulgrltk/split/split_dim:output:0&zgrdiwrovx/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zgrdiwrovx/wdwulgrltk/split¶
$zgrdiwrovx/wdwulgrltk/ReadVariableOpReadVariableOp-zgrdiwrovx_wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02&
$zgrdiwrovx/wdwulgrltk/ReadVariableOpº
zgrdiwrovx/wdwulgrltk/mulMul,zgrdiwrovx/wdwulgrltk/ReadVariableOp:value:0zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mulº
zgrdiwrovx/wdwulgrltk/add_1AddV2$zgrdiwrovx/wdwulgrltk/split:output:0zgrdiwrovx/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_1
zgrdiwrovx/wdwulgrltk/SigmoidSigmoidzgrdiwrovx/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/Sigmoid¼
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1ReadVariableOp/zgrdiwrovx_wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1À
zgrdiwrovx/wdwulgrltk/mul_1Mul.zgrdiwrovx/wdwulgrltk/ReadVariableOp_1:value:0zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_1¼
zgrdiwrovx/wdwulgrltk/add_2AddV2$zgrdiwrovx/wdwulgrltk/split:output:1zgrdiwrovx/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_2 
zgrdiwrovx/wdwulgrltk/Sigmoid_1Sigmoidzgrdiwrovx/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zgrdiwrovx/wdwulgrltk/Sigmoid_1µ
zgrdiwrovx/wdwulgrltk/mul_2Mul#zgrdiwrovx/wdwulgrltk/Sigmoid_1:y:0zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_2
zgrdiwrovx/wdwulgrltk/TanhTanh$zgrdiwrovx/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/Tanh¶
zgrdiwrovx/wdwulgrltk/mul_3Mul!zgrdiwrovx/wdwulgrltk/Sigmoid:y:0zgrdiwrovx/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_3·
zgrdiwrovx/wdwulgrltk/add_3AddV2zgrdiwrovx/wdwulgrltk/mul_2:z:0zgrdiwrovx/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_3¼
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2ReadVariableOp/zgrdiwrovx_wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2Ä
zgrdiwrovx/wdwulgrltk/mul_4Mul.zgrdiwrovx/wdwulgrltk/ReadVariableOp_2:value:0zgrdiwrovx/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_4¼
zgrdiwrovx/wdwulgrltk/add_4AddV2$zgrdiwrovx/wdwulgrltk/split:output:3zgrdiwrovx/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_4 
zgrdiwrovx/wdwulgrltk/Sigmoid_2Sigmoidzgrdiwrovx/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zgrdiwrovx/wdwulgrltk/Sigmoid_2
zgrdiwrovx/wdwulgrltk/Tanh_1Tanhzgrdiwrovx/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/Tanh_1º
zgrdiwrovx/wdwulgrltk/mul_5Mul#zgrdiwrovx/wdwulgrltk/Sigmoid_2:y:0 zgrdiwrovx/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_5¥
(zgrdiwrovx/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(zgrdiwrovx/TensorArrayV2_1/element_shapeä
zgrdiwrovx/TensorArrayV2_1TensorListReserve1zgrdiwrovx/TensorArrayV2_1/element_shape:output:0#zgrdiwrovx/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zgrdiwrovx/TensorArrayV2_1d
zgrdiwrovx/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/time
#zgrdiwrovx/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#zgrdiwrovx/while/maximum_iterations
zgrdiwrovx/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/while/loop_counter²
zgrdiwrovx/whileWhile&zgrdiwrovx/while/loop_counter:output:0,zgrdiwrovx/while/maximum_iterations:output:0zgrdiwrovx/time:output:0#zgrdiwrovx/TensorArrayV2_1:handle:0zgrdiwrovx/zeros:output:0zgrdiwrovx/zeros_1:output:0#zgrdiwrovx/strided_slice_1:output:0Bzgrdiwrovx/TensorArrayUnstack/TensorListFromTensor:output_handle:04zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource6zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource5zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource-zgrdiwrovx_wdwulgrltk_readvariableop_resource/zgrdiwrovx_wdwulgrltk_readvariableop_1_resource/zgrdiwrovx_wdwulgrltk_readvariableop_2_resource*
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
zgrdiwrovx_while_body_2160954*)
cond!R
zgrdiwrovx_while_cond_2160953*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
zgrdiwrovx/whileË
;zgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;zgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shape
-zgrdiwrovx/TensorArrayV2Stack/TensorListStackTensorListStackzgrdiwrovx/while:output:3Dzgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-zgrdiwrovx/TensorArrayV2Stack/TensorListStack
 zgrdiwrovx/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 zgrdiwrovx/strided_slice_3/stack
"zgrdiwrovx/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"zgrdiwrovx/strided_slice_3/stack_1
"zgrdiwrovx/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_3/stack_2Ü
zgrdiwrovx/strided_slice_3StridedSlice6zgrdiwrovx/TensorArrayV2Stack/TensorListStack:tensor:0)zgrdiwrovx/strided_slice_3/stack:output:0+zgrdiwrovx/strided_slice_3/stack_1:output:0+zgrdiwrovx/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
zgrdiwrovx/strided_slice_3
zgrdiwrovx/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zgrdiwrovx/transpose_1/permÑ
zgrdiwrovx/transpose_1	Transpose6zgrdiwrovx/TensorArrayV2Stack/TensorListStack:tensor:0$zgrdiwrovx/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/transpose_1®
 kekwghyimt/MatMul/ReadVariableOpReadVariableOp)kekwghyimt_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 kekwghyimt/MatMul/ReadVariableOp±
kekwghyimt/MatMulMatMul#zgrdiwrovx/strided_slice_3:output:0(kekwghyimt/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kekwghyimt/MatMul­
!kekwghyimt/BiasAdd/ReadVariableOpReadVariableOp*kekwghyimt_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!kekwghyimt/BiasAdd/ReadVariableOp­
kekwghyimt/BiasAddBiasAddkekwghyimt/MatMul:product:0)kekwghyimt/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kekwghyimt/BiasAddÏ
IdentityIdentitykekwghyimt/BiasAdd:output:0"^kekwghyimt/BiasAdd/ReadVariableOp!^kekwghyimt/MatMul/ReadVariableOp.^vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp5^vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp-^vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp,^vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp.^vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp%^vhacowjcza/jczmzyhsca/ReadVariableOp'^vhacowjcza/jczmzyhsca/ReadVariableOp_1'^vhacowjcza/jczmzyhsca/ReadVariableOp_2^vhacowjcza/while-^zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp,^zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp.^zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp%^zgrdiwrovx/wdwulgrltk/ReadVariableOp'^zgrdiwrovx/wdwulgrltk/ReadVariableOp_1'^zgrdiwrovx/wdwulgrltk/ReadVariableOp_2^zgrdiwrovx/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2F
!kekwghyimt/BiasAdd/ReadVariableOp!kekwghyimt/BiasAdd/ReadVariableOp2D
 kekwghyimt/MatMul/ReadVariableOp kekwghyimt/MatMul/ReadVariableOp2^
-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp2l
4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp2Z
+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp2^
-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp2L
$vhacowjcza/jczmzyhsca/ReadVariableOp$vhacowjcza/jczmzyhsca/ReadVariableOp2P
&vhacowjcza/jczmzyhsca/ReadVariableOp_1&vhacowjcza/jczmzyhsca/ReadVariableOp_12P
&vhacowjcza/jczmzyhsca/ReadVariableOp_2&vhacowjcza/jczmzyhsca/ReadVariableOp_22$
vhacowjcza/whilevhacowjcza/while2\
,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp2Z
+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp2^
-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp2L
$zgrdiwrovx/wdwulgrltk/ReadVariableOp$zgrdiwrovx/wdwulgrltk/ReadVariableOp2P
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1&zgrdiwrovx/wdwulgrltk/ReadVariableOp_12P
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2&zgrdiwrovx/wdwulgrltk/ReadVariableOp_22$
zgrdiwrovx/whilezgrdiwrovx/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç)
Ò
while_body_2158366
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_wdwulgrltk_2158390_0:	 -
while_wdwulgrltk_2158392_0:	 )
while_wdwulgrltk_2158394_0:	(
while_wdwulgrltk_2158396_0: (
while_wdwulgrltk_2158398_0: (
while_wdwulgrltk_2158400_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_wdwulgrltk_2158390:	 +
while_wdwulgrltk_2158392:	 '
while_wdwulgrltk_2158394:	&
while_wdwulgrltk_2158396: &
while_wdwulgrltk_2158398: &
while_wdwulgrltk_2158400: ¢(while/wdwulgrltk/StatefulPartitionedCallÃ
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
(while/wdwulgrltk/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_wdwulgrltk_2158390_0while_wdwulgrltk_2158392_0while_wdwulgrltk_2158394_0while_wdwulgrltk_2158396_0while_wdwulgrltk_2158398_0while_wdwulgrltk_2158400_0*
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_21583462*
(while/wdwulgrltk/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/wdwulgrltk/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/wdwulgrltk/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/wdwulgrltk/StatefulPartitionedCall:output:1)^while/wdwulgrltk/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/wdwulgrltk/StatefulPartitionedCall:output:2)^while/wdwulgrltk/StatefulPartitionedCall*
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
while_wdwulgrltk_2158390while_wdwulgrltk_2158390_0"6
while_wdwulgrltk_2158392while_wdwulgrltk_2158392_0"6
while_wdwulgrltk_2158394while_wdwulgrltk_2158394_0"6
while_wdwulgrltk_2158396while_wdwulgrltk_2158396_0"6
while_wdwulgrltk_2158398while_wdwulgrltk_2158398_0"6
while_wdwulgrltk_2158400while_wdwulgrltk_2158400_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/wdwulgrltk/StatefulPartitionedCall(while/wdwulgrltk/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
(sequential_vhacowjcza_while_cond_2157217H
Dsequential_vhacowjcza_while_sequential_vhacowjcza_while_loop_counterN
Jsequential_vhacowjcza_while_sequential_vhacowjcza_while_maximum_iterations+
'sequential_vhacowjcza_while_placeholder-
)sequential_vhacowjcza_while_placeholder_1-
)sequential_vhacowjcza_while_placeholder_2-
)sequential_vhacowjcza_while_placeholder_3J
Fsequential_vhacowjcza_while_less_sequential_vhacowjcza_strided_slice_1a
]sequential_vhacowjcza_while_sequential_vhacowjcza_while_cond_2157217___redundant_placeholder0a
]sequential_vhacowjcza_while_sequential_vhacowjcza_while_cond_2157217___redundant_placeholder1a
]sequential_vhacowjcza_while_sequential_vhacowjcza_while_cond_2157217___redundant_placeholder2a
]sequential_vhacowjcza_while_sequential_vhacowjcza_while_cond_2157217___redundant_placeholder3a
]sequential_vhacowjcza_while_sequential_vhacowjcza_while_cond_2157217___redundant_placeholder4a
]sequential_vhacowjcza_while_sequential_vhacowjcza_while_cond_2157217___redundant_placeholder5a
]sequential_vhacowjcza_while_sequential_vhacowjcza_while_cond_2157217___redundant_placeholder6(
$sequential_vhacowjcza_while_identity
Þ
 sequential/vhacowjcza/while/LessLess'sequential_vhacowjcza_while_placeholderFsequential_vhacowjcza_while_less_sequential_vhacowjcza_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/vhacowjcza/while/Less
$sequential/vhacowjcza/while/IdentityIdentity$sequential/vhacowjcza/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/vhacowjcza/while/Identity"U
$sequential_vhacowjcza_while_identity-sequential/vhacowjcza/while/Identity:output:0*(
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
zgrdiwrovx_while_body_21609542
.zgrdiwrovx_while_zgrdiwrovx_while_loop_counter8
4zgrdiwrovx_while_zgrdiwrovx_while_maximum_iterations 
zgrdiwrovx_while_placeholder"
zgrdiwrovx_while_placeholder_1"
zgrdiwrovx_while_placeholder_2"
zgrdiwrovx_while_placeholder_31
-zgrdiwrovx_while_zgrdiwrovx_strided_slice_1_0m
izgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0O
<zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0:	 Q
>zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 L
=zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0:	C
5zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0: E
7zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0: E
7zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0: 
zgrdiwrovx_while_identity
zgrdiwrovx_while_identity_1
zgrdiwrovx_while_identity_2
zgrdiwrovx_while_identity_3
zgrdiwrovx_while_identity_4
zgrdiwrovx_while_identity_5/
+zgrdiwrovx_while_zgrdiwrovx_strided_slice_1k
gzgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensorM
:zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource:	 O
<zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource:	 J
;zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource:	A
3zgrdiwrovx_while_wdwulgrltk_readvariableop_resource: C
5zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource: C
5zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource: ¢2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp¢1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp¢3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp¢*zgrdiwrovx/while/wdwulgrltk/ReadVariableOp¢,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1¢,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2Ù
Bzgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bzgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shape
4zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemizgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0zgrdiwrovx_while_placeholderKzgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItemä
1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp<zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOpý
"zgrdiwrovx/while/wdwulgrltk/MatMulMatMul;zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem:item:09zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"zgrdiwrovx/while/wdwulgrltk/MatMulê
3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp>zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOpæ
$zgrdiwrovx/while/wdwulgrltk/MatMul_1MatMulzgrdiwrovx_while_placeholder_2;zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$zgrdiwrovx/while/wdwulgrltk/MatMul_1Ü
zgrdiwrovx/while/wdwulgrltk/addAddV2,zgrdiwrovx/while/wdwulgrltk/MatMul:product:0.zgrdiwrovx/while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
zgrdiwrovx/while/wdwulgrltk/addã
2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp=zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOpé
#zgrdiwrovx/while/wdwulgrltk/BiasAddBiasAdd#zgrdiwrovx/while/wdwulgrltk/add:z:0:zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#zgrdiwrovx/while/wdwulgrltk/BiasAdd
+zgrdiwrovx/while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+zgrdiwrovx/while/wdwulgrltk/split/split_dim¯
!zgrdiwrovx/while/wdwulgrltk/splitSplit4zgrdiwrovx/while/wdwulgrltk/split/split_dim:output:0,zgrdiwrovx/while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!zgrdiwrovx/while/wdwulgrltk/splitÊ
*zgrdiwrovx/while/wdwulgrltk/ReadVariableOpReadVariableOp5zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*zgrdiwrovx/while/wdwulgrltk/ReadVariableOpÏ
zgrdiwrovx/while/wdwulgrltk/mulMul2zgrdiwrovx/while/wdwulgrltk/ReadVariableOp:value:0zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zgrdiwrovx/while/wdwulgrltk/mulÒ
!zgrdiwrovx/while/wdwulgrltk/add_1AddV2*zgrdiwrovx/while/wdwulgrltk/split:output:0#zgrdiwrovx/while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_1®
#zgrdiwrovx/while/wdwulgrltk/SigmoidSigmoid%zgrdiwrovx/while/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#zgrdiwrovx/while/wdwulgrltk/SigmoidÐ
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1ReadVariableOp7zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1Õ
!zgrdiwrovx/while/wdwulgrltk/mul_1Mul4zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1:value:0zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_1Ô
!zgrdiwrovx/while/wdwulgrltk/add_2AddV2*zgrdiwrovx/while/wdwulgrltk/split:output:1%zgrdiwrovx/while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_2²
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_1Sigmoid%zgrdiwrovx/while/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_1Ê
!zgrdiwrovx/while/wdwulgrltk/mul_2Mul)zgrdiwrovx/while/wdwulgrltk/Sigmoid_1:y:0zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_2ª
 zgrdiwrovx/while/wdwulgrltk/TanhTanh*zgrdiwrovx/while/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 zgrdiwrovx/while/wdwulgrltk/TanhÎ
!zgrdiwrovx/while/wdwulgrltk/mul_3Mul'zgrdiwrovx/while/wdwulgrltk/Sigmoid:y:0$zgrdiwrovx/while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_3Ï
!zgrdiwrovx/while/wdwulgrltk/add_3AddV2%zgrdiwrovx/while/wdwulgrltk/mul_2:z:0%zgrdiwrovx/while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_3Ð
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2ReadVariableOp7zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2Ü
!zgrdiwrovx/while/wdwulgrltk/mul_4Mul4zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2:value:0%zgrdiwrovx/while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_4Ô
!zgrdiwrovx/while/wdwulgrltk/add_4AddV2*zgrdiwrovx/while/wdwulgrltk/split:output:3%zgrdiwrovx/while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_4²
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_2Sigmoid%zgrdiwrovx/while/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_2©
"zgrdiwrovx/while/wdwulgrltk/Tanh_1Tanh%zgrdiwrovx/while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"zgrdiwrovx/while/wdwulgrltk/Tanh_1Ò
!zgrdiwrovx/while/wdwulgrltk/mul_5Mul)zgrdiwrovx/while/wdwulgrltk/Sigmoid_2:y:0&zgrdiwrovx/while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_5
5zgrdiwrovx/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemzgrdiwrovx_while_placeholder_1zgrdiwrovx_while_placeholder%zgrdiwrovx/while/wdwulgrltk/mul_5:z:0*
_output_shapes
: *
element_dtype027
5zgrdiwrovx/while/TensorArrayV2Write/TensorListSetItemr
zgrdiwrovx/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
zgrdiwrovx/while/add/y
zgrdiwrovx/while/addAddV2zgrdiwrovx_while_placeholderzgrdiwrovx/while/add/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/while/addv
zgrdiwrovx/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
zgrdiwrovx/while/add_1/y­
zgrdiwrovx/while/add_1AddV2.zgrdiwrovx_while_zgrdiwrovx_while_loop_counter!zgrdiwrovx/while/add_1/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/while/add_1©
zgrdiwrovx/while/IdentityIdentityzgrdiwrovx/while/add_1:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/IdentityÇ
zgrdiwrovx/while/Identity_1Identity4zgrdiwrovx_while_zgrdiwrovx_while_maximum_iterations3^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Identity_1«
zgrdiwrovx/while/Identity_2Identityzgrdiwrovx/while/add:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Identity_2Ø
zgrdiwrovx/while/Identity_3IdentityEzgrdiwrovx/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Identity_3É
zgrdiwrovx/while/Identity_4Identity%zgrdiwrovx/while/wdwulgrltk/mul_5:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/while/Identity_4É
zgrdiwrovx/while/Identity_5Identity%zgrdiwrovx/while/wdwulgrltk/add_3:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/while/Identity_5"?
zgrdiwrovx_while_identity"zgrdiwrovx/while/Identity:output:0"C
zgrdiwrovx_while_identity_1$zgrdiwrovx/while/Identity_1:output:0"C
zgrdiwrovx_while_identity_2$zgrdiwrovx/while/Identity_2:output:0"C
zgrdiwrovx_while_identity_3$zgrdiwrovx/while/Identity_3:output:0"C
zgrdiwrovx_while_identity_4$zgrdiwrovx/while/Identity_4:output:0"C
zgrdiwrovx_while_identity_5$zgrdiwrovx/while/Identity_5:output:0"Ô
gzgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensorizgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0"|
;zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource=zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0"~
<zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource>zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0"z
:zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource<zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0"p
5zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource7zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0"p
5zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource7zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0"l
3zgrdiwrovx_while_wdwulgrltk_readvariableop_resource5zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0"\
+zgrdiwrovx_while_zgrdiwrovx_strided_slice_1-zgrdiwrovx_while_zgrdiwrovx_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2f
1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp2j
3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp2X
*zgrdiwrovx/while/wdwulgrltk/ReadVariableOp*zgrdiwrovx/while/wdwulgrltk/ReadVariableOp2\
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_12\
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2162065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2162065___redundant_placeholder05
1while_while_cond_2162065___redundant_placeholder15
1while_while_cond_2162065___redundant_placeholder25
1while_while_cond_2162065___redundant_placeholder35
1while_while_cond_2162065___redundant_placeholder45
1while_while_cond_2162065___redundant_placeholder55
1while_while_cond_2162065___redundant_placeholder6
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
while_body_2162426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_wdwulgrltk_matmul_readvariableop_resource_0:	 F
3while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 A
2while_wdwulgrltk_biasadd_readvariableop_resource_0:	8
*while_wdwulgrltk_readvariableop_resource_0: :
,while_wdwulgrltk_readvariableop_1_resource_0: :
,while_wdwulgrltk_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_wdwulgrltk_matmul_readvariableop_resource:	 D
1while_wdwulgrltk_matmul_1_readvariableop_resource:	 ?
0while_wdwulgrltk_biasadd_readvariableop_resource:	6
(while_wdwulgrltk_readvariableop_resource: 8
*while_wdwulgrltk_readvariableop_1_resource: 8
*while_wdwulgrltk_readvariableop_2_resource: ¢'while/wdwulgrltk/BiasAdd/ReadVariableOp¢&while/wdwulgrltk/MatMul/ReadVariableOp¢(while/wdwulgrltk/MatMul_1/ReadVariableOp¢while/wdwulgrltk/ReadVariableOp¢!while/wdwulgrltk/ReadVariableOp_1¢!while/wdwulgrltk/ReadVariableOp_2Ã
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
&while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp1while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/wdwulgrltk/MatMul/ReadVariableOpÑ
while/wdwulgrltk/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMulÉ
(while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp3while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/wdwulgrltk/MatMul_1/ReadVariableOpº
while/wdwulgrltk/MatMul_1MatMulwhile_placeholder_20while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMul_1°
while/wdwulgrltk/addAddV2!while/wdwulgrltk/MatMul:product:0#while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/addÂ
'while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp2while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/wdwulgrltk/BiasAdd/ReadVariableOp½
while/wdwulgrltk/BiasAddBiasAddwhile/wdwulgrltk/add:z:0/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/BiasAdd
 while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/wdwulgrltk/split/split_dim
while/wdwulgrltk/splitSplit)while/wdwulgrltk/split/split_dim:output:0!while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/wdwulgrltk/split©
while/wdwulgrltk/ReadVariableOpReadVariableOp*while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/wdwulgrltk/ReadVariableOp£
while/wdwulgrltk/mulMul'while/wdwulgrltk/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul¦
while/wdwulgrltk/add_1AddV2while/wdwulgrltk/split:output:0while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_1
while/wdwulgrltk/SigmoidSigmoidwhile/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid¯
!while/wdwulgrltk/ReadVariableOp_1ReadVariableOp,while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_1©
while/wdwulgrltk/mul_1Mul)while/wdwulgrltk/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_1¨
while/wdwulgrltk/add_2AddV2while/wdwulgrltk/split:output:1while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_2
while/wdwulgrltk/Sigmoid_1Sigmoidwhile/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_1
while/wdwulgrltk/mul_2Mulwhile/wdwulgrltk/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_2
while/wdwulgrltk/TanhTanhwhile/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh¢
while/wdwulgrltk/mul_3Mulwhile/wdwulgrltk/Sigmoid:y:0while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_3£
while/wdwulgrltk/add_3AddV2while/wdwulgrltk/mul_2:z:0while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_3¯
!while/wdwulgrltk/ReadVariableOp_2ReadVariableOp,while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_2°
while/wdwulgrltk/mul_4Mul)while/wdwulgrltk/ReadVariableOp_2:value:0while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_4¨
while/wdwulgrltk/add_4AddV2while/wdwulgrltk/split:output:3while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_4
while/wdwulgrltk/Sigmoid_2Sigmoidwhile/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_2
while/wdwulgrltk/Tanh_1Tanhwhile/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh_1¦
while/wdwulgrltk/mul_5Mulwhile/wdwulgrltk/Sigmoid_2:y:0while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/wdwulgrltk/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/wdwulgrltk/mul_5:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/wdwulgrltk/add_3:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
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
0while_wdwulgrltk_biasadd_readvariableop_resource2while_wdwulgrltk_biasadd_readvariableop_resource_0"h
1while_wdwulgrltk_matmul_1_readvariableop_resource3while_wdwulgrltk_matmul_1_readvariableop_resource_0"d
/while_wdwulgrltk_matmul_readvariableop_resource1while_wdwulgrltk_matmul_readvariableop_resource_0"Z
*while_wdwulgrltk_readvariableop_1_resource,while_wdwulgrltk_readvariableop_1_resource_0"Z
*while_wdwulgrltk_readvariableop_2_resource,while_wdwulgrltk_readvariableop_2_resource_0"V
(while_wdwulgrltk_readvariableop_resource*while_wdwulgrltk_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/wdwulgrltk/BiasAdd/ReadVariableOp'while/wdwulgrltk/BiasAdd/ReadVariableOp2P
&while/wdwulgrltk/MatMul/ReadVariableOp&while/wdwulgrltk/MatMul/ReadVariableOp2T
(while/wdwulgrltk/MatMul_1/ReadVariableOp(while/wdwulgrltk/MatMul_1/ReadVariableOp2B
while/wdwulgrltk/ReadVariableOpwhile/wdwulgrltk/ReadVariableOp2F
!while/wdwulgrltk/ReadVariableOp_1!while/wdwulgrltk/ReadVariableOp_12F
!while/wdwulgrltk/ReadVariableOp_2!while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_sequential_layer_call_fn_2161098

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
G__inference_sequential_layer_call_and_return_conditional_losses_21594852
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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_2160657

inputsL
6vfwtupxpzf_conv1d_expanddims_1_readvariableop_resource:K
=vfwtupxpzf_squeeze_batch_dims_biasadd_readvariableop_resource:G
4vhacowjcza_jczmzyhsca_matmul_readvariableop_resource:	I
6vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource:	 D
5vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource:	;
-vhacowjcza_jczmzyhsca_readvariableop_resource: =
/vhacowjcza_jczmzyhsca_readvariableop_1_resource: =
/vhacowjcza_jczmzyhsca_readvariableop_2_resource: G
4zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource:	 I
6zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource:	 D
5zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource:	;
-zgrdiwrovx_wdwulgrltk_readvariableop_resource: =
/zgrdiwrovx_wdwulgrltk_readvariableop_1_resource: =
/zgrdiwrovx_wdwulgrltk_readvariableop_2_resource: ;
)kekwghyimt_matmul_readvariableop_resource: 8
*kekwghyimt_biasadd_readvariableop_resource:
identity¢!kekwghyimt/BiasAdd/ReadVariableOp¢ kekwghyimt/MatMul/ReadVariableOp¢-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp¢4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp¢+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp¢-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp¢$vhacowjcza/jczmzyhsca/ReadVariableOp¢&vhacowjcza/jczmzyhsca/ReadVariableOp_1¢&vhacowjcza/jczmzyhsca/ReadVariableOp_2¢vhacowjcza/while¢,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp¢+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp¢-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp¢$zgrdiwrovx/wdwulgrltk/ReadVariableOp¢&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1¢&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2¢zgrdiwrovx/while
 vfwtupxpzf/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 vfwtupxpzf/conv1d/ExpandDims/dim»
vfwtupxpzf/conv1d/ExpandDims
ExpandDimsinputs)vfwtupxpzf/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/ExpandDimsÙ
-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6vfwtupxpzf_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp
"vfwtupxpzf/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"vfwtupxpzf/conv1d/ExpandDims_1/dimã
vfwtupxpzf/conv1d/ExpandDims_1
ExpandDims5vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp:value:0+vfwtupxpzf/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
vfwtupxpzf/conv1d/ExpandDims_1
vfwtupxpzf/conv1d/ShapeShape%vfwtupxpzf/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
vfwtupxpzf/conv1d/Shape
%vfwtupxpzf/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%vfwtupxpzf/conv1d/strided_slice/stack¥
'vfwtupxpzf/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'vfwtupxpzf/conv1d/strided_slice/stack_1
'vfwtupxpzf/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'vfwtupxpzf/conv1d/strided_slice/stack_2Ì
vfwtupxpzf/conv1d/strided_sliceStridedSlice vfwtupxpzf/conv1d/Shape:output:0.vfwtupxpzf/conv1d/strided_slice/stack:output:00vfwtupxpzf/conv1d/strided_slice/stack_1:output:00vfwtupxpzf/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
vfwtupxpzf/conv1d/strided_slice
vfwtupxpzf/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
vfwtupxpzf/conv1d/Reshape/shapeÌ
vfwtupxpzf/conv1d/ReshapeReshape%vfwtupxpzf/conv1d/ExpandDims:output:0(vfwtupxpzf/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/Reshapeî
vfwtupxpzf/conv1d/Conv2DConv2D"vfwtupxpzf/conv1d/Reshape:output:0'vfwtupxpzf/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
vfwtupxpzf/conv1d/Conv2D
!vfwtupxpzf/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!vfwtupxpzf/conv1d/concat/values_1
vfwtupxpzf/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/concat/axisì
vfwtupxpzf/conv1d/concatConcatV2(vfwtupxpzf/conv1d/strided_slice:output:0*vfwtupxpzf/conv1d/concat/values_1:output:0&vfwtupxpzf/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
vfwtupxpzf/conv1d/concatÉ
vfwtupxpzf/conv1d/Reshape_1Reshape!vfwtupxpzf/conv1d/Conv2D:output:0!vfwtupxpzf/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/Reshape_1Á
vfwtupxpzf/conv1d/SqueezeSqueeze$vfwtupxpzf/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
vfwtupxpzf/conv1d/Squeeze
#vfwtupxpzf/squeeze_batch_dims/ShapeShape"vfwtupxpzf/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#vfwtupxpzf/squeeze_batch_dims/Shape°
1vfwtupxpzf/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1vfwtupxpzf/squeeze_batch_dims/strided_slice/stack½
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1´
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2
+vfwtupxpzf/squeeze_batch_dims/strided_sliceStridedSlice,vfwtupxpzf/squeeze_batch_dims/Shape:output:0:vfwtupxpzf/squeeze_batch_dims/strided_slice/stack:output:0<vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_1:output:0<vfwtupxpzf/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+vfwtupxpzf/squeeze_batch_dims/strided_slice¯
+vfwtupxpzf/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+vfwtupxpzf/squeeze_batch_dims/Reshape/shapeé
%vfwtupxpzf/squeeze_batch_dims/ReshapeReshape"vfwtupxpzf/conv1d/Squeeze:output:04vfwtupxpzf/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%vfwtupxpzf/squeeze_batch_dims/Reshapeæ
4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=vfwtupxpzf_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%vfwtupxpzf/squeeze_batch_dims/BiasAddBiasAdd.vfwtupxpzf/squeeze_batch_dims/Reshape:output:0<vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%vfwtupxpzf/squeeze_batch_dims/BiasAdd¯
-vfwtupxpzf/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-vfwtupxpzf/squeeze_batch_dims/concat/values_1¡
)vfwtupxpzf/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)vfwtupxpzf/squeeze_batch_dims/concat/axis¨
$vfwtupxpzf/squeeze_batch_dims/concatConcatV24vfwtupxpzf/squeeze_batch_dims/strided_slice:output:06vfwtupxpzf/squeeze_batch_dims/concat/values_1:output:02vfwtupxpzf/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$vfwtupxpzf/squeeze_batch_dims/concatö
'vfwtupxpzf/squeeze_batch_dims/Reshape_1Reshape.vfwtupxpzf/squeeze_batch_dims/BiasAdd:output:0-vfwtupxpzf/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'vfwtupxpzf/squeeze_batch_dims/Reshape_1
ojzbgzevue/ShapeShape0vfwtupxpzf/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ojzbgzevue/Shape
ojzbgzevue/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ojzbgzevue/strided_slice/stack
 ojzbgzevue/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojzbgzevue/strided_slice/stack_1
 ojzbgzevue/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ojzbgzevue/strided_slice/stack_2¤
ojzbgzevue/strided_sliceStridedSliceojzbgzevue/Shape:output:0'ojzbgzevue/strided_slice/stack:output:0)ojzbgzevue/strided_slice/stack_1:output:0)ojzbgzevue/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ojzbgzevue/strided_slicez
ojzbgzevue/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ojzbgzevue/Reshape/shape/1z
ojzbgzevue/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ojzbgzevue/Reshape/shape/2×
ojzbgzevue/Reshape/shapePack!ojzbgzevue/strided_slice:output:0#ojzbgzevue/Reshape/shape/1:output:0#ojzbgzevue/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ojzbgzevue/Reshape/shape¾
ojzbgzevue/ReshapeReshape0vfwtupxpzf/squeeze_batch_dims/Reshape_1:output:0!ojzbgzevue/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ojzbgzevue/Reshapeo
vhacowjcza/ShapeShapeojzbgzevue/Reshape:output:0*
T0*
_output_shapes
:2
vhacowjcza/Shape
vhacowjcza/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
vhacowjcza/strided_slice/stack
 vhacowjcza/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 vhacowjcza/strided_slice/stack_1
 vhacowjcza/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 vhacowjcza/strided_slice/stack_2¤
vhacowjcza/strided_sliceStridedSlicevhacowjcza/Shape:output:0'vhacowjcza/strided_slice/stack:output:0)vhacowjcza/strided_slice/stack_1:output:0)vhacowjcza/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vhacowjcza/strided_slicer
vhacowjcza/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros/mul/y
vhacowjcza/zeros/mulMul!vhacowjcza/strided_slice:output:0vhacowjcza/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros/mulu
vhacowjcza/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vhacowjcza/zeros/Less/y
vhacowjcza/zeros/LessLessvhacowjcza/zeros/mul:z:0 vhacowjcza/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros/Lessx
vhacowjcza/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros/packed/1¯
vhacowjcza/zeros/packedPack!vhacowjcza/strided_slice:output:0"vhacowjcza/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
vhacowjcza/zeros/packedu
vhacowjcza/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vhacowjcza/zeros/Const¡
vhacowjcza/zerosFill vhacowjcza/zeros/packed:output:0vhacowjcza/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/zerosv
vhacowjcza/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros_1/mul/y
vhacowjcza/zeros_1/mulMul!vhacowjcza/strided_slice:output:0!vhacowjcza/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros_1/muly
vhacowjcza/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
vhacowjcza/zeros_1/Less/y
vhacowjcza/zeros_1/LessLessvhacowjcza/zeros_1/mul:z:0"vhacowjcza/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/zeros_1/Less|
vhacowjcza/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/zeros_1/packed/1µ
vhacowjcza/zeros_1/packedPack!vhacowjcza/strided_slice:output:0$vhacowjcza/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
vhacowjcza/zeros_1/packedy
vhacowjcza/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
vhacowjcza/zeros_1/Const©
vhacowjcza/zeros_1Fill"vhacowjcza/zeros_1/packed:output:0!vhacowjcza/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/zeros_1
vhacowjcza/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vhacowjcza/transpose/perm°
vhacowjcza/transpose	Transposeojzbgzevue/Reshape:output:0"vhacowjcza/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/transposep
vhacowjcza/Shape_1Shapevhacowjcza/transpose:y:0*
T0*
_output_shapes
:2
vhacowjcza/Shape_1
 vhacowjcza/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vhacowjcza/strided_slice_1/stack
"vhacowjcza/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_1/stack_1
"vhacowjcza/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_1/stack_2°
vhacowjcza/strided_slice_1StridedSlicevhacowjcza/Shape_1:output:0)vhacowjcza/strided_slice_1/stack:output:0+vhacowjcza/strided_slice_1/stack_1:output:0+vhacowjcza/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
vhacowjcza/strided_slice_1
&vhacowjcza/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&vhacowjcza/TensorArrayV2/element_shapeÞ
vhacowjcza/TensorArrayV2TensorListReserve/vhacowjcza/TensorArrayV2/element_shape:output:0#vhacowjcza/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vhacowjcza/TensorArrayV2Õ
@vhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@vhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2vhacowjcza/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorvhacowjcza/transpose:y:0Ivhacowjcza/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2vhacowjcza/TensorArrayUnstack/TensorListFromTensor
 vhacowjcza/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 vhacowjcza/strided_slice_2/stack
"vhacowjcza/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_2/stack_1
"vhacowjcza/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_2/stack_2¾
vhacowjcza/strided_slice_2StridedSlicevhacowjcza/transpose:y:0)vhacowjcza/strided_slice_2/stack:output:0+vhacowjcza/strided_slice_2/stack_1:output:0+vhacowjcza/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
vhacowjcza/strided_slice_2Ð
+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp4vhacowjcza_jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOpÓ
vhacowjcza/jczmzyhsca/MatMulMatMul#vhacowjcza/strided_slice_2:output:03vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/jczmzyhsca/MatMulÖ
-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp6vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOpÏ
vhacowjcza/jczmzyhsca/MatMul_1MatMulvhacowjcza/zeros:output:05vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
vhacowjcza/jczmzyhsca/MatMul_1Ä
vhacowjcza/jczmzyhsca/addAddV2&vhacowjcza/jczmzyhsca/MatMul:product:0(vhacowjcza/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/jczmzyhsca/addÏ
,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp5vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOpÑ
vhacowjcza/jczmzyhsca/BiasAddBiasAddvhacowjcza/jczmzyhsca/add:z:04vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
vhacowjcza/jczmzyhsca/BiasAdd
%vhacowjcza/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%vhacowjcza/jczmzyhsca/split/split_dim
vhacowjcza/jczmzyhsca/splitSplit.vhacowjcza/jczmzyhsca/split/split_dim:output:0&vhacowjcza/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
vhacowjcza/jczmzyhsca/split¶
$vhacowjcza/jczmzyhsca/ReadVariableOpReadVariableOp-vhacowjcza_jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02&
$vhacowjcza/jczmzyhsca/ReadVariableOpº
vhacowjcza/jczmzyhsca/mulMul,vhacowjcza/jczmzyhsca/ReadVariableOp:value:0vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mulº
vhacowjcza/jczmzyhsca/add_1AddV2$vhacowjcza/jczmzyhsca/split:output:0vhacowjcza/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_1
vhacowjcza/jczmzyhsca/SigmoidSigmoidvhacowjcza/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/Sigmoid¼
&vhacowjcza/jczmzyhsca/ReadVariableOp_1ReadVariableOp/vhacowjcza_jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&vhacowjcza/jczmzyhsca/ReadVariableOp_1À
vhacowjcza/jczmzyhsca/mul_1Mul.vhacowjcza/jczmzyhsca/ReadVariableOp_1:value:0vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_1¼
vhacowjcza/jczmzyhsca/add_2AddV2$vhacowjcza/jczmzyhsca/split:output:1vhacowjcza/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_2 
vhacowjcza/jczmzyhsca/Sigmoid_1Sigmoidvhacowjcza/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vhacowjcza/jczmzyhsca/Sigmoid_1µ
vhacowjcza/jczmzyhsca/mul_2Mul#vhacowjcza/jczmzyhsca/Sigmoid_1:y:0vhacowjcza/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_2
vhacowjcza/jczmzyhsca/TanhTanh$vhacowjcza/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/Tanh¶
vhacowjcza/jczmzyhsca/mul_3Mul!vhacowjcza/jczmzyhsca/Sigmoid:y:0vhacowjcza/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_3·
vhacowjcza/jczmzyhsca/add_3AddV2vhacowjcza/jczmzyhsca/mul_2:z:0vhacowjcza/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_3¼
&vhacowjcza/jczmzyhsca/ReadVariableOp_2ReadVariableOp/vhacowjcza_jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&vhacowjcza/jczmzyhsca/ReadVariableOp_2Ä
vhacowjcza/jczmzyhsca/mul_4Mul.vhacowjcza/jczmzyhsca/ReadVariableOp_2:value:0vhacowjcza/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_4¼
vhacowjcza/jczmzyhsca/add_4AddV2$vhacowjcza/jczmzyhsca/split:output:3vhacowjcza/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/add_4 
vhacowjcza/jczmzyhsca/Sigmoid_2Sigmoidvhacowjcza/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vhacowjcza/jczmzyhsca/Sigmoid_2
vhacowjcza/jczmzyhsca/Tanh_1Tanhvhacowjcza/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/Tanh_1º
vhacowjcza/jczmzyhsca/mul_5Mul#vhacowjcza/jczmzyhsca/Sigmoid_2:y:0 vhacowjcza/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/jczmzyhsca/mul_5¥
(vhacowjcza/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(vhacowjcza/TensorArrayV2_1/element_shapeä
vhacowjcza/TensorArrayV2_1TensorListReserve1vhacowjcza/TensorArrayV2_1/element_shape:output:0#vhacowjcza/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
vhacowjcza/TensorArrayV2_1d
vhacowjcza/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/time
#vhacowjcza/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#vhacowjcza/while/maximum_iterations
vhacowjcza/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
vhacowjcza/while/loop_counter²
vhacowjcza/whileWhile&vhacowjcza/while/loop_counter:output:0,vhacowjcza/while/maximum_iterations:output:0vhacowjcza/time:output:0#vhacowjcza/TensorArrayV2_1:handle:0vhacowjcza/zeros:output:0vhacowjcza/zeros_1:output:0#vhacowjcza/strided_slice_1:output:0Bvhacowjcza/TensorArrayUnstack/TensorListFromTensor:output_handle:04vhacowjcza_jczmzyhsca_matmul_readvariableop_resource6vhacowjcza_jczmzyhsca_matmul_1_readvariableop_resource5vhacowjcza_jczmzyhsca_biasadd_readvariableop_resource-vhacowjcza_jczmzyhsca_readvariableop_resource/vhacowjcza_jczmzyhsca_readvariableop_1_resource/vhacowjcza_jczmzyhsca_readvariableop_2_resource*
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
vhacowjcza_while_body_2160374*)
cond!R
vhacowjcza_while_cond_2160373*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
vhacowjcza/whileË
;vhacowjcza/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;vhacowjcza/TensorArrayV2Stack/TensorListStack/element_shape
-vhacowjcza/TensorArrayV2Stack/TensorListStackTensorListStackvhacowjcza/while:output:3Dvhacowjcza/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-vhacowjcza/TensorArrayV2Stack/TensorListStack
 vhacowjcza/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 vhacowjcza/strided_slice_3/stack
"vhacowjcza/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"vhacowjcza/strided_slice_3/stack_1
"vhacowjcza/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"vhacowjcza/strided_slice_3/stack_2Ü
vhacowjcza/strided_slice_3StridedSlice6vhacowjcza/TensorArrayV2Stack/TensorListStack:tensor:0)vhacowjcza/strided_slice_3/stack:output:0+vhacowjcza/strided_slice_3/stack_1:output:0+vhacowjcza/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
vhacowjcza/strided_slice_3
vhacowjcza/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
vhacowjcza/transpose_1/permÑ
vhacowjcza/transpose_1	Transpose6vhacowjcza/TensorArrayV2Stack/TensorListStack:tensor:0$vhacowjcza/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/transpose_1n
zgrdiwrovx/ShapeShapevhacowjcza/transpose_1:y:0*
T0*
_output_shapes
:2
zgrdiwrovx/Shape
zgrdiwrovx/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
zgrdiwrovx/strided_slice/stack
 zgrdiwrovx/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 zgrdiwrovx/strided_slice/stack_1
 zgrdiwrovx/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 zgrdiwrovx/strided_slice/stack_2¤
zgrdiwrovx/strided_sliceStridedSlicezgrdiwrovx/Shape:output:0'zgrdiwrovx/strided_slice/stack:output:0)zgrdiwrovx/strided_slice/stack_1:output:0)zgrdiwrovx/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zgrdiwrovx/strided_slicer
zgrdiwrovx/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros/mul/y
zgrdiwrovx/zeros/mulMul!zgrdiwrovx/strided_slice:output:0zgrdiwrovx/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros/mulu
zgrdiwrovx/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zgrdiwrovx/zeros/Less/y
zgrdiwrovx/zeros/LessLesszgrdiwrovx/zeros/mul:z:0 zgrdiwrovx/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros/Lessx
zgrdiwrovx/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros/packed/1¯
zgrdiwrovx/zeros/packedPack!zgrdiwrovx/strided_slice:output:0"zgrdiwrovx/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zgrdiwrovx/zeros/packedu
zgrdiwrovx/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zgrdiwrovx/zeros/Const¡
zgrdiwrovx/zerosFill zgrdiwrovx/zeros/packed:output:0zgrdiwrovx/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/zerosv
zgrdiwrovx/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros_1/mul/y
zgrdiwrovx/zeros_1/mulMul!zgrdiwrovx/strided_slice:output:0!zgrdiwrovx/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros_1/muly
zgrdiwrovx/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zgrdiwrovx/zeros_1/Less/y
zgrdiwrovx/zeros_1/LessLesszgrdiwrovx/zeros_1/mul:z:0"zgrdiwrovx/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/zeros_1/Less|
zgrdiwrovx/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/zeros_1/packed/1µ
zgrdiwrovx/zeros_1/packedPack!zgrdiwrovx/strided_slice:output:0$zgrdiwrovx/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zgrdiwrovx/zeros_1/packedy
zgrdiwrovx/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zgrdiwrovx/zeros_1/Const©
zgrdiwrovx/zeros_1Fill"zgrdiwrovx/zeros_1/packed:output:0!zgrdiwrovx/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/zeros_1
zgrdiwrovx/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zgrdiwrovx/transpose/perm¯
zgrdiwrovx/transpose	Transposevhacowjcza/transpose_1:y:0"zgrdiwrovx/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/transposep
zgrdiwrovx/Shape_1Shapezgrdiwrovx/transpose:y:0*
T0*
_output_shapes
:2
zgrdiwrovx/Shape_1
 zgrdiwrovx/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zgrdiwrovx/strided_slice_1/stack
"zgrdiwrovx/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_1/stack_1
"zgrdiwrovx/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_1/stack_2°
zgrdiwrovx/strided_slice_1StridedSlicezgrdiwrovx/Shape_1:output:0)zgrdiwrovx/strided_slice_1/stack:output:0+zgrdiwrovx/strided_slice_1/stack_1:output:0+zgrdiwrovx/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zgrdiwrovx/strided_slice_1
&zgrdiwrovx/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&zgrdiwrovx/TensorArrayV2/element_shapeÞ
zgrdiwrovx/TensorArrayV2TensorListReserve/zgrdiwrovx/TensorArrayV2/element_shape:output:0#zgrdiwrovx/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zgrdiwrovx/TensorArrayV2Õ
@zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2zgrdiwrovx/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorzgrdiwrovx/transpose:y:0Izgrdiwrovx/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2zgrdiwrovx/TensorArrayUnstack/TensorListFromTensor
 zgrdiwrovx/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zgrdiwrovx/strided_slice_2/stack
"zgrdiwrovx/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_2/stack_1
"zgrdiwrovx/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_2/stack_2¾
zgrdiwrovx/strided_slice_2StridedSlicezgrdiwrovx/transpose:y:0)zgrdiwrovx/strided_slice_2/stack:output:0+zgrdiwrovx/strided_slice_2/stack_1:output:0+zgrdiwrovx/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
zgrdiwrovx/strided_slice_2Ð
+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp4zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOpÓ
zgrdiwrovx/wdwulgrltk/MatMulMatMul#zgrdiwrovx/strided_slice_2:output:03zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zgrdiwrovx/wdwulgrltk/MatMulÖ
-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp6zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOpÏ
zgrdiwrovx/wdwulgrltk/MatMul_1MatMulzgrdiwrovx/zeros:output:05zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
zgrdiwrovx/wdwulgrltk/MatMul_1Ä
zgrdiwrovx/wdwulgrltk/addAddV2&zgrdiwrovx/wdwulgrltk/MatMul:product:0(zgrdiwrovx/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zgrdiwrovx/wdwulgrltk/addÏ
,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp5zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOpÑ
zgrdiwrovx/wdwulgrltk/BiasAddBiasAddzgrdiwrovx/wdwulgrltk/add:z:04zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zgrdiwrovx/wdwulgrltk/BiasAdd
%zgrdiwrovx/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%zgrdiwrovx/wdwulgrltk/split/split_dim
zgrdiwrovx/wdwulgrltk/splitSplit.zgrdiwrovx/wdwulgrltk/split/split_dim:output:0&zgrdiwrovx/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zgrdiwrovx/wdwulgrltk/split¶
$zgrdiwrovx/wdwulgrltk/ReadVariableOpReadVariableOp-zgrdiwrovx_wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02&
$zgrdiwrovx/wdwulgrltk/ReadVariableOpº
zgrdiwrovx/wdwulgrltk/mulMul,zgrdiwrovx/wdwulgrltk/ReadVariableOp:value:0zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mulº
zgrdiwrovx/wdwulgrltk/add_1AddV2$zgrdiwrovx/wdwulgrltk/split:output:0zgrdiwrovx/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_1
zgrdiwrovx/wdwulgrltk/SigmoidSigmoidzgrdiwrovx/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/Sigmoid¼
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1ReadVariableOp/zgrdiwrovx_wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1À
zgrdiwrovx/wdwulgrltk/mul_1Mul.zgrdiwrovx/wdwulgrltk/ReadVariableOp_1:value:0zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_1¼
zgrdiwrovx/wdwulgrltk/add_2AddV2$zgrdiwrovx/wdwulgrltk/split:output:1zgrdiwrovx/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_2 
zgrdiwrovx/wdwulgrltk/Sigmoid_1Sigmoidzgrdiwrovx/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zgrdiwrovx/wdwulgrltk/Sigmoid_1µ
zgrdiwrovx/wdwulgrltk/mul_2Mul#zgrdiwrovx/wdwulgrltk/Sigmoid_1:y:0zgrdiwrovx/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_2
zgrdiwrovx/wdwulgrltk/TanhTanh$zgrdiwrovx/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/Tanh¶
zgrdiwrovx/wdwulgrltk/mul_3Mul!zgrdiwrovx/wdwulgrltk/Sigmoid:y:0zgrdiwrovx/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_3·
zgrdiwrovx/wdwulgrltk/add_3AddV2zgrdiwrovx/wdwulgrltk/mul_2:z:0zgrdiwrovx/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_3¼
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2ReadVariableOp/zgrdiwrovx_wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2Ä
zgrdiwrovx/wdwulgrltk/mul_4Mul.zgrdiwrovx/wdwulgrltk/ReadVariableOp_2:value:0zgrdiwrovx/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_4¼
zgrdiwrovx/wdwulgrltk/add_4AddV2$zgrdiwrovx/wdwulgrltk/split:output:3zgrdiwrovx/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/add_4 
zgrdiwrovx/wdwulgrltk/Sigmoid_2Sigmoidzgrdiwrovx/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zgrdiwrovx/wdwulgrltk/Sigmoid_2
zgrdiwrovx/wdwulgrltk/Tanh_1Tanhzgrdiwrovx/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/Tanh_1º
zgrdiwrovx/wdwulgrltk/mul_5Mul#zgrdiwrovx/wdwulgrltk/Sigmoid_2:y:0 zgrdiwrovx/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/wdwulgrltk/mul_5¥
(zgrdiwrovx/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(zgrdiwrovx/TensorArrayV2_1/element_shapeä
zgrdiwrovx/TensorArrayV2_1TensorListReserve1zgrdiwrovx/TensorArrayV2_1/element_shape:output:0#zgrdiwrovx/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zgrdiwrovx/TensorArrayV2_1d
zgrdiwrovx/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/time
#zgrdiwrovx/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#zgrdiwrovx/while/maximum_iterations
zgrdiwrovx/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
zgrdiwrovx/while/loop_counter²
zgrdiwrovx/whileWhile&zgrdiwrovx/while/loop_counter:output:0,zgrdiwrovx/while/maximum_iterations:output:0zgrdiwrovx/time:output:0#zgrdiwrovx/TensorArrayV2_1:handle:0zgrdiwrovx/zeros:output:0zgrdiwrovx/zeros_1:output:0#zgrdiwrovx/strided_slice_1:output:0Bzgrdiwrovx/TensorArrayUnstack/TensorListFromTensor:output_handle:04zgrdiwrovx_wdwulgrltk_matmul_readvariableop_resource6zgrdiwrovx_wdwulgrltk_matmul_1_readvariableop_resource5zgrdiwrovx_wdwulgrltk_biasadd_readvariableop_resource-zgrdiwrovx_wdwulgrltk_readvariableop_resource/zgrdiwrovx_wdwulgrltk_readvariableop_1_resource/zgrdiwrovx_wdwulgrltk_readvariableop_2_resource*
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
zgrdiwrovx_while_body_2160550*)
cond!R
zgrdiwrovx_while_cond_2160549*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
zgrdiwrovx/whileË
;zgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;zgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shape
-zgrdiwrovx/TensorArrayV2Stack/TensorListStackTensorListStackzgrdiwrovx/while:output:3Dzgrdiwrovx/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-zgrdiwrovx/TensorArrayV2Stack/TensorListStack
 zgrdiwrovx/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 zgrdiwrovx/strided_slice_3/stack
"zgrdiwrovx/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"zgrdiwrovx/strided_slice_3/stack_1
"zgrdiwrovx/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zgrdiwrovx/strided_slice_3/stack_2Ü
zgrdiwrovx/strided_slice_3StridedSlice6zgrdiwrovx/TensorArrayV2Stack/TensorListStack:tensor:0)zgrdiwrovx/strided_slice_3/stack:output:0+zgrdiwrovx/strided_slice_3/stack_1:output:0+zgrdiwrovx/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
zgrdiwrovx/strided_slice_3
zgrdiwrovx/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zgrdiwrovx/transpose_1/permÑ
zgrdiwrovx/transpose_1	Transpose6zgrdiwrovx/TensorArrayV2Stack/TensorListStack:tensor:0$zgrdiwrovx/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/transpose_1®
 kekwghyimt/MatMul/ReadVariableOpReadVariableOp)kekwghyimt_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 kekwghyimt/MatMul/ReadVariableOp±
kekwghyimt/MatMulMatMul#zgrdiwrovx/strided_slice_3:output:0(kekwghyimt/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kekwghyimt/MatMul­
!kekwghyimt/BiasAdd/ReadVariableOpReadVariableOp*kekwghyimt_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!kekwghyimt/BiasAdd/ReadVariableOp­
kekwghyimt/BiasAddBiasAddkekwghyimt/MatMul:product:0)kekwghyimt/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kekwghyimt/BiasAddÏ
IdentityIdentitykekwghyimt/BiasAdd:output:0"^kekwghyimt/BiasAdd/ReadVariableOp!^kekwghyimt/MatMul/ReadVariableOp.^vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp5^vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp-^vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp,^vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp.^vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp%^vhacowjcza/jczmzyhsca/ReadVariableOp'^vhacowjcza/jczmzyhsca/ReadVariableOp_1'^vhacowjcza/jczmzyhsca/ReadVariableOp_2^vhacowjcza/while-^zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp,^zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp.^zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp%^zgrdiwrovx/wdwulgrltk/ReadVariableOp'^zgrdiwrovx/wdwulgrltk/ReadVariableOp_1'^zgrdiwrovx/wdwulgrltk/ReadVariableOp_2^zgrdiwrovx/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2F
!kekwghyimt/BiasAdd/ReadVariableOp!kekwghyimt/BiasAdd/ReadVariableOp2D
 kekwghyimt/MatMul/ReadVariableOp kekwghyimt/MatMul/ReadVariableOp2^
-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp-vfwtupxpzf/conv1d/ExpandDims_1/ReadVariableOp2l
4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp4vfwtupxpzf/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp,vhacowjcza/jczmzyhsca/BiasAdd/ReadVariableOp2Z
+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp+vhacowjcza/jczmzyhsca/MatMul/ReadVariableOp2^
-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp-vhacowjcza/jczmzyhsca/MatMul_1/ReadVariableOp2L
$vhacowjcza/jczmzyhsca/ReadVariableOp$vhacowjcza/jczmzyhsca/ReadVariableOp2P
&vhacowjcza/jczmzyhsca/ReadVariableOp_1&vhacowjcza/jczmzyhsca/ReadVariableOp_12P
&vhacowjcza/jczmzyhsca/ReadVariableOp_2&vhacowjcza/jczmzyhsca/ReadVariableOp_22$
vhacowjcza/whilevhacowjcza/while2\
,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp,zgrdiwrovx/wdwulgrltk/BiasAdd/ReadVariableOp2Z
+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp+zgrdiwrovx/wdwulgrltk/MatMul/ReadVariableOp2^
-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp-zgrdiwrovx/wdwulgrltk/MatMul_1/ReadVariableOp2L
$zgrdiwrovx/wdwulgrltk/ReadVariableOp$zgrdiwrovx/wdwulgrltk/ReadVariableOp2P
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_1&zgrdiwrovx/wdwulgrltk/ReadVariableOp_12P
&zgrdiwrovx/wdwulgrltk/ReadVariableOp_2&zgrdiwrovx/wdwulgrltk/ReadVariableOp_22$
zgrdiwrovx/whilezgrdiwrovx/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
H
,__inference_ojzbgzevue_layer_call_fn_2161199

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
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_21590802
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
while_cond_2157607
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2157607___redundant_placeholder05
1while_while_cond_2157607___redundant_placeholder15
1while_while_cond_2157607___redundant_placeholder25
1while_while_cond_2157607___redundant_placeholder35
1while_while_cond_2157607___redundant_placeholder45
1while_while_cond_2157607___redundant_placeholder55
1while_while_cond_2157607___redundant_placeholder6
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
³F
ê
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2157951

inputs%
jczmzyhsca_2157852:	%
jczmzyhsca_2157854:	 !
jczmzyhsca_2157856:	 
jczmzyhsca_2157858:  
jczmzyhsca_2157860:  
jczmzyhsca_2157862: 
identity¢"jczmzyhsca/StatefulPartitionedCall¢whileD
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
"jczmzyhsca/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0jczmzyhsca_2157852jczmzyhsca_2157854jczmzyhsca_2157856jczmzyhsca_2157858jczmzyhsca_2157860jczmzyhsca_2157862*
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_21577752$
"jczmzyhsca/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0jczmzyhsca_2157852jczmzyhsca_2157854jczmzyhsca_2157856jczmzyhsca_2157858jczmzyhsca_2157860jczmzyhsca_2157862*
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
while_body_2157871*
condR
while_cond_2157870*Q
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
IdentityIdentitytranspose_1:y:0#^jczmzyhsca/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"jczmzyhsca/StatefulPartitionedCall"jczmzyhsca/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
p
Ê
zgrdiwrovx_while_body_21605502
.zgrdiwrovx_while_zgrdiwrovx_while_loop_counter8
4zgrdiwrovx_while_zgrdiwrovx_while_maximum_iterations 
zgrdiwrovx_while_placeholder"
zgrdiwrovx_while_placeholder_1"
zgrdiwrovx_while_placeholder_2"
zgrdiwrovx_while_placeholder_31
-zgrdiwrovx_while_zgrdiwrovx_strided_slice_1_0m
izgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0O
<zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0:	 Q
>zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 L
=zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0:	C
5zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0: E
7zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0: E
7zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0: 
zgrdiwrovx_while_identity
zgrdiwrovx_while_identity_1
zgrdiwrovx_while_identity_2
zgrdiwrovx_while_identity_3
zgrdiwrovx_while_identity_4
zgrdiwrovx_while_identity_5/
+zgrdiwrovx_while_zgrdiwrovx_strided_slice_1k
gzgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensorM
:zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource:	 O
<zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource:	 J
;zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource:	A
3zgrdiwrovx_while_wdwulgrltk_readvariableop_resource: C
5zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource: C
5zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource: ¢2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp¢1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp¢3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp¢*zgrdiwrovx/while/wdwulgrltk/ReadVariableOp¢,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1¢,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2Ù
Bzgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bzgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shape
4zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemizgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0zgrdiwrovx_while_placeholderKzgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItemä
1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp<zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOpý
"zgrdiwrovx/while/wdwulgrltk/MatMulMatMul;zgrdiwrovx/while/TensorArrayV2Read/TensorListGetItem:item:09zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"zgrdiwrovx/while/wdwulgrltk/MatMulê
3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp>zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOpæ
$zgrdiwrovx/while/wdwulgrltk/MatMul_1MatMulzgrdiwrovx_while_placeholder_2;zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$zgrdiwrovx/while/wdwulgrltk/MatMul_1Ü
zgrdiwrovx/while/wdwulgrltk/addAddV2,zgrdiwrovx/while/wdwulgrltk/MatMul:product:0.zgrdiwrovx/while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
zgrdiwrovx/while/wdwulgrltk/addã
2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp=zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOpé
#zgrdiwrovx/while/wdwulgrltk/BiasAddBiasAdd#zgrdiwrovx/while/wdwulgrltk/add:z:0:zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#zgrdiwrovx/while/wdwulgrltk/BiasAdd
+zgrdiwrovx/while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+zgrdiwrovx/while/wdwulgrltk/split/split_dim¯
!zgrdiwrovx/while/wdwulgrltk/splitSplit4zgrdiwrovx/while/wdwulgrltk/split/split_dim:output:0,zgrdiwrovx/while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!zgrdiwrovx/while/wdwulgrltk/splitÊ
*zgrdiwrovx/while/wdwulgrltk/ReadVariableOpReadVariableOp5zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*zgrdiwrovx/while/wdwulgrltk/ReadVariableOpÏ
zgrdiwrovx/while/wdwulgrltk/mulMul2zgrdiwrovx/while/wdwulgrltk/ReadVariableOp:value:0zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zgrdiwrovx/while/wdwulgrltk/mulÒ
!zgrdiwrovx/while/wdwulgrltk/add_1AddV2*zgrdiwrovx/while/wdwulgrltk/split:output:0#zgrdiwrovx/while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_1®
#zgrdiwrovx/while/wdwulgrltk/SigmoidSigmoid%zgrdiwrovx/while/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#zgrdiwrovx/while/wdwulgrltk/SigmoidÐ
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1ReadVariableOp7zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1Õ
!zgrdiwrovx/while/wdwulgrltk/mul_1Mul4zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1:value:0zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_1Ô
!zgrdiwrovx/while/wdwulgrltk/add_2AddV2*zgrdiwrovx/while/wdwulgrltk/split:output:1%zgrdiwrovx/while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_2²
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_1Sigmoid%zgrdiwrovx/while/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_1Ê
!zgrdiwrovx/while/wdwulgrltk/mul_2Mul)zgrdiwrovx/while/wdwulgrltk/Sigmoid_1:y:0zgrdiwrovx_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_2ª
 zgrdiwrovx/while/wdwulgrltk/TanhTanh*zgrdiwrovx/while/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 zgrdiwrovx/while/wdwulgrltk/TanhÎ
!zgrdiwrovx/while/wdwulgrltk/mul_3Mul'zgrdiwrovx/while/wdwulgrltk/Sigmoid:y:0$zgrdiwrovx/while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_3Ï
!zgrdiwrovx/while/wdwulgrltk/add_3AddV2%zgrdiwrovx/while/wdwulgrltk/mul_2:z:0%zgrdiwrovx/while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_3Ð
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2ReadVariableOp7zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2Ü
!zgrdiwrovx/while/wdwulgrltk/mul_4Mul4zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2:value:0%zgrdiwrovx/while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_4Ô
!zgrdiwrovx/while/wdwulgrltk/add_4AddV2*zgrdiwrovx/while/wdwulgrltk/split:output:3%zgrdiwrovx/while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/add_4²
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_2Sigmoid%zgrdiwrovx/while/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zgrdiwrovx/while/wdwulgrltk/Sigmoid_2©
"zgrdiwrovx/while/wdwulgrltk/Tanh_1Tanh%zgrdiwrovx/while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"zgrdiwrovx/while/wdwulgrltk/Tanh_1Ò
!zgrdiwrovx/while/wdwulgrltk/mul_5Mul)zgrdiwrovx/while/wdwulgrltk/Sigmoid_2:y:0&zgrdiwrovx/while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zgrdiwrovx/while/wdwulgrltk/mul_5
5zgrdiwrovx/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemzgrdiwrovx_while_placeholder_1zgrdiwrovx_while_placeholder%zgrdiwrovx/while/wdwulgrltk/mul_5:z:0*
_output_shapes
: *
element_dtype027
5zgrdiwrovx/while/TensorArrayV2Write/TensorListSetItemr
zgrdiwrovx/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
zgrdiwrovx/while/add/y
zgrdiwrovx/while/addAddV2zgrdiwrovx_while_placeholderzgrdiwrovx/while/add/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/while/addv
zgrdiwrovx/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
zgrdiwrovx/while/add_1/y­
zgrdiwrovx/while/add_1AddV2.zgrdiwrovx_while_zgrdiwrovx_while_loop_counter!zgrdiwrovx/while/add_1/y:output:0*
T0*
_output_shapes
: 2
zgrdiwrovx/while/add_1©
zgrdiwrovx/while/IdentityIdentityzgrdiwrovx/while/add_1:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/IdentityÇ
zgrdiwrovx/while/Identity_1Identity4zgrdiwrovx_while_zgrdiwrovx_while_maximum_iterations3^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Identity_1«
zgrdiwrovx/while/Identity_2Identityzgrdiwrovx/while/add:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Identity_2Ø
zgrdiwrovx/while/Identity_3IdentityEzgrdiwrovx/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Identity_3É
zgrdiwrovx/while/Identity_4Identity%zgrdiwrovx/while/wdwulgrltk/mul_5:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/while/Identity_4É
zgrdiwrovx/while/Identity_5Identity%zgrdiwrovx/while/wdwulgrltk/add_3:z:03^zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2^zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp4^zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp+^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1-^zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zgrdiwrovx/while/Identity_5"?
zgrdiwrovx_while_identity"zgrdiwrovx/while/Identity:output:0"C
zgrdiwrovx_while_identity_1$zgrdiwrovx/while/Identity_1:output:0"C
zgrdiwrovx_while_identity_2$zgrdiwrovx/while/Identity_2:output:0"C
zgrdiwrovx_while_identity_3$zgrdiwrovx/while/Identity_3:output:0"C
zgrdiwrovx_while_identity_4$zgrdiwrovx/while/Identity_4:output:0"C
zgrdiwrovx_while_identity_5$zgrdiwrovx/while/Identity_5:output:0"Ô
gzgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensorizgrdiwrovx_while_tensorarrayv2read_tensorlistgetitem_zgrdiwrovx_tensorarrayunstack_tensorlistfromtensor_0"|
;zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource=zgrdiwrovx_while_wdwulgrltk_biasadd_readvariableop_resource_0"~
<zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource>zgrdiwrovx_while_wdwulgrltk_matmul_1_readvariableop_resource_0"z
:zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource<zgrdiwrovx_while_wdwulgrltk_matmul_readvariableop_resource_0"p
5zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource7zgrdiwrovx_while_wdwulgrltk_readvariableop_1_resource_0"p
5zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource7zgrdiwrovx_while_wdwulgrltk_readvariableop_2_resource_0"l
3zgrdiwrovx_while_wdwulgrltk_readvariableop_resource5zgrdiwrovx_while_wdwulgrltk_readvariableop_resource_0"\
+zgrdiwrovx_while_zgrdiwrovx_strided_slice_1-zgrdiwrovx_while_zgrdiwrovx_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2zgrdiwrovx/while/wdwulgrltk/BiasAdd/ReadVariableOp2f
1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp1zgrdiwrovx/while/wdwulgrltk/MatMul/ReadVariableOp2j
3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp3zgrdiwrovx/while/wdwulgrltk/MatMul_1/ReadVariableOp2X
*zgrdiwrovx/while/wdwulgrltk/ReadVariableOp*zgrdiwrovx/while/wdwulgrltk/ReadVariableOp2\
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_1,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_12\
,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2,zgrdiwrovx/while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2162882

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
zgrdiwrovx_while_cond_21609532
.zgrdiwrovx_while_zgrdiwrovx_while_loop_counter8
4zgrdiwrovx_while_zgrdiwrovx_while_maximum_iterations 
zgrdiwrovx_while_placeholder"
zgrdiwrovx_while_placeholder_1"
zgrdiwrovx_while_placeholder_2"
zgrdiwrovx_while_placeholder_34
0zgrdiwrovx_while_less_zgrdiwrovx_strided_slice_1K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160953___redundant_placeholder0K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160953___redundant_placeholder1K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160953___redundant_placeholder2K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160953___redundant_placeholder3K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160953___redundant_placeholder4K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160953___redundant_placeholder5K
Gzgrdiwrovx_while_zgrdiwrovx_while_cond_2160953___redundant_placeholder6
zgrdiwrovx_while_identity
§
zgrdiwrovx/while/LessLesszgrdiwrovx_while_placeholder0zgrdiwrovx_while_less_zgrdiwrovx_strided_slice_1*
T0*
_output_shapes
: 2
zgrdiwrovx/while/Less~
zgrdiwrovx/while/IdentityIdentityzgrdiwrovx/while/Less:z:0*
T0
*
_output_shapes
: 2
zgrdiwrovx/while/Identity"?
zgrdiwrovx_while_identity"zgrdiwrovx/while/Identity:output:0*(
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
while_body_2161458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_jczmzyhsca_matmul_readvariableop_resource_0:	F
3while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 A
2while_jczmzyhsca_biasadd_readvariableop_resource_0:	8
*while_jczmzyhsca_readvariableop_resource_0: :
,while_jczmzyhsca_readvariableop_1_resource_0: :
,while_jczmzyhsca_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_jczmzyhsca_matmul_readvariableop_resource:	D
1while_jczmzyhsca_matmul_1_readvariableop_resource:	 ?
0while_jczmzyhsca_biasadd_readvariableop_resource:	6
(while_jczmzyhsca_readvariableop_resource: 8
*while_jczmzyhsca_readvariableop_1_resource: 8
*while_jczmzyhsca_readvariableop_2_resource: ¢'while/jczmzyhsca/BiasAdd/ReadVariableOp¢&while/jczmzyhsca/MatMul/ReadVariableOp¢(while/jczmzyhsca/MatMul_1/ReadVariableOp¢while/jczmzyhsca/ReadVariableOp¢!while/jczmzyhsca/ReadVariableOp_1¢!while/jczmzyhsca/ReadVariableOp_2Ã
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
&while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp1while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/jczmzyhsca/MatMul/ReadVariableOpÑ
while/jczmzyhsca/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMulÉ
(while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp3while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/jczmzyhsca/MatMul_1/ReadVariableOpº
while/jczmzyhsca/MatMul_1MatMulwhile_placeholder_20while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/MatMul_1°
while/jczmzyhsca/addAddV2!while/jczmzyhsca/MatMul:product:0#while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/addÂ
'while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp2while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/jczmzyhsca/BiasAdd/ReadVariableOp½
while/jczmzyhsca/BiasAddBiasAddwhile/jczmzyhsca/add:z:0/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/jczmzyhsca/BiasAdd
 while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/jczmzyhsca/split/split_dim
while/jczmzyhsca/splitSplit)while/jczmzyhsca/split/split_dim:output:0!while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/jczmzyhsca/split©
while/jczmzyhsca/ReadVariableOpReadVariableOp*while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/jczmzyhsca/ReadVariableOp£
while/jczmzyhsca/mulMul'while/jczmzyhsca/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul¦
while/jczmzyhsca/add_1AddV2while/jczmzyhsca/split:output:0while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_1
while/jczmzyhsca/SigmoidSigmoidwhile/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid¯
!while/jczmzyhsca/ReadVariableOp_1ReadVariableOp,while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_1©
while/jczmzyhsca/mul_1Mul)while/jczmzyhsca/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_1¨
while/jczmzyhsca/add_2AddV2while/jczmzyhsca/split:output:1while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_2
while/jczmzyhsca/Sigmoid_1Sigmoidwhile/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_1
while/jczmzyhsca/mul_2Mulwhile/jczmzyhsca/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_2
while/jczmzyhsca/TanhTanhwhile/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh¢
while/jczmzyhsca/mul_3Mulwhile/jczmzyhsca/Sigmoid:y:0while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_3£
while/jczmzyhsca/add_3AddV2while/jczmzyhsca/mul_2:z:0while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_3¯
!while/jczmzyhsca/ReadVariableOp_2ReadVariableOp,while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/jczmzyhsca/ReadVariableOp_2°
while/jczmzyhsca/mul_4Mul)while/jczmzyhsca/ReadVariableOp_2:value:0while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_4¨
while/jczmzyhsca/add_4AddV2while/jczmzyhsca/split:output:3while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/add_4
while/jczmzyhsca/Sigmoid_2Sigmoidwhile/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Sigmoid_2
while/jczmzyhsca/Tanh_1Tanhwhile/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/Tanh_1¦
while/jczmzyhsca/mul_5Mulwhile/jczmzyhsca/Sigmoid_2:y:0while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/jczmzyhsca/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/jczmzyhsca/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/jczmzyhsca/mul_5:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/jczmzyhsca/add_3:z:0(^while/jczmzyhsca/BiasAdd/ReadVariableOp'^while/jczmzyhsca/MatMul/ReadVariableOp)^while/jczmzyhsca/MatMul_1/ReadVariableOp ^while/jczmzyhsca/ReadVariableOp"^while/jczmzyhsca/ReadVariableOp_1"^while/jczmzyhsca/ReadVariableOp_2*
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
0while_jczmzyhsca_biasadd_readvariableop_resource2while_jczmzyhsca_biasadd_readvariableop_resource_0"h
1while_jczmzyhsca_matmul_1_readvariableop_resource3while_jczmzyhsca_matmul_1_readvariableop_resource_0"d
/while_jczmzyhsca_matmul_readvariableop_resource1while_jczmzyhsca_matmul_readvariableop_resource_0"Z
*while_jczmzyhsca_readvariableop_1_resource,while_jczmzyhsca_readvariableop_1_resource_0"Z
*while_jczmzyhsca_readvariableop_2_resource,while_jczmzyhsca_readvariableop_2_resource_0"V
(while_jczmzyhsca_readvariableop_resource*while_jczmzyhsca_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/jczmzyhsca/BiasAdd/ReadVariableOp'while/jczmzyhsca/BiasAdd/ReadVariableOp2P
&while/jczmzyhsca/MatMul/ReadVariableOp&while/jczmzyhsca/MatMul/ReadVariableOp2T
(while/jczmzyhsca/MatMul_1/ReadVariableOp(while/jczmzyhsca/MatMul_1/ReadVariableOp2B
while/jczmzyhsca/ReadVariableOpwhile/jczmzyhsca/ReadVariableOp2F
!while/jczmzyhsca/ReadVariableOp_1!while/jczmzyhsca/ReadVariableOp_12F
!while/jczmzyhsca/ReadVariableOp_2!while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2162245
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2162245___redundant_placeholder05
1while_while_cond_2162245___redundant_placeholder15
1while_while_cond_2162245___redundant_placeholder25
1while_while_cond_2162245___redundant_placeholder35
1while_while_cond_2162245___redundant_placeholder45
1while_while_cond_2162245___redundant_placeholder55
1while_while_cond_2162245___redundant_placeholder6
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
vhacowjcza_while_body_21603742
.vhacowjcza_while_vhacowjcza_while_loop_counter8
4vhacowjcza_while_vhacowjcza_while_maximum_iterations 
vhacowjcza_while_placeholder"
vhacowjcza_while_placeholder_1"
vhacowjcza_while_placeholder_2"
vhacowjcza_while_placeholder_31
-vhacowjcza_while_vhacowjcza_strided_slice_1_0m
ivhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0O
<vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0:	Q
>vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 L
=vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0:	C
5vhacowjcza_while_jczmzyhsca_readvariableop_resource_0: E
7vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0: E
7vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0: 
vhacowjcza_while_identity
vhacowjcza_while_identity_1
vhacowjcza_while_identity_2
vhacowjcza_while_identity_3
vhacowjcza_while_identity_4
vhacowjcza_while_identity_5/
+vhacowjcza_while_vhacowjcza_strided_slice_1k
gvhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensorM
:vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource:	O
<vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource:	 J
;vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource:	A
3vhacowjcza_while_jczmzyhsca_readvariableop_resource: C
5vhacowjcza_while_jczmzyhsca_readvariableop_1_resource: C
5vhacowjcza_while_jczmzyhsca_readvariableop_2_resource: ¢2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp¢1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp¢3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp¢*vhacowjcza/while/jczmzyhsca/ReadVariableOp¢,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1¢,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2Ù
Bvhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bvhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shape
4vhacowjcza/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemivhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0vhacowjcza_while_placeholderKvhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4vhacowjcza/while/TensorArrayV2Read/TensorListGetItemä
1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp<vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOpý
"vhacowjcza/while/jczmzyhsca/MatMulMatMul;vhacowjcza/while/TensorArrayV2Read/TensorListGetItem:item:09vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"vhacowjcza/while/jczmzyhsca/MatMulê
3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp>vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOpæ
$vhacowjcza/while/jczmzyhsca/MatMul_1MatMulvhacowjcza_while_placeholder_2;vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$vhacowjcza/while/jczmzyhsca/MatMul_1Ü
vhacowjcza/while/jczmzyhsca/addAddV2,vhacowjcza/while/jczmzyhsca/MatMul:product:0.vhacowjcza/while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
vhacowjcza/while/jczmzyhsca/addã
2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp=vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOpé
#vhacowjcza/while/jczmzyhsca/BiasAddBiasAdd#vhacowjcza/while/jczmzyhsca/add:z:0:vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#vhacowjcza/while/jczmzyhsca/BiasAdd
+vhacowjcza/while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+vhacowjcza/while/jczmzyhsca/split/split_dim¯
!vhacowjcza/while/jczmzyhsca/splitSplit4vhacowjcza/while/jczmzyhsca/split/split_dim:output:0,vhacowjcza/while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!vhacowjcza/while/jczmzyhsca/splitÊ
*vhacowjcza/while/jczmzyhsca/ReadVariableOpReadVariableOp5vhacowjcza_while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*vhacowjcza/while/jczmzyhsca/ReadVariableOpÏ
vhacowjcza/while/jczmzyhsca/mulMul2vhacowjcza/while/jczmzyhsca/ReadVariableOp:value:0vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vhacowjcza/while/jczmzyhsca/mulÒ
!vhacowjcza/while/jczmzyhsca/add_1AddV2*vhacowjcza/while/jczmzyhsca/split:output:0#vhacowjcza/while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_1®
#vhacowjcza/while/jczmzyhsca/SigmoidSigmoid%vhacowjcza/while/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#vhacowjcza/while/jczmzyhsca/SigmoidÐ
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1ReadVariableOp7vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1Õ
!vhacowjcza/while/jczmzyhsca/mul_1Mul4vhacowjcza/while/jczmzyhsca/ReadVariableOp_1:value:0vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_1Ô
!vhacowjcza/while/jczmzyhsca/add_2AddV2*vhacowjcza/while/jczmzyhsca/split:output:1%vhacowjcza/while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_2²
%vhacowjcza/while/jczmzyhsca/Sigmoid_1Sigmoid%vhacowjcza/while/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vhacowjcza/while/jczmzyhsca/Sigmoid_1Ê
!vhacowjcza/while/jczmzyhsca/mul_2Mul)vhacowjcza/while/jczmzyhsca/Sigmoid_1:y:0vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_2ª
 vhacowjcza/while/jczmzyhsca/TanhTanh*vhacowjcza/while/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 vhacowjcza/while/jczmzyhsca/TanhÎ
!vhacowjcza/while/jczmzyhsca/mul_3Mul'vhacowjcza/while/jczmzyhsca/Sigmoid:y:0$vhacowjcza/while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_3Ï
!vhacowjcza/while/jczmzyhsca/add_3AddV2%vhacowjcza/while/jczmzyhsca/mul_2:z:0%vhacowjcza/while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_3Ð
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2ReadVariableOp7vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2Ü
!vhacowjcza/while/jczmzyhsca/mul_4Mul4vhacowjcza/while/jczmzyhsca/ReadVariableOp_2:value:0%vhacowjcza/while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_4Ô
!vhacowjcza/while/jczmzyhsca/add_4AddV2*vhacowjcza/while/jczmzyhsca/split:output:3%vhacowjcza/while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_4²
%vhacowjcza/while/jczmzyhsca/Sigmoid_2Sigmoid%vhacowjcza/while/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vhacowjcza/while/jczmzyhsca/Sigmoid_2©
"vhacowjcza/while/jczmzyhsca/Tanh_1Tanh%vhacowjcza/while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"vhacowjcza/while/jczmzyhsca/Tanh_1Ò
!vhacowjcza/while/jczmzyhsca/mul_5Mul)vhacowjcza/while/jczmzyhsca/Sigmoid_2:y:0&vhacowjcza/while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_5
5vhacowjcza/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemvhacowjcza_while_placeholder_1vhacowjcza_while_placeholder%vhacowjcza/while/jczmzyhsca/mul_5:z:0*
_output_shapes
: *
element_dtype027
5vhacowjcza/while/TensorArrayV2Write/TensorListSetItemr
vhacowjcza/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
vhacowjcza/while/add/y
vhacowjcza/while/addAddV2vhacowjcza_while_placeholdervhacowjcza/while/add/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/while/addv
vhacowjcza/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
vhacowjcza/while/add_1/y­
vhacowjcza/while/add_1AddV2.vhacowjcza_while_vhacowjcza_while_loop_counter!vhacowjcza/while/add_1/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/while/add_1©
vhacowjcza/while/IdentityIdentityvhacowjcza/while/add_1:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/IdentityÇ
vhacowjcza/while/Identity_1Identity4vhacowjcza_while_vhacowjcza_while_maximum_iterations3^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/Identity_1«
vhacowjcza/while/Identity_2Identityvhacowjcza/while/add:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/Identity_2Ø
vhacowjcza/while/Identity_3IdentityEvhacowjcza/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/Identity_3É
vhacowjcza/while/Identity_4Identity%vhacowjcza/while/jczmzyhsca/mul_5:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/while/Identity_4É
vhacowjcza/while/Identity_5Identity%vhacowjcza/while/jczmzyhsca/add_3:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/while/Identity_5"?
vhacowjcza_while_identity"vhacowjcza/while/Identity:output:0"C
vhacowjcza_while_identity_1$vhacowjcza/while/Identity_1:output:0"C
vhacowjcza_while_identity_2$vhacowjcza/while/Identity_2:output:0"C
vhacowjcza_while_identity_3$vhacowjcza/while/Identity_3:output:0"C
vhacowjcza_while_identity_4$vhacowjcza/while/Identity_4:output:0"C
vhacowjcza_while_identity_5$vhacowjcza/while/Identity_5:output:0"|
;vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource=vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0"~
<vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource>vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0"z
:vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource<vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0"p
5vhacowjcza_while_jczmzyhsca_readvariableop_1_resource7vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0"p
5vhacowjcza_while_jczmzyhsca_readvariableop_2_resource7vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0"l
3vhacowjcza_while_jczmzyhsca_readvariableop_resource5vhacowjcza_while_jczmzyhsca_readvariableop_resource_0"Ô
gvhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensorivhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0"\
+vhacowjcza_while_vhacowjcza_strided_slice_1-vhacowjcza_while_vhacowjcza_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2f
1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp2j
3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp2X
*vhacowjcza/while/jczmzyhsca/ReadVariableOp*vhacowjcza/while/jczmzyhsca/ReadVariableOp2\
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1,vhacowjcza/while/jczmzyhsca/ReadVariableOp_12\
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_2159353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_wdwulgrltk_matmul_readvariableop_resource_0:	 F
3while_wdwulgrltk_matmul_1_readvariableop_resource_0:	 A
2while_wdwulgrltk_biasadd_readvariableop_resource_0:	8
*while_wdwulgrltk_readvariableop_resource_0: :
,while_wdwulgrltk_readvariableop_1_resource_0: :
,while_wdwulgrltk_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_wdwulgrltk_matmul_readvariableop_resource:	 D
1while_wdwulgrltk_matmul_1_readvariableop_resource:	 ?
0while_wdwulgrltk_biasadd_readvariableop_resource:	6
(while_wdwulgrltk_readvariableop_resource: 8
*while_wdwulgrltk_readvariableop_1_resource: 8
*while_wdwulgrltk_readvariableop_2_resource: ¢'while/wdwulgrltk/BiasAdd/ReadVariableOp¢&while/wdwulgrltk/MatMul/ReadVariableOp¢(while/wdwulgrltk/MatMul_1/ReadVariableOp¢while/wdwulgrltk/ReadVariableOp¢!while/wdwulgrltk/ReadVariableOp_1¢!while/wdwulgrltk/ReadVariableOp_2Ã
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
&while/wdwulgrltk/MatMul/ReadVariableOpReadVariableOp1while_wdwulgrltk_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/wdwulgrltk/MatMul/ReadVariableOpÑ
while/wdwulgrltk/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMulÉ
(while/wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp3while_wdwulgrltk_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/wdwulgrltk/MatMul_1/ReadVariableOpº
while/wdwulgrltk/MatMul_1MatMulwhile_placeholder_20while/wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/MatMul_1°
while/wdwulgrltk/addAddV2!while/wdwulgrltk/MatMul:product:0#while/wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/addÂ
'while/wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp2while_wdwulgrltk_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/wdwulgrltk/BiasAdd/ReadVariableOp½
while/wdwulgrltk/BiasAddBiasAddwhile/wdwulgrltk/add:z:0/while/wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/wdwulgrltk/BiasAdd
 while/wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/wdwulgrltk/split/split_dim
while/wdwulgrltk/splitSplit)while/wdwulgrltk/split/split_dim:output:0!while/wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/wdwulgrltk/split©
while/wdwulgrltk/ReadVariableOpReadVariableOp*while_wdwulgrltk_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/wdwulgrltk/ReadVariableOp£
while/wdwulgrltk/mulMul'while/wdwulgrltk/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul¦
while/wdwulgrltk/add_1AddV2while/wdwulgrltk/split:output:0while/wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_1
while/wdwulgrltk/SigmoidSigmoidwhile/wdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid¯
!while/wdwulgrltk/ReadVariableOp_1ReadVariableOp,while_wdwulgrltk_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_1©
while/wdwulgrltk/mul_1Mul)while/wdwulgrltk/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_1¨
while/wdwulgrltk/add_2AddV2while/wdwulgrltk/split:output:1while/wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_2
while/wdwulgrltk/Sigmoid_1Sigmoidwhile/wdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_1
while/wdwulgrltk/mul_2Mulwhile/wdwulgrltk/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_2
while/wdwulgrltk/TanhTanhwhile/wdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh¢
while/wdwulgrltk/mul_3Mulwhile/wdwulgrltk/Sigmoid:y:0while/wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_3£
while/wdwulgrltk/add_3AddV2while/wdwulgrltk/mul_2:z:0while/wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_3¯
!while/wdwulgrltk/ReadVariableOp_2ReadVariableOp,while_wdwulgrltk_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/wdwulgrltk/ReadVariableOp_2°
while/wdwulgrltk/mul_4Mul)while/wdwulgrltk/ReadVariableOp_2:value:0while/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_4¨
while/wdwulgrltk/add_4AddV2while/wdwulgrltk/split:output:3while/wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/add_4
while/wdwulgrltk/Sigmoid_2Sigmoidwhile/wdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Sigmoid_2
while/wdwulgrltk/Tanh_1Tanhwhile/wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/Tanh_1¦
while/wdwulgrltk/mul_5Mulwhile/wdwulgrltk/Sigmoid_2:y:0while/wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/wdwulgrltk/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/wdwulgrltk/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/wdwulgrltk/mul_5:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/wdwulgrltk/add_3:z:0(^while/wdwulgrltk/BiasAdd/ReadVariableOp'^while/wdwulgrltk/MatMul/ReadVariableOp)^while/wdwulgrltk/MatMul_1/ReadVariableOp ^while/wdwulgrltk/ReadVariableOp"^while/wdwulgrltk/ReadVariableOp_1"^while/wdwulgrltk/ReadVariableOp_2*
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
0while_wdwulgrltk_biasadd_readvariableop_resource2while_wdwulgrltk_biasadd_readvariableop_resource_0"h
1while_wdwulgrltk_matmul_1_readvariableop_resource3while_wdwulgrltk_matmul_1_readvariableop_resource_0"d
/while_wdwulgrltk_matmul_readvariableop_resource1while_wdwulgrltk_matmul_readvariableop_resource_0"Z
*while_wdwulgrltk_readvariableop_1_resource,while_wdwulgrltk_readvariableop_1_resource_0"Z
*while_wdwulgrltk_readvariableop_2_resource,while_wdwulgrltk_readvariableop_2_resource_0"V
(while_wdwulgrltk_readvariableop_resource*while_wdwulgrltk_readvariableop_resource_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/wdwulgrltk/BiasAdd/ReadVariableOp'while/wdwulgrltk/BiasAdd/ReadVariableOp2P
&while/wdwulgrltk/MatMul/ReadVariableOp&while/wdwulgrltk/MatMul/ReadVariableOp2T
(while/wdwulgrltk/MatMul_1/ReadVariableOp(while/wdwulgrltk/MatMul_1/ReadVariableOp2B
while/wdwulgrltk/ReadVariableOpwhile/wdwulgrltk/ReadVariableOp2F
!while/wdwulgrltk/ReadVariableOp_1!while/wdwulgrltk/ReadVariableOp_12F
!while/wdwulgrltk/ReadVariableOp_2!while/wdwulgrltk/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162167
inputs_0<
)wdwulgrltk_matmul_readvariableop_resource:	 >
+wdwulgrltk_matmul_1_readvariableop_resource:	 9
*wdwulgrltk_biasadd_readvariableop_resource:	0
"wdwulgrltk_readvariableop_resource: 2
$wdwulgrltk_readvariableop_1_resource: 2
$wdwulgrltk_readvariableop_2_resource: 
identity¢!wdwulgrltk/BiasAdd/ReadVariableOp¢ wdwulgrltk/MatMul/ReadVariableOp¢"wdwulgrltk/MatMul_1/ReadVariableOp¢wdwulgrltk/ReadVariableOp¢wdwulgrltk/ReadVariableOp_1¢wdwulgrltk/ReadVariableOp_2¢whileF
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
 wdwulgrltk/MatMul/ReadVariableOpReadVariableOp)wdwulgrltk_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 wdwulgrltk/MatMul/ReadVariableOp§
wdwulgrltk/MatMulMatMulstrided_slice_2:output:0(wdwulgrltk/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMulµ
"wdwulgrltk/MatMul_1/ReadVariableOpReadVariableOp+wdwulgrltk_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"wdwulgrltk/MatMul_1/ReadVariableOp£
wdwulgrltk/MatMul_1MatMulzeros:output:0*wdwulgrltk/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/MatMul_1
wdwulgrltk/addAddV2wdwulgrltk/MatMul:product:0wdwulgrltk/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/add®
!wdwulgrltk/BiasAdd/ReadVariableOpReadVariableOp*wdwulgrltk_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!wdwulgrltk/BiasAdd/ReadVariableOp¥
wdwulgrltk/BiasAddBiasAddwdwulgrltk/add:z:0)wdwulgrltk/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
wdwulgrltk/BiasAddz
wdwulgrltk/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
wdwulgrltk/split/split_dimë
wdwulgrltk/splitSplit#wdwulgrltk/split/split_dim:output:0wdwulgrltk/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
wdwulgrltk/split
wdwulgrltk/ReadVariableOpReadVariableOp"wdwulgrltk_readvariableop_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp
wdwulgrltk/mulMul!wdwulgrltk/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul
wdwulgrltk/add_1AddV2wdwulgrltk/split:output:0wdwulgrltk/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_1{
wdwulgrltk/SigmoidSigmoidwdwulgrltk/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid
wdwulgrltk/ReadVariableOp_1ReadVariableOp$wdwulgrltk_readvariableop_1_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_1
wdwulgrltk/mul_1Mul#wdwulgrltk/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_1
wdwulgrltk/add_2AddV2wdwulgrltk/split:output:1wdwulgrltk/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_2
wdwulgrltk/Sigmoid_1Sigmoidwdwulgrltk/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_1
wdwulgrltk/mul_2Mulwdwulgrltk/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_2w
wdwulgrltk/TanhTanhwdwulgrltk/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh
wdwulgrltk/mul_3Mulwdwulgrltk/Sigmoid:y:0wdwulgrltk/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_3
wdwulgrltk/add_3AddV2wdwulgrltk/mul_2:z:0wdwulgrltk/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_3
wdwulgrltk/ReadVariableOp_2ReadVariableOp$wdwulgrltk_readvariableop_2_resource*
_output_shapes
: *
dtype02
wdwulgrltk/ReadVariableOp_2
wdwulgrltk/mul_4Mul#wdwulgrltk/ReadVariableOp_2:value:0wdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_4
wdwulgrltk/add_4AddV2wdwulgrltk/split:output:3wdwulgrltk/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/add_4
wdwulgrltk/Sigmoid_2Sigmoidwdwulgrltk/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Sigmoid_2v
wdwulgrltk/Tanh_1Tanhwdwulgrltk/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/Tanh_1
wdwulgrltk/mul_5Mulwdwulgrltk/Sigmoid_2:y:0wdwulgrltk/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
wdwulgrltk/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)wdwulgrltk_matmul_readvariableop_resource+wdwulgrltk_matmul_1_readvariableop_resource*wdwulgrltk_biasadd_readvariableop_resource"wdwulgrltk_readvariableop_resource$wdwulgrltk_readvariableop_1_resource$wdwulgrltk_readvariableop_2_resource*
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
while_body_2162066*
condR
while_cond_2162065*Q
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
IdentityIdentitystrided_slice_3:output:0"^wdwulgrltk/BiasAdd/ReadVariableOp!^wdwulgrltk/MatMul/ReadVariableOp#^wdwulgrltk/MatMul_1/ReadVariableOp^wdwulgrltk/ReadVariableOp^wdwulgrltk/ReadVariableOp_1^wdwulgrltk/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!wdwulgrltk/BiasAdd/ReadVariableOp!wdwulgrltk/BiasAdd/ReadVariableOp2D
 wdwulgrltk/MatMul/ReadVariableOp wdwulgrltk/MatMul/ReadVariableOp2H
"wdwulgrltk/MatMul_1/ReadVariableOp"wdwulgrltk/MatMul_1/ReadVariableOp26
wdwulgrltk/ReadVariableOpwdwulgrltk/ReadVariableOp2:
wdwulgrltk/ReadVariableOp_1wdwulgrltk/ReadVariableOp_12:
wdwulgrltk/ReadVariableOp_2wdwulgrltk/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
p
Ê
vhacowjcza_while_body_21607782
.vhacowjcza_while_vhacowjcza_while_loop_counter8
4vhacowjcza_while_vhacowjcza_while_maximum_iterations 
vhacowjcza_while_placeholder"
vhacowjcza_while_placeholder_1"
vhacowjcza_while_placeholder_2"
vhacowjcza_while_placeholder_31
-vhacowjcza_while_vhacowjcza_strided_slice_1_0m
ivhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0O
<vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0:	Q
>vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0:	 L
=vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0:	C
5vhacowjcza_while_jczmzyhsca_readvariableop_resource_0: E
7vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0: E
7vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0: 
vhacowjcza_while_identity
vhacowjcza_while_identity_1
vhacowjcza_while_identity_2
vhacowjcza_while_identity_3
vhacowjcza_while_identity_4
vhacowjcza_while_identity_5/
+vhacowjcza_while_vhacowjcza_strided_slice_1k
gvhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensorM
:vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource:	O
<vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource:	 J
;vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource:	A
3vhacowjcza_while_jczmzyhsca_readvariableop_resource: C
5vhacowjcza_while_jczmzyhsca_readvariableop_1_resource: C
5vhacowjcza_while_jczmzyhsca_readvariableop_2_resource: ¢2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp¢1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp¢3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp¢*vhacowjcza/while/jczmzyhsca/ReadVariableOp¢,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1¢,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2Ù
Bvhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bvhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shape
4vhacowjcza/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemivhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0vhacowjcza_while_placeholderKvhacowjcza/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4vhacowjcza/while/TensorArrayV2Read/TensorListGetItemä
1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOpReadVariableOp<vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOpý
"vhacowjcza/while/jczmzyhsca/MatMulMatMul;vhacowjcza/while/TensorArrayV2Read/TensorListGetItem:item:09vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"vhacowjcza/while/jczmzyhsca/MatMulê
3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp>vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOpæ
$vhacowjcza/while/jczmzyhsca/MatMul_1MatMulvhacowjcza_while_placeholder_2;vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$vhacowjcza/while/jczmzyhsca/MatMul_1Ü
vhacowjcza/while/jczmzyhsca/addAddV2,vhacowjcza/while/jczmzyhsca/MatMul:product:0.vhacowjcza/while/jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
vhacowjcza/while/jczmzyhsca/addã
2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp=vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOpé
#vhacowjcza/while/jczmzyhsca/BiasAddBiasAdd#vhacowjcza/while/jczmzyhsca/add:z:0:vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#vhacowjcza/while/jczmzyhsca/BiasAdd
+vhacowjcza/while/jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+vhacowjcza/while/jczmzyhsca/split/split_dim¯
!vhacowjcza/while/jczmzyhsca/splitSplit4vhacowjcza/while/jczmzyhsca/split/split_dim:output:0,vhacowjcza/while/jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!vhacowjcza/while/jczmzyhsca/splitÊ
*vhacowjcza/while/jczmzyhsca/ReadVariableOpReadVariableOp5vhacowjcza_while_jczmzyhsca_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*vhacowjcza/while/jczmzyhsca/ReadVariableOpÏ
vhacowjcza/while/jczmzyhsca/mulMul2vhacowjcza/while/jczmzyhsca/ReadVariableOp:value:0vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
vhacowjcza/while/jczmzyhsca/mulÒ
!vhacowjcza/while/jczmzyhsca/add_1AddV2*vhacowjcza/while/jczmzyhsca/split:output:0#vhacowjcza/while/jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_1®
#vhacowjcza/while/jczmzyhsca/SigmoidSigmoid%vhacowjcza/while/jczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#vhacowjcza/while/jczmzyhsca/SigmoidÐ
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1ReadVariableOp7vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1Õ
!vhacowjcza/while/jczmzyhsca/mul_1Mul4vhacowjcza/while/jczmzyhsca/ReadVariableOp_1:value:0vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_1Ô
!vhacowjcza/while/jczmzyhsca/add_2AddV2*vhacowjcza/while/jczmzyhsca/split:output:1%vhacowjcza/while/jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_2²
%vhacowjcza/while/jczmzyhsca/Sigmoid_1Sigmoid%vhacowjcza/while/jczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vhacowjcza/while/jczmzyhsca/Sigmoid_1Ê
!vhacowjcza/while/jczmzyhsca/mul_2Mul)vhacowjcza/while/jczmzyhsca/Sigmoid_1:y:0vhacowjcza_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_2ª
 vhacowjcza/while/jczmzyhsca/TanhTanh*vhacowjcza/while/jczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 vhacowjcza/while/jczmzyhsca/TanhÎ
!vhacowjcza/while/jczmzyhsca/mul_3Mul'vhacowjcza/while/jczmzyhsca/Sigmoid:y:0$vhacowjcza/while/jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_3Ï
!vhacowjcza/while/jczmzyhsca/add_3AddV2%vhacowjcza/while/jczmzyhsca/mul_2:z:0%vhacowjcza/while/jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_3Ð
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2ReadVariableOp7vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2Ü
!vhacowjcza/while/jczmzyhsca/mul_4Mul4vhacowjcza/while/jczmzyhsca/ReadVariableOp_2:value:0%vhacowjcza/while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_4Ô
!vhacowjcza/while/jczmzyhsca/add_4AddV2*vhacowjcza/while/jczmzyhsca/split:output:3%vhacowjcza/while/jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/add_4²
%vhacowjcza/while/jczmzyhsca/Sigmoid_2Sigmoid%vhacowjcza/while/jczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%vhacowjcza/while/jczmzyhsca/Sigmoid_2©
"vhacowjcza/while/jczmzyhsca/Tanh_1Tanh%vhacowjcza/while/jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"vhacowjcza/while/jczmzyhsca/Tanh_1Ò
!vhacowjcza/while/jczmzyhsca/mul_5Mul)vhacowjcza/while/jczmzyhsca/Sigmoid_2:y:0&vhacowjcza/while/jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!vhacowjcza/while/jczmzyhsca/mul_5
5vhacowjcza/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemvhacowjcza_while_placeholder_1vhacowjcza_while_placeholder%vhacowjcza/while/jczmzyhsca/mul_5:z:0*
_output_shapes
: *
element_dtype027
5vhacowjcza/while/TensorArrayV2Write/TensorListSetItemr
vhacowjcza/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
vhacowjcza/while/add/y
vhacowjcza/while/addAddV2vhacowjcza_while_placeholdervhacowjcza/while/add/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/while/addv
vhacowjcza/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
vhacowjcza/while/add_1/y­
vhacowjcza/while/add_1AddV2.vhacowjcza_while_vhacowjcza_while_loop_counter!vhacowjcza/while/add_1/y:output:0*
T0*
_output_shapes
: 2
vhacowjcza/while/add_1©
vhacowjcza/while/IdentityIdentityvhacowjcza/while/add_1:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/IdentityÇ
vhacowjcza/while/Identity_1Identity4vhacowjcza_while_vhacowjcza_while_maximum_iterations3^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/Identity_1«
vhacowjcza/while/Identity_2Identityvhacowjcza/while/add:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/Identity_2Ø
vhacowjcza/while/Identity_3IdentityEvhacowjcza/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*
_output_shapes
: 2
vhacowjcza/while/Identity_3É
vhacowjcza/while/Identity_4Identity%vhacowjcza/while/jczmzyhsca/mul_5:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/while/Identity_4É
vhacowjcza/while/Identity_5Identity%vhacowjcza/while/jczmzyhsca/add_3:z:03^vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2^vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp4^vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp+^vhacowjcza/while/jczmzyhsca/ReadVariableOp-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_1-^vhacowjcza/while/jczmzyhsca/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
vhacowjcza/while/Identity_5"?
vhacowjcza_while_identity"vhacowjcza/while/Identity:output:0"C
vhacowjcza_while_identity_1$vhacowjcza/while/Identity_1:output:0"C
vhacowjcza_while_identity_2$vhacowjcza/while/Identity_2:output:0"C
vhacowjcza_while_identity_3$vhacowjcza/while/Identity_3:output:0"C
vhacowjcza_while_identity_4$vhacowjcza/while/Identity_4:output:0"C
vhacowjcza_while_identity_5$vhacowjcza/while/Identity_5:output:0"|
;vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource=vhacowjcza_while_jczmzyhsca_biasadd_readvariableop_resource_0"~
<vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource>vhacowjcza_while_jczmzyhsca_matmul_1_readvariableop_resource_0"z
:vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource<vhacowjcza_while_jczmzyhsca_matmul_readvariableop_resource_0"p
5vhacowjcza_while_jczmzyhsca_readvariableop_1_resource7vhacowjcza_while_jczmzyhsca_readvariableop_1_resource_0"p
5vhacowjcza_while_jczmzyhsca_readvariableop_2_resource7vhacowjcza_while_jczmzyhsca_readvariableop_2_resource_0"l
3vhacowjcza_while_jczmzyhsca_readvariableop_resource5vhacowjcza_while_jczmzyhsca_readvariableop_resource_0"Ô
gvhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensorivhacowjcza_while_tensorarrayv2read_tensorlistgetitem_vhacowjcza_tensorarrayunstack_tensorlistfromtensor_0"\
+vhacowjcza_while_vhacowjcza_strided_slice_1-vhacowjcza_while_vhacowjcza_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2vhacowjcza/while/jczmzyhsca/BiasAdd/ReadVariableOp2f
1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp1vhacowjcza/while/jczmzyhsca/MatMul/ReadVariableOp2j
3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp3vhacowjcza/while/jczmzyhsca/MatMul_1/ReadVariableOp2X
*vhacowjcza/while/jczmzyhsca/ReadVariableOp*vhacowjcza/while/jczmzyhsca/ReadVariableOp2\
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_1,vhacowjcza/while/jczmzyhsca/ReadVariableOp_12\
,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2,vhacowjcza/while/jczmzyhsca/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2159261

inputs<
)jczmzyhsca_matmul_readvariableop_resource:	>
+jczmzyhsca_matmul_1_readvariableop_resource:	 9
*jczmzyhsca_biasadd_readvariableop_resource:	0
"jczmzyhsca_readvariableop_resource: 2
$jczmzyhsca_readvariableop_1_resource: 2
$jczmzyhsca_readvariableop_2_resource: 
identity¢!jczmzyhsca/BiasAdd/ReadVariableOp¢ jczmzyhsca/MatMul/ReadVariableOp¢"jczmzyhsca/MatMul_1/ReadVariableOp¢jczmzyhsca/ReadVariableOp¢jczmzyhsca/ReadVariableOp_1¢jczmzyhsca/ReadVariableOp_2¢whileD
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
 jczmzyhsca/MatMul/ReadVariableOpReadVariableOp)jczmzyhsca_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 jczmzyhsca/MatMul/ReadVariableOp§
jczmzyhsca/MatMulMatMulstrided_slice_2:output:0(jczmzyhsca/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMulµ
"jczmzyhsca/MatMul_1/ReadVariableOpReadVariableOp+jczmzyhsca_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"jczmzyhsca/MatMul_1/ReadVariableOp£
jczmzyhsca/MatMul_1MatMulzeros:output:0*jczmzyhsca/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/MatMul_1
jczmzyhsca/addAddV2jczmzyhsca/MatMul:product:0jczmzyhsca/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/add®
!jczmzyhsca/BiasAdd/ReadVariableOpReadVariableOp*jczmzyhsca_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!jczmzyhsca/BiasAdd/ReadVariableOp¥
jczmzyhsca/BiasAddBiasAddjczmzyhsca/add:z:0)jczmzyhsca/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
jczmzyhsca/BiasAddz
jczmzyhsca/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
jczmzyhsca/split/split_dimë
jczmzyhsca/splitSplit#jczmzyhsca/split/split_dim:output:0jczmzyhsca/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
jczmzyhsca/split
jczmzyhsca/ReadVariableOpReadVariableOp"jczmzyhsca_readvariableop_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp
jczmzyhsca/mulMul!jczmzyhsca/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul
jczmzyhsca/add_1AddV2jczmzyhsca/split:output:0jczmzyhsca/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_1{
jczmzyhsca/SigmoidSigmoidjczmzyhsca/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid
jczmzyhsca/ReadVariableOp_1ReadVariableOp$jczmzyhsca_readvariableop_1_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_1
jczmzyhsca/mul_1Mul#jczmzyhsca/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_1
jczmzyhsca/add_2AddV2jczmzyhsca/split:output:1jczmzyhsca/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_2
jczmzyhsca/Sigmoid_1Sigmoidjczmzyhsca/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_1
jczmzyhsca/mul_2Muljczmzyhsca/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_2w
jczmzyhsca/TanhTanhjczmzyhsca/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh
jczmzyhsca/mul_3Muljczmzyhsca/Sigmoid:y:0jczmzyhsca/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_3
jczmzyhsca/add_3AddV2jczmzyhsca/mul_2:z:0jczmzyhsca/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_3
jczmzyhsca/ReadVariableOp_2ReadVariableOp$jczmzyhsca_readvariableop_2_resource*
_output_shapes
: *
dtype02
jczmzyhsca/ReadVariableOp_2
jczmzyhsca/mul_4Mul#jczmzyhsca/ReadVariableOp_2:value:0jczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_4
jczmzyhsca/add_4AddV2jczmzyhsca/split:output:3jczmzyhsca/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/add_4
jczmzyhsca/Sigmoid_2Sigmoidjczmzyhsca/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Sigmoid_2v
jczmzyhsca/Tanh_1Tanhjczmzyhsca/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/Tanh_1
jczmzyhsca/mul_5Muljczmzyhsca/Sigmoid_2:y:0jczmzyhsca/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
jczmzyhsca/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)jczmzyhsca_matmul_readvariableop_resource+jczmzyhsca_matmul_1_readvariableop_resource*jczmzyhsca_biasadd_readvariableop_resource"jczmzyhsca_readvariableop_resource$jczmzyhsca_readvariableop_1_resource$jczmzyhsca_readvariableop_2_resource*
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
while_body_2159160*
condR
while_cond_2159159*Q
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
IdentityIdentitytranspose_1:y:0"^jczmzyhsca/BiasAdd/ReadVariableOp!^jczmzyhsca/MatMul/ReadVariableOp#^jczmzyhsca/MatMul_1/ReadVariableOp^jczmzyhsca/ReadVariableOp^jczmzyhsca/ReadVariableOp_1^jczmzyhsca/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!jczmzyhsca/BiasAdd/ReadVariableOp!jczmzyhsca/BiasAdd/ReadVariableOp2D
 jczmzyhsca/MatMul/ReadVariableOp jczmzyhsca/MatMul/ReadVariableOp2H
"jczmzyhsca/MatMul_1/ReadVariableOp"jczmzyhsca/MatMul_1/ReadVariableOp26
jczmzyhsca/ReadVariableOpjczmzyhsca/ReadVariableOp2:
jczmzyhsca/ReadVariableOp_1jczmzyhsca/ReadVariableOp_12:
jczmzyhsca/ReadVariableOp_2jczmzyhsca/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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

aveeivcxur;
serving_default_aveeivcxur:0ÿÿÿÿÿÿÿÿÿ>

kekwghyimt0
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
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
_default_save_signature
+&call_and_return_all_conditional_losses
__call__"ÂA
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aveeivcxur"}}, {"class_name": "Conv1D", "config": {"name": "vfwtupxpzf", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "ojzbgzevue", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "vhacowjcza", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jczmzyhsca", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "zgrdiwrovx", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "wdwulgrltk", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "kekwghyimt", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "aveeivcxur"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aveeivcxur"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "vfwtupxpzf", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "ojzbgzevue", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "vhacowjcza", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jczmzyhsca", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "zgrdiwrovx", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "wdwulgrltk", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "kekwghyimt", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
Ì

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"¥

_tf_keras_layer
{"name": "vfwtupxpzf", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "vfwtupxpzf", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"name": "ojzbgzevue", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "ojzbgzevue", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
­
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layerä{"name": "vhacowjcza", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "vhacowjcza", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jczmzyhsca", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
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
_tf_keras_rnn_layerê{"name": "zgrdiwrovx", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "zgrdiwrovx", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "wdwulgrltk", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "kekwghyimt", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "kekwghyimt", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
':%2vfwtupxpzf/kernel
:2vfwtupxpzf/bias
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
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"name": "jczmzyhsca", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "jczmzyhsca", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
+&call_and_return_all_conditional_losses
__call__"Ú
_tf_keras_layerÀ{"name": "wdwulgrltk", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "wdwulgrltk", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
#:! 2kekwghyimt/kernel
:2kekwghyimt/bias
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
/:-	2vhacowjcza/jczmzyhsca/kernel
9:7	 2&vhacowjcza/jczmzyhsca/recurrent_kernel
):'2vhacowjcza/jczmzyhsca/bias
?:= 21vhacowjcza/jczmzyhsca/input_gate_peephole_weights
@:> 22vhacowjcza/jczmzyhsca/forget_gate_peephole_weights
@:> 22vhacowjcza/jczmzyhsca/output_gate_peephole_weights
/:-	 2zgrdiwrovx/wdwulgrltk/kernel
9:7	 2&zgrdiwrovx/wdwulgrltk/recurrent_kernel
):'2zgrdiwrovx/wdwulgrltk/bias
?:= 21zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights
@:> 22zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights
@:> 22zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights
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
1:/2RMSprop/vfwtupxpzf/kernel/rms
':%2RMSprop/vfwtupxpzf/bias/rms
-:+ 2RMSprop/kekwghyimt/kernel/rms
':%2RMSprop/kekwghyimt/bias/rms
9:7	2(RMSprop/vhacowjcza/jczmzyhsca/kernel/rms
C:A	 22RMSprop/vhacowjcza/jczmzyhsca/recurrent_kernel/rms
3:12&RMSprop/vhacowjcza/jczmzyhsca/bias/rms
I:G 2=RMSprop/vhacowjcza/jczmzyhsca/input_gate_peephole_weights/rms
J:H 2>RMSprop/vhacowjcza/jczmzyhsca/forget_gate_peephole_weights/rms
J:H 2>RMSprop/vhacowjcza/jczmzyhsca/output_gate_peephole_weights/rms
9:7	 2(RMSprop/zgrdiwrovx/wdwulgrltk/kernel/rms
C:A	 22RMSprop/zgrdiwrovx/wdwulgrltk/recurrent_kernel/rms
3:12&RMSprop/zgrdiwrovx/wdwulgrltk/bias/rms
I:G 2=RMSprop/zgrdiwrovx/wdwulgrltk/input_gate_peephole_weights/rms
J:H 2>RMSprop/zgrdiwrovx/wdwulgrltk/forget_gate_peephole_weights/rms
J:H 2>RMSprop/zgrdiwrovx/wdwulgrltk/output_gate_peephole_weights/rms
ë2è
"__inference__wrapped_model_2157501Á
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

aveeivcxurÿÿÿÿÿÿÿÿÿ
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_2160657
G__inference_sequential_layer_call_and_return_conditional_losses_2161061
G__inference_sequential_layer_call_and_return_conditional_losses_2160167
G__inference_sequential_layer_call_and_return_conditional_losses_2160208À
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
,__inference_sequential_layer_call_fn_2159520
,__inference_sequential_layer_call_fn_2161098
,__inference_sequential_layer_call_fn_2161135
,__inference_sequential_layer_call_fn_2160126À
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
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_2161172¢
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
,__inference_vfwtupxpzf_layer_call_fn_2161181¢
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
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_2161194¢
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
,__inference_ojzbgzevue_layer_call_fn_2161199¢
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161379
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161559
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161739
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161919æ
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
,__inference_vhacowjcza_layer_call_fn_2161936
,__inference_vhacowjcza_layer_call_fn_2161953
,__inference_vhacowjcza_layer_call_fn_2161970
,__inference_vhacowjcza_layer_call_fn_2161987æ
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162167
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162347
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162527
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162707æ
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
,__inference_zgrdiwrovx_layer_call_fn_2162724
,__inference_zgrdiwrovx_layer_call_fn_2162741
,__inference_zgrdiwrovx_layer_call_fn_2162758
,__inference_zgrdiwrovx_layer_call_fn_2162775æ
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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_2162785¢
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
,__inference_kekwghyimt_layer_call_fn_2162794¢
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
%__inference_signature_wrapper_2160253
aveeivcxur"
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2162838
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2162882¾
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
,__inference_jczmzyhsca_layer_call_fn_2162905
,__inference_jczmzyhsca_layer_call_fn_2162928¾
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2162972
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2163016¾
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
,__inference_wdwulgrltk_layer_call_fn_2163039
,__inference_wdwulgrltk_layer_call_fn_2163062¾
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
"__inference__wrapped_model_2157501-./012345678"#;¢8
1¢.
,)

aveeivcxurÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

kekwghyimt$!

kekwghyimtÿÿÿÿÿÿÿÿÿÌ
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2162838-./012¢}
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
G__inference_jczmzyhsca_layer_call_and_return_conditional_losses_2162882-./012¢}
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
,__inference_jczmzyhsca_layer_call_fn_2162905ð-./012¢}
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
,__inference_jczmzyhsca_layer_call_fn_2162928ð-./012¢}
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
G__inference_kekwghyimt_layer_call_and_return_conditional_losses_2162785\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_kekwghyimt_layer_call_fn_2162794O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¯
G__inference_ojzbgzevue_layer_call_and_return_conditional_losses_2161194d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_ojzbgzevue_layer_call_fn_2161199W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÉ
G__inference_sequential_layer_call_and_return_conditional_losses_2160167~-./012345678"#C¢@
9¢6
,)

aveeivcxurÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_layer_call_and_return_conditional_losses_2160208~-./012345678"#C¢@
9¢6
,)

aveeivcxurÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_2160657z-./012345678"#?¢<
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
G__inference_sequential_layer_call_and_return_conditional_losses_2161061z-./012345678"#?¢<
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
,__inference_sequential_layer_call_fn_2159520q-./012345678"#C¢@
9¢6
,)

aveeivcxurÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_layer_call_fn_2160126q-./012345678"#C¢@
9¢6
,)

aveeivcxurÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_2161098m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_2161135m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
%__inference_signature_wrapper_2160253-./012345678"#I¢F
¢ 
?ª<
:

aveeivcxur,)

aveeivcxurÿÿÿÿÿÿÿÿÿ"7ª4
2

kekwghyimt$!

kekwghyimtÿÿÿÿÿÿÿÿÿ·
G__inference_vfwtupxpzf_layer_call_and_return_conditional_losses_2161172l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_vfwtupxpzf_layer_call_fn_2161181_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÝ
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161379-./012S¢P
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161559-./012S¢P
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161739x-./012C¢@
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
G__inference_vhacowjcza_layer_call_and_return_conditional_losses_2161919x-./012C¢@
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
,__inference_vhacowjcza_layer_call_fn_2161936-./012S¢P
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
,__inference_vhacowjcza_layer_call_fn_2161953-./012S¢P
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
,__inference_vhacowjcza_layer_call_fn_2161970k-./012C¢@
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
,__inference_vhacowjcza_layer_call_fn_2161987k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ Ì
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2162972345678¢}
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
G__inference_wdwulgrltk_layer_call_and_return_conditional_losses_2163016345678¢}
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
,__inference_wdwulgrltk_layer_call_fn_2163039ð345678¢}
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
,__inference_wdwulgrltk_layer_call_fn_2163062ð345678¢}
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162167345678S¢P
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162347345678S¢P
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162527t345678C¢@
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
G__inference_zgrdiwrovx_layer_call_and_return_conditional_losses_2162707t345678C¢@
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
,__inference_zgrdiwrovx_layer_call_fn_2162724w345678S¢P
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
,__inference_zgrdiwrovx_layer_call_fn_2162741w345678S¢P
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
,__inference_zgrdiwrovx_layer_call_fn_2162758g345678C¢@
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
,__inference_zgrdiwrovx_layer_call_fn_2162775g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ 