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
ekzorghjta/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameekzorghjta/kernel
{
%ekzorghjta/kernel/Read/ReadVariableOpReadVariableOpekzorghjta/kernel*"
_output_shapes
:*
dtype0
v
ekzorghjta/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameekzorghjta/bias
o
#ekzorghjta/bias/Read/ReadVariableOpReadVariableOpekzorghjta/bias*
_output_shapes
:*
dtype0
~
pemnqztknd/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namepemnqztknd/kernel
w
%pemnqztknd/kernel/Read/ReadVariableOpReadVariableOppemnqztknd/kernel*
_output_shapes

: *
dtype0
v
pemnqztknd/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namepemnqztknd/bias
o
#pemnqztknd/bias/Read/ReadVariableOpReadVariableOppemnqztknd/bias*
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
zeuewnmlut/kfdgklwsil/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namezeuewnmlut/kfdgklwsil/kernel

0zeuewnmlut/kfdgklwsil/kernel/Read/ReadVariableOpReadVariableOpzeuewnmlut/kfdgklwsil/kernel*
_output_shapes
:	*
dtype0
©
&zeuewnmlut/kfdgklwsil/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&zeuewnmlut/kfdgklwsil/recurrent_kernel
¢
:zeuewnmlut/kfdgklwsil/recurrent_kernel/Read/ReadVariableOpReadVariableOp&zeuewnmlut/kfdgklwsil/recurrent_kernel*
_output_shapes
:	 *
dtype0

zeuewnmlut/kfdgklwsil/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namezeuewnmlut/kfdgklwsil/bias

.zeuewnmlut/kfdgklwsil/bias/Read/ReadVariableOpReadVariableOpzeuewnmlut/kfdgklwsil/bias*
_output_shapes	
:*
dtype0
º
1zeuewnmlut/kfdgklwsil/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31zeuewnmlut/kfdgklwsil/input_gate_peephole_weights
³
Ezeuewnmlut/kfdgklwsil/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1zeuewnmlut/kfdgklwsil/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2zeuewnmlut/kfdgklwsil/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights
µ
Fzeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2zeuewnmlut/kfdgklwsil/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42zeuewnmlut/kfdgklwsil/output_gate_peephole_weights
µ
Fzeuewnmlut/kfdgklwsil/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2zeuewnmlut/kfdgklwsil/output_gate_peephole_weights*
_output_shapes
: *
dtype0

ipeigywbwc/bdgxhkvdqy/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_nameipeigywbwc/bdgxhkvdqy/kernel

0ipeigywbwc/bdgxhkvdqy/kernel/Read/ReadVariableOpReadVariableOpipeigywbwc/bdgxhkvdqy/kernel*
_output_shapes
:	 *
dtype0
©
&ipeigywbwc/bdgxhkvdqy/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&ipeigywbwc/bdgxhkvdqy/recurrent_kernel
¢
:ipeigywbwc/bdgxhkvdqy/recurrent_kernel/Read/ReadVariableOpReadVariableOp&ipeigywbwc/bdgxhkvdqy/recurrent_kernel*
_output_shapes
:	 *
dtype0

ipeigywbwc/bdgxhkvdqy/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameipeigywbwc/bdgxhkvdqy/bias

.ipeigywbwc/bdgxhkvdqy/bias/Read/ReadVariableOpReadVariableOpipeigywbwc/bdgxhkvdqy/bias*
_output_shapes	
:*
dtype0
º
1ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights
³
Eipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights
µ
Fipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights
µ
Fipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights*
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
RMSprop/ekzorghjta/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/ekzorghjta/kernel/rms

1RMSprop/ekzorghjta/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/ekzorghjta/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/ekzorghjta/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/ekzorghjta/bias/rms

/RMSprop/ekzorghjta/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/ekzorghjta/bias/rms*
_output_shapes
:*
dtype0

RMSprop/pemnqztknd/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/pemnqztknd/kernel/rms

1RMSprop/pemnqztknd/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/pemnqztknd/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/pemnqztknd/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/pemnqztknd/bias/rms

/RMSprop/pemnqztknd/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/pemnqztknd/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/zeuewnmlut/kfdgklwsil/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/zeuewnmlut/kfdgklwsil/kernel/rms
¦
<RMSprop/zeuewnmlut/kfdgklwsil/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/zeuewnmlut/kfdgklwsil/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rms
º
FRMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/zeuewnmlut/kfdgklwsil/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/zeuewnmlut/kfdgklwsil/bias/rms

:RMSprop/zeuewnmlut/kfdgklwsil/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/zeuewnmlut/kfdgklwsil/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rms
Ë
QRMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rms
Í
RRMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rms
Í
RRMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rms
¦
<RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rms
º
FRMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rms

:RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rms
Ë
QRMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rms
Í
RRMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rms
Í
RRMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rms*
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
VARIABLE_VALUEekzorghjta/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEekzorghjta/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEpemnqztknd/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEpemnqztknd/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEzeuewnmlut/kfdgklwsil/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&zeuewnmlut/kfdgklwsil/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEzeuewnmlut/kfdgklwsil/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1zeuewnmlut/kfdgklwsil/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2zeuewnmlut/kfdgklwsil/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEipeigywbwc/bdgxhkvdqy/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&ipeigywbwc/bdgxhkvdqy/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEipeigywbwc/bdgxhkvdqy/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUERMSprop/ekzorghjta/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/ekzorghjta/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/pemnqztknd/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/pemnqztknd/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/zeuewnmlut/kfdgklwsil/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/zeuewnmlut/kfdgklwsil/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_kjggqknufbPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_kjggqknufbekzorghjta/kernelekzorghjta/biaszeuewnmlut/kfdgklwsil/kernel&zeuewnmlut/kfdgklwsil/recurrent_kernelzeuewnmlut/kfdgklwsil/bias1zeuewnmlut/kfdgklwsil/input_gate_peephole_weights2zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights2zeuewnmlut/kfdgklwsil/output_gate_peephole_weightsipeigywbwc/bdgxhkvdqy/kernel&ipeigywbwc/bdgxhkvdqy/recurrent_kernelipeigywbwc/bdgxhkvdqy/bias1ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights2ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights2ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weightspemnqztknd/kernelpemnqztknd/bias*
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
$__inference_signature_wrapper_363893
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%ekzorghjta/kernel/Read/ReadVariableOp#ekzorghjta/bias/Read/ReadVariableOp%pemnqztknd/kernel/Read/ReadVariableOp#pemnqztknd/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0zeuewnmlut/kfdgklwsil/kernel/Read/ReadVariableOp:zeuewnmlut/kfdgklwsil/recurrent_kernel/Read/ReadVariableOp.zeuewnmlut/kfdgklwsil/bias/Read/ReadVariableOpEzeuewnmlut/kfdgklwsil/input_gate_peephole_weights/Read/ReadVariableOpFzeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/Read/ReadVariableOpFzeuewnmlut/kfdgklwsil/output_gate_peephole_weights/Read/ReadVariableOp0ipeigywbwc/bdgxhkvdqy/kernel/Read/ReadVariableOp:ipeigywbwc/bdgxhkvdqy/recurrent_kernel/Read/ReadVariableOp.ipeigywbwc/bdgxhkvdqy/bias/Read/ReadVariableOpEipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/Read/ReadVariableOpFipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/Read/ReadVariableOpFipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/ekzorghjta/kernel/rms/Read/ReadVariableOp/RMSprop/ekzorghjta/bias/rms/Read/ReadVariableOp1RMSprop/pemnqztknd/kernel/rms/Read/ReadVariableOp/RMSprop/pemnqztknd/bias/rms/Read/ReadVariableOp<RMSprop/zeuewnmlut/kfdgklwsil/kernel/rms/Read/ReadVariableOpFRMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/zeuewnmlut/kfdgklwsil/bias/rms/Read/ReadVariableOpQRMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rms/Read/ReadVariableOpFRMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rms/Read/ReadVariableOpQRMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
__inference__traced_save_366842
å
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameekzorghjta/kernelekzorghjta/biaspemnqztknd/kernelpemnqztknd/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhozeuewnmlut/kfdgklwsil/kernel&zeuewnmlut/kfdgklwsil/recurrent_kernelzeuewnmlut/kfdgklwsil/bias1zeuewnmlut/kfdgklwsil/input_gate_peephole_weights2zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights2zeuewnmlut/kfdgklwsil/output_gate_peephole_weightsipeigywbwc/bdgxhkvdqy/kernel&ipeigywbwc/bdgxhkvdqy/recurrent_kernelipeigywbwc/bdgxhkvdqy/bias1ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights2ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights2ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weightstotalcountRMSprop/ekzorghjta/kernel/rmsRMSprop/ekzorghjta/bias/rmsRMSprop/pemnqztknd/kernel/rmsRMSprop/pemnqztknd/bias/rms(RMSprop/zeuewnmlut/kfdgklwsil/kernel/rms2RMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rms&RMSprop/zeuewnmlut/kfdgklwsil/bias/rms=RMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rms>RMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rms>RMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rms(RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rms2RMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rms&RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rms=RMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rms>RMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rms>RMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rms*3
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
"__inference__traced_restore_366969¥Û-


å
while_cond_362992
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362992___redundant_placeholder04
0while_while_cond_362992___redundant_placeholder14
0while_while_cond_362992___redundant_placeholder24
0while_while_cond_362992___redundant_placeholder34
0while_while_cond_362992___redundant_placeholder44
0while_while_cond_362992___redundant_placeholder54
0while_while_cond_362992___redundant_placeholder6
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
h

F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_363583

inputs<
)kfdgklwsil_matmul_readvariableop_resource:	>
+kfdgklwsil_matmul_1_readvariableop_resource:	 9
*kfdgklwsil_biasadd_readvariableop_resource:	0
"kfdgklwsil_readvariableop_resource: 2
$kfdgklwsil_readvariableop_1_resource: 2
$kfdgklwsil_readvariableop_2_resource: 
identity¢!kfdgklwsil/BiasAdd/ReadVariableOp¢ kfdgklwsil/MatMul/ReadVariableOp¢"kfdgklwsil/MatMul_1/ReadVariableOp¢kfdgklwsil/ReadVariableOp¢kfdgklwsil/ReadVariableOp_1¢kfdgklwsil/ReadVariableOp_2¢whileD
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
 kfdgklwsil/MatMul/ReadVariableOpReadVariableOp)kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 kfdgklwsil/MatMul/ReadVariableOp§
kfdgklwsil/MatMulMatMulstrided_slice_2:output:0(kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMulµ
"kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp+kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kfdgklwsil/MatMul_1/ReadVariableOp£
kfdgklwsil/MatMul_1MatMulzeros:output:0*kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMul_1
kfdgklwsil/addAddV2kfdgklwsil/MatMul:product:0kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/add®
!kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp*kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kfdgklwsil/BiasAdd/ReadVariableOp¥
kfdgklwsil/BiasAddBiasAddkfdgklwsil/add:z:0)kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/BiasAddz
kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kfdgklwsil/split/split_dimë
kfdgklwsil/splitSplit#kfdgklwsil/split/split_dim:output:0kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kfdgklwsil/split
kfdgklwsil/ReadVariableOpReadVariableOp"kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp
kfdgklwsil/mulMul!kfdgklwsil/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul
kfdgklwsil/add_1AddV2kfdgklwsil/split:output:0kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_1{
kfdgklwsil/SigmoidSigmoidkfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid
kfdgklwsil/ReadVariableOp_1ReadVariableOp$kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_1
kfdgklwsil/mul_1Mul#kfdgklwsil/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_1
kfdgklwsil/add_2AddV2kfdgklwsil/split:output:1kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_2
kfdgklwsil/Sigmoid_1Sigmoidkfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_1
kfdgklwsil/mul_2Mulkfdgklwsil/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_2w
kfdgklwsil/TanhTanhkfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh
kfdgklwsil/mul_3Mulkfdgklwsil/Sigmoid:y:0kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_3
kfdgklwsil/add_3AddV2kfdgklwsil/mul_2:z:0kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_3
kfdgklwsil/ReadVariableOp_2ReadVariableOp$kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_2
kfdgklwsil/mul_4Mul#kfdgklwsil/ReadVariableOp_2:value:0kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_4
kfdgklwsil/add_4AddV2kfdgklwsil/split:output:3kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_4
kfdgklwsil/Sigmoid_2Sigmoidkfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_2v
kfdgklwsil/Tanh_1Tanhkfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh_1
kfdgklwsil/mul_5Mulkfdgklwsil/Sigmoid_2:y:0kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kfdgklwsil_matmul_readvariableop_resource+kfdgklwsil_matmul_1_readvariableop_resource*kfdgklwsil_biasadd_readvariableop_resource"kfdgklwsil_readvariableop_resource$kfdgklwsil_readvariableop_1_resource$kfdgklwsil_readvariableop_2_resource*
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
while_body_363482*
condR
while_cond_363481*Q
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
IdentityIdentitytranspose_1:y:0"^kfdgklwsil/BiasAdd/ReadVariableOp!^kfdgklwsil/MatMul/ReadVariableOp#^kfdgklwsil/MatMul_1/ReadVariableOp^kfdgklwsil/ReadVariableOp^kfdgklwsil/ReadVariableOp_1^kfdgklwsil/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!kfdgklwsil/BiasAdd/ReadVariableOp!kfdgklwsil/BiasAdd/ReadVariableOp2D
 kfdgklwsil/MatMul/ReadVariableOp kfdgklwsil/MatMul/ReadVariableOp2H
"kfdgklwsil/MatMul_1/ReadVariableOp"kfdgklwsil/MatMul_1/ReadVariableOp26
kfdgklwsil/ReadVariableOpkfdgklwsil/ReadVariableOp2:
kfdgklwsil/ReadVariableOp_1kfdgklwsil/ReadVariableOp_12:
kfdgklwsil/ReadVariableOp_2kfdgklwsil/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ßY

while_body_366066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 F
3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 A
2while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	8
*while_bdgxhkvdqy_readvariableop_resource_0: :
,while_bdgxhkvdqy_readvariableop_1_resource_0: :
,while_bdgxhkvdqy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_bdgxhkvdqy_matmul_readvariableop_resource:	 D
1while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 ?
0while_bdgxhkvdqy_biasadd_readvariableop_resource:	6
(while_bdgxhkvdqy_readvariableop_resource: 8
*while_bdgxhkvdqy_readvariableop_1_resource: 8
*while_bdgxhkvdqy_readvariableop_2_resource: ¢'while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢&while/bdgxhkvdqy/MatMul/ReadVariableOp¢(while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢while/bdgxhkvdqy/ReadVariableOp¢!while/bdgxhkvdqy/ReadVariableOp_1¢!while/bdgxhkvdqy/ReadVariableOp_2Ã
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
&while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp1while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/bdgxhkvdqy/MatMul/ReadVariableOpÑ
while/bdgxhkvdqy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMulÉ
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpº
while/bdgxhkvdqy/MatMul_1MatMulwhile_placeholder_20while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMul_1°
while/bdgxhkvdqy/addAddV2!while/bdgxhkvdqy/MatMul:product:0#while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/addÂ
'while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp2while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp½
while/bdgxhkvdqy/BiasAddBiasAddwhile/bdgxhkvdqy/add:z:0/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/BiasAdd
 while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/bdgxhkvdqy/split/split_dim
while/bdgxhkvdqy/splitSplit)while/bdgxhkvdqy/split/split_dim:output:0!while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/bdgxhkvdqy/split©
while/bdgxhkvdqy/ReadVariableOpReadVariableOp*while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/bdgxhkvdqy/ReadVariableOp£
while/bdgxhkvdqy/mulMul'while/bdgxhkvdqy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul¦
while/bdgxhkvdqy/add_1AddV2while/bdgxhkvdqy/split:output:0while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_1
while/bdgxhkvdqy/SigmoidSigmoidwhile/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid¯
!while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp,while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_1©
while/bdgxhkvdqy/mul_1Mul)while/bdgxhkvdqy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_1¨
while/bdgxhkvdqy/add_2AddV2while/bdgxhkvdqy/split:output:1while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_2
while/bdgxhkvdqy/Sigmoid_1Sigmoidwhile/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_1
while/bdgxhkvdqy/mul_2Mulwhile/bdgxhkvdqy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_2
while/bdgxhkvdqy/TanhTanhwhile/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh¢
while/bdgxhkvdqy/mul_3Mulwhile/bdgxhkvdqy/Sigmoid:y:0while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_3£
while/bdgxhkvdqy/add_3AddV2while/bdgxhkvdqy/mul_2:z:0while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_3¯
!while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp,while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_2°
while/bdgxhkvdqy/mul_4Mul)while/bdgxhkvdqy/ReadVariableOp_2:value:0while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_4¨
while/bdgxhkvdqy/add_4AddV2while/bdgxhkvdqy/split:output:3while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_4
while/bdgxhkvdqy/Sigmoid_2Sigmoidwhile/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_2
while/bdgxhkvdqy/Tanh_1Tanhwhile/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh_1¦
while/bdgxhkvdqy/mul_5Mulwhile/bdgxhkvdqy/Sigmoid_2:y:0while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/bdgxhkvdqy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/bdgxhkvdqy/mul_5:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/bdgxhkvdqy/add_3:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_bdgxhkvdqy_biasadd_readvariableop_resource2while_bdgxhkvdqy_biasadd_readvariableop_resource_0"h
1while_bdgxhkvdqy_matmul_1_readvariableop_resource3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"d
/while_bdgxhkvdqy_matmul_readvariableop_resource1while_bdgxhkvdqy_matmul_readvariableop_resource_0"Z
*while_bdgxhkvdqy_readvariableop_1_resource,while_bdgxhkvdqy_readvariableop_1_resource_0"Z
*while_bdgxhkvdqy_readvariableop_2_resource,while_bdgxhkvdqy_readvariableop_2_resource_0"V
(while_bdgxhkvdqy_readvariableop_resource*while_bdgxhkvdqy_readvariableop_resource_0")
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
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp'while/bdgxhkvdqy/BiasAdd/ReadVariableOp2P
&while/bdgxhkvdqy/MatMul/ReadVariableOp&while/bdgxhkvdqy/MatMul/ReadVariableOp2T
(while/bdgxhkvdqy/MatMul_1/ReadVariableOp(while/bdgxhkvdqy/MatMul_1/ReadVariableOp2B
while/bdgxhkvdqy/ReadVariableOpwhile/bdgxhkvdqy/ReadVariableOp2F
!while/bdgxhkvdqy/ReadVariableOp_1!while/bdgxhkvdqy/ReadVariableOp_12F
!while/bdgxhkvdqy/ReadVariableOp_2!while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
p
É
zeuewnmlut_while_body_3644182
.zeuewnmlut_while_zeuewnmlut_while_loop_counter8
4zeuewnmlut_while_zeuewnmlut_while_maximum_iterations 
zeuewnmlut_while_placeholder"
zeuewnmlut_while_placeholder_1"
zeuewnmlut_while_placeholder_2"
zeuewnmlut_while_placeholder_31
-zeuewnmlut_while_zeuewnmlut_strided_slice_1_0m
izeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0O
<zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0:	Q
>zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 L
=zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0:	C
5zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0: E
7zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0: E
7zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0: 
zeuewnmlut_while_identity
zeuewnmlut_while_identity_1
zeuewnmlut_while_identity_2
zeuewnmlut_while_identity_3
zeuewnmlut_while_identity_4
zeuewnmlut_while_identity_5/
+zeuewnmlut_while_zeuewnmlut_strided_slice_1k
gzeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensorM
:zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource:	O
<zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource:	 J
;zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource:	A
3zeuewnmlut_while_kfdgklwsil_readvariableop_resource: C
5zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource: C
5zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource: ¢2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp¢1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp¢3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp¢*zeuewnmlut/while/kfdgklwsil/ReadVariableOp¢,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1¢,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2Ù
Bzeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bzeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shape
4zeuewnmlut/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemizeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0zeuewnmlut_while_placeholderKzeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4zeuewnmlut/while/TensorArrayV2Read/TensorListGetItemä
1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp<zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOpý
"zeuewnmlut/while/kfdgklwsil/MatMulMatMul;zeuewnmlut/while/TensorArrayV2Read/TensorListGetItem:item:09zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"zeuewnmlut/while/kfdgklwsil/MatMulê
3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp>zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOpæ
$zeuewnmlut/while/kfdgklwsil/MatMul_1MatMulzeuewnmlut_while_placeholder_2;zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$zeuewnmlut/while/kfdgklwsil/MatMul_1Ü
zeuewnmlut/while/kfdgklwsil/addAddV2,zeuewnmlut/while/kfdgklwsil/MatMul:product:0.zeuewnmlut/while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
zeuewnmlut/while/kfdgklwsil/addã
2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp=zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOpé
#zeuewnmlut/while/kfdgklwsil/BiasAddBiasAdd#zeuewnmlut/while/kfdgklwsil/add:z:0:zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#zeuewnmlut/while/kfdgklwsil/BiasAdd
+zeuewnmlut/while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+zeuewnmlut/while/kfdgklwsil/split/split_dim¯
!zeuewnmlut/while/kfdgklwsil/splitSplit4zeuewnmlut/while/kfdgklwsil/split/split_dim:output:0,zeuewnmlut/while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!zeuewnmlut/while/kfdgklwsil/splitÊ
*zeuewnmlut/while/kfdgklwsil/ReadVariableOpReadVariableOp5zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*zeuewnmlut/while/kfdgklwsil/ReadVariableOpÏ
zeuewnmlut/while/kfdgklwsil/mulMul2zeuewnmlut/while/kfdgklwsil/ReadVariableOp:value:0zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zeuewnmlut/while/kfdgklwsil/mulÒ
!zeuewnmlut/while/kfdgklwsil/add_1AddV2*zeuewnmlut/while/kfdgklwsil/split:output:0#zeuewnmlut/while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_1®
#zeuewnmlut/while/kfdgklwsil/SigmoidSigmoid%zeuewnmlut/while/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#zeuewnmlut/while/kfdgklwsil/SigmoidÐ
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1ReadVariableOp7zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1Õ
!zeuewnmlut/while/kfdgklwsil/mul_1Mul4zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1:value:0zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_1Ô
!zeuewnmlut/while/kfdgklwsil/add_2AddV2*zeuewnmlut/while/kfdgklwsil/split:output:1%zeuewnmlut/while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_2²
%zeuewnmlut/while/kfdgklwsil/Sigmoid_1Sigmoid%zeuewnmlut/while/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zeuewnmlut/while/kfdgklwsil/Sigmoid_1Ê
!zeuewnmlut/while/kfdgklwsil/mul_2Mul)zeuewnmlut/while/kfdgklwsil/Sigmoid_1:y:0zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_2ª
 zeuewnmlut/while/kfdgklwsil/TanhTanh*zeuewnmlut/while/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 zeuewnmlut/while/kfdgklwsil/TanhÎ
!zeuewnmlut/while/kfdgklwsil/mul_3Mul'zeuewnmlut/while/kfdgklwsil/Sigmoid:y:0$zeuewnmlut/while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_3Ï
!zeuewnmlut/while/kfdgklwsil/add_3AddV2%zeuewnmlut/while/kfdgklwsil/mul_2:z:0%zeuewnmlut/while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_3Ð
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2ReadVariableOp7zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2Ü
!zeuewnmlut/while/kfdgklwsil/mul_4Mul4zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2:value:0%zeuewnmlut/while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_4Ô
!zeuewnmlut/while/kfdgklwsil/add_4AddV2*zeuewnmlut/while/kfdgklwsil/split:output:3%zeuewnmlut/while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_4²
%zeuewnmlut/while/kfdgklwsil/Sigmoid_2Sigmoid%zeuewnmlut/while/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zeuewnmlut/while/kfdgklwsil/Sigmoid_2©
"zeuewnmlut/while/kfdgklwsil/Tanh_1Tanh%zeuewnmlut/while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"zeuewnmlut/while/kfdgklwsil/Tanh_1Ò
!zeuewnmlut/while/kfdgklwsil/mul_5Mul)zeuewnmlut/while/kfdgklwsil/Sigmoid_2:y:0&zeuewnmlut/while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_5
5zeuewnmlut/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemzeuewnmlut_while_placeholder_1zeuewnmlut_while_placeholder%zeuewnmlut/while/kfdgklwsil/mul_5:z:0*
_output_shapes
: *
element_dtype027
5zeuewnmlut/while/TensorArrayV2Write/TensorListSetItemr
zeuewnmlut/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeuewnmlut/while/add/y
zeuewnmlut/while/addAddV2zeuewnmlut_while_placeholderzeuewnmlut/while/add/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/while/addv
zeuewnmlut/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeuewnmlut/while/add_1/y­
zeuewnmlut/while/add_1AddV2.zeuewnmlut_while_zeuewnmlut_while_loop_counter!zeuewnmlut/while/add_1/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/while/add_1©
zeuewnmlut/while/IdentityIdentityzeuewnmlut/while/add_1:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/IdentityÇ
zeuewnmlut/while/Identity_1Identity4zeuewnmlut_while_zeuewnmlut_while_maximum_iterations3^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/Identity_1«
zeuewnmlut/while/Identity_2Identityzeuewnmlut/while/add:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/Identity_2Ø
zeuewnmlut/while/Identity_3IdentityEzeuewnmlut/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/Identity_3É
zeuewnmlut/while/Identity_4Identity%zeuewnmlut/while/kfdgklwsil/mul_5:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/while/Identity_4É
zeuewnmlut/while/Identity_5Identity%zeuewnmlut/while/kfdgklwsil/add_3:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/while/Identity_5"?
zeuewnmlut_while_identity"zeuewnmlut/while/Identity:output:0"C
zeuewnmlut_while_identity_1$zeuewnmlut/while/Identity_1:output:0"C
zeuewnmlut_while_identity_2$zeuewnmlut/while/Identity_2:output:0"C
zeuewnmlut_while_identity_3$zeuewnmlut/while/Identity_3:output:0"C
zeuewnmlut_while_identity_4$zeuewnmlut/while/Identity_4:output:0"C
zeuewnmlut_while_identity_5$zeuewnmlut/while/Identity_5:output:0"|
;zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource=zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0"~
<zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource>zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0"z
:zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource<zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0"p
5zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource7zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0"p
5zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource7zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0"l
3zeuewnmlut_while_kfdgklwsil_readvariableop_resource5zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0"Ô
gzeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensorizeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0"\
+zeuewnmlut_while_zeuewnmlut_strided_slice_1-zeuewnmlut_while_zeuewnmlut_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2f
1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp2j
3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp2X
*zeuewnmlut/while/kfdgklwsil/ReadVariableOp*zeuewnmlut/while/kfdgklwsil/ReadVariableOp2\
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_12\
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_sequential_layer_call_and_return_conditional_losses_363848

kjggqknufb'
ekzorghjta_363810:
ekzorghjta_363812:$
zeuewnmlut_363816:	$
zeuewnmlut_363818:	  
zeuewnmlut_363820:	
zeuewnmlut_363822: 
zeuewnmlut_363824: 
zeuewnmlut_363826: $
ipeigywbwc_363829:	 $
ipeigywbwc_363831:	  
ipeigywbwc_363833:	
ipeigywbwc_363835: 
ipeigywbwc_363837: 
ipeigywbwc_363839: #
pemnqztknd_363842: 
pemnqztknd_363844:
identity¢"ekzorghjta/StatefulPartitionedCall¢"ipeigywbwc/StatefulPartitionedCall¢"pemnqztknd/StatefulPartitionedCall¢"zeuewnmlut/StatefulPartitionedCall­
"ekzorghjta/StatefulPartitionedCallStatefulPartitionedCall
kjggqknufbekzorghjta_363810ekzorghjta_363812*
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
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_3627012$
"ekzorghjta/StatefulPartitionedCall
xsvuntduhq/PartitionedCallPartitionedCall+ekzorghjta/StatefulPartitionedCall:output:0*
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
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_3627202
xsvuntduhq/PartitionedCall
"zeuewnmlut/StatefulPartitionedCallStatefulPartitionedCall#xsvuntduhq/PartitionedCall:output:0zeuewnmlut_363816zeuewnmlut_363818zeuewnmlut_363820zeuewnmlut_363822zeuewnmlut_363824zeuewnmlut_363826*
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3635832$
"zeuewnmlut/StatefulPartitionedCall
"ipeigywbwc/StatefulPartitionedCallStatefulPartitionedCall+zeuewnmlut/StatefulPartitionedCall:output:0ipeigywbwc_363829ipeigywbwc_363831ipeigywbwc_363833ipeigywbwc_363835ipeigywbwc_363837ipeigywbwc_363839*
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3633692$
"ipeigywbwc/StatefulPartitionedCallÆ
"pemnqztknd/StatefulPartitionedCallStatefulPartitionedCall+ipeigywbwc/StatefulPartitionedCall:output:0pemnqztknd_363842pemnqztknd_363844*
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
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_3631182$
"pemnqztknd/StatefulPartitionedCall
IdentityIdentity+pemnqztknd/StatefulPartitionedCall:output:0#^ekzorghjta/StatefulPartitionedCall#^ipeigywbwc/StatefulPartitionedCall#^pemnqztknd/StatefulPartitionedCall#^zeuewnmlut/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ekzorghjta/StatefulPartitionedCall"ekzorghjta/StatefulPartitionedCall2H
"ipeigywbwc/StatefulPartitionedCall"ipeigywbwc/StatefulPartitionedCall2H
"pemnqztknd/StatefulPartitionedCall"pemnqztknd/StatefulPartitionedCall2H
"zeuewnmlut/StatefulPartitionedCall"zeuewnmlut/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
kjggqknufb
Ç)
Å
while_body_361248
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_kfdgklwsil_361272_0:	,
while_kfdgklwsil_361274_0:	 (
while_kfdgklwsil_361276_0:	'
while_kfdgklwsil_361278_0: '
while_kfdgklwsil_361280_0: '
while_kfdgklwsil_361282_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_kfdgklwsil_361272:	*
while_kfdgklwsil_361274:	 &
while_kfdgklwsil_361276:	%
while_kfdgklwsil_361278: %
while_kfdgklwsil_361280: %
while_kfdgklwsil_361282: ¢(while/kfdgklwsil/StatefulPartitionedCallÃ
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
(while/kfdgklwsil/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_kfdgklwsil_361272_0while_kfdgklwsil_361274_0while_kfdgklwsil_361276_0while_kfdgklwsil_361278_0while_kfdgklwsil_361280_0while_kfdgklwsil_361282_0*
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_3612282*
(while/kfdgklwsil/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/kfdgklwsil/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/kfdgklwsil/StatefulPartitionedCall:output:1)^while/kfdgklwsil/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/kfdgklwsil/StatefulPartitionedCall:output:2)^while/kfdgklwsil/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_kfdgklwsil_361272while_kfdgklwsil_361272_0"4
while_kfdgklwsil_361274while_kfdgklwsil_361274_0"4
while_kfdgklwsil_361276while_kfdgklwsil_361276_0"4
while_kfdgklwsil_361278while_kfdgklwsil_361278_0"4
while_kfdgklwsil_361280while_kfdgklwsil_361280_0"4
while_kfdgklwsil_361282while_kfdgklwsil_361282_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/kfdgklwsil/StatefulPartitionedCall(while/kfdgklwsil/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_362349

inputs$
bdgxhkvdqy_362250:	 $
bdgxhkvdqy_362252:	  
bdgxhkvdqy_362254:	
bdgxhkvdqy_362256: 
bdgxhkvdqy_362258: 
bdgxhkvdqy_362260: 
identity¢"bdgxhkvdqy/StatefulPartitionedCall¢whileD
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
"bdgxhkvdqy/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0bdgxhkvdqy_362250bdgxhkvdqy_362252bdgxhkvdqy_362254bdgxhkvdqy_362256bdgxhkvdqy_362258bdgxhkvdqy_362260*
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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_3621732$
"bdgxhkvdqy/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0bdgxhkvdqy_362250bdgxhkvdqy_362252bdgxhkvdqy_362254bdgxhkvdqy_362256bdgxhkvdqy_362258bdgxhkvdqy_362260*
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
while_body_362269*
condR
while_cond_362268*Q
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
IdentityIdentitystrided_slice_3:output:0#^bdgxhkvdqy/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"bdgxhkvdqy/StatefulPartitionedCall"bdgxhkvdqy/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


zeuewnmlut_while_cond_3640132
.zeuewnmlut_while_zeuewnmlut_while_loop_counter8
4zeuewnmlut_while_zeuewnmlut_while_maximum_iterations 
zeuewnmlut_while_placeholder"
zeuewnmlut_while_placeholder_1"
zeuewnmlut_while_placeholder_2"
zeuewnmlut_while_placeholder_34
0zeuewnmlut_while_less_zeuewnmlut_strided_slice_1J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364013___redundant_placeholder0J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364013___redundant_placeholder1J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364013___redundant_placeholder2J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364013___redundant_placeholder3J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364013___redundant_placeholder4J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364013___redundant_placeholder5J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364013___redundant_placeholder6
zeuewnmlut_while_identity
§
zeuewnmlut/while/LessLesszeuewnmlut_while_placeholder0zeuewnmlut_while_less_zeuewnmlut_strided_slice_1*
T0*
_output_shapes
: 2
zeuewnmlut/while/Less~
zeuewnmlut/while/IdentityIdentityzeuewnmlut/while/Less:z:0*
T0
*
_output_shapes
: 2
zeuewnmlut/while/Identity"?
zeuewnmlut_while_identity"zeuewnmlut/while/Identity:output:0*(
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
while_cond_361247
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_361247___redundant_placeholder04
0while_while_cond_361247___redundant_placeholder14
0while_while_cond_361247___redundant_placeholder24
0while_while_cond_361247___redundant_placeholder34
0while_while_cond_361247___redundant_placeholder44
0while_while_cond_361247___redundant_placeholder54
0while_while_cond_361247___redundant_placeholder6
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
h

F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365559

inputs<
)kfdgklwsil_matmul_readvariableop_resource:	>
+kfdgklwsil_matmul_1_readvariableop_resource:	 9
*kfdgklwsil_biasadd_readvariableop_resource:	0
"kfdgklwsil_readvariableop_resource: 2
$kfdgklwsil_readvariableop_1_resource: 2
$kfdgklwsil_readvariableop_2_resource: 
identity¢!kfdgklwsil/BiasAdd/ReadVariableOp¢ kfdgklwsil/MatMul/ReadVariableOp¢"kfdgklwsil/MatMul_1/ReadVariableOp¢kfdgklwsil/ReadVariableOp¢kfdgklwsil/ReadVariableOp_1¢kfdgklwsil/ReadVariableOp_2¢whileD
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
 kfdgklwsil/MatMul/ReadVariableOpReadVariableOp)kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 kfdgklwsil/MatMul/ReadVariableOp§
kfdgklwsil/MatMulMatMulstrided_slice_2:output:0(kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMulµ
"kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp+kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kfdgklwsil/MatMul_1/ReadVariableOp£
kfdgklwsil/MatMul_1MatMulzeros:output:0*kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMul_1
kfdgklwsil/addAddV2kfdgklwsil/MatMul:product:0kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/add®
!kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp*kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kfdgklwsil/BiasAdd/ReadVariableOp¥
kfdgklwsil/BiasAddBiasAddkfdgklwsil/add:z:0)kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/BiasAddz
kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kfdgklwsil/split/split_dimë
kfdgklwsil/splitSplit#kfdgklwsil/split/split_dim:output:0kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kfdgklwsil/split
kfdgklwsil/ReadVariableOpReadVariableOp"kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp
kfdgklwsil/mulMul!kfdgklwsil/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul
kfdgklwsil/add_1AddV2kfdgklwsil/split:output:0kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_1{
kfdgklwsil/SigmoidSigmoidkfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid
kfdgklwsil/ReadVariableOp_1ReadVariableOp$kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_1
kfdgklwsil/mul_1Mul#kfdgklwsil/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_1
kfdgklwsil/add_2AddV2kfdgklwsil/split:output:1kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_2
kfdgklwsil/Sigmoid_1Sigmoidkfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_1
kfdgklwsil/mul_2Mulkfdgklwsil/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_2w
kfdgklwsil/TanhTanhkfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh
kfdgklwsil/mul_3Mulkfdgklwsil/Sigmoid:y:0kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_3
kfdgklwsil/add_3AddV2kfdgklwsil/mul_2:z:0kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_3
kfdgklwsil/ReadVariableOp_2ReadVariableOp$kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_2
kfdgklwsil/mul_4Mul#kfdgklwsil/ReadVariableOp_2:value:0kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_4
kfdgklwsil/add_4AddV2kfdgklwsil/split:output:3kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_4
kfdgklwsil/Sigmoid_2Sigmoidkfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_2v
kfdgklwsil/Tanh_1Tanhkfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh_1
kfdgklwsil/mul_5Mulkfdgklwsil/Sigmoid_2:y:0kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kfdgklwsil_matmul_readvariableop_resource+kfdgklwsil_matmul_1_readvariableop_resource*kfdgklwsil_biasadd_readvariableop_resource"kfdgklwsil_readvariableop_resource$kfdgklwsil_readvariableop_1_resource$kfdgklwsil_readvariableop_2_resource*
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
while_body_365458*
condR
while_cond_365457*Q
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
IdentityIdentitytranspose_1:y:0"^kfdgklwsil/BiasAdd/ReadVariableOp!^kfdgklwsil/MatMul/ReadVariableOp#^kfdgklwsil/MatMul_1/ReadVariableOp^kfdgklwsil/ReadVariableOp^kfdgklwsil/ReadVariableOp_1^kfdgklwsil/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!kfdgklwsil/BiasAdd/ReadVariableOp!kfdgklwsil/BiasAdd/ReadVariableOp2D
 kfdgklwsil/MatMul/ReadVariableOp kfdgklwsil/MatMul/ReadVariableOp2H
"kfdgklwsil/MatMul_1/ReadVariableOp"kfdgklwsil/MatMul_1/ReadVariableOp26
kfdgklwsil/ReadVariableOpkfdgklwsil/ReadVariableOp2:
kfdgklwsil/ReadVariableOp_1kfdgklwsil/ReadVariableOp_12:
kfdgklwsil/ReadVariableOp_2kfdgklwsil/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_365705
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365705___redundant_placeholder04
0while_while_cond_365705___redundant_placeholder14
0while_while_cond_365705___redundant_placeholder24
0while_while_cond_365705___redundant_placeholder34
0while_while_cond_365705___redundant_placeholder44
0while_while_cond_365705___redundant_placeholder54
0while_while_cond_365705___redundant_placeholder6
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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_362173

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


+__inference_sequential_layer_call_fn_363160

kjggqknufb
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
kjggqknufbunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
F__inference_sequential_layer_call_and_return_conditional_losses_3631252
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
kjggqknufb
p
É
zeuewnmlut_while_body_3640142
.zeuewnmlut_while_zeuewnmlut_while_loop_counter8
4zeuewnmlut_while_zeuewnmlut_while_maximum_iterations 
zeuewnmlut_while_placeholder"
zeuewnmlut_while_placeholder_1"
zeuewnmlut_while_placeholder_2"
zeuewnmlut_while_placeholder_31
-zeuewnmlut_while_zeuewnmlut_strided_slice_1_0m
izeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0O
<zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0:	Q
>zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 L
=zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0:	C
5zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0: E
7zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0: E
7zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0: 
zeuewnmlut_while_identity
zeuewnmlut_while_identity_1
zeuewnmlut_while_identity_2
zeuewnmlut_while_identity_3
zeuewnmlut_while_identity_4
zeuewnmlut_while_identity_5/
+zeuewnmlut_while_zeuewnmlut_strided_slice_1k
gzeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensorM
:zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource:	O
<zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource:	 J
;zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource:	A
3zeuewnmlut_while_kfdgklwsil_readvariableop_resource: C
5zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource: C
5zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource: ¢2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp¢1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp¢3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp¢*zeuewnmlut/while/kfdgklwsil/ReadVariableOp¢,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1¢,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2Ù
Bzeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bzeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shape
4zeuewnmlut/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemizeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0zeuewnmlut_while_placeholderKzeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4zeuewnmlut/while/TensorArrayV2Read/TensorListGetItemä
1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp<zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOpý
"zeuewnmlut/while/kfdgklwsil/MatMulMatMul;zeuewnmlut/while/TensorArrayV2Read/TensorListGetItem:item:09zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"zeuewnmlut/while/kfdgklwsil/MatMulê
3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp>zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOpæ
$zeuewnmlut/while/kfdgklwsil/MatMul_1MatMulzeuewnmlut_while_placeholder_2;zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$zeuewnmlut/while/kfdgklwsil/MatMul_1Ü
zeuewnmlut/while/kfdgklwsil/addAddV2,zeuewnmlut/while/kfdgklwsil/MatMul:product:0.zeuewnmlut/while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
zeuewnmlut/while/kfdgklwsil/addã
2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp=zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOpé
#zeuewnmlut/while/kfdgklwsil/BiasAddBiasAdd#zeuewnmlut/while/kfdgklwsil/add:z:0:zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#zeuewnmlut/while/kfdgklwsil/BiasAdd
+zeuewnmlut/while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+zeuewnmlut/while/kfdgklwsil/split/split_dim¯
!zeuewnmlut/while/kfdgklwsil/splitSplit4zeuewnmlut/while/kfdgklwsil/split/split_dim:output:0,zeuewnmlut/while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!zeuewnmlut/while/kfdgklwsil/splitÊ
*zeuewnmlut/while/kfdgklwsil/ReadVariableOpReadVariableOp5zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*zeuewnmlut/while/kfdgklwsil/ReadVariableOpÏ
zeuewnmlut/while/kfdgklwsil/mulMul2zeuewnmlut/while/kfdgklwsil/ReadVariableOp:value:0zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zeuewnmlut/while/kfdgklwsil/mulÒ
!zeuewnmlut/while/kfdgklwsil/add_1AddV2*zeuewnmlut/while/kfdgklwsil/split:output:0#zeuewnmlut/while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_1®
#zeuewnmlut/while/kfdgklwsil/SigmoidSigmoid%zeuewnmlut/while/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#zeuewnmlut/while/kfdgklwsil/SigmoidÐ
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1ReadVariableOp7zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1Õ
!zeuewnmlut/while/kfdgklwsil/mul_1Mul4zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1:value:0zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_1Ô
!zeuewnmlut/while/kfdgklwsil/add_2AddV2*zeuewnmlut/while/kfdgklwsil/split:output:1%zeuewnmlut/while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_2²
%zeuewnmlut/while/kfdgklwsil/Sigmoid_1Sigmoid%zeuewnmlut/while/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zeuewnmlut/while/kfdgklwsil/Sigmoid_1Ê
!zeuewnmlut/while/kfdgklwsil/mul_2Mul)zeuewnmlut/while/kfdgklwsil/Sigmoid_1:y:0zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_2ª
 zeuewnmlut/while/kfdgklwsil/TanhTanh*zeuewnmlut/while/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 zeuewnmlut/while/kfdgklwsil/TanhÎ
!zeuewnmlut/while/kfdgklwsil/mul_3Mul'zeuewnmlut/while/kfdgklwsil/Sigmoid:y:0$zeuewnmlut/while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_3Ï
!zeuewnmlut/while/kfdgklwsil/add_3AddV2%zeuewnmlut/while/kfdgklwsil/mul_2:z:0%zeuewnmlut/while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_3Ð
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2ReadVariableOp7zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2Ü
!zeuewnmlut/while/kfdgklwsil/mul_4Mul4zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2:value:0%zeuewnmlut/while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_4Ô
!zeuewnmlut/while/kfdgklwsil/add_4AddV2*zeuewnmlut/while/kfdgklwsil/split:output:3%zeuewnmlut/while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/add_4²
%zeuewnmlut/while/kfdgklwsil/Sigmoid_2Sigmoid%zeuewnmlut/while/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%zeuewnmlut/while/kfdgklwsil/Sigmoid_2©
"zeuewnmlut/while/kfdgklwsil/Tanh_1Tanh%zeuewnmlut/while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"zeuewnmlut/while/kfdgklwsil/Tanh_1Ò
!zeuewnmlut/while/kfdgklwsil/mul_5Mul)zeuewnmlut/while/kfdgklwsil/Sigmoid_2:y:0&zeuewnmlut/while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!zeuewnmlut/while/kfdgklwsil/mul_5
5zeuewnmlut/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemzeuewnmlut_while_placeholder_1zeuewnmlut_while_placeholder%zeuewnmlut/while/kfdgklwsil/mul_5:z:0*
_output_shapes
: *
element_dtype027
5zeuewnmlut/while/TensorArrayV2Write/TensorListSetItemr
zeuewnmlut/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeuewnmlut/while/add/y
zeuewnmlut/while/addAddV2zeuewnmlut_while_placeholderzeuewnmlut/while/add/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/while/addv
zeuewnmlut/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeuewnmlut/while/add_1/y­
zeuewnmlut/while/add_1AddV2.zeuewnmlut_while_zeuewnmlut_while_loop_counter!zeuewnmlut/while/add_1/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/while/add_1©
zeuewnmlut/while/IdentityIdentityzeuewnmlut/while/add_1:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/IdentityÇ
zeuewnmlut/while/Identity_1Identity4zeuewnmlut_while_zeuewnmlut_while_maximum_iterations3^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/Identity_1«
zeuewnmlut/while/Identity_2Identityzeuewnmlut/while/add:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/Identity_2Ø
zeuewnmlut/while/Identity_3IdentityEzeuewnmlut/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
zeuewnmlut/while/Identity_3É
zeuewnmlut/while/Identity_4Identity%zeuewnmlut/while/kfdgklwsil/mul_5:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/while/Identity_4É
zeuewnmlut/while/Identity_5Identity%zeuewnmlut/while/kfdgklwsil/add_3:z:03^zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2^zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp4^zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp+^zeuewnmlut/while/kfdgklwsil/ReadVariableOp-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1-^zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/while/Identity_5"?
zeuewnmlut_while_identity"zeuewnmlut/while/Identity:output:0"C
zeuewnmlut_while_identity_1$zeuewnmlut/while/Identity_1:output:0"C
zeuewnmlut_while_identity_2$zeuewnmlut/while/Identity_2:output:0"C
zeuewnmlut_while_identity_3$zeuewnmlut/while/Identity_3:output:0"C
zeuewnmlut_while_identity_4$zeuewnmlut/while/Identity_4:output:0"C
zeuewnmlut_while_identity_5$zeuewnmlut/while/Identity_5:output:0"|
;zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource=zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0"~
<zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource>zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0"z
:zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource<zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0"p
5zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource7zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0"p
5zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource7zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0"l
3zeuewnmlut_while_kfdgklwsil_readvariableop_resource5zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0"Ô
gzeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensorizeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0"\
+zeuewnmlut_while_zeuewnmlut_strided_slice_1-zeuewnmlut_while_zeuewnmlut_strided_slice_1_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2f
1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp1zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp2j
3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp3zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp2X
*zeuewnmlut/while/kfdgklwsil/ReadVariableOp*zeuewnmlut/while/kfdgklwsil/ReadVariableOp2\
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_12\
,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2,zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_362086

inputs$
bdgxhkvdqy_361987:	 $
bdgxhkvdqy_361989:	  
bdgxhkvdqy_361991:	
bdgxhkvdqy_361993: 
bdgxhkvdqy_361995: 
bdgxhkvdqy_361997: 
identity¢"bdgxhkvdqy/StatefulPartitionedCall¢whileD
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
"bdgxhkvdqy/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0bdgxhkvdqy_361987bdgxhkvdqy_361989bdgxhkvdqy_361991bdgxhkvdqy_361993bdgxhkvdqy_361995bdgxhkvdqy_361997*
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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_3619862$
"bdgxhkvdqy/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0bdgxhkvdqy_361987bdgxhkvdqy_361989bdgxhkvdqy_361991bdgxhkvdqy_361993bdgxhkvdqy_361995bdgxhkvdqy_361997*
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
while_body_362006*
condR
while_cond_362005*Q
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
IdentityIdentitystrided_slice_3:output:0#^bdgxhkvdqy/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"bdgxhkvdqy/StatefulPartitionedCall"bdgxhkvdqy/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


å
while_cond_365885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365885___redundant_placeholder04
0while_while_cond_365885___redundant_placeholder14
0while_while_cond_365885___redundant_placeholder24
0while_while_cond_365885___redundant_placeholder34
0while_while_cond_365885___redundant_placeholder44
0while_while_cond_365885___redundant_placeholder54
0while_while_cond_365885___redundant_placeholder6
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
®
Ö
!__inference__wrapped_model_361141

kjggqknufbW
Asequential_ekzorghjta_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_ekzorghjta_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource:	T
Asequential_zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource:	 O
@sequential_zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource:	F
8sequential_zeuewnmlut_kfdgklwsil_readvariableop_resource: H
:sequential_zeuewnmlut_kfdgklwsil_readvariableop_1_resource: H
:sequential_zeuewnmlut_kfdgklwsil_readvariableop_2_resource: R
?sequential_ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource:	 T
Asequential_ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource:	 O
@sequential_ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource:	F
8sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_resource: H
:sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource: H
:sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource: F
4sequential_pemnqztknd_matmul_readvariableop_resource: C
5sequential_pemnqztknd_biasadd_readvariableop_resource:
identity¢8sequential/ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp¢6sequential/ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp¢8sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp¢/sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp¢1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1¢1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2¢sequential/ipeigywbwc/while¢,sequential/pemnqztknd/BiasAdd/ReadVariableOp¢+sequential/pemnqztknd/MatMul/ReadVariableOp¢7sequential/zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp¢6sequential/zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp¢8sequential/zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp¢/sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp¢1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_1¢1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_2¢sequential/zeuewnmlut/while¥
+sequential/ekzorghjta/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/ekzorghjta/conv1d/ExpandDims/dimà
'sequential/ekzorghjta/conv1d/ExpandDims
ExpandDims
kjggqknufb4sequential/ekzorghjta/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/ekzorghjta/conv1d/ExpandDimsú
8sequential/ekzorghjta/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_ekzorghjta_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/ekzorghjta/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/ekzorghjta/conv1d/ExpandDims_1/dim
)sequential/ekzorghjta/conv1d/ExpandDims_1
ExpandDims@sequential/ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/ekzorghjta/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/ekzorghjta/conv1d/ExpandDims_1¨
"sequential/ekzorghjta/conv1d/ShapeShape0sequential/ekzorghjta/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/ekzorghjta/conv1d/Shape®
0sequential/ekzorghjta/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/ekzorghjta/conv1d/strided_slice/stack»
2sequential/ekzorghjta/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/ekzorghjta/conv1d/strided_slice/stack_1²
2sequential/ekzorghjta/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/ekzorghjta/conv1d/strided_slice/stack_2
*sequential/ekzorghjta/conv1d/strided_sliceStridedSlice+sequential/ekzorghjta/conv1d/Shape:output:09sequential/ekzorghjta/conv1d/strided_slice/stack:output:0;sequential/ekzorghjta/conv1d/strided_slice/stack_1:output:0;sequential/ekzorghjta/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/ekzorghjta/conv1d/strided_slice±
*sequential/ekzorghjta/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/ekzorghjta/conv1d/Reshape/shapeø
$sequential/ekzorghjta/conv1d/ReshapeReshape0sequential/ekzorghjta/conv1d/ExpandDims:output:03sequential/ekzorghjta/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/ekzorghjta/conv1d/Reshape
#sequential/ekzorghjta/conv1d/Conv2DConv2D-sequential/ekzorghjta/conv1d/Reshape:output:02sequential/ekzorghjta/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/ekzorghjta/conv1d/Conv2D±
,sequential/ekzorghjta/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/ekzorghjta/conv1d/concat/values_1
(sequential/ekzorghjta/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/ekzorghjta/conv1d/concat/axis£
#sequential/ekzorghjta/conv1d/concatConcatV23sequential/ekzorghjta/conv1d/strided_slice:output:05sequential/ekzorghjta/conv1d/concat/values_1:output:01sequential/ekzorghjta/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/ekzorghjta/conv1d/concatõ
&sequential/ekzorghjta/conv1d/Reshape_1Reshape,sequential/ekzorghjta/conv1d/Conv2D:output:0,sequential/ekzorghjta/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/ekzorghjta/conv1d/Reshape_1â
$sequential/ekzorghjta/conv1d/SqueezeSqueeze/sequential/ekzorghjta/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/ekzorghjta/conv1d/Squeeze½
.sequential/ekzorghjta/squeeze_batch_dims/ShapeShape-sequential/ekzorghjta/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/ekzorghjta/squeeze_batch_dims/ShapeÆ
<sequential/ekzorghjta/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/ekzorghjta/squeeze_batch_dims/strided_slice/stackÓ
>sequential/ekzorghjta/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/ekzorghjta/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/ekzorghjta/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/ekzorghjta/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/ekzorghjta/squeeze_batch_dims/strided_sliceStridedSlice7sequential/ekzorghjta/squeeze_batch_dims/Shape:output:0Esequential/ekzorghjta/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/ekzorghjta/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/ekzorghjta/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/ekzorghjta/squeeze_batch_dims/strided_sliceÅ
6sequential/ekzorghjta/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/ekzorghjta/squeeze_batch_dims/Reshape/shape
0sequential/ekzorghjta/squeeze_batch_dims/ReshapeReshape-sequential/ekzorghjta/conv1d/Squeeze:output:0?sequential/ekzorghjta/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/ekzorghjta/squeeze_batch_dims/Reshape
?sequential/ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_ekzorghjta_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/ekzorghjta/squeeze_batch_dims/BiasAddBiasAdd9sequential/ekzorghjta/squeeze_batch_dims/Reshape:output:0Gsequential/ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/ekzorghjta/squeeze_batch_dims/BiasAddÅ
8sequential/ekzorghjta/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/ekzorghjta/squeeze_batch_dims/concat/values_1·
4sequential/ekzorghjta/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/ekzorghjta/squeeze_batch_dims/concat/axisß
/sequential/ekzorghjta/squeeze_batch_dims/concatConcatV2?sequential/ekzorghjta/squeeze_batch_dims/strided_slice:output:0Asequential/ekzorghjta/squeeze_batch_dims/concat/values_1:output:0=sequential/ekzorghjta/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/ekzorghjta/squeeze_batch_dims/concat¢
2sequential/ekzorghjta/squeeze_batch_dims/Reshape_1Reshape9sequential/ekzorghjta/squeeze_batch_dims/BiasAdd:output:08sequential/ekzorghjta/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/ekzorghjta/squeeze_batch_dims/Reshape_1¥
sequential/xsvuntduhq/ShapeShape;sequential/ekzorghjta/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/xsvuntduhq/Shape 
)sequential/xsvuntduhq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/xsvuntduhq/strided_slice/stack¤
+sequential/xsvuntduhq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xsvuntduhq/strided_slice/stack_1¤
+sequential/xsvuntduhq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/xsvuntduhq/strided_slice/stack_2æ
#sequential/xsvuntduhq/strided_sliceStridedSlice$sequential/xsvuntduhq/Shape:output:02sequential/xsvuntduhq/strided_slice/stack:output:04sequential/xsvuntduhq/strided_slice/stack_1:output:04sequential/xsvuntduhq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/xsvuntduhq/strided_slice
%sequential/xsvuntduhq/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/xsvuntduhq/Reshape/shape/1
%sequential/xsvuntduhq/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/xsvuntduhq/Reshape/shape/2
#sequential/xsvuntduhq/Reshape/shapePack,sequential/xsvuntduhq/strided_slice:output:0.sequential/xsvuntduhq/Reshape/shape/1:output:0.sequential/xsvuntduhq/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/xsvuntduhq/Reshape/shapeê
sequential/xsvuntduhq/ReshapeReshape;sequential/ekzorghjta/squeeze_batch_dims/Reshape_1:output:0,sequential/xsvuntduhq/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/xsvuntduhq/Reshape
sequential/zeuewnmlut/ShapeShape&sequential/xsvuntduhq/Reshape:output:0*
T0*
_output_shapes
:2
sequential/zeuewnmlut/Shape 
)sequential/zeuewnmlut/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/zeuewnmlut/strided_slice/stack¤
+sequential/zeuewnmlut/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/zeuewnmlut/strided_slice/stack_1¤
+sequential/zeuewnmlut/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/zeuewnmlut/strided_slice/stack_2æ
#sequential/zeuewnmlut/strided_sliceStridedSlice$sequential/zeuewnmlut/Shape:output:02sequential/zeuewnmlut/strided_slice/stack:output:04sequential/zeuewnmlut/strided_slice/stack_1:output:04sequential/zeuewnmlut/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/zeuewnmlut/strided_slice
!sequential/zeuewnmlut/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/zeuewnmlut/zeros/mul/yÄ
sequential/zeuewnmlut/zeros/mulMul,sequential/zeuewnmlut/strided_slice:output:0*sequential/zeuewnmlut/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/zeuewnmlut/zeros/mul
"sequential/zeuewnmlut/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/zeuewnmlut/zeros/Less/y¿
 sequential/zeuewnmlut/zeros/LessLess#sequential/zeuewnmlut/zeros/mul:z:0+sequential/zeuewnmlut/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/zeuewnmlut/zeros/Less
$sequential/zeuewnmlut/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/zeuewnmlut/zeros/packed/1Û
"sequential/zeuewnmlut/zeros/packedPack,sequential/zeuewnmlut/strided_slice:output:0-sequential/zeuewnmlut/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/zeuewnmlut/zeros/packed
!sequential/zeuewnmlut/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/zeuewnmlut/zeros/ConstÍ
sequential/zeuewnmlut/zerosFill+sequential/zeuewnmlut/zeros/packed:output:0*sequential/zeuewnmlut/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/zeuewnmlut/zeros
#sequential/zeuewnmlut/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/zeuewnmlut/zeros_1/mul/yÊ
!sequential/zeuewnmlut/zeros_1/mulMul,sequential/zeuewnmlut/strided_slice:output:0,sequential/zeuewnmlut/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/zeuewnmlut/zeros_1/mul
$sequential/zeuewnmlut/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/zeuewnmlut/zeros_1/Less/yÇ
"sequential/zeuewnmlut/zeros_1/LessLess%sequential/zeuewnmlut/zeros_1/mul:z:0-sequential/zeuewnmlut/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/zeuewnmlut/zeros_1/Less
&sequential/zeuewnmlut/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/zeuewnmlut/zeros_1/packed/1á
$sequential/zeuewnmlut/zeros_1/packedPack,sequential/zeuewnmlut/strided_slice:output:0/sequential/zeuewnmlut/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/zeuewnmlut/zeros_1/packed
#sequential/zeuewnmlut/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/zeuewnmlut/zeros_1/ConstÕ
sequential/zeuewnmlut/zeros_1Fill-sequential/zeuewnmlut/zeros_1/packed:output:0,sequential/zeuewnmlut/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/zeuewnmlut/zeros_1¡
$sequential/zeuewnmlut/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/zeuewnmlut/transpose/permÜ
sequential/zeuewnmlut/transpose	Transpose&sequential/xsvuntduhq/Reshape:output:0-sequential/zeuewnmlut/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/zeuewnmlut/transpose
sequential/zeuewnmlut/Shape_1Shape#sequential/zeuewnmlut/transpose:y:0*
T0*
_output_shapes
:2
sequential/zeuewnmlut/Shape_1¤
+sequential/zeuewnmlut/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/zeuewnmlut/strided_slice_1/stack¨
-sequential/zeuewnmlut/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zeuewnmlut/strided_slice_1/stack_1¨
-sequential/zeuewnmlut/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zeuewnmlut/strided_slice_1/stack_2ò
%sequential/zeuewnmlut/strided_slice_1StridedSlice&sequential/zeuewnmlut/Shape_1:output:04sequential/zeuewnmlut/strided_slice_1/stack:output:06sequential/zeuewnmlut/strided_slice_1/stack_1:output:06sequential/zeuewnmlut/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/zeuewnmlut/strided_slice_1±
1sequential/zeuewnmlut/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/zeuewnmlut/TensorArrayV2/element_shape
#sequential/zeuewnmlut/TensorArrayV2TensorListReserve:sequential/zeuewnmlut/TensorArrayV2/element_shape:output:0.sequential/zeuewnmlut/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/zeuewnmlut/TensorArrayV2ë
Ksequential/zeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/zeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/zeuewnmlut/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/zeuewnmlut/transpose:y:0Tsequential/zeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/zeuewnmlut/TensorArrayUnstack/TensorListFromTensor¤
+sequential/zeuewnmlut/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/zeuewnmlut/strided_slice_2/stack¨
-sequential/zeuewnmlut/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zeuewnmlut/strided_slice_2/stack_1¨
-sequential/zeuewnmlut/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zeuewnmlut/strided_slice_2/stack_2
%sequential/zeuewnmlut/strided_slice_2StridedSlice#sequential/zeuewnmlut/transpose:y:04sequential/zeuewnmlut/strided_slice_2/stack:output:06sequential/zeuewnmlut/strided_slice_2/stack_1:output:06sequential/zeuewnmlut/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/zeuewnmlut/strided_slice_2ñ
6sequential/zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp?sequential_zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOpÿ
'sequential/zeuewnmlut/kfdgklwsil/MatMulMatMul.sequential/zeuewnmlut/strided_slice_2:output:0>sequential/zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/zeuewnmlut/kfdgklwsil/MatMul÷
8sequential/zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOpAsequential_zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOpû
)sequential/zeuewnmlut/kfdgklwsil/MatMul_1MatMul$sequential/zeuewnmlut/zeros:output:0@sequential/zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/zeuewnmlut/kfdgklwsil/MatMul_1ð
$sequential/zeuewnmlut/kfdgklwsil/addAddV21sequential/zeuewnmlut/kfdgklwsil/MatMul:product:03sequential/zeuewnmlut/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/zeuewnmlut/kfdgklwsil/addð
7sequential/zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp@sequential_zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOpý
(sequential/zeuewnmlut/kfdgklwsil/BiasAddBiasAdd(sequential/zeuewnmlut/kfdgklwsil/add:z:0?sequential/zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/zeuewnmlut/kfdgklwsil/BiasAdd¦
0sequential/zeuewnmlut/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/zeuewnmlut/kfdgklwsil/split/split_dimÃ
&sequential/zeuewnmlut/kfdgklwsil/splitSplit9sequential/zeuewnmlut/kfdgklwsil/split/split_dim:output:01sequential/zeuewnmlut/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/zeuewnmlut/kfdgklwsil/split×
/sequential/zeuewnmlut/kfdgklwsil/ReadVariableOpReadVariableOp8sequential_zeuewnmlut_kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/zeuewnmlut/kfdgklwsil/ReadVariableOpæ
$sequential/zeuewnmlut/kfdgklwsil/mulMul7sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp:value:0&sequential/zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/zeuewnmlut/kfdgklwsil/mulæ
&sequential/zeuewnmlut/kfdgklwsil/add_1AddV2/sequential/zeuewnmlut/kfdgklwsil/split:output:0(sequential/zeuewnmlut/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/add_1½
(sequential/zeuewnmlut/kfdgklwsil/SigmoidSigmoid*sequential/zeuewnmlut/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/zeuewnmlut/kfdgklwsil/SigmoidÝ
1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_1ReadVariableOp:sequential_zeuewnmlut_kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_1ì
&sequential/zeuewnmlut/kfdgklwsil/mul_1Mul9sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_1:value:0&sequential/zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/mul_1è
&sequential/zeuewnmlut/kfdgklwsil/add_2AddV2/sequential/zeuewnmlut/kfdgklwsil/split:output:1*sequential/zeuewnmlut/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/add_2Á
*sequential/zeuewnmlut/kfdgklwsil/Sigmoid_1Sigmoid*sequential/zeuewnmlut/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/zeuewnmlut/kfdgklwsil/Sigmoid_1á
&sequential/zeuewnmlut/kfdgklwsil/mul_2Mul.sequential/zeuewnmlut/kfdgklwsil/Sigmoid_1:y:0&sequential/zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/mul_2¹
%sequential/zeuewnmlut/kfdgklwsil/TanhTanh/sequential/zeuewnmlut/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/zeuewnmlut/kfdgklwsil/Tanhâ
&sequential/zeuewnmlut/kfdgklwsil/mul_3Mul,sequential/zeuewnmlut/kfdgklwsil/Sigmoid:y:0)sequential/zeuewnmlut/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/mul_3ã
&sequential/zeuewnmlut/kfdgklwsil/add_3AddV2*sequential/zeuewnmlut/kfdgklwsil/mul_2:z:0*sequential/zeuewnmlut/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/add_3Ý
1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_2ReadVariableOp:sequential_zeuewnmlut_kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_2ð
&sequential/zeuewnmlut/kfdgklwsil/mul_4Mul9sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_2:value:0*sequential/zeuewnmlut/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/mul_4è
&sequential/zeuewnmlut/kfdgklwsil/add_4AddV2/sequential/zeuewnmlut/kfdgklwsil/split:output:3*sequential/zeuewnmlut/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/add_4Á
*sequential/zeuewnmlut/kfdgklwsil/Sigmoid_2Sigmoid*sequential/zeuewnmlut/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/zeuewnmlut/kfdgklwsil/Sigmoid_2¸
'sequential/zeuewnmlut/kfdgklwsil/Tanh_1Tanh*sequential/zeuewnmlut/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/zeuewnmlut/kfdgklwsil/Tanh_1æ
&sequential/zeuewnmlut/kfdgklwsil/mul_5Mul.sequential/zeuewnmlut/kfdgklwsil/Sigmoid_2:y:0+sequential/zeuewnmlut/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/kfdgklwsil/mul_5»
3sequential/zeuewnmlut/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/zeuewnmlut/TensorArrayV2_1/element_shape
%sequential/zeuewnmlut/TensorArrayV2_1TensorListReserve<sequential/zeuewnmlut/TensorArrayV2_1/element_shape:output:0.sequential/zeuewnmlut/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/zeuewnmlut/TensorArrayV2_1z
sequential/zeuewnmlut/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/zeuewnmlut/time«
.sequential/zeuewnmlut/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/zeuewnmlut/while/maximum_iterations
(sequential/zeuewnmlut/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/zeuewnmlut/while/loop_counterö	
sequential/zeuewnmlut/whileWhile1sequential/zeuewnmlut/while/loop_counter:output:07sequential/zeuewnmlut/while/maximum_iterations:output:0#sequential/zeuewnmlut/time:output:0.sequential/zeuewnmlut/TensorArrayV2_1:handle:0$sequential/zeuewnmlut/zeros:output:0&sequential/zeuewnmlut/zeros_1:output:0.sequential/zeuewnmlut/strided_slice_1:output:0Msequential/zeuewnmlut/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_zeuewnmlut_kfdgklwsil_matmul_readvariableop_resourceAsequential_zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource@sequential_zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource8sequential_zeuewnmlut_kfdgklwsil_readvariableop_resource:sequential_zeuewnmlut_kfdgklwsil_readvariableop_1_resource:sequential_zeuewnmlut_kfdgklwsil_readvariableop_2_resource*
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
'sequential_zeuewnmlut_while_body_360858*3
cond+R)
'sequential_zeuewnmlut_while_cond_360857*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/zeuewnmlut/whileá
Fsequential/zeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/zeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/zeuewnmlut/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/zeuewnmlut/while:output:3Osequential/zeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/zeuewnmlut/TensorArrayV2Stack/TensorListStack­
+sequential/zeuewnmlut/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/zeuewnmlut/strided_slice_3/stack¨
-sequential/zeuewnmlut/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/zeuewnmlut/strided_slice_3/stack_1¨
-sequential/zeuewnmlut/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/zeuewnmlut/strided_slice_3/stack_2
%sequential/zeuewnmlut/strided_slice_3StridedSliceAsequential/zeuewnmlut/TensorArrayV2Stack/TensorListStack:tensor:04sequential/zeuewnmlut/strided_slice_3/stack:output:06sequential/zeuewnmlut/strided_slice_3/stack_1:output:06sequential/zeuewnmlut/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/zeuewnmlut/strided_slice_3¥
&sequential/zeuewnmlut/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/zeuewnmlut/transpose_1/permý
!sequential/zeuewnmlut/transpose_1	TransposeAsequential/zeuewnmlut/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/zeuewnmlut/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/zeuewnmlut/transpose_1
sequential/ipeigywbwc/ShapeShape%sequential/zeuewnmlut/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/ipeigywbwc/Shape 
)sequential/ipeigywbwc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/ipeigywbwc/strided_slice/stack¤
+sequential/ipeigywbwc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ipeigywbwc/strided_slice/stack_1¤
+sequential/ipeigywbwc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ipeigywbwc/strided_slice/stack_2æ
#sequential/ipeigywbwc/strided_sliceStridedSlice$sequential/ipeigywbwc/Shape:output:02sequential/ipeigywbwc/strided_slice/stack:output:04sequential/ipeigywbwc/strided_slice/stack_1:output:04sequential/ipeigywbwc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/ipeigywbwc/strided_slice
!sequential/ipeigywbwc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/ipeigywbwc/zeros/mul/yÄ
sequential/ipeigywbwc/zeros/mulMul,sequential/ipeigywbwc/strided_slice:output:0*sequential/ipeigywbwc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/ipeigywbwc/zeros/mul
"sequential/ipeigywbwc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/ipeigywbwc/zeros/Less/y¿
 sequential/ipeigywbwc/zeros/LessLess#sequential/ipeigywbwc/zeros/mul:z:0+sequential/ipeigywbwc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/ipeigywbwc/zeros/Less
$sequential/ipeigywbwc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/ipeigywbwc/zeros/packed/1Û
"sequential/ipeigywbwc/zeros/packedPack,sequential/ipeigywbwc/strided_slice:output:0-sequential/ipeigywbwc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/ipeigywbwc/zeros/packed
!sequential/ipeigywbwc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/ipeigywbwc/zeros/ConstÍ
sequential/ipeigywbwc/zerosFill+sequential/ipeigywbwc/zeros/packed:output:0*sequential/ipeigywbwc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/ipeigywbwc/zeros
#sequential/ipeigywbwc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/ipeigywbwc/zeros_1/mul/yÊ
!sequential/ipeigywbwc/zeros_1/mulMul,sequential/ipeigywbwc/strided_slice:output:0,sequential/ipeigywbwc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/ipeigywbwc/zeros_1/mul
$sequential/ipeigywbwc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/ipeigywbwc/zeros_1/Less/yÇ
"sequential/ipeigywbwc/zeros_1/LessLess%sequential/ipeigywbwc/zeros_1/mul:z:0-sequential/ipeigywbwc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/ipeigywbwc/zeros_1/Less
&sequential/ipeigywbwc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/ipeigywbwc/zeros_1/packed/1á
$sequential/ipeigywbwc/zeros_1/packedPack,sequential/ipeigywbwc/strided_slice:output:0/sequential/ipeigywbwc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/ipeigywbwc/zeros_1/packed
#sequential/ipeigywbwc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/ipeigywbwc/zeros_1/ConstÕ
sequential/ipeigywbwc/zeros_1Fill-sequential/ipeigywbwc/zeros_1/packed:output:0,sequential/ipeigywbwc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/ipeigywbwc/zeros_1¡
$sequential/ipeigywbwc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/ipeigywbwc/transpose/permÛ
sequential/ipeigywbwc/transpose	Transpose%sequential/zeuewnmlut/transpose_1:y:0-sequential/ipeigywbwc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/ipeigywbwc/transpose
sequential/ipeigywbwc/Shape_1Shape#sequential/ipeigywbwc/transpose:y:0*
T0*
_output_shapes
:2
sequential/ipeigywbwc/Shape_1¤
+sequential/ipeigywbwc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/ipeigywbwc/strided_slice_1/stack¨
-sequential/ipeigywbwc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ipeigywbwc/strided_slice_1/stack_1¨
-sequential/ipeigywbwc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ipeigywbwc/strided_slice_1/stack_2ò
%sequential/ipeigywbwc/strided_slice_1StridedSlice&sequential/ipeigywbwc/Shape_1:output:04sequential/ipeigywbwc/strided_slice_1/stack:output:06sequential/ipeigywbwc/strided_slice_1/stack_1:output:06sequential/ipeigywbwc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/ipeigywbwc/strided_slice_1±
1sequential/ipeigywbwc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/ipeigywbwc/TensorArrayV2/element_shape
#sequential/ipeigywbwc/TensorArrayV2TensorListReserve:sequential/ipeigywbwc/TensorArrayV2/element_shape:output:0.sequential/ipeigywbwc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/ipeigywbwc/TensorArrayV2ë
Ksequential/ipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/ipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/ipeigywbwc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/ipeigywbwc/transpose:y:0Tsequential/ipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/ipeigywbwc/TensorArrayUnstack/TensorListFromTensor¤
+sequential/ipeigywbwc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/ipeigywbwc/strided_slice_2/stack¨
-sequential/ipeigywbwc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ipeigywbwc/strided_slice_2/stack_1¨
-sequential/ipeigywbwc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ipeigywbwc/strided_slice_2/stack_2
%sequential/ipeigywbwc/strided_slice_2StridedSlice#sequential/ipeigywbwc/transpose:y:04sequential/ipeigywbwc/strided_slice_2/stack:output:06sequential/ipeigywbwc/strided_slice_2/stack_1:output:06sequential/ipeigywbwc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/ipeigywbwc/strided_slice_2ñ
6sequential/ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp?sequential_ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOpÿ
'sequential/ipeigywbwc/bdgxhkvdqy/MatMulMatMul.sequential/ipeigywbwc/strided_slice_2:output:0>sequential/ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/ipeigywbwc/bdgxhkvdqy/MatMul÷
8sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOpAsequential_ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOpû
)sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1MatMul$sequential/ipeigywbwc/zeros:output:0@sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1ð
$sequential/ipeigywbwc/bdgxhkvdqy/addAddV21sequential/ipeigywbwc/bdgxhkvdqy/MatMul:product:03sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/ipeigywbwc/bdgxhkvdqy/addð
7sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp@sequential_ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOpý
(sequential/ipeigywbwc/bdgxhkvdqy/BiasAddBiasAdd(sequential/ipeigywbwc/bdgxhkvdqy/add:z:0?sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd¦
0sequential/ipeigywbwc/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/ipeigywbwc/bdgxhkvdqy/split/split_dimÃ
&sequential/ipeigywbwc/bdgxhkvdqy/splitSplit9sequential/ipeigywbwc/bdgxhkvdqy/split/split_dim:output:01sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/ipeigywbwc/bdgxhkvdqy/split×
/sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOpReadVariableOp8sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOpæ
$sequential/ipeigywbwc/bdgxhkvdqy/mulMul7sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp:value:0&sequential/ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/ipeigywbwc/bdgxhkvdqy/mulæ
&sequential/ipeigywbwc/bdgxhkvdqy/add_1AddV2/sequential/ipeigywbwc/bdgxhkvdqy/split:output:0(sequential/ipeigywbwc/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/add_1½
(sequential/ipeigywbwc/bdgxhkvdqy/SigmoidSigmoid*sequential/ipeigywbwc/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/ipeigywbwc/bdgxhkvdqy/SigmoidÝ
1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp:sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1ì
&sequential/ipeigywbwc/bdgxhkvdqy/mul_1Mul9sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1:value:0&sequential/ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/mul_1è
&sequential/ipeigywbwc/bdgxhkvdqy/add_2AddV2/sequential/ipeigywbwc/bdgxhkvdqy/split:output:1*sequential/ipeigywbwc/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/add_2Á
*sequential/ipeigywbwc/bdgxhkvdqy/Sigmoid_1Sigmoid*sequential/ipeigywbwc/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/ipeigywbwc/bdgxhkvdqy/Sigmoid_1á
&sequential/ipeigywbwc/bdgxhkvdqy/mul_2Mul.sequential/ipeigywbwc/bdgxhkvdqy/Sigmoid_1:y:0&sequential/ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/mul_2¹
%sequential/ipeigywbwc/bdgxhkvdqy/TanhTanh/sequential/ipeigywbwc/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/ipeigywbwc/bdgxhkvdqy/Tanhâ
&sequential/ipeigywbwc/bdgxhkvdqy/mul_3Mul,sequential/ipeigywbwc/bdgxhkvdqy/Sigmoid:y:0)sequential/ipeigywbwc/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/mul_3ã
&sequential/ipeigywbwc/bdgxhkvdqy/add_3AddV2*sequential/ipeigywbwc/bdgxhkvdqy/mul_2:z:0*sequential/ipeigywbwc/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/add_3Ý
1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp:sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2ð
&sequential/ipeigywbwc/bdgxhkvdqy/mul_4Mul9sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2:value:0*sequential/ipeigywbwc/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/mul_4è
&sequential/ipeigywbwc/bdgxhkvdqy/add_4AddV2/sequential/ipeigywbwc/bdgxhkvdqy/split:output:3*sequential/ipeigywbwc/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/add_4Á
*sequential/ipeigywbwc/bdgxhkvdqy/Sigmoid_2Sigmoid*sequential/ipeigywbwc/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/ipeigywbwc/bdgxhkvdqy/Sigmoid_2¸
'sequential/ipeigywbwc/bdgxhkvdqy/Tanh_1Tanh*sequential/ipeigywbwc/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/ipeigywbwc/bdgxhkvdqy/Tanh_1æ
&sequential/ipeigywbwc/bdgxhkvdqy/mul_5Mul.sequential/ipeigywbwc/bdgxhkvdqy/Sigmoid_2:y:0+sequential/ipeigywbwc/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/bdgxhkvdqy/mul_5»
3sequential/ipeigywbwc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/ipeigywbwc/TensorArrayV2_1/element_shape
%sequential/ipeigywbwc/TensorArrayV2_1TensorListReserve<sequential/ipeigywbwc/TensorArrayV2_1/element_shape:output:0.sequential/ipeigywbwc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/ipeigywbwc/TensorArrayV2_1z
sequential/ipeigywbwc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/ipeigywbwc/time«
.sequential/ipeigywbwc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/ipeigywbwc/while/maximum_iterations
(sequential/ipeigywbwc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/ipeigywbwc/while/loop_counterö	
sequential/ipeigywbwc/whileWhile1sequential/ipeigywbwc/while/loop_counter:output:07sequential/ipeigywbwc/while/maximum_iterations:output:0#sequential/ipeigywbwc/time:output:0.sequential/ipeigywbwc/TensorArrayV2_1:handle:0$sequential/ipeigywbwc/zeros:output:0&sequential/ipeigywbwc/zeros_1:output:0.sequential/ipeigywbwc/strided_slice_1:output:0Msequential/ipeigywbwc/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resourceAsequential_ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource@sequential_ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource8sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_resource:sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource:sequential_ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource*
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
'sequential_ipeigywbwc_while_body_361034*3
cond+R)
'sequential_ipeigywbwc_while_cond_361033*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/ipeigywbwc/whileá
Fsequential/ipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/ipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/ipeigywbwc/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/ipeigywbwc/while:output:3Osequential/ipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/ipeigywbwc/TensorArrayV2Stack/TensorListStack­
+sequential/ipeigywbwc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/ipeigywbwc/strided_slice_3/stack¨
-sequential/ipeigywbwc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/ipeigywbwc/strided_slice_3/stack_1¨
-sequential/ipeigywbwc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/ipeigywbwc/strided_slice_3/stack_2
%sequential/ipeigywbwc/strided_slice_3StridedSliceAsequential/ipeigywbwc/TensorArrayV2Stack/TensorListStack:tensor:04sequential/ipeigywbwc/strided_slice_3/stack:output:06sequential/ipeigywbwc/strided_slice_3/stack_1:output:06sequential/ipeigywbwc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/ipeigywbwc/strided_slice_3¥
&sequential/ipeigywbwc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/ipeigywbwc/transpose_1/permý
!sequential/ipeigywbwc/transpose_1	TransposeAsequential/ipeigywbwc/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/ipeigywbwc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/ipeigywbwc/transpose_1Ï
+sequential/pemnqztknd/MatMul/ReadVariableOpReadVariableOp4sequential_pemnqztknd_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/pemnqztknd/MatMul/ReadVariableOpÝ
sequential/pemnqztknd/MatMulMatMul.sequential/ipeigywbwc/strided_slice_3:output:03sequential/pemnqztknd/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/pemnqztknd/MatMulÎ
,sequential/pemnqztknd/BiasAdd/ReadVariableOpReadVariableOp5sequential_pemnqztknd_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/pemnqztknd/BiasAdd/ReadVariableOpÙ
sequential/pemnqztknd/BiasAddBiasAdd&sequential/pemnqztknd/MatMul:product:04sequential/pemnqztknd/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/pemnqztknd/BiasAdd 
IdentityIdentity&sequential/pemnqztknd/BiasAdd:output:09^sequential/ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp@^sequential/ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp8^sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp7^sequential/ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp9^sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp0^sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp2^sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_12^sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2^sequential/ipeigywbwc/while-^sequential/pemnqztknd/BiasAdd/ReadVariableOp,^sequential/pemnqztknd/MatMul/ReadVariableOp8^sequential/zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp7^sequential/zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp9^sequential/zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp0^sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp2^sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_12^sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_2^sequential/zeuewnmlut/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2t
8sequential/ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp8sequential/ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp7sequential/ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp2p
6sequential/ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp6sequential/ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp2t
8sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp8sequential/ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp2b
/sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp/sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp2f
1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_11sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_12f
1sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_21sequential/ipeigywbwc/bdgxhkvdqy/ReadVariableOp_22:
sequential/ipeigywbwc/whilesequential/ipeigywbwc/while2\
,sequential/pemnqztknd/BiasAdd/ReadVariableOp,sequential/pemnqztknd/BiasAdd/ReadVariableOp2Z
+sequential/pemnqztknd/MatMul/ReadVariableOp+sequential/pemnqztknd/MatMul/ReadVariableOp2r
7sequential/zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp7sequential/zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp2p
6sequential/zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp6sequential/zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp2t
8sequential/zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp8sequential/zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp2b
/sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp/sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp2f
1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_11sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_12f
1sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_21sequential/zeuewnmlut/kfdgklwsil/ReadVariableOp_22:
sequential/zeuewnmlut/whilesequential/zeuewnmlut/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
kjggqknufb

¡	
'sequential_zeuewnmlut_while_cond_360857H
Dsequential_zeuewnmlut_while_sequential_zeuewnmlut_while_loop_counterN
Jsequential_zeuewnmlut_while_sequential_zeuewnmlut_while_maximum_iterations+
'sequential_zeuewnmlut_while_placeholder-
)sequential_zeuewnmlut_while_placeholder_1-
)sequential_zeuewnmlut_while_placeholder_2-
)sequential_zeuewnmlut_while_placeholder_3J
Fsequential_zeuewnmlut_while_less_sequential_zeuewnmlut_strided_slice_1`
\sequential_zeuewnmlut_while_sequential_zeuewnmlut_while_cond_360857___redundant_placeholder0`
\sequential_zeuewnmlut_while_sequential_zeuewnmlut_while_cond_360857___redundant_placeholder1`
\sequential_zeuewnmlut_while_sequential_zeuewnmlut_while_cond_360857___redundant_placeholder2`
\sequential_zeuewnmlut_while_sequential_zeuewnmlut_while_cond_360857___redundant_placeholder3`
\sequential_zeuewnmlut_while_sequential_zeuewnmlut_while_cond_360857___redundant_placeholder4`
\sequential_zeuewnmlut_while_sequential_zeuewnmlut_while_cond_360857___redundant_placeholder5`
\sequential_zeuewnmlut_while_sequential_zeuewnmlut_while_cond_360857___redundant_placeholder6(
$sequential_zeuewnmlut_while_identity
Þ
 sequential/zeuewnmlut/while/LessLess'sequential_zeuewnmlut_while_placeholderFsequential_zeuewnmlut_while_less_sequential_zeuewnmlut_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/zeuewnmlut/while/Less
$sequential/zeuewnmlut/while/IdentityIdentity$sequential/zeuewnmlut/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/zeuewnmlut/while/Identity"U
$sequential_zeuewnmlut_while_identity-sequential/zeuewnmlut/while/Identity:output:0*(
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
Ò	
÷
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_366425

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
ÙÊ

F__inference_sequential_layer_call_and_return_conditional_losses_364297

inputsL
6ekzorghjta_conv1d_expanddims_1_readvariableop_resource:K
=ekzorghjta_squeeze_batch_dims_biasadd_readvariableop_resource:G
4zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource:	I
6zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource:	 D
5zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource:	;
-zeuewnmlut_kfdgklwsil_readvariableop_resource: =
/zeuewnmlut_kfdgklwsil_readvariableop_1_resource: =
/zeuewnmlut_kfdgklwsil_readvariableop_2_resource: G
4ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource:	 I
6ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource:	 D
5ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource:	;
-ipeigywbwc_bdgxhkvdqy_readvariableop_resource: =
/ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource: =
/ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource: ;
)pemnqztknd_matmul_readvariableop_resource: 8
*pemnqztknd_biasadd_readvariableop_resource:
identity¢-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp¢4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp¢+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp¢-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp¢$ipeigywbwc/bdgxhkvdqy/ReadVariableOp¢&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1¢&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2¢ipeigywbwc/while¢!pemnqztknd/BiasAdd/ReadVariableOp¢ pemnqztknd/MatMul/ReadVariableOp¢,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp¢+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp¢-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp¢$zeuewnmlut/kfdgklwsil/ReadVariableOp¢&zeuewnmlut/kfdgklwsil/ReadVariableOp_1¢&zeuewnmlut/kfdgklwsil/ReadVariableOp_2¢zeuewnmlut/while
 ekzorghjta/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 ekzorghjta/conv1d/ExpandDims/dim»
ekzorghjta/conv1d/ExpandDims
ExpandDimsinputs)ekzorghjta/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/ExpandDimsÙ
-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6ekzorghjta_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp
"ekzorghjta/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ekzorghjta/conv1d/ExpandDims_1/dimã
ekzorghjta/conv1d/ExpandDims_1
ExpandDims5ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp:value:0+ekzorghjta/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
ekzorghjta/conv1d/ExpandDims_1
ekzorghjta/conv1d/ShapeShape%ekzorghjta/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
ekzorghjta/conv1d/Shape
%ekzorghjta/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%ekzorghjta/conv1d/strided_slice/stack¥
'ekzorghjta/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'ekzorghjta/conv1d/strided_slice/stack_1
'ekzorghjta/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'ekzorghjta/conv1d/strided_slice/stack_2Ì
ekzorghjta/conv1d/strided_sliceStridedSlice ekzorghjta/conv1d/Shape:output:0.ekzorghjta/conv1d/strided_slice/stack:output:00ekzorghjta/conv1d/strided_slice/stack_1:output:00ekzorghjta/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
ekzorghjta/conv1d/strided_slice
ekzorghjta/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
ekzorghjta/conv1d/Reshape/shapeÌ
ekzorghjta/conv1d/ReshapeReshape%ekzorghjta/conv1d/ExpandDims:output:0(ekzorghjta/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/Reshapeî
ekzorghjta/conv1d/Conv2DConv2D"ekzorghjta/conv1d/Reshape:output:0'ekzorghjta/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ekzorghjta/conv1d/Conv2D
!ekzorghjta/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!ekzorghjta/conv1d/concat/values_1
ekzorghjta/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/concat/axisì
ekzorghjta/conv1d/concatConcatV2(ekzorghjta/conv1d/strided_slice:output:0*ekzorghjta/conv1d/concat/values_1:output:0&ekzorghjta/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ekzorghjta/conv1d/concatÉ
ekzorghjta/conv1d/Reshape_1Reshape!ekzorghjta/conv1d/Conv2D:output:0!ekzorghjta/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/Reshape_1Á
ekzorghjta/conv1d/SqueezeSqueeze$ekzorghjta/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/Squeeze
#ekzorghjta/squeeze_batch_dims/ShapeShape"ekzorghjta/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#ekzorghjta/squeeze_batch_dims/Shape°
1ekzorghjta/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1ekzorghjta/squeeze_batch_dims/strided_slice/stack½
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_1´
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_2
+ekzorghjta/squeeze_batch_dims/strided_sliceStridedSlice,ekzorghjta/squeeze_batch_dims/Shape:output:0:ekzorghjta/squeeze_batch_dims/strided_slice/stack:output:0<ekzorghjta/squeeze_batch_dims/strided_slice/stack_1:output:0<ekzorghjta/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+ekzorghjta/squeeze_batch_dims/strided_slice¯
+ekzorghjta/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+ekzorghjta/squeeze_batch_dims/Reshape/shapeé
%ekzorghjta/squeeze_batch_dims/ReshapeReshape"ekzorghjta/conv1d/Squeeze:output:04ekzorghjta/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ekzorghjta/squeeze_batch_dims/Reshapeæ
4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=ekzorghjta_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%ekzorghjta/squeeze_batch_dims/BiasAddBiasAdd.ekzorghjta/squeeze_batch_dims/Reshape:output:0<ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ekzorghjta/squeeze_batch_dims/BiasAdd¯
-ekzorghjta/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-ekzorghjta/squeeze_batch_dims/concat/values_1¡
)ekzorghjta/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)ekzorghjta/squeeze_batch_dims/concat/axis¨
$ekzorghjta/squeeze_batch_dims/concatConcatV24ekzorghjta/squeeze_batch_dims/strided_slice:output:06ekzorghjta/squeeze_batch_dims/concat/values_1:output:02ekzorghjta/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$ekzorghjta/squeeze_batch_dims/concatö
'ekzorghjta/squeeze_batch_dims/Reshape_1Reshape.ekzorghjta/squeeze_batch_dims/BiasAdd:output:0-ekzorghjta/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'ekzorghjta/squeeze_batch_dims/Reshape_1
xsvuntduhq/ShapeShape0ekzorghjta/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
xsvuntduhq/Shape
xsvuntduhq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xsvuntduhq/strided_slice/stack
 xsvuntduhq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xsvuntduhq/strided_slice/stack_1
 xsvuntduhq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xsvuntduhq/strided_slice/stack_2¤
xsvuntduhq/strided_sliceStridedSlicexsvuntduhq/Shape:output:0'xsvuntduhq/strided_slice/stack:output:0)xsvuntduhq/strided_slice/stack_1:output:0)xsvuntduhq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xsvuntduhq/strided_slicez
xsvuntduhq/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
xsvuntduhq/Reshape/shape/1z
xsvuntduhq/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
xsvuntduhq/Reshape/shape/2×
xsvuntduhq/Reshape/shapePack!xsvuntduhq/strided_slice:output:0#xsvuntduhq/Reshape/shape/1:output:0#xsvuntduhq/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
xsvuntduhq/Reshape/shape¾
xsvuntduhq/ReshapeReshape0ekzorghjta/squeeze_batch_dims/Reshape_1:output:0!xsvuntduhq/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xsvuntduhq/Reshapeo
zeuewnmlut/ShapeShapexsvuntduhq/Reshape:output:0*
T0*
_output_shapes
:2
zeuewnmlut/Shape
zeuewnmlut/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
zeuewnmlut/strided_slice/stack
 zeuewnmlut/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 zeuewnmlut/strided_slice/stack_1
 zeuewnmlut/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 zeuewnmlut/strided_slice/stack_2¤
zeuewnmlut/strided_sliceStridedSlicezeuewnmlut/Shape:output:0'zeuewnmlut/strided_slice/stack:output:0)zeuewnmlut/strided_slice/stack_1:output:0)zeuewnmlut/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zeuewnmlut/strided_slicer
zeuewnmlut/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros/mul/y
zeuewnmlut/zeros/mulMul!zeuewnmlut/strided_slice:output:0zeuewnmlut/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros/mulu
zeuewnmlut/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeuewnmlut/zeros/Less/y
zeuewnmlut/zeros/LessLesszeuewnmlut/zeros/mul:z:0 zeuewnmlut/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros/Lessx
zeuewnmlut/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros/packed/1¯
zeuewnmlut/zeros/packedPack!zeuewnmlut/strided_slice:output:0"zeuewnmlut/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeuewnmlut/zeros/packedu
zeuewnmlut/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeuewnmlut/zeros/Const¡
zeuewnmlut/zerosFill zeuewnmlut/zeros/packed:output:0zeuewnmlut/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/zerosv
zeuewnmlut/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros_1/mul/y
zeuewnmlut/zeros_1/mulMul!zeuewnmlut/strided_slice:output:0!zeuewnmlut/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros_1/muly
zeuewnmlut/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeuewnmlut/zeros_1/Less/y
zeuewnmlut/zeros_1/LessLesszeuewnmlut/zeros_1/mul:z:0"zeuewnmlut/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros_1/Less|
zeuewnmlut/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros_1/packed/1µ
zeuewnmlut/zeros_1/packedPack!zeuewnmlut/strided_slice:output:0$zeuewnmlut/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeuewnmlut/zeros_1/packedy
zeuewnmlut/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeuewnmlut/zeros_1/Const©
zeuewnmlut/zeros_1Fill"zeuewnmlut/zeros_1/packed:output:0!zeuewnmlut/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/zeros_1
zeuewnmlut/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zeuewnmlut/transpose/perm°
zeuewnmlut/transpose	Transposexsvuntduhq/Reshape:output:0"zeuewnmlut/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/transposep
zeuewnmlut/Shape_1Shapezeuewnmlut/transpose:y:0*
T0*
_output_shapes
:2
zeuewnmlut/Shape_1
 zeuewnmlut/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zeuewnmlut/strided_slice_1/stack
"zeuewnmlut/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_1/stack_1
"zeuewnmlut/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_1/stack_2°
zeuewnmlut/strided_slice_1StridedSlicezeuewnmlut/Shape_1:output:0)zeuewnmlut/strided_slice_1/stack:output:0+zeuewnmlut/strided_slice_1/stack_1:output:0+zeuewnmlut/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zeuewnmlut/strided_slice_1
&zeuewnmlut/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&zeuewnmlut/TensorArrayV2/element_shapeÞ
zeuewnmlut/TensorArrayV2TensorListReserve/zeuewnmlut/TensorArrayV2/element_shape:output:0#zeuewnmlut/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zeuewnmlut/TensorArrayV2Õ
@zeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@zeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2zeuewnmlut/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorzeuewnmlut/transpose:y:0Izeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2zeuewnmlut/TensorArrayUnstack/TensorListFromTensor
 zeuewnmlut/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zeuewnmlut/strided_slice_2/stack
"zeuewnmlut/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_2/stack_1
"zeuewnmlut/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_2/stack_2¾
zeuewnmlut/strided_slice_2StridedSlicezeuewnmlut/transpose:y:0)zeuewnmlut/strided_slice_2/stack:output:0+zeuewnmlut/strided_slice_2/stack_1:output:0+zeuewnmlut/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
zeuewnmlut/strided_slice_2Ð
+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp4zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOpÓ
zeuewnmlut/kfdgklwsil/MatMulMatMul#zeuewnmlut/strided_slice_2:output:03zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/kfdgklwsil/MatMulÖ
-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp6zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOpÏ
zeuewnmlut/kfdgklwsil/MatMul_1MatMulzeuewnmlut/zeros:output:05zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
zeuewnmlut/kfdgklwsil/MatMul_1Ä
zeuewnmlut/kfdgklwsil/addAddV2&zeuewnmlut/kfdgklwsil/MatMul:product:0(zeuewnmlut/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/kfdgklwsil/addÏ
,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp5zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOpÑ
zeuewnmlut/kfdgklwsil/BiasAddBiasAddzeuewnmlut/kfdgklwsil/add:z:04zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/kfdgklwsil/BiasAdd
%zeuewnmlut/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%zeuewnmlut/kfdgklwsil/split/split_dim
zeuewnmlut/kfdgklwsil/splitSplit.zeuewnmlut/kfdgklwsil/split/split_dim:output:0&zeuewnmlut/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zeuewnmlut/kfdgklwsil/split¶
$zeuewnmlut/kfdgklwsil/ReadVariableOpReadVariableOp-zeuewnmlut_kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02&
$zeuewnmlut/kfdgklwsil/ReadVariableOpº
zeuewnmlut/kfdgklwsil/mulMul,zeuewnmlut/kfdgklwsil/ReadVariableOp:value:0zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mulº
zeuewnmlut/kfdgklwsil/add_1AddV2$zeuewnmlut/kfdgklwsil/split:output:0zeuewnmlut/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_1
zeuewnmlut/kfdgklwsil/SigmoidSigmoidzeuewnmlut/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/Sigmoid¼
&zeuewnmlut/kfdgklwsil/ReadVariableOp_1ReadVariableOp/zeuewnmlut_kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&zeuewnmlut/kfdgklwsil/ReadVariableOp_1À
zeuewnmlut/kfdgklwsil/mul_1Mul.zeuewnmlut/kfdgklwsil/ReadVariableOp_1:value:0zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_1¼
zeuewnmlut/kfdgklwsil/add_2AddV2$zeuewnmlut/kfdgklwsil/split:output:1zeuewnmlut/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_2 
zeuewnmlut/kfdgklwsil/Sigmoid_1Sigmoidzeuewnmlut/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zeuewnmlut/kfdgklwsil/Sigmoid_1µ
zeuewnmlut/kfdgklwsil/mul_2Mul#zeuewnmlut/kfdgklwsil/Sigmoid_1:y:0zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_2
zeuewnmlut/kfdgklwsil/TanhTanh$zeuewnmlut/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/Tanh¶
zeuewnmlut/kfdgklwsil/mul_3Mul!zeuewnmlut/kfdgklwsil/Sigmoid:y:0zeuewnmlut/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_3·
zeuewnmlut/kfdgklwsil/add_3AddV2zeuewnmlut/kfdgklwsil/mul_2:z:0zeuewnmlut/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_3¼
&zeuewnmlut/kfdgklwsil/ReadVariableOp_2ReadVariableOp/zeuewnmlut_kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&zeuewnmlut/kfdgklwsil/ReadVariableOp_2Ä
zeuewnmlut/kfdgklwsil/mul_4Mul.zeuewnmlut/kfdgklwsil/ReadVariableOp_2:value:0zeuewnmlut/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_4¼
zeuewnmlut/kfdgklwsil/add_4AddV2$zeuewnmlut/kfdgklwsil/split:output:3zeuewnmlut/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_4 
zeuewnmlut/kfdgklwsil/Sigmoid_2Sigmoidzeuewnmlut/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zeuewnmlut/kfdgklwsil/Sigmoid_2
zeuewnmlut/kfdgklwsil/Tanh_1Tanhzeuewnmlut/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/Tanh_1º
zeuewnmlut/kfdgklwsil/mul_5Mul#zeuewnmlut/kfdgklwsil/Sigmoid_2:y:0 zeuewnmlut/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_5¥
(zeuewnmlut/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(zeuewnmlut/TensorArrayV2_1/element_shapeä
zeuewnmlut/TensorArrayV2_1TensorListReserve1zeuewnmlut/TensorArrayV2_1/element_shape:output:0#zeuewnmlut/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zeuewnmlut/TensorArrayV2_1d
zeuewnmlut/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/time
#zeuewnmlut/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#zeuewnmlut/while/maximum_iterations
zeuewnmlut/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/while/loop_counter°
zeuewnmlut/whileWhile&zeuewnmlut/while/loop_counter:output:0,zeuewnmlut/while/maximum_iterations:output:0zeuewnmlut/time:output:0#zeuewnmlut/TensorArrayV2_1:handle:0zeuewnmlut/zeros:output:0zeuewnmlut/zeros_1:output:0#zeuewnmlut/strided_slice_1:output:0Bzeuewnmlut/TensorArrayUnstack/TensorListFromTensor:output_handle:04zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource6zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource5zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource-zeuewnmlut_kfdgklwsil_readvariableop_resource/zeuewnmlut_kfdgklwsil_readvariableop_1_resource/zeuewnmlut_kfdgklwsil_readvariableop_2_resource*
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
zeuewnmlut_while_body_364014*(
cond R
zeuewnmlut_while_cond_364013*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
zeuewnmlut/whileË
;zeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;zeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shape
-zeuewnmlut/TensorArrayV2Stack/TensorListStackTensorListStackzeuewnmlut/while:output:3Dzeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-zeuewnmlut/TensorArrayV2Stack/TensorListStack
 zeuewnmlut/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 zeuewnmlut/strided_slice_3/stack
"zeuewnmlut/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"zeuewnmlut/strided_slice_3/stack_1
"zeuewnmlut/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_3/stack_2Ü
zeuewnmlut/strided_slice_3StridedSlice6zeuewnmlut/TensorArrayV2Stack/TensorListStack:tensor:0)zeuewnmlut/strided_slice_3/stack:output:0+zeuewnmlut/strided_slice_3/stack_1:output:0+zeuewnmlut/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
zeuewnmlut/strided_slice_3
zeuewnmlut/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zeuewnmlut/transpose_1/permÑ
zeuewnmlut/transpose_1	Transpose6zeuewnmlut/TensorArrayV2Stack/TensorListStack:tensor:0$zeuewnmlut/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/transpose_1n
ipeigywbwc/ShapeShapezeuewnmlut/transpose_1:y:0*
T0*
_output_shapes
:2
ipeigywbwc/Shape
ipeigywbwc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ipeigywbwc/strided_slice/stack
 ipeigywbwc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ipeigywbwc/strided_slice/stack_1
 ipeigywbwc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ipeigywbwc/strided_slice/stack_2¤
ipeigywbwc/strided_sliceStridedSliceipeigywbwc/Shape:output:0'ipeigywbwc/strided_slice/stack:output:0)ipeigywbwc/strided_slice/stack_1:output:0)ipeigywbwc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ipeigywbwc/strided_slicer
ipeigywbwc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros/mul/y
ipeigywbwc/zeros/mulMul!ipeigywbwc/strided_slice:output:0ipeigywbwc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros/mulu
ipeigywbwc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ipeigywbwc/zeros/Less/y
ipeigywbwc/zeros/LessLessipeigywbwc/zeros/mul:z:0 ipeigywbwc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros/Lessx
ipeigywbwc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros/packed/1¯
ipeigywbwc/zeros/packedPack!ipeigywbwc/strided_slice:output:0"ipeigywbwc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
ipeigywbwc/zeros/packedu
ipeigywbwc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ipeigywbwc/zeros/Const¡
ipeigywbwc/zerosFill ipeigywbwc/zeros/packed:output:0ipeigywbwc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/zerosv
ipeigywbwc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros_1/mul/y
ipeigywbwc/zeros_1/mulMul!ipeigywbwc/strided_slice:output:0!ipeigywbwc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros_1/muly
ipeigywbwc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ipeigywbwc/zeros_1/Less/y
ipeigywbwc/zeros_1/LessLessipeigywbwc/zeros_1/mul:z:0"ipeigywbwc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros_1/Less|
ipeigywbwc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros_1/packed/1µ
ipeigywbwc/zeros_1/packedPack!ipeigywbwc/strided_slice:output:0$ipeigywbwc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
ipeigywbwc/zeros_1/packedy
ipeigywbwc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ipeigywbwc/zeros_1/Const©
ipeigywbwc/zeros_1Fill"ipeigywbwc/zeros_1/packed:output:0!ipeigywbwc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/zeros_1
ipeigywbwc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ipeigywbwc/transpose/perm¯
ipeigywbwc/transpose	Transposezeuewnmlut/transpose_1:y:0"ipeigywbwc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/transposep
ipeigywbwc/Shape_1Shapeipeigywbwc/transpose:y:0*
T0*
_output_shapes
:2
ipeigywbwc/Shape_1
 ipeigywbwc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ipeigywbwc/strided_slice_1/stack
"ipeigywbwc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_1/stack_1
"ipeigywbwc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_1/stack_2°
ipeigywbwc/strided_slice_1StridedSliceipeigywbwc/Shape_1:output:0)ipeigywbwc/strided_slice_1/stack:output:0+ipeigywbwc/strided_slice_1/stack_1:output:0+ipeigywbwc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ipeigywbwc/strided_slice_1
&ipeigywbwc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&ipeigywbwc/TensorArrayV2/element_shapeÞ
ipeigywbwc/TensorArrayV2TensorListReserve/ipeigywbwc/TensorArrayV2/element_shape:output:0#ipeigywbwc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ipeigywbwc/TensorArrayV2Õ
@ipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@ipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2ipeigywbwc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoripeigywbwc/transpose:y:0Iipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2ipeigywbwc/TensorArrayUnstack/TensorListFromTensor
 ipeigywbwc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ipeigywbwc/strided_slice_2/stack
"ipeigywbwc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_2/stack_1
"ipeigywbwc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_2/stack_2¾
ipeigywbwc/strided_slice_2StridedSliceipeigywbwc/transpose:y:0)ipeigywbwc/strided_slice_2/stack:output:0+ipeigywbwc/strided_slice_2/stack_1:output:0+ipeigywbwc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ipeigywbwc/strided_slice_2Ð
+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp4ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOpÓ
ipeigywbwc/bdgxhkvdqy/MatMulMatMul#ipeigywbwc/strided_slice_2:output:03ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ipeigywbwc/bdgxhkvdqy/MatMulÖ
-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp6ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOpÏ
ipeigywbwc/bdgxhkvdqy/MatMul_1MatMulipeigywbwc/zeros:output:05ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
ipeigywbwc/bdgxhkvdqy/MatMul_1Ä
ipeigywbwc/bdgxhkvdqy/addAddV2&ipeigywbwc/bdgxhkvdqy/MatMul:product:0(ipeigywbwc/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ipeigywbwc/bdgxhkvdqy/addÏ
,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp5ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOpÑ
ipeigywbwc/bdgxhkvdqy/BiasAddBiasAddipeigywbwc/bdgxhkvdqy/add:z:04ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ipeigywbwc/bdgxhkvdqy/BiasAdd
%ipeigywbwc/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%ipeigywbwc/bdgxhkvdqy/split/split_dim
ipeigywbwc/bdgxhkvdqy/splitSplit.ipeigywbwc/bdgxhkvdqy/split/split_dim:output:0&ipeigywbwc/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ipeigywbwc/bdgxhkvdqy/split¶
$ipeigywbwc/bdgxhkvdqy/ReadVariableOpReadVariableOp-ipeigywbwc_bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02&
$ipeigywbwc/bdgxhkvdqy/ReadVariableOpº
ipeigywbwc/bdgxhkvdqy/mulMul,ipeigywbwc/bdgxhkvdqy/ReadVariableOp:value:0ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mulº
ipeigywbwc/bdgxhkvdqy/add_1AddV2$ipeigywbwc/bdgxhkvdqy/split:output:0ipeigywbwc/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_1
ipeigywbwc/bdgxhkvdqy/SigmoidSigmoidipeigywbwc/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/Sigmoid¼
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp/ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1À
ipeigywbwc/bdgxhkvdqy/mul_1Mul.ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1:value:0ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_1¼
ipeigywbwc/bdgxhkvdqy/add_2AddV2$ipeigywbwc/bdgxhkvdqy/split:output:1ipeigywbwc/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_2 
ipeigywbwc/bdgxhkvdqy/Sigmoid_1Sigmoidipeigywbwc/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ipeigywbwc/bdgxhkvdqy/Sigmoid_1µ
ipeigywbwc/bdgxhkvdqy/mul_2Mul#ipeigywbwc/bdgxhkvdqy/Sigmoid_1:y:0ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_2
ipeigywbwc/bdgxhkvdqy/TanhTanh$ipeigywbwc/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/Tanh¶
ipeigywbwc/bdgxhkvdqy/mul_3Mul!ipeigywbwc/bdgxhkvdqy/Sigmoid:y:0ipeigywbwc/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_3·
ipeigywbwc/bdgxhkvdqy/add_3AddV2ipeigywbwc/bdgxhkvdqy/mul_2:z:0ipeigywbwc/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_3¼
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp/ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2Ä
ipeigywbwc/bdgxhkvdqy/mul_4Mul.ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2:value:0ipeigywbwc/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_4¼
ipeigywbwc/bdgxhkvdqy/add_4AddV2$ipeigywbwc/bdgxhkvdqy/split:output:3ipeigywbwc/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_4 
ipeigywbwc/bdgxhkvdqy/Sigmoid_2Sigmoidipeigywbwc/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ipeigywbwc/bdgxhkvdqy/Sigmoid_2
ipeigywbwc/bdgxhkvdqy/Tanh_1Tanhipeigywbwc/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/Tanh_1º
ipeigywbwc/bdgxhkvdqy/mul_5Mul#ipeigywbwc/bdgxhkvdqy/Sigmoid_2:y:0 ipeigywbwc/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_5¥
(ipeigywbwc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(ipeigywbwc/TensorArrayV2_1/element_shapeä
ipeigywbwc/TensorArrayV2_1TensorListReserve1ipeigywbwc/TensorArrayV2_1/element_shape:output:0#ipeigywbwc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ipeigywbwc/TensorArrayV2_1d
ipeigywbwc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/time
#ipeigywbwc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#ipeigywbwc/while/maximum_iterations
ipeigywbwc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/while/loop_counter°
ipeigywbwc/whileWhile&ipeigywbwc/while/loop_counter:output:0,ipeigywbwc/while/maximum_iterations:output:0ipeigywbwc/time:output:0#ipeigywbwc/TensorArrayV2_1:handle:0ipeigywbwc/zeros:output:0ipeigywbwc/zeros_1:output:0#ipeigywbwc/strided_slice_1:output:0Bipeigywbwc/TensorArrayUnstack/TensorListFromTensor:output_handle:04ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource6ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource5ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource-ipeigywbwc_bdgxhkvdqy_readvariableop_resource/ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource/ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource*
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
ipeigywbwc_while_body_364190*(
cond R
ipeigywbwc_while_cond_364189*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
ipeigywbwc/whileË
;ipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;ipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shape
-ipeigywbwc/TensorArrayV2Stack/TensorListStackTensorListStackipeigywbwc/while:output:3Dipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-ipeigywbwc/TensorArrayV2Stack/TensorListStack
 ipeigywbwc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ipeigywbwc/strided_slice_3/stack
"ipeigywbwc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ipeigywbwc/strided_slice_3/stack_1
"ipeigywbwc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_3/stack_2Ü
ipeigywbwc/strided_slice_3StridedSlice6ipeigywbwc/TensorArrayV2Stack/TensorListStack:tensor:0)ipeigywbwc/strided_slice_3/stack:output:0+ipeigywbwc/strided_slice_3/stack_1:output:0+ipeigywbwc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ipeigywbwc/strided_slice_3
ipeigywbwc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ipeigywbwc/transpose_1/permÑ
ipeigywbwc/transpose_1	Transpose6ipeigywbwc/TensorArrayV2Stack/TensorListStack:tensor:0$ipeigywbwc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/transpose_1®
 pemnqztknd/MatMul/ReadVariableOpReadVariableOp)pemnqztknd_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 pemnqztknd/MatMul/ReadVariableOp±
pemnqztknd/MatMulMatMul#ipeigywbwc/strided_slice_3:output:0(pemnqztknd/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pemnqztknd/MatMul­
!pemnqztknd/BiasAdd/ReadVariableOpReadVariableOp*pemnqztknd_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!pemnqztknd/BiasAdd/ReadVariableOp­
pemnqztknd/BiasAddBiasAddpemnqztknd/MatMul:product:0)pemnqztknd/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pemnqztknd/BiasAddÏ
IdentityIdentitypemnqztknd/BiasAdd:output:0.^ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp5^ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp-^ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp,^ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp.^ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp%^ipeigywbwc/bdgxhkvdqy/ReadVariableOp'^ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1'^ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2^ipeigywbwc/while"^pemnqztknd/BiasAdd/ReadVariableOp!^pemnqztknd/MatMul/ReadVariableOp-^zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp,^zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp.^zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp%^zeuewnmlut/kfdgklwsil/ReadVariableOp'^zeuewnmlut/kfdgklwsil/ReadVariableOp_1'^zeuewnmlut/kfdgklwsil/ReadVariableOp_2^zeuewnmlut/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp2l
4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp2Z
+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp2^
-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp2L
$ipeigywbwc/bdgxhkvdqy/ReadVariableOp$ipeigywbwc/bdgxhkvdqy/ReadVariableOp2P
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_12P
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_22$
ipeigywbwc/whileipeigywbwc/while2F
!pemnqztknd/BiasAdd/ReadVariableOp!pemnqztknd/BiasAdd/ReadVariableOp2D
 pemnqztknd/MatMul/ReadVariableOp pemnqztknd/MatMul/ReadVariableOp2\
,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp2Z
+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp2^
-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp2L
$zeuewnmlut/kfdgklwsil/ReadVariableOp$zeuewnmlut/kfdgklwsil/ReadVariableOp2P
&zeuewnmlut/kfdgklwsil/ReadVariableOp_1&zeuewnmlut/kfdgklwsil/ReadVariableOp_12P
&zeuewnmlut/kfdgklwsil/ReadVariableOp_2&zeuewnmlut/kfdgklwsil/ReadVariableOp_22$
zeuewnmlut/whilezeuewnmlut/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


zeuewnmlut_while_cond_3644172
.zeuewnmlut_while_zeuewnmlut_while_loop_counter8
4zeuewnmlut_while_zeuewnmlut_while_maximum_iterations 
zeuewnmlut_while_placeholder"
zeuewnmlut_while_placeholder_1"
zeuewnmlut_while_placeholder_2"
zeuewnmlut_while_placeholder_34
0zeuewnmlut_while_less_zeuewnmlut_strided_slice_1J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364417___redundant_placeholder0J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364417___redundant_placeholder1J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364417___redundant_placeholder2J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364417___redundant_placeholder3J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364417___redundant_placeholder4J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364417___redundant_placeholder5J
Fzeuewnmlut_while_zeuewnmlut_while_cond_364417___redundant_placeholder6
zeuewnmlut_while_identity
§
zeuewnmlut/while/LessLesszeuewnmlut_while_placeholder0zeuewnmlut_while_less_zeuewnmlut_strided_slice_1*
T0*
_output_shapes
: 2
zeuewnmlut/while/Less~
zeuewnmlut/while/IdentityIdentityzeuewnmlut/while/Less:z:0*
T0
*
_output_shapes
: 2
zeuewnmlut/while/Identity"?
zeuewnmlut_while_identity"zeuewnmlut/while/Identity:output:0*(
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_363094

inputs<
)bdgxhkvdqy_matmul_readvariableop_resource:	 >
+bdgxhkvdqy_matmul_1_readvariableop_resource:	 9
*bdgxhkvdqy_biasadd_readvariableop_resource:	0
"bdgxhkvdqy_readvariableop_resource: 2
$bdgxhkvdqy_readvariableop_1_resource: 2
$bdgxhkvdqy_readvariableop_2_resource: 
identity¢!bdgxhkvdqy/BiasAdd/ReadVariableOp¢ bdgxhkvdqy/MatMul/ReadVariableOp¢"bdgxhkvdqy/MatMul_1/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp_1¢bdgxhkvdqy/ReadVariableOp_2¢whileD
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
 bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp)bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 bdgxhkvdqy/MatMul/ReadVariableOp§
bdgxhkvdqy/MatMulMatMulstrided_slice_2:output:0(bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMulµ
"bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp+bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"bdgxhkvdqy/MatMul_1/ReadVariableOp£
bdgxhkvdqy/MatMul_1MatMulzeros:output:0*bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMul_1
bdgxhkvdqy/addAddV2bdgxhkvdqy/MatMul:product:0bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/add®
!bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp*bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!bdgxhkvdqy/BiasAdd/ReadVariableOp¥
bdgxhkvdqy/BiasAddBiasAddbdgxhkvdqy/add:z:0)bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/BiasAddz
bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
bdgxhkvdqy/split/split_dimë
bdgxhkvdqy/splitSplit#bdgxhkvdqy/split/split_dim:output:0bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
bdgxhkvdqy/split
bdgxhkvdqy/ReadVariableOpReadVariableOp"bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp
bdgxhkvdqy/mulMul!bdgxhkvdqy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul
bdgxhkvdqy/add_1AddV2bdgxhkvdqy/split:output:0bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_1{
bdgxhkvdqy/SigmoidSigmoidbdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid
bdgxhkvdqy/ReadVariableOp_1ReadVariableOp$bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_1
bdgxhkvdqy/mul_1Mul#bdgxhkvdqy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_1
bdgxhkvdqy/add_2AddV2bdgxhkvdqy/split:output:1bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_2
bdgxhkvdqy/Sigmoid_1Sigmoidbdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_1
bdgxhkvdqy/mul_2Mulbdgxhkvdqy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_2w
bdgxhkvdqy/TanhTanhbdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh
bdgxhkvdqy/mul_3Mulbdgxhkvdqy/Sigmoid:y:0bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_3
bdgxhkvdqy/add_3AddV2bdgxhkvdqy/mul_2:z:0bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_3
bdgxhkvdqy/ReadVariableOp_2ReadVariableOp$bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_2
bdgxhkvdqy/mul_4Mul#bdgxhkvdqy/ReadVariableOp_2:value:0bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_4
bdgxhkvdqy/add_4AddV2bdgxhkvdqy/split:output:3bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_4
bdgxhkvdqy/Sigmoid_2Sigmoidbdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_2v
bdgxhkvdqy/Tanh_1Tanhbdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh_1
bdgxhkvdqy/mul_5Mulbdgxhkvdqy/Sigmoid_2:y:0bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)bdgxhkvdqy_matmul_readvariableop_resource+bdgxhkvdqy_matmul_1_readvariableop_resource*bdgxhkvdqy_biasadd_readvariableop_resource"bdgxhkvdqy_readvariableop_resource$bdgxhkvdqy_readvariableop_1_resource$bdgxhkvdqy_readvariableop_2_resource*
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
while_body_362993*
condR
while_cond_362992*Q
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
IdentityIdentitystrided_slice_3:output:0"^bdgxhkvdqy/BiasAdd/ReadVariableOp!^bdgxhkvdqy/MatMul/ReadVariableOp#^bdgxhkvdqy/MatMul_1/ReadVariableOp^bdgxhkvdqy/ReadVariableOp^bdgxhkvdqy/ReadVariableOp_1^bdgxhkvdqy/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!bdgxhkvdqy/BiasAdd/ReadVariableOp!bdgxhkvdqy/BiasAdd/ReadVariableOp2D
 bdgxhkvdqy/MatMul/ReadVariableOp bdgxhkvdqy/MatMul/ReadVariableOp2H
"bdgxhkvdqy/MatMul_1/ReadVariableOp"bdgxhkvdqy/MatMul_1/ReadVariableOp26
bdgxhkvdqy/ReadVariableOpbdgxhkvdqy/ReadVariableOp2:
bdgxhkvdqy/ReadVariableOp_1bdgxhkvdqy/ReadVariableOp_12:
bdgxhkvdqy/ReadVariableOp_2bdgxhkvdqy/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


+__inference_sequential_layer_call_fn_364738

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
F__inference_sequential_layer_call_and_return_conditional_losses_3631252
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
while_body_364918
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kfdgklwsil_matmul_readvariableop_resource_0:	F
3while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 A
2while_kfdgklwsil_biasadd_readvariableop_resource_0:	8
*while_kfdgklwsil_readvariableop_resource_0: :
,while_kfdgklwsil_readvariableop_1_resource_0: :
,while_kfdgklwsil_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kfdgklwsil_matmul_readvariableop_resource:	D
1while_kfdgklwsil_matmul_1_readvariableop_resource:	 ?
0while_kfdgklwsil_biasadd_readvariableop_resource:	6
(while_kfdgklwsil_readvariableop_resource: 8
*while_kfdgklwsil_readvariableop_1_resource: 8
*while_kfdgklwsil_readvariableop_2_resource: ¢'while/kfdgklwsil/BiasAdd/ReadVariableOp¢&while/kfdgklwsil/MatMul/ReadVariableOp¢(while/kfdgklwsil/MatMul_1/ReadVariableOp¢while/kfdgklwsil/ReadVariableOp¢!while/kfdgklwsil/ReadVariableOp_1¢!while/kfdgklwsil/ReadVariableOp_2Ã
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
&while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp1while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/kfdgklwsil/MatMul/ReadVariableOpÑ
while/kfdgklwsil/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMulÉ
(while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp3while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kfdgklwsil/MatMul_1/ReadVariableOpº
while/kfdgklwsil/MatMul_1MatMulwhile_placeholder_20while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMul_1°
while/kfdgklwsil/addAddV2!while/kfdgklwsil/MatMul:product:0#while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/addÂ
'while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp2while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kfdgklwsil/BiasAdd/ReadVariableOp½
while/kfdgklwsil/BiasAddBiasAddwhile/kfdgklwsil/add:z:0/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/BiasAdd
 while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kfdgklwsil/split/split_dim
while/kfdgklwsil/splitSplit)while/kfdgklwsil/split/split_dim:output:0!while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kfdgklwsil/split©
while/kfdgklwsil/ReadVariableOpReadVariableOp*while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kfdgklwsil/ReadVariableOp£
while/kfdgklwsil/mulMul'while/kfdgklwsil/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul¦
while/kfdgklwsil/add_1AddV2while/kfdgklwsil/split:output:0while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_1
while/kfdgklwsil/SigmoidSigmoidwhile/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid¯
!while/kfdgklwsil/ReadVariableOp_1ReadVariableOp,while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_1©
while/kfdgklwsil/mul_1Mul)while/kfdgklwsil/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_1¨
while/kfdgklwsil/add_2AddV2while/kfdgklwsil/split:output:1while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_2
while/kfdgklwsil/Sigmoid_1Sigmoidwhile/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_1
while/kfdgklwsil/mul_2Mulwhile/kfdgklwsil/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_2
while/kfdgklwsil/TanhTanhwhile/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh¢
while/kfdgklwsil/mul_3Mulwhile/kfdgklwsil/Sigmoid:y:0while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_3£
while/kfdgklwsil/add_3AddV2while/kfdgklwsil/mul_2:z:0while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_3¯
!while/kfdgklwsil/ReadVariableOp_2ReadVariableOp,while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_2°
while/kfdgklwsil/mul_4Mul)while/kfdgklwsil/ReadVariableOp_2:value:0while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_4¨
while/kfdgklwsil/add_4AddV2while/kfdgklwsil/split:output:3while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_4
while/kfdgklwsil/Sigmoid_2Sigmoidwhile/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_2
while/kfdgklwsil/Tanh_1Tanhwhile/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh_1¦
while/kfdgklwsil/mul_5Mulwhile/kfdgklwsil/Sigmoid_2:y:0while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kfdgklwsil/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kfdgklwsil/mul_5:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kfdgklwsil/add_3:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
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
0while_kfdgklwsil_biasadd_readvariableop_resource2while_kfdgklwsil_biasadd_readvariableop_resource_0"h
1while_kfdgklwsil_matmul_1_readvariableop_resource3while_kfdgklwsil_matmul_1_readvariableop_resource_0"d
/while_kfdgklwsil_matmul_readvariableop_resource1while_kfdgklwsil_matmul_readvariableop_resource_0"Z
*while_kfdgklwsil_readvariableop_1_resource,while_kfdgklwsil_readvariableop_1_resource_0"Z
*while_kfdgklwsil_readvariableop_2_resource,while_kfdgklwsil_readvariableop_2_resource_0"V
(while_kfdgklwsil_readvariableop_resource*while_kfdgklwsil_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kfdgklwsil/BiasAdd/ReadVariableOp'while/kfdgklwsil/BiasAdd/ReadVariableOp2P
&while/kfdgklwsil/MatMul/ReadVariableOp&while/kfdgklwsil/MatMul/ReadVariableOp2T
(while/kfdgklwsil/MatMul_1/ReadVariableOp(while/kfdgklwsil/MatMul_1/ReadVariableOp2B
while/kfdgklwsil/ReadVariableOpwhile/kfdgklwsil/ReadVariableOp2F
!while/kfdgklwsil/ReadVariableOp_1!while/kfdgklwsil/ReadVariableOp_12F
!while/kfdgklwsil/ReadVariableOp_2!while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Ñ

+__inference_ipeigywbwc_layer_call_fn_366398

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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3630942
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
ßY

while_body_366246
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 F
3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 A
2while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	8
*while_bdgxhkvdqy_readvariableop_resource_0: :
,while_bdgxhkvdqy_readvariableop_1_resource_0: :
,while_bdgxhkvdqy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_bdgxhkvdqy_matmul_readvariableop_resource:	 D
1while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 ?
0while_bdgxhkvdqy_biasadd_readvariableop_resource:	6
(while_bdgxhkvdqy_readvariableop_resource: 8
*while_bdgxhkvdqy_readvariableop_1_resource: 8
*while_bdgxhkvdqy_readvariableop_2_resource: ¢'while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢&while/bdgxhkvdqy/MatMul/ReadVariableOp¢(while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢while/bdgxhkvdqy/ReadVariableOp¢!while/bdgxhkvdqy/ReadVariableOp_1¢!while/bdgxhkvdqy/ReadVariableOp_2Ã
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
&while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp1while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/bdgxhkvdqy/MatMul/ReadVariableOpÑ
while/bdgxhkvdqy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMulÉ
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpº
while/bdgxhkvdqy/MatMul_1MatMulwhile_placeholder_20while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMul_1°
while/bdgxhkvdqy/addAddV2!while/bdgxhkvdqy/MatMul:product:0#while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/addÂ
'while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp2while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp½
while/bdgxhkvdqy/BiasAddBiasAddwhile/bdgxhkvdqy/add:z:0/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/BiasAdd
 while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/bdgxhkvdqy/split/split_dim
while/bdgxhkvdqy/splitSplit)while/bdgxhkvdqy/split/split_dim:output:0!while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/bdgxhkvdqy/split©
while/bdgxhkvdqy/ReadVariableOpReadVariableOp*while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/bdgxhkvdqy/ReadVariableOp£
while/bdgxhkvdqy/mulMul'while/bdgxhkvdqy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul¦
while/bdgxhkvdqy/add_1AddV2while/bdgxhkvdqy/split:output:0while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_1
while/bdgxhkvdqy/SigmoidSigmoidwhile/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid¯
!while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp,while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_1©
while/bdgxhkvdqy/mul_1Mul)while/bdgxhkvdqy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_1¨
while/bdgxhkvdqy/add_2AddV2while/bdgxhkvdqy/split:output:1while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_2
while/bdgxhkvdqy/Sigmoid_1Sigmoidwhile/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_1
while/bdgxhkvdqy/mul_2Mulwhile/bdgxhkvdqy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_2
while/bdgxhkvdqy/TanhTanhwhile/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh¢
while/bdgxhkvdqy/mul_3Mulwhile/bdgxhkvdqy/Sigmoid:y:0while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_3£
while/bdgxhkvdqy/add_3AddV2while/bdgxhkvdqy/mul_2:z:0while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_3¯
!while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp,while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_2°
while/bdgxhkvdqy/mul_4Mul)while/bdgxhkvdqy/ReadVariableOp_2:value:0while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_4¨
while/bdgxhkvdqy/add_4AddV2while/bdgxhkvdqy/split:output:3while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_4
while/bdgxhkvdqy/Sigmoid_2Sigmoidwhile/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_2
while/bdgxhkvdqy/Tanh_1Tanhwhile/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh_1¦
while/bdgxhkvdqy/mul_5Mulwhile/bdgxhkvdqy/Sigmoid_2:y:0while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/bdgxhkvdqy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/bdgxhkvdqy/mul_5:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/bdgxhkvdqy/add_3:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_bdgxhkvdqy_biasadd_readvariableop_resource2while_bdgxhkvdqy_biasadd_readvariableop_resource_0"h
1while_bdgxhkvdqy_matmul_1_readvariableop_resource3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"d
/while_bdgxhkvdqy_matmul_readvariableop_resource1while_bdgxhkvdqy_matmul_readvariableop_resource_0"Z
*while_bdgxhkvdqy_readvariableop_1_resource,while_bdgxhkvdqy_readvariableop_1_resource_0"Z
*while_bdgxhkvdqy_readvariableop_2_resource,while_bdgxhkvdqy_readvariableop_2_resource_0"V
(while_bdgxhkvdqy_readvariableop_resource*while_bdgxhkvdqy_readvariableop_resource_0")
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
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp'while/bdgxhkvdqy/BiasAdd/ReadVariableOp2P
&while/bdgxhkvdqy/MatMul/ReadVariableOp&while/bdgxhkvdqy/MatMul/ReadVariableOp2T
(while/bdgxhkvdqy/MatMul_1/ReadVariableOp(while/bdgxhkvdqy/MatMul_1/ReadVariableOp2B
while/bdgxhkvdqy/ReadVariableOpwhile/bdgxhkvdqy/ReadVariableOp2F
!while/bdgxhkvdqy/ReadVariableOp_1!while/bdgxhkvdqy/ReadVariableOp_12F
!while/bdgxhkvdqy/ReadVariableOp_2!while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
¸'
´
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_366612

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
while_body_362800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kfdgklwsil_matmul_readvariableop_resource_0:	F
3while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 A
2while_kfdgklwsil_biasadd_readvariableop_resource_0:	8
*while_kfdgklwsil_readvariableop_resource_0: :
,while_kfdgklwsil_readvariableop_1_resource_0: :
,while_kfdgklwsil_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kfdgklwsil_matmul_readvariableop_resource:	D
1while_kfdgklwsil_matmul_1_readvariableop_resource:	 ?
0while_kfdgklwsil_biasadd_readvariableop_resource:	6
(while_kfdgklwsil_readvariableop_resource: 8
*while_kfdgklwsil_readvariableop_1_resource: 8
*while_kfdgklwsil_readvariableop_2_resource: ¢'while/kfdgklwsil/BiasAdd/ReadVariableOp¢&while/kfdgklwsil/MatMul/ReadVariableOp¢(while/kfdgklwsil/MatMul_1/ReadVariableOp¢while/kfdgklwsil/ReadVariableOp¢!while/kfdgklwsil/ReadVariableOp_1¢!while/kfdgklwsil/ReadVariableOp_2Ã
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
&while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp1while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/kfdgklwsil/MatMul/ReadVariableOpÑ
while/kfdgklwsil/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMulÉ
(while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp3while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kfdgklwsil/MatMul_1/ReadVariableOpº
while/kfdgklwsil/MatMul_1MatMulwhile_placeholder_20while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMul_1°
while/kfdgklwsil/addAddV2!while/kfdgklwsil/MatMul:product:0#while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/addÂ
'while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp2while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kfdgklwsil/BiasAdd/ReadVariableOp½
while/kfdgklwsil/BiasAddBiasAddwhile/kfdgklwsil/add:z:0/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/BiasAdd
 while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kfdgklwsil/split/split_dim
while/kfdgklwsil/splitSplit)while/kfdgklwsil/split/split_dim:output:0!while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kfdgklwsil/split©
while/kfdgklwsil/ReadVariableOpReadVariableOp*while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kfdgklwsil/ReadVariableOp£
while/kfdgklwsil/mulMul'while/kfdgklwsil/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul¦
while/kfdgklwsil/add_1AddV2while/kfdgklwsil/split:output:0while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_1
while/kfdgklwsil/SigmoidSigmoidwhile/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid¯
!while/kfdgklwsil/ReadVariableOp_1ReadVariableOp,while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_1©
while/kfdgklwsil/mul_1Mul)while/kfdgklwsil/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_1¨
while/kfdgklwsil/add_2AddV2while/kfdgklwsil/split:output:1while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_2
while/kfdgklwsil/Sigmoid_1Sigmoidwhile/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_1
while/kfdgklwsil/mul_2Mulwhile/kfdgklwsil/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_2
while/kfdgklwsil/TanhTanhwhile/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh¢
while/kfdgklwsil/mul_3Mulwhile/kfdgklwsil/Sigmoid:y:0while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_3£
while/kfdgklwsil/add_3AddV2while/kfdgklwsil/mul_2:z:0while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_3¯
!while/kfdgklwsil/ReadVariableOp_2ReadVariableOp,while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_2°
while/kfdgklwsil/mul_4Mul)while/kfdgklwsil/ReadVariableOp_2:value:0while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_4¨
while/kfdgklwsil/add_4AddV2while/kfdgklwsil/split:output:3while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_4
while/kfdgklwsil/Sigmoid_2Sigmoidwhile/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_2
while/kfdgklwsil/Tanh_1Tanhwhile/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh_1¦
while/kfdgklwsil/mul_5Mulwhile/kfdgklwsil/Sigmoid_2:y:0while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kfdgklwsil/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kfdgklwsil/mul_5:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kfdgklwsil/add_3:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
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
0while_kfdgklwsil_biasadd_readvariableop_resource2while_kfdgklwsil_biasadd_readvariableop_resource_0"h
1while_kfdgklwsil_matmul_1_readvariableop_resource3while_kfdgklwsil_matmul_1_readvariableop_resource_0"d
/while_kfdgklwsil_matmul_readvariableop_resource1while_kfdgklwsil_matmul_readvariableop_resource_0"Z
*while_kfdgklwsil_readvariableop_1_resource,while_kfdgklwsil_readvariableop_1_resource_0"Z
*while_kfdgklwsil_readvariableop_2_resource,while_kfdgklwsil_readvariableop_2_resource_0"V
(while_kfdgklwsil_readvariableop_resource*while_kfdgklwsil_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kfdgklwsil/BiasAdd/ReadVariableOp'while/kfdgklwsil/BiasAdd/ReadVariableOp2P
&while/kfdgklwsil/MatMul/ReadVariableOp&while/kfdgklwsil/MatMul/ReadVariableOp2T
(while/kfdgklwsil/MatMul_1/ReadVariableOp(while/kfdgklwsil/MatMul_1/ReadVariableOp2B
while/kfdgklwsil/ReadVariableOpwhile/kfdgklwsil/ReadVariableOp2F
!while/kfdgklwsil/ReadVariableOp_1!while/kfdgklwsil/ReadVariableOp_12F
!while/kfdgklwsil/ReadVariableOp_2!while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_361415

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
ßY

while_body_365278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kfdgklwsil_matmul_readvariableop_resource_0:	F
3while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 A
2while_kfdgklwsil_biasadd_readvariableop_resource_0:	8
*while_kfdgklwsil_readvariableop_resource_0: :
,while_kfdgklwsil_readvariableop_1_resource_0: :
,while_kfdgklwsil_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kfdgklwsil_matmul_readvariableop_resource:	D
1while_kfdgklwsil_matmul_1_readvariableop_resource:	 ?
0while_kfdgklwsil_biasadd_readvariableop_resource:	6
(while_kfdgklwsil_readvariableop_resource: 8
*while_kfdgklwsil_readvariableop_1_resource: 8
*while_kfdgklwsil_readvariableop_2_resource: ¢'while/kfdgklwsil/BiasAdd/ReadVariableOp¢&while/kfdgklwsil/MatMul/ReadVariableOp¢(while/kfdgklwsil/MatMul_1/ReadVariableOp¢while/kfdgklwsil/ReadVariableOp¢!while/kfdgklwsil/ReadVariableOp_1¢!while/kfdgklwsil/ReadVariableOp_2Ã
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
&while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp1while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/kfdgklwsil/MatMul/ReadVariableOpÑ
while/kfdgklwsil/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMulÉ
(while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp3while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kfdgklwsil/MatMul_1/ReadVariableOpº
while/kfdgklwsil/MatMul_1MatMulwhile_placeholder_20while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMul_1°
while/kfdgklwsil/addAddV2!while/kfdgklwsil/MatMul:product:0#while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/addÂ
'while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp2while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kfdgklwsil/BiasAdd/ReadVariableOp½
while/kfdgklwsil/BiasAddBiasAddwhile/kfdgklwsil/add:z:0/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/BiasAdd
 while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kfdgklwsil/split/split_dim
while/kfdgklwsil/splitSplit)while/kfdgklwsil/split/split_dim:output:0!while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kfdgklwsil/split©
while/kfdgklwsil/ReadVariableOpReadVariableOp*while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kfdgklwsil/ReadVariableOp£
while/kfdgklwsil/mulMul'while/kfdgklwsil/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul¦
while/kfdgklwsil/add_1AddV2while/kfdgklwsil/split:output:0while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_1
while/kfdgklwsil/SigmoidSigmoidwhile/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid¯
!while/kfdgklwsil/ReadVariableOp_1ReadVariableOp,while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_1©
while/kfdgklwsil/mul_1Mul)while/kfdgklwsil/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_1¨
while/kfdgklwsil/add_2AddV2while/kfdgklwsil/split:output:1while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_2
while/kfdgklwsil/Sigmoid_1Sigmoidwhile/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_1
while/kfdgklwsil/mul_2Mulwhile/kfdgklwsil/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_2
while/kfdgklwsil/TanhTanhwhile/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh¢
while/kfdgklwsil/mul_3Mulwhile/kfdgklwsil/Sigmoid:y:0while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_3£
while/kfdgklwsil/add_3AddV2while/kfdgklwsil/mul_2:z:0while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_3¯
!while/kfdgklwsil/ReadVariableOp_2ReadVariableOp,while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_2°
while/kfdgklwsil/mul_4Mul)while/kfdgklwsil/ReadVariableOp_2:value:0while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_4¨
while/kfdgklwsil/add_4AddV2while/kfdgklwsil/split:output:3while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_4
while/kfdgklwsil/Sigmoid_2Sigmoidwhile/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_2
while/kfdgklwsil/Tanh_1Tanhwhile/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh_1¦
while/kfdgklwsil/mul_5Mulwhile/kfdgklwsil/Sigmoid_2:y:0while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kfdgklwsil/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kfdgklwsil/mul_5:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kfdgklwsil/add_3:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
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
0while_kfdgklwsil_biasadd_readvariableop_resource2while_kfdgklwsil_biasadd_readvariableop_resource_0"h
1while_kfdgklwsil_matmul_1_readvariableop_resource3while_kfdgklwsil_matmul_1_readvariableop_resource_0"d
/while_kfdgklwsil_matmul_readvariableop_resource1while_kfdgklwsil_matmul_readvariableop_resource_0"Z
*while_kfdgklwsil_readvariableop_1_resource,while_kfdgklwsil_readvariableop_1_resource_0"Z
*while_kfdgklwsil_readvariableop_2_resource,while_kfdgklwsil_readvariableop_2_resource_0"V
(while_kfdgklwsil_readvariableop_resource*while_kfdgklwsil_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kfdgklwsil/BiasAdd/ReadVariableOp'while/kfdgklwsil/BiasAdd/ReadVariableOp2P
&while/kfdgklwsil/MatMul/ReadVariableOp&while/kfdgklwsil/MatMul/ReadVariableOp2T
(while/kfdgklwsil/MatMul_1/ReadVariableOp(while/kfdgklwsil/MatMul_1/ReadVariableOp2B
while/kfdgklwsil/ReadVariableOpwhile/kfdgklwsil/ReadVariableOp2F
!while/kfdgklwsil/ReadVariableOp_1!while/kfdgklwsil/ReadVariableOp_12F
!while/kfdgklwsil/ReadVariableOp_2!while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Û

'sequential_ipeigywbwc_while_body_361034H
Dsequential_ipeigywbwc_while_sequential_ipeigywbwc_while_loop_counterN
Jsequential_ipeigywbwc_while_sequential_ipeigywbwc_while_maximum_iterations+
'sequential_ipeigywbwc_while_placeholder-
)sequential_ipeigywbwc_while_placeholder_1-
)sequential_ipeigywbwc_while_placeholder_2-
)sequential_ipeigywbwc_while_placeholder_3G
Csequential_ipeigywbwc_while_sequential_ipeigywbwc_strided_slice_1_0
sequential_ipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_sequential_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 \
Isequential_ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 W
Hsequential_ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	N
@sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0: P
Bsequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0: P
Bsequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0: (
$sequential_ipeigywbwc_while_identity*
&sequential_ipeigywbwc_while_identity_1*
&sequential_ipeigywbwc_while_identity_2*
&sequential_ipeigywbwc_while_identity_3*
&sequential_ipeigywbwc_while_identity_4*
&sequential_ipeigywbwc_while_identity_5E
Asequential_ipeigywbwc_while_sequential_ipeigywbwc_strided_slice_1
}sequential_ipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_sequential_ipeigywbwc_tensorarrayunstack_tensorlistfromtensorX
Esequential_ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource:	 Z
Gsequential_ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 U
Fsequential_ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource:	L
>sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource: N
@sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource: N
@sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource: ¢=sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢<sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp¢>sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢5sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp¢7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1¢7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2ï
Msequential/ipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/ipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/ipeigywbwc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_ipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_sequential_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0'sequential_ipeigywbwc_while_placeholderVsequential/ipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/ipeigywbwc/while/TensorArrayV2Read/TensorListGetItem
<sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOpGsequential_ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp©
-sequential/ipeigywbwc/while/bdgxhkvdqy/MatMulMatMulFsequential/ipeigywbwc/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul
>sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOpIsequential_ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp
/sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1MatMul)sequential_ipeigywbwc_while_placeholder_2Fsequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1
*sequential/ipeigywbwc/while/bdgxhkvdqy/addAddV27sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul:product:09sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/ipeigywbwc/while/bdgxhkvdqy/add
=sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOpHsequential_ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp
.sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAddBiasAdd.sequential/ipeigywbwc/while/bdgxhkvdqy/add:z:0Esequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd²
6sequential/ipeigywbwc/while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/ipeigywbwc/while/bdgxhkvdqy/split/split_dimÛ
,sequential/ipeigywbwc/while/bdgxhkvdqy/splitSplit?sequential/ipeigywbwc/while/bdgxhkvdqy/split/split_dim:output:07sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/splitë
5sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOpReadVariableOp@sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOpû
*sequential/ipeigywbwc/while/bdgxhkvdqy/mulMul=sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp:value:0)sequential_ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/ipeigywbwc/while/bdgxhkvdqy/mulþ
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_1AddV25sequential/ipeigywbwc/while/bdgxhkvdqy/split:output:0.sequential/ipeigywbwc/while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_1Ï
.sequential/ipeigywbwc/while/bdgxhkvdqy/SigmoidSigmoid0sequential/ipeigywbwc/while/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoidñ
7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOpBsequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_1Mul?sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1:value:0)sequential_ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_1
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_2AddV25sequential/ipeigywbwc/while/bdgxhkvdqy/split:output:10sequential/ipeigywbwc/while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_2Ó
0sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1Sigmoid0sequential/ipeigywbwc/while/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1ö
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_2Mul4sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1:y:0)sequential_ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_2Ë
+sequential/ipeigywbwc/while/bdgxhkvdqy/TanhTanh5sequential/ipeigywbwc/while/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/ipeigywbwc/while/bdgxhkvdqy/Tanhú
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_3Mul2sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoid:y:0/sequential/ipeigywbwc/while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_3û
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_3AddV20sequential/ipeigywbwc/while/bdgxhkvdqy/mul_2:z:00sequential/ipeigywbwc/while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_3ñ
7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOpBsequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_4Mul?sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2:value:00sequential/ipeigywbwc/while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_4
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_4AddV25sequential/ipeigywbwc/while/bdgxhkvdqy/split:output:30sequential/ipeigywbwc/while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/add_4Ó
0sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2Sigmoid0sequential/ipeigywbwc/while/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2Ê
-sequential/ipeigywbwc/while/bdgxhkvdqy/Tanh_1Tanh0sequential/ipeigywbwc/while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/ipeigywbwc/while/bdgxhkvdqy/Tanh_1þ
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_5Mul4sequential/ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2:y:01sequential/ipeigywbwc/while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/ipeigywbwc/while/bdgxhkvdqy/mul_5Ì
@sequential/ipeigywbwc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_ipeigywbwc_while_placeholder_1'sequential_ipeigywbwc_while_placeholder0sequential/ipeigywbwc/while/bdgxhkvdqy/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/ipeigywbwc/while/TensorArrayV2Write/TensorListSetItem
!sequential/ipeigywbwc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/ipeigywbwc/while/add/yÁ
sequential/ipeigywbwc/while/addAddV2'sequential_ipeigywbwc_while_placeholder*sequential/ipeigywbwc/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/ipeigywbwc/while/add
#sequential/ipeigywbwc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/ipeigywbwc/while/add_1/yä
!sequential/ipeigywbwc/while/add_1AddV2Dsequential_ipeigywbwc_while_sequential_ipeigywbwc_while_loop_counter,sequential/ipeigywbwc/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/ipeigywbwc/while/add_1
$sequential/ipeigywbwc/while/IdentityIdentity%sequential/ipeigywbwc/while/add_1:z:0>^sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp=^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp?^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp6^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp8^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_18^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/ipeigywbwc/while/Identityµ
&sequential/ipeigywbwc/while/Identity_1IdentityJsequential_ipeigywbwc_while_sequential_ipeigywbwc_while_maximum_iterations>^sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp=^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp?^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp6^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp8^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_18^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/ipeigywbwc/while/Identity_1
&sequential/ipeigywbwc/while/Identity_2Identity#sequential/ipeigywbwc/while/add:z:0>^sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp=^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp?^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp6^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp8^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_18^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/ipeigywbwc/while/Identity_2»
&sequential/ipeigywbwc/while/Identity_3IdentityPsequential/ipeigywbwc/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp=^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp?^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp6^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp8^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_18^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/ipeigywbwc/while/Identity_3¬
&sequential/ipeigywbwc/while/Identity_4Identity0sequential/ipeigywbwc/while/bdgxhkvdqy/mul_5:z:0>^sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp=^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp?^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp6^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp8^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_18^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/while/Identity_4¬
&sequential/ipeigywbwc/while/Identity_5Identity0sequential/ipeigywbwc/while/bdgxhkvdqy/add_3:z:0>^sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp=^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp?^sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp6^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp8^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_18^sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/ipeigywbwc/while/Identity_5"
Fsequential_ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resourceHsequential_ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0"
Gsequential_ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resourceIsequential_ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"
Esequential_ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resourceGsequential_ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0"
@sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resourceBsequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0"
@sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resourceBsequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0"
>sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource@sequential_ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0"U
$sequential_ipeigywbwc_while_identity-sequential/ipeigywbwc/while/Identity:output:0"Y
&sequential_ipeigywbwc_while_identity_1/sequential/ipeigywbwc/while/Identity_1:output:0"Y
&sequential_ipeigywbwc_while_identity_2/sequential/ipeigywbwc/while/Identity_2:output:0"Y
&sequential_ipeigywbwc_while_identity_3/sequential/ipeigywbwc/while/Identity_3:output:0"Y
&sequential_ipeigywbwc_while_identity_4/sequential/ipeigywbwc/while/Identity_4:output:0"Y
&sequential_ipeigywbwc_while_identity_5/sequential/ipeigywbwc/while/Identity_5:output:0"
Asequential_ipeigywbwc_while_sequential_ipeigywbwc_strided_slice_1Csequential_ipeigywbwc_while_sequential_ipeigywbwc_strided_slice_1_0"
}sequential_ipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_sequential_ipeigywbwc_tensorarrayunstack_tensorlistfromtensorsequential_ipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_sequential_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp=sequential/ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2|
<sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp<sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp2
>sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp>sequential/ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp2n
5sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp5sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp2r
7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_17sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_12r
7sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_27sequential/ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_sequential_layer_call_and_return_conditional_losses_363807

kjggqknufb'
ekzorghjta_363769:
ekzorghjta_363771:$
zeuewnmlut_363775:	$
zeuewnmlut_363777:	  
zeuewnmlut_363779:	
zeuewnmlut_363781: 
zeuewnmlut_363783: 
zeuewnmlut_363785: $
ipeigywbwc_363788:	 $
ipeigywbwc_363790:	  
ipeigywbwc_363792:	
ipeigywbwc_363794: 
ipeigywbwc_363796: 
ipeigywbwc_363798: #
pemnqztknd_363801: 
pemnqztknd_363803:
identity¢"ekzorghjta/StatefulPartitionedCall¢"ipeigywbwc/StatefulPartitionedCall¢"pemnqztknd/StatefulPartitionedCall¢"zeuewnmlut/StatefulPartitionedCall­
"ekzorghjta/StatefulPartitionedCallStatefulPartitionedCall
kjggqknufbekzorghjta_363769ekzorghjta_363771*
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
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_3627012$
"ekzorghjta/StatefulPartitionedCall
xsvuntduhq/PartitionedCallPartitionedCall+ekzorghjta/StatefulPartitionedCall:output:0*
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
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_3627202
xsvuntduhq/PartitionedCall
"zeuewnmlut/StatefulPartitionedCallStatefulPartitionedCall#xsvuntduhq/PartitionedCall:output:0zeuewnmlut_363775zeuewnmlut_363777zeuewnmlut_363779zeuewnmlut_363781zeuewnmlut_363783zeuewnmlut_363785*
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3629012$
"zeuewnmlut/StatefulPartitionedCall
"ipeigywbwc/StatefulPartitionedCallStatefulPartitionedCall+zeuewnmlut/StatefulPartitionedCall:output:0ipeigywbwc_363788ipeigywbwc_363790ipeigywbwc_363792ipeigywbwc_363794ipeigywbwc_363796ipeigywbwc_363798*
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3630942$
"ipeigywbwc/StatefulPartitionedCallÆ
"pemnqztknd/StatefulPartitionedCallStatefulPartitionedCall+ipeigywbwc/StatefulPartitionedCall:output:0pemnqztknd_363801pemnqztknd_363803*
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
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_3631182$
"pemnqztknd/StatefulPartitionedCall
IdentityIdentity+pemnqztknd/StatefulPartitionedCall:output:0#^ekzorghjta/StatefulPartitionedCall#^ipeigywbwc/StatefulPartitionedCall#^pemnqztknd/StatefulPartitionedCall#^zeuewnmlut/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ekzorghjta/StatefulPartitionedCall"ekzorghjta/StatefulPartitionedCall2H
"ipeigywbwc/StatefulPartitionedCall"ipeigywbwc/StatefulPartitionedCall2H
"pemnqztknd/StatefulPartitionedCall"pemnqztknd/StatefulPartitionedCall2H
"zeuewnmlut/StatefulPartitionedCall"zeuewnmlut/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
kjggqknufb
Ò	
÷
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_363118

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
°'
²
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_361228

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
ßY

while_body_365098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kfdgklwsil_matmul_readvariableop_resource_0:	F
3while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 A
2while_kfdgklwsil_biasadd_readvariableop_resource_0:	8
*while_kfdgklwsil_readvariableop_resource_0: :
,while_kfdgklwsil_readvariableop_1_resource_0: :
,while_kfdgklwsil_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kfdgklwsil_matmul_readvariableop_resource:	D
1while_kfdgklwsil_matmul_1_readvariableop_resource:	 ?
0while_kfdgklwsil_biasadd_readvariableop_resource:	6
(while_kfdgklwsil_readvariableop_resource: 8
*while_kfdgklwsil_readvariableop_1_resource: 8
*while_kfdgklwsil_readvariableop_2_resource: ¢'while/kfdgklwsil/BiasAdd/ReadVariableOp¢&while/kfdgklwsil/MatMul/ReadVariableOp¢(while/kfdgklwsil/MatMul_1/ReadVariableOp¢while/kfdgklwsil/ReadVariableOp¢!while/kfdgklwsil/ReadVariableOp_1¢!while/kfdgklwsil/ReadVariableOp_2Ã
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
&while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp1while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/kfdgklwsil/MatMul/ReadVariableOpÑ
while/kfdgklwsil/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMulÉ
(while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp3while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kfdgklwsil/MatMul_1/ReadVariableOpº
while/kfdgklwsil/MatMul_1MatMulwhile_placeholder_20while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMul_1°
while/kfdgklwsil/addAddV2!while/kfdgklwsil/MatMul:product:0#while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/addÂ
'while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp2while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kfdgklwsil/BiasAdd/ReadVariableOp½
while/kfdgklwsil/BiasAddBiasAddwhile/kfdgklwsil/add:z:0/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/BiasAdd
 while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kfdgklwsil/split/split_dim
while/kfdgklwsil/splitSplit)while/kfdgklwsil/split/split_dim:output:0!while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kfdgklwsil/split©
while/kfdgklwsil/ReadVariableOpReadVariableOp*while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kfdgklwsil/ReadVariableOp£
while/kfdgklwsil/mulMul'while/kfdgklwsil/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul¦
while/kfdgklwsil/add_1AddV2while/kfdgklwsil/split:output:0while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_1
while/kfdgklwsil/SigmoidSigmoidwhile/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid¯
!while/kfdgklwsil/ReadVariableOp_1ReadVariableOp,while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_1©
while/kfdgklwsil/mul_1Mul)while/kfdgklwsil/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_1¨
while/kfdgklwsil/add_2AddV2while/kfdgklwsil/split:output:1while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_2
while/kfdgklwsil/Sigmoid_1Sigmoidwhile/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_1
while/kfdgklwsil/mul_2Mulwhile/kfdgklwsil/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_2
while/kfdgklwsil/TanhTanhwhile/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh¢
while/kfdgklwsil/mul_3Mulwhile/kfdgklwsil/Sigmoid:y:0while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_3£
while/kfdgklwsil/add_3AddV2while/kfdgklwsil/mul_2:z:0while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_3¯
!while/kfdgklwsil/ReadVariableOp_2ReadVariableOp,while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_2°
while/kfdgklwsil/mul_4Mul)while/kfdgklwsil/ReadVariableOp_2:value:0while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_4¨
while/kfdgklwsil/add_4AddV2while/kfdgklwsil/split:output:3while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_4
while/kfdgklwsil/Sigmoid_2Sigmoidwhile/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_2
while/kfdgklwsil/Tanh_1Tanhwhile/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh_1¦
while/kfdgklwsil/mul_5Mulwhile/kfdgklwsil/Sigmoid_2:y:0while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kfdgklwsil/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kfdgklwsil/mul_5:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kfdgklwsil/add_3:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
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
0while_kfdgklwsil_biasadd_readvariableop_resource2while_kfdgklwsil_biasadd_readvariableop_resource_0"h
1while_kfdgklwsil_matmul_1_readvariableop_resource3while_kfdgklwsil_matmul_1_readvariableop_resource_0"d
/while_kfdgklwsil_matmul_readvariableop_resource1while_kfdgklwsil_matmul_readvariableop_resource_0"Z
*while_kfdgklwsil_readvariableop_1_resource,while_kfdgklwsil_readvariableop_1_resource_0"Z
*while_kfdgklwsil_readvariableop_2_resource,while_kfdgklwsil_readvariableop_2_resource_0"V
(while_kfdgklwsil_readvariableop_resource*while_kfdgklwsil_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kfdgklwsil/BiasAdd/ReadVariableOp'while/kfdgklwsil/BiasAdd/ReadVariableOp2P
&while/kfdgklwsil/MatMul/ReadVariableOp&while/kfdgklwsil/MatMul/ReadVariableOp2T
(while/kfdgklwsil/MatMul_1/ReadVariableOp(while/kfdgklwsil/MatMul_1/ReadVariableOp2B
while/kfdgklwsil/ReadVariableOpwhile/kfdgklwsil/ReadVariableOp2F
!while/kfdgklwsil/ReadVariableOp_1!while/kfdgklwsil/ReadVariableOp_12F
!while/kfdgklwsil/ReadVariableOp_2!while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
p
É
ipeigywbwc_while_body_3645942
.ipeigywbwc_while_ipeigywbwc_while_loop_counter8
4ipeigywbwc_while_ipeigywbwc_while_maximum_iterations 
ipeigywbwc_while_placeholder"
ipeigywbwc_while_placeholder_1"
ipeigywbwc_while_placeholder_2"
ipeigywbwc_while_placeholder_31
-ipeigywbwc_while_ipeigywbwc_strided_slice_1_0m
iipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0O
<ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 Q
>ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 L
=ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	C
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0: E
7ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0: E
7ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0: 
ipeigywbwc_while_identity
ipeigywbwc_while_identity_1
ipeigywbwc_while_identity_2
ipeigywbwc_while_identity_3
ipeigywbwc_while_identity_4
ipeigywbwc_while_identity_5/
+ipeigywbwc_while_ipeigywbwc_strided_slice_1k
gipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensorM
:ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource:	 O
<ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 J
;ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource:	A
3ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource: C
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource: C
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource: ¢2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp¢3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp¢,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1¢,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2Ù
Bipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4ipeigywbwc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0ipeigywbwc_while_placeholderKipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4ipeigywbwc/while/TensorArrayV2Read/TensorListGetItemä
1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp<ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOpý
"ipeigywbwc/while/bdgxhkvdqy/MatMulMatMul;ipeigywbwc/while/TensorArrayV2Read/TensorListGetItem:item:09ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"ipeigywbwc/while/bdgxhkvdqy/MatMulê
3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp>ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOpæ
$ipeigywbwc/while/bdgxhkvdqy/MatMul_1MatMulipeigywbwc_while_placeholder_2;ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$ipeigywbwc/while/bdgxhkvdqy/MatMul_1Ü
ipeigywbwc/while/bdgxhkvdqy/addAddV2,ipeigywbwc/while/bdgxhkvdqy/MatMul:product:0.ipeigywbwc/while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
ipeigywbwc/while/bdgxhkvdqy/addã
2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp=ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOpé
#ipeigywbwc/while/bdgxhkvdqy/BiasAddBiasAdd#ipeigywbwc/while/bdgxhkvdqy/add:z:0:ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ipeigywbwc/while/bdgxhkvdqy/BiasAdd
+ipeigywbwc/while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+ipeigywbwc/while/bdgxhkvdqy/split/split_dim¯
!ipeigywbwc/while/bdgxhkvdqy/splitSplit4ipeigywbwc/while/bdgxhkvdqy/split/split_dim:output:0,ipeigywbwc/while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!ipeigywbwc/while/bdgxhkvdqy/splitÊ
*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOpReadVariableOp5ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOpÏ
ipeigywbwc/while/bdgxhkvdqy/mulMul2ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp:value:0ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ipeigywbwc/while/bdgxhkvdqy/mulÒ
!ipeigywbwc/while/bdgxhkvdqy/add_1AddV2*ipeigywbwc/while/bdgxhkvdqy/split:output:0#ipeigywbwc/while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_1®
#ipeigywbwc/while/bdgxhkvdqy/SigmoidSigmoid%ipeigywbwc/while/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#ipeigywbwc/while/bdgxhkvdqy/SigmoidÐ
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp7ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1Õ
!ipeigywbwc/while/bdgxhkvdqy/mul_1Mul4ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1:value:0ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_1Ô
!ipeigywbwc/while/bdgxhkvdqy/add_2AddV2*ipeigywbwc/while/bdgxhkvdqy/split:output:1%ipeigywbwc/while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_2²
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1Sigmoid%ipeigywbwc/while/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1Ê
!ipeigywbwc/while/bdgxhkvdqy/mul_2Mul)ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1:y:0ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_2ª
 ipeigywbwc/while/bdgxhkvdqy/TanhTanh*ipeigywbwc/while/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 ipeigywbwc/while/bdgxhkvdqy/TanhÎ
!ipeigywbwc/while/bdgxhkvdqy/mul_3Mul'ipeigywbwc/while/bdgxhkvdqy/Sigmoid:y:0$ipeigywbwc/while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_3Ï
!ipeigywbwc/while/bdgxhkvdqy/add_3AddV2%ipeigywbwc/while/bdgxhkvdqy/mul_2:z:0%ipeigywbwc/while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_3Ð
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp7ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2Ü
!ipeigywbwc/while/bdgxhkvdqy/mul_4Mul4ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2:value:0%ipeigywbwc/while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_4Ô
!ipeigywbwc/while/bdgxhkvdqy/add_4AddV2*ipeigywbwc/while/bdgxhkvdqy/split:output:3%ipeigywbwc/while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_4²
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2Sigmoid%ipeigywbwc/while/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2©
"ipeigywbwc/while/bdgxhkvdqy/Tanh_1Tanh%ipeigywbwc/while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"ipeigywbwc/while/bdgxhkvdqy/Tanh_1Ò
!ipeigywbwc/while/bdgxhkvdqy/mul_5Mul)ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2:y:0&ipeigywbwc/while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_5
5ipeigywbwc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemipeigywbwc_while_placeholder_1ipeigywbwc_while_placeholder%ipeigywbwc/while/bdgxhkvdqy/mul_5:z:0*
_output_shapes
: *
element_dtype027
5ipeigywbwc/while/TensorArrayV2Write/TensorListSetItemr
ipeigywbwc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
ipeigywbwc/while/add/y
ipeigywbwc/while/addAddV2ipeigywbwc_while_placeholderipeigywbwc/while/add/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/while/addv
ipeigywbwc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
ipeigywbwc/while/add_1/y­
ipeigywbwc/while/add_1AddV2.ipeigywbwc_while_ipeigywbwc_while_loop_counter!ipeigywbwc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/while/add_1©
ipeigywbwc/while/IdentityIdentityipeigywbwc/while/add_1:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/IdentityÇ
ipeigywbwc/while/Identity_1Identity4ipeigywbwc_while_ipeigywbwc_while_maximum_iterations3^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/Identity_1«
ipeigywbwc/while/Identity_2Identityipeigywbwc/while/add:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/Identity_2Ø
ipeigywbwc/while/Identity_3IdentityEipeigywbwc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/Identity_3É
ipeigywbwc/while/Identity_4Identity%ipeigywbwc/while/bdgxhkvdqy/mul_5:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/while/Identity_4É
ipeigywbwc/while/Identity_5Identity%ipeigywbwc/while/bdgxhkvdqy/add_3:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/while/Identity_5"|
;ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource=ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0"~
<ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource>ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"z
:ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource<ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0"p
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource7ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0"p
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource7ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0"l
3ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource5ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0"?
ipeigywbwc_while_identity"ipeigywbwc/while/Identity:output:0"C
ipeigywbwc_while_identity_1$ipeigywbwc/while/Identity_1:output:0"C
ipeigywbwc_while_identity_2$ipeigywbwc/while/Identity_2:output:0"C
ipeigywbwc_while_identity_3$ipeigywbwc/while/Identity_3:output:0"C
ipeigywbwc_while_identity_4$ipeigywbwc/while/Identity_4:output:0"C
ipeigywbwc_while_identity_5$ipeigywbwc/while/Identity_5:output:0"\
+ipeigywbwc_while_ipeigywbwc_strided_slice_1-ipeigywbwc_while_ipeigywbwc_strided_slice_1_0"Ô
gipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensoriipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2f
1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp2j
3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp2X
*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp2\
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_12\
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_363268
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 F
3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 A
2while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	8
*while_bdgxhkvdqy_readvariableop_resource_0: :
,while_bdgxhkvdqy_readvariableop_1_resource_0: :
,while_bdgxhkvdqy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_bdgxhkvdqy_matmul_readvariableop_resource:	 D
1while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 ?
0while_bdgxhkvdqy_biasadd_readvariableop_resource:	6
(while_bdgxhkvdqy_readvariableop_resource: 8
*while_bdgxhkvdqy_readvariableop_1_resource: 8
*while_bdgxhkvdqy_readvariableop_2_resource: ¢'while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢&while/bdgxhkvdqy/MatMul/ReadVariableOp¢(while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢while/bdgxhkvdqy/ReadVariableOp¢!while/bdgxhkvdqy/ReadVariableOp_1¢!while/bdgxhkvdqy/ReadVariableOp_2Ã
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
&while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp1while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/bdgxhkvdqy/MatMul/ReadVariableOpÑ
while/bdgxhkvdqy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMulÉ
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpº
while/bdgxhkvdqy/MatMul_1MatMulwhile_placeholder_20while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMul_1°
while/bdgxhkvdqy/addAddV2!while/bdgxhkvdqy/MatMul:product:0#while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/addÂ
'while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp2while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp½
while/bdgxhkvdqy/BiasAddBiasAddwhile/bdgxhkvdqy/add:z:0/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/BiasAdd
 while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/bdgxhkvdqy/split/split_dim
while/bdgxhkvdqy/splitSplit)while/bdgxhkvdqy/split/split_dim:output:0!while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/bdgxhkvdqy/split©
while/bdgxhkvdqy/ReadVariableOpReadVariableOp*while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/bdgxhkvdqy/ReadVariableOp£
while/bdgxhkvdqy/mulMul'while/bdgxhkvdqy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul¦
while/bdgxhkvdqy/add_1AddV2while/bdgxhkvdqy/split:output:0while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_1
while/bdgxhkvdqy/SigmoidSigmoidwhile/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid¯
!while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp,while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_1©
while/bdgxhkvdqy/mul_1Mul)while/bdgxhkvdqy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_1¨
while/bdgxhkvdqy/add_2AddV2while/bdgxhkvdqy/split:output:1while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_2
while/bdgxhkvdqy/Sigmoid_1Sigmoidwhile/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_1
while/bdgxhkvdqy/mul_2Mulwhile/bdgxhkvdqy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_2
while/bdgxhkvdqy/TanhTanhwhile/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh¢
while/bdgxhkvdqy/mul_3Mulwhile/bdgxhkvdqy/Sigmoid:y:0while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_3£
while/bdgxhkvdqy/add_3AddV2while/bdgxhkvdqy/mul_2:z:0while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_3¯
!while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp,while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_2°
while/bdgxhkvdqy/mul_4Mul)while/bdgxhkvdqy/ReadVariableOp_2:value:0while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_4¨
while/bdgxhkvdqy/add_4AddV2while/bdgxhkvdqy/split:output:3while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_4
while/bdgxhkvdqy/Sigmoid_2Sigmoidwhile/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_2
while/bdgxhkvdqy/Tanh_1Tanhwhile/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh_1¦
while/bdgxhkvdqy/mul_5Mulwhile/bdgxhkvdqy/Sigmoid_2:y:0while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/bdgxhkvdqy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/bdgxhkvdqy/mul_5:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/bdgxhkvdqy/add_3:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_bdgxhkvdqy_biasadd_readvariableop_resource2while_bdgxhkvdqy_biasadd_readvariableop_resource_0"h
1while_bdgxhkvdqy_matmul_1_readvariableop_resource3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"d
/while_bdgxhkvdqy_matmul_readvariableop_resource1while_bdgxhkvdqy_matmul_readvariableop_resource_0"Z
*while_bdgxhkvdqy_readvariableop_1_resource,while_bdgxhkvdqy_readvariableop_1_resource_0"Z
*while_bdgxhkvdqy_readvariableop_2_resource,while_bdgxhkvdqy_readvariableop_2_resource_0"V
(while_bdgxhkvdqy_readvariableop_resource*while_bdgxhkvdqy_readvariableop_resource_0")
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
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp'while/bdgxhkvdqy/BiasAdd/ReadVariableOp2P
&while/bdgxhkvdqy/MatMul/ReadVariableOp&while/bdgxhkvdqy/MatMul/ReadVariableOp2T
(while/bdgxhkvdqy/MatMul_1/ReadVariableOp(while/bdgxhkvdqy/MatMul_1/ReadVariableOp2B
while/bdgxhkvdqy/ReadVariableOpwhile/bdgxhkvdqy/ReadVariableOp2F
!while/bdgxhkvdqy/ReadVariableOp_1!while/bdgxhkvdqy/ReadVariableOp_12F
!while/bdgxhkvdqy/ReadVariableOp_2!while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_362269
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_bdgxhkvdqy_362293_0:	 ,
while_bdgxhkvdqy_362295_0:	 (
while_bdgxhkvdqy_362297_0:	'
while_bdgxhkvdqy_362299_0: '
while_bdgxhkvdqy_362301_0: '
while_bdgxhkvdqy_362303_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_bdgxhkvdqy_362293:	 *
while_bdgxhkvdqy_362295:	 &
while_bdgxhkvdqy_362297:	%
while_bdgxhkvdqy_362299: %
while_bdgxhkvdqy_362301: %
while_bdgxhkvdqy_362303: ¢(while/bdgxhkvdqy/StatefulPartitionedCallÃ
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
(while/bdgxhkvdqy/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_bdgxhkvdqy_362293_0while_bdgxhkvdqy_362295_0while_bdgxhkvdqy_362297_0while_bdgxhkvdqy_362299_0while_bdgxhkvdqy_362301_0while_bdgxhkvdqy_362303_0*
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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_3621732*
(while/bdgxhkvdqy/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/bdgxhkvdqy/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/bdgxhkvdqy/StatefulPartitionedCall:output:1)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/bdgxhkvdqy/StatefulPartitionedCall:output:2)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_bdgxhkvdqy_362293while_bdgxhkvdqy_362293_0"4
while_bdgxhkvdqy_362295while_bdgxhkvdqy_362295_0"4
while_bdgxhkvdqy_362297while_bdgxhkvdqy_362297_0"4
while_bdgxhkvdqy_362299while_bdgxhkvdqy_362299_0"4
while_bdgxhkvdqy_362301while_bdgxhkvdqy_362301_0"4
while_bdgxhkvdqy_362303while_bdgxhkvdqy_362303_0")
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
(while/bdgxhkvdqy/StatefulPartitionedCall(while/bdgxhkvdqy/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
¸'
´
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_366656

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
while_body_365458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kfdgklwsil_matmul_readvariableop_resource_0:	F
3while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 A
2while_kfdgklwsil_biasadd_readvariableop_resource_0:	8
*while_kfdgklwsil_readvariableop_resource_0: :
,while_kfdgklwsil_readvariableop_1_resource_0: :
,while_kfdgklwsil_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kfdgklwsil_matmul_readvariableop_resource:	D
1while_kfdgklwsil_matmul_1_readvariableop_resource:	 ?
0while_kfdgklwsil_biasadd_readvariableop_resource:	6
(while_kfdgklwsil_readvariableop_resource: 8
*while_kfdgklwsil_readvariableop_1_resource: 8
*while_kfdgklwsil_readvariableop_2_resource: ¢'while/kfdgklwsil/BiasAdd/ReadVariableOp¢&while/kfdgklwsil/MatMul/ReadVariableOp¢(while/kfdgklwsil/MatMul_1/ReadVariableOp¢while/kfdgklwsil/ReadVariableOp¢!while/kfdgklwsil/ReadVariableOp_1¢!while/kfdgklwsil/ReadVariableOp_2Ã
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
&while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp1while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/kfdgklwsil/MatMul/ReadVariableOpÑ
while/kfdgklwsil/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMulÉ
(while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp3while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kfdgklwsil/MatMul_1/ReadVariableOpº
while/kfdgklwsil/MatMul_1MatMulwhile_placeholder_20while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMul_1°
while/kfdgklwsil/addAddV2!while/kfdgklwsil/MatMul:product:0#while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/addÂ
'while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp2while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kfdgklwsil/BiasAdd/ReadVariableOp½
while/kfdgklwsil/BiasAddBiasAddwhile/kfdgklwsil/add:z:0/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/BiasAdd
 while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kfdgklwsil/split/split_dim
while/kfdgklwsil/splitSplit)while/kfdgklwsil/split/split_dim:output:0!while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kfdgklwsil/split©
while/kfdgklwsil/ReadVariableOpReadVariableOp*while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kfdgklwsil/ReadVariableOp£
while/kfdgklwsil/mulMul'while/kfdgklwsil/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul¦
while/kfdgklwsil/add_1AddV2while/kfdgklwsil/split:output:0while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_1
while/kfdgklwsil/SigmoidSigmoidwhile/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid¯
!while/kfdgklwsil/ReadVariableOp_1ReadVariableOp,while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_1©
while/kfdgklwsil/mul_1Mul)while/kfdgklwsil/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_1¨
while/kfdgklwsil/add_2AddV2while/kfdgklwsil/split:output:1while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_2
while/kfdgklwsil/Sigmoid_1Sigmoidwhile/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_1
while/kfdgklwsil/mul_2Mulwhile/kfdgklwsil/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_2
while/kfdgklwsil/TanhTanhwhile/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh¢
while/kfdgklwsil/mul_3Mulwhile/kfdgklwsil/Sigmoid:y:0while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_3£
while/kfdgklwsil/add_3AddV2while/kfdgklwsil/mul_2:z:0while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_3¯
!while/kfdgklwsil/ReadVariableOp_2ReadVariableOp,while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_2°
while/kfdgklwsil/mul_4Mul)while/kfdgklwsil/ReadVariableOp_2:value:0while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_4¨
while/kfdgklwsil/add_4AddV2while/kfdgklwsil/split:output:3while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_4
while/kfdgklwsil/Sigmoid_2Sigmoidwhile/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_2
while/kfdgklwsil/Tanh_1Tanhwhile/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh_1¦
while/kfdgklwsil/mul_5Mulwhile/kfdgklwsil/Sigmoid_2:y:0while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kfdgklwsil/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kfdgklwsil/mul_5:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kfdgklwsil/add_3:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
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
0while_kfdgklwsil_biasadd_readvariableop_resource2while_kfdgklwsil_biasadd_readvariableop_resource_0"h
1while_kfdgklwsil_matmul_1_readvariableop_resource3while_kfdgklwsil_matmul_1_readvariableop_resource_0"d
/while_kfdgklwsil_matmul_readvariableop_resource1while_kfdgklwsil_matmul_readvariableop_resource_0"Z
*while_kfdgklwsil_readvariableop_1_resource,while_kfdgklwsil_readvariableop_1_resource_0"Z
*while_kfdgklwsil_readvariableop_2_resource,while_kfdgklwsil_readvariableop_2_resource_0"V
(while_kfdgklwsil_readvariableop_resource*while_kfdgklwsil_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kfdgklwsil/BiasAdd/ReadVariableOp'while/kfdgklwsil/BiasAdd/ReadVariableOp2P
&while/kfdgklwsil/MatMul/ReadVariableOp&while/kfdgklwsil/MatMul/ReadVariableOp2T
(while/kfdgklwsil/MatMul_1/ReadVariableOp(while/kfdgklwsil/MatMul_1/ReadVariableOp2B
while/kfdgklwsil/ReadVariableOpwhile/kfdgklwsil/ReadVariableOp2F
!while/kfdgklwsil/ReadVariableOp_1!while/kfdgklwsil/ReadVariableOp_12F
!while/kfdgklwsil/ReadVariableOp_2!while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_sequential_layer_call_and_return_conditional_losses_364701

inputsL
6ekzorghjta_conv1d_expanddims_1_readvariableop_resource:K
=ekzorghjta_squeeze_batch_dims_biasadd_readvariableop_resource:G
4zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource:	I
6zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource:	 D
5zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource:	;
-zeuewnmlut_kfdgklwsil_readvariableop_resource: =
/zeuewnmlut_kfdgklwsil_readvariableop_1_resource: =
/zeuewnmlut_kfdgklwsil_readvariableop_2_resource: G
4ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource:	 I
6ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource:	 D
5ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource:	;
-ipeigywbwc_bdgxhkvdqy_readvariableop_resource: =
/ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource: =
/ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource: ;
)pemnqztknd_matmul_readvariableop_resource: 8
*pemnqztknd_biasadd_readvariableop_resource:
identity¢-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp¢4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp¢+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp¢-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp¢$ipeigywbwc/bdgxhkvdqy/ReadVariableOp¢&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1¢&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2¢ipeigywbwc/while¢!pemnqztknd/BiasAdd/ReadVariableOp¢ pemnqztknd/MatMul/ReadVariableOp¢,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp¢+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp¢-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp¢$zeuewnmlut/kfdgklwsil/ReadVariableOp¢&zeuewnmlut/kfdgklwsil/ReadVariableOp_1¢&zeuewnmlut/kfdgklwsil/ReadVariableOp_2¢zeuewnmlut/while
 ekzorghjta/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 ekzorghjta/conv1d/ExpandDims/dim»
ekzorghjta/conv1d/ExpandDims
ExpandDimsinputs)ekzorghjta/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/ExpandDimsÙ
-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6ekzorghjta_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp
"ekzorghjta/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"ekzorghjta/conv1d/ExpandDims_1/dimã
ekzorghjta/conv1d/ExpandDims_1
ExpandDims5ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp:value:0+ekzorghjta/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
ekzorghjta/conv1d/ExpandDims_1
ekzorghjta/conv1d/ShapeShape%ekzorghjta/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
ekzorghjta/conv1d/Shape
%ekzorghjta/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%ekzorghjta/conv1d/strided_slice/stack¥
'ekzorghjta/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'ekzorghjta/conv1d/strided_slice/stack_1
'ekzorghjta/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'ekzorghjta/conv1d/strided_slice/stack_2Ì
ekzorghjta/conv1d/strided_sliceStridedSlice ekzorghjta/conv1d/Shape:output:0.ekzorghjta/conv1d/strided_slice/stack:output:00ekzorghjta/conv1d/strided_slice/stack_1:output:00ekzorghjta/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
ekzorghjta/conv1d/strided_slice
ekzorghjta/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
ekzorghjta/conv1d/Reshape/shapeÌ
ekzorghjta/conv1d/ReshapeReshape%ekzorghjta/conv1d/ExpandDims:output:0(ekzorghjta/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/Reshapeî
ekzorghjta/conv1d/Conv2DConv2D"ekzorghjta/conv1d/Reshape:output:0'ekzorghjta/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ekzorghjta/conv1d/Conv2D
!ekzorghjta/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!ekzorghjta/conv1d/concat/values_1
ekzorghjta/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/concat/axisì
ekzorghjta/conv1d/concatConcatV2(ekzorghjta/conv1d/strided_slice:output:0*ekzorghjta/conv1d/concat/values_1:output:0&ekzorghjta/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
ekzorghjta/conv1d/concatÉ
ekzorghjta/conv1d/Reshape_1Reshape!ekzorghjta/conv1d/Conv2D:output:0!ekzorghjta/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/Reshape_1Á
ekzorghjta/conv1d/SqueezeSqueeze$ekzorghjta/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
ekzorghjta/conv1d/Squeeze
#ekzorghjta/squeeze_batch_dims/ShapeShape"ekzorghjta/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#ekzorghjta/squeeze_batch_dims/Shape°
1ekzorghjta/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1ekzorghjta/squeeze_batch_dims/strided_slice/stack½
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_1´
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3ekzorghjta/squeeze_batch_dims/strided_slice/stack_2
+ekzorghjta/squeeze_batch_dims/strided_sliceStridedSlice,ekzorghjta/squeeze_batch_dims/Shape:output:0:ekzorghjta/squeeze_batch_dims/strided_slice/stack:output:0<ekzorghjta/squeeze_batch_dims/strided_slice/stack_1:output:0<ekzorghjta/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+ekzorghjta/squeeze_batch_dims/strided_slice¯
+ekzorghjta/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+ekzorghjta/squeeze_batch_dims/Reshape/shapeé
%ekzorghjta/squeeze_batch_dims/ReshapeReshape"ekzorghjta/conv1d/Squeeze:output:04ekzorghjta/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ekzorghjta/squeeze_batch_dims/Reshapeæ
4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=ekzorghjta_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%ekzorghjta/squeeze_batch_dims/BiasAddBiasAdd.ekzorghjta/squeeze_batch_dims/Reshape:output:0<ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%ekzorghjta/squeeze_batch_dims/BiasAdd¯
-ekzorghjta/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-ekzorghjta/squeeze_batch_dims/concat/values_1¡
)ekzorghjta/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)ekzorghjta/squeeze_batch_dims/concat/axis¨
$ekzorghjta/squeeze_batch_dims/concatConcatV24ekzorghjta/squeeze_batch_dims/strided_slice:output:06ekzorghjta/squeeze_batch_dims/concat/values_1:output:02ekzorghjta/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$ekzorghjta/squeeze_batch_dims/concatö
'ekzorghjta/squeeze_batch_dims/Reshape_1Reshape.ekzorghjta/squeeze_batch_dims/BiasAdd:output:0-ekzorghjta/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'ekzorghjta/squeeze_batch_dims/Reshape_1
xsvuntduhq/ShapeShape0ekzorghjta/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
xsvuntduhq/Shape
xsvuntduhq/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
xsvuntduhq/strided_slice/stack
 xsvuntduhq/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 xsvuntduhq/strided_slice/stack_1
 xsvuntduhq/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 xsvuntduhq/strided_slice/stack_2¤
xsvuntduhq/strided_sliceStridedSlicexsvuntduhq/Shape:output:0'xsvuntduhq/strided_slice/stack:output:0)xsvuntduhq/strided_slice/stack_1:output:0)xsvuntduhq/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
xsvuntduhq/strided_slicez
xsvuntduhq/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
xsvuntduhq/Reshape/shape/1z
xsvuntduhq/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
xsvuntduhq/Reshape/shape/2×
xsvuntduhq/Reshape/shapePack!xsvuntduhq/strided_slice:output:0#xsvuntduhq/Reshape/shape/1:output:0#xsvuntduhq/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
xsvuntduhq/Reshape/shape¾
xsvuntduhq/ReshapeReshape0ekzorghjta/squeeze_batch_dims/Reshape_1:output:0!xsvuntduhq/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
xsvuntduhq/Reshapeo
zeuewnmlut/ShapeShapexsvuntduhq/Reshape:output:0*
T0*
_output_shapes
:2
zeuewnmlut/Shape
zeuewnmlut/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
zeuewnmlut/strided_slice/stack
 zeuewnmlut/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 zeuewnmlut/strided_slice/stack_1
 zeuewnmlut/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 zeuewnmlut/strided_slice/stack_2¤
zeuewnmlut/strided_sliceStridedSlicezeuewnmlut/Shape:output:0'zeuewnmlut/strided_slice/stack:output:0)zeuewnmlut/strided_slice/stack_1:output:0)zeuewnmlut/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zeuewnmlut/strided_slicer
zeuewnmlut/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros/mul/y
zeuewnmlut/zeros/mulMul!zeuewnmlut/strided_slice:output:0zeuewnmlut/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros/mulu
zeuewnmlut/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeuewnmlut/zeros/Less/y
zeuewnmlut/zeros/LessLesszeuewnmlut/zeros/mul:z:0 zeuewnmlut/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros/Lessx
zeuewnmlut/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros/packed/1¯
zeuewnmlut/zeros/packedPack!zeuewnmlut/strided_slice:output:0"zeuewnmlut/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeuewnmlut/zeros/packedu
zeuewnmlut/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeuewnmlut/zeros/Const¡
zeuewnmlut/zerosFill zeuewnmlut/zeros/packed:output:0zeuewnmlut/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/zerosv
zeuewnmlut/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros_1/mul/y
zeuewnmlut/zeros_1/mulMul!zeuewnmlut/strided_slice:output:0!zeuewnmlut/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros_1/muly
zeuewnmlut/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeuewnmlut/zeros_1/Less/y
zeuewnmlut/zeros_1/LessLesszeuewnmlut/zeros_1/mul:z:0"zeuewnmlut/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeuewnmlut/zeros_1/Less|
zeuewnmlut/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/zeros_1/packed/1µ
zeuewnmlut/zeros_1/packedPack!zeuewnmlut/strided_slice:output:0$zeuewnmlut/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeuewnmlut/zeros_1/packedy
zeuewnmlut/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeuewnmlut/zeros_1/Const©
zeuewnmlut/zeros_1Fill"zeuewnmlut/zeros_1/packed:output:0!zeuewnmlut/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/zeros_1
zeuewnmlut/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zeuewnmlut/transpose/perm°
zeuewnmlut/transpose	Transposexsvuntduhq/Reshape:output:0"zeuewnmlut/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/transposep
zeuewnmlut/Shape_1Shapezeuewnmlut/transpose:y:0*
T0*
_output_shapes
:2
zeuewnmlut/Shape_1
 zeuewnmlut/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zeuewnmlut/strided_slice_1/stack
"zeuewnmlut/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_1/stack_1
"zeuewnmlut/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_1/stack_2°
zeuewnmlut/strided_slice_1StridedSlicezeuewnmlut/Shape_1:output:0)zeuewnmlut/strided_slice_1/stack:output:0+zeuewnmlut/strided_slice_1/stack_1:output:0+zeuewnmlut/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zeuewnmlut/strided_slice_1
&zeuewnmlut/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&zeuewnmlut/TensorArrayV2/element_shapeÞ
zeuewnmlut/TensorArrayV2TensorListReserve/zeuewnmlut/TensorArrayV2/element_shape:output:0#zeuewnmlut/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zeuewnmlut/TensorArrayV2Õ
@zeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@zeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2zeuewnmlut/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorzeuewnmlut/transpose:y:0Izeuewnmlut/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2zeuewnmlut/TensorArrayUnstack/TensorListFromTensor
 zeuewnmlut/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 zeuewnmlut/strided_slice_2/stack
"zeuewnmlut/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_2/stack_1
"zeuewnmlut/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_2/stack_2¾
zeuewnmlut/strided_slice_2StridedSlicezeuewnmlut/transpose:y:0)zeuewnmlut/strided_slice_2/stack:output:0+zeuewnmlut/strided_slice_2/stack_1:output:0+zeuewnmlut/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
zeuewnmlut/strided_slice_2Ð
+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp4zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOpÓ
zeuewnmlut/kfdgklwsil/MatMulMatMul#zeuewnmlut/strided_slice_2:output:03zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/kfdgklwsil/MatMulÖ
-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp6zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOpÏ
zeuewnmlut/kfdgklwsil/MatMul_1MatMulzeuewnmlut/zeros:output:05zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
zeuewnmlut/kfdgklwsil/MatMul_1Ä
zeuewnmlut/kfdgklwsil/addAddV2&zeuewnmlut/kfdgklwsil/MatMul:product:0(zeuewnmlut/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/kfdgklwsil/addÏ
,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp5zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOpÑ
zeuewnmlut/kfdgklwsil/BiasAddBiasAddzeuewnmlut/kfdgklwsil/add:z:04zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zeuewnmlut/kfdgklwsil/BiasAdd
%zeuewnmlut/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%zeuewnmlut/kfdgklwsil/split/split_dim
zeuewnmlut/kfdgklwsil/splitSplit.zeuewnmlut/kfdgklwsil/split/split_dim:output:0&zeuewnmlut/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
zeuewnmlut/kfdgklwsil/split¶
$zeuewnmlut/kfdgklwsil/ReadVariableOpReadVariableOp-zeuewnmlut_kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02&
$zeuewnmlut/kfdgklwsil/ReadVariableOpº
zeuewnmlut/kfdgklwsil/mulMul,zeuewnmlut/kfdgklwsil/ReadVariableOp:value:0zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mulº
zeuewnmlut/kfdgklwsil/add_1AddV2$zeuewnmlut/kfdgklwsil/split:output:0zeuewnmlut/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_1
zeuewnmlut/kfdgklwsil/SigmoidSigmoidzeuewnmlut/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/Sigmoid¼
&zeuewnmlut/kfdgklwsil/ReadVariableOp_1ReadVariableOp/zeuewnmlut_kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&zeuewnmlut/kfdgklwsil/ReadVariableOp_1À
zeuewnmlut/kfdgklwsil/mul_1Mul.zeuewnmlut/kfdgklwsil/ReadVariableOp_1:value:0zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_1¼
zeuewnmlut/kfdgklwsil/add_2AddV2$zeuewnmlut/kfdgklwsil/split:output:1zeuewnmlut/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_2 
zeuewnmlut/kfdgklwsil/Sigmoid_1Sigmoidzeuewnmlut/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zeuewnmlut/kfdgklwsil/Sigmoid_1µ
zeuewnmlut/kfdgklwsil/mul_2Mul#zeuewnmlut/kfdgklwsil/Sigmoid_1:y:0zeuewnmlut/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_2
zeuewnmlut/kfdgklwsil/TanhTanh$zeuewnmlut/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/Tanh¶
zeuewnmlut/kfdgklwsil/mul_3Mul!zeuewnmlut/kfdgklwsil/Sigmoid:y:0zeuewnmlut/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_3·
zeuewnmlut/kfdgklwsil/add_3AddV2zeuewnmlut/kfdgklwsil/mul_2:z:0zeuewnmlut/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_3¼
&zeuewnmlut/kfdgklwsil/ReadVariableOp_2ReadVariableOp/zeuewnmlut_kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&zeuewnmlut/kfdgklwsil/ReadVariableOp_2Ä
zeuewnmlut/kfdgklwsil/mul_4Mul.zeuewnmlut/kfdgklwsil/ReadVariableOp_2:value:0zeuewnmlut/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_4¼
zeuewnmlut/kfdgklwsil/add_4AddV2$zeuewnmlut/kfdgklwsil/split:output:3zeuewnmlut/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/add_4 
zeuewnmlut/kfdgklwsil/Sigmoid_2Sigmoidzeuewnmlut/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
zeuewnmlut/kfdgklwsil/Sigmoid_2
zeuewnmlut/kfdgklwsil/Tanh_1Tanhzeuewnmlut/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/Tanh_1º
zeuewnmlut/kfdgklwsil/mul_5Mul#zeuewnmlut/kfdgklwsil/Sigmoid_2:y:0 zeuewnmlut/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/kfdgklwsil/mul_5¥
(zeuewnmlut/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(zeuewnmlut/TensorArrayV2_1/element_shapeä
zeuewnmlut/TensorArrayV2_1TensorListReserve1zeuewnmlut/TensorArrayV2_1/element_shape:output:0#zeuewnmlut/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
zeuewnmlut/TensorArrayV2_1d
zeuewnmlut/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/time
#zeuewnmlut/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#zeuewnmlut/while/maximum_iterations
zeuewnmlut/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
zeuewnmlut/while/loop_counter°
zeuewnmlut/whileWhile&zeuewnmlut/while/loop_counter:output:0,zeuewnmlut/while/maximum_iterations:output:0zeuewnmlut/time:output:0#zeuewnmlut/TensorArrayV2_1:handle:0zeuewnmlut/zeros:output:0zeuewnmlut/zeros_1:output:0#zeuewnmlut/strided_slice_1:output:0Bzeuewnmlut/TensorArrayUnstack/TensorListFromTensor:output_handle:04zeuewnmlut_kfdgklwsil_matmul_readvariableop_resource6zeuewnmlut_kfdgklwsil_matmul_1_readvariableop_resource5zeuewnmlut_kfdgklwsil_biasadd_readvariableop_resource-zeuewnmlut_kfdgklwsil_readvariableop_resource/zeuewnmlut_kfdgklwsil_readvariableop_1_resource/zeuewnmlut_kfdgklwsil_readvariableop_2_resource*
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
zeuewnmlut_while_body_364418*(
cond R
zeuewnmlut_while_cond_364417*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
zeuewnmlut/whileË
;zeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;zeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shape
-zeuewnmlut/TensorArrayV2Stack/TensorListStackTensorListStackzeuewnmlut/while:output:3Dzeuewnmlut/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-zeuewnmlut/TensorArrayV2Stack/TensorListStack
 zeuewnmlut/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 zeuewnmlut/strided_slice_3/stack
"zeuewnmlut/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"zeuewnmlut/strided_slice_3/stack_1
"zeuewnmlut/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"zeuewnmlut/strided_slice_3/stack_2Ü
zeuewnmlut/strided_slice_3StridedSlice6zeuewnmlut/TensorArrayV2Stack/TensorListStack:tensor:0)zeuewnmlut/strided_slice_3/stack:output:0+zeuewnmlut/strided_slice_3/stack_1:output:0+zeuewnmlut/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
zeuewnmlut/strided_slice_3
zeuewnmlut/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
zeuewnmlut/transpose_1/permÑ
zeuewnmlut/transpose_1	Transpose6zeuewnmlut/TensorArrayV2Stack/TensorListStack:tensor:0$zeuewnmlut/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zeuewnmlut/transpose_1n
ipeigywbwc/ShapeShapezeuewnmlut/transpose_1:y:0*
T0*
_output_shapes
:2
ipeigywbwc/Shape
ipeigywbwc/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ipeigywbwc/strided_slice/stack
 ipeigywbwc/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ipeigywbwc/strided_slice/stack_1
 ipeigywbwc/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ipeigywbwc/strided_slice/stack_2¤
ipeigywbwc/strided_sliceStridedSliceipeigywbwc/Shape:output:0'ipeigywbwc/strided_slice/stack:output:0)ipeigywbwc/strided_slice/stack_1:output:0)ipeigywbwc/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ipeigywbwc/strided_slicer
ipeigywbwc/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros/mul/y
ipeigywbwc/zeros/mulMul!ipeigywbwc/strided_slice:output:0ipeigywbwc/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros/mulu
ipeigywbwc/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ipeigywbwc/zeros/Less/y
ipeigywbwc/zeros/LessLessipeigywbwc/zeros/mul:z:0 ipeigywbwc/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros/Lessx
ipeigywbwc/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros/packed/1¯
ipeigywbwc/zeros/packedPack!ipeigywbwc/strided_slice:output:0"ipeigywbwc/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
ipeigywbwc/zeros/packedu
ipeigywbwc/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ipeigywbwc/zeros/Const¡
ipeigywbwc/zerosFill ipeigywbwc/zeros/packed:output:0ipeigywbwc/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/zerosv
ipeigywbwc/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros_1/mul/y
ipeigywbwc/zeros_1/mulMul!ipeigywbwc/strided_slice:output:0!ipeigywbwc/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros_1/muly
ipeigywbwc/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
ipeigywbwc/zeros_1/Less/y
ipeigywbwc/zeros_1/LessLessipeigywbwc/zeros_1/mul:z:0"ipeigywbwc/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/zeros_1/Less|
ipeigywbwc/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/zeros_1/packed/1µ
ipeigywbwc/zeros_1/packedPack!ipeigywbwc/strided_slice:output:0$ipeigywbwc/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
ipeigywbwc/zeros_1/packedy
ipeigywbwc/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ipeigywbwc/zeros_1/Const©
ipeigywbwc/zeros_1Fill"ipeigywbwc/zeros_1/packed:output:0!ipeigywbwc/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/zeros_1
ipeigywbwc/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ipeigywbwc/transpose/perm¯
ipeigywbwc/transpose	Transposezeuewnmlut/transpose_1:y:0"ipeigywbwc/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/transposep
ipeigywbwc/Shape_1Shapeipeigywbwc/transpose:y:0*
T0*
_output_shapes
:2
ipeigywbwc/Shape_1
 ipeigywbwc/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ipeigywbwc/strided_slice_1/stack
"ipeigywbwc/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_1/stack_1
"ipeigywbwc/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_1/stack_2°
ipeigywbwc/strided_slice_1StridedSliceipeigywbwc/Shape_1:output:0)ipeigywbwc/strided_slice_1/stack:output:0+ipeigywbwc/strided_slice_1/stack_1:output:0+ipeigywbwc/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ipeigywbwc/strided_slice_1
&ipeigywbwc/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&ipeigywbwc/TensorArrayV2/element_shapeÞ
ipeigywbwc/TensorArrayV2TensorListReserve/ipeigywbwc/TensorArrayV2/element_shape:output:0#ipeigywbwc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ipeigywbwc/TensorArrayV2Õ
@ipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@ipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2ipeigywbwc/TensorArrayUnstack/TensorListFromTensorTensorListFromTensoripeigywbwc/transpose:y:0Iipeigywbwc/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2ipeigywbwc/TensorArrayUnstack/TensorListFromTensor
 ipeigywbwc/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 ipeigywbwc/strided_slice_2/stack
"ipeigywbwc/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_2/stack_1
"ipeigywbwc/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_2/stack_2¾
ipeigywbwc/strided_slice_2StridedSliceipeigywbwc/transpose:y:0)ipeigywbwc/strided_slice_2/stack:output:0+ipeigywbwc/strided_slice_2/stack_1:output:0+ipeigywbwc/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ipeigywbwc/strided_slice_2Ð
+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp4ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOpÓ
ipeigywbwc/bdgxhkvdqy/MatMulMatMul#ipeigywbwc/strided_slice_2:output:03ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ipeigywbwc/bdgxhkvdqy/MatMulÖ
-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp6ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOpÏ
ipeigywbwc/bdgxhkvdqy/MatMul_1MatMulipeigywbwc/zeros:output:05ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
ipeigywbwc/bdgxhkvdqy/MatMul_1Ä
ipeigywbwc/bdgxhkvdqy/addAddV2&ipeigywbwc/bdgxhkvdqy/MatMul:product:0(ipeigywbwc/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ipeigywbwc/bdgxhkvdqy/addÏ
,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp5ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOpÑ
ipeigywbwc/bdgxhkvdqy/BiasAddBiasAddipeigywbwc/bdgxhkvdqy/add:z:04ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ipeigywbwc/bdgxhkvdqy/BiasAdd
%ipeigywbwc/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%ipeigywbwc/bdgxhkvdqy/split/split_dim
ipeigywbwc/bdgxhkvdqy/splitSplit.ipeigywbwc/bdgxhkvdqy/split/split_dim:output:0&ipeigywbwc/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
ipeigywbwc/bdgxhkvdqy/split¶
$ipeigywbwc/bdgxhkvdqy/ReadVariableOpReadVariableOp-ipeigywbwc_bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02&
$ipeigywbwc/bdgxhkvdqy/ReadVariableOpº
ipeigywbwc/bdgxhkvdqy/mulMul,ipeigywbwc/bdgxhkvdqy/ReadVariableOp:value:0ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mulº
ipeigywbwc/bdgxhkvdqy/add_1AddV2$ipeigywbwc/bdgxhkvdqy/split:output:0ipeigywbwc/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_1
ipeigywbwc/bdgxhkvdqy/SigmoidSigmoidipeigywbwc/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/Sigmoid¼
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp/ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1À
ipeigywbwc/bdgxhkvdqy/mul_1Mul.ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1:value:0ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_1¼
ipeigywbwc/bdgxhkvdqy/add_2AddV2$ipeigywbwc/bdgxhkvdqy/split:output:1ipeigywbwc/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_2 
ipeigywbwc/bdgxhkvdqy/Sigmoid_1Sigmoidipeigywbwc/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ipeigywbwc/bdgxhkvdqy/Sigmoid_1µ
ipeigywbwc/bdgxhkvdqy/mul_2Mul#ipeigywbwc/bdgxhkvdqy/Sigmoid_1:y:0ipeigywbwc/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_2
ipeigywbwc/bdgxhkvdqy/TanhTanh$ipeigywbwc/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/Tanh¶
ipeigywbwc/bdgxhkvdqy/mul_3Mul!ipeigywbwc/bdgxhkvdqy/Sigmoid:y:0ipeigywbwc/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_3·
ipeigywbwc/bdgxhkvdqy/add_3AddV2ipeigywbwc/bdgxhkvdqy/mul_2:z:0ipeigywbwc/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_3¼
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp/ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2Ä
ipeigywbwc/bdgxhkvdqy/mul_4Mul.ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2:value:0ipeigywbwc/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_4¼
ipeigywbwc/bdgxhkvdqy/add_4AddV2$ipeigywbwc/bdgxhkvdqy/split:output:3ipeigywbwc/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/add_4 
ipeigywbwc/bdgxhkvdqy/Sigmoid_2Sigmoidipeigywbwc/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ipeigywbwc/bdgxhkvdqy/Sigmoid_2
ipeigywbwc/bdgxhkvdqy/Tanh_1Tanhipeigywbwc/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/Tanh_1º
ipeigywbwc/bdgxhkvdqy/mul_5Mul#ipeigywbwc/bdgxhkvdqy/Sigmoid_2:y:0 ipeigywbwc/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/bdgxhkvdqy/mul_5¥
(ipeigywbwc/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(ipeigywbwc/TensorArrayV2_1/element_shapeä
ipeigywbwc/TensorArrayV2_1TensorListReserve1ipeigywbwc/TensorArrayV2_1/element_shape:output:0#ipeigywbwc/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
ipeigywbwc/TensorArrayV2_1d
ipeigywbwc/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/time
#ipeigywbwc/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#ipeigywbwc/while/maximum_iterations
ipeigywbwc/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
ipeigywbwc/while/loop_counter°
ipeigywbwc/whileWhile&ipeigywbwc/while/loop_counter:output:0,ipeigywbwc/while/maximum_iterations:output:0ipeigywbwc/time:output:0#ipeigywbwc/TensorArrayV2_1:handle:0ipeigywbwc/zeros:output:0ipeigywbwc/zeros_1:output:0#ipeigywbwc/strided_slice_1:output:0Bipeigywbwc/TensorArrayUnstack/TensorListFromTensor:output_handle:04ipeigywbwc_bdgxhkvdqy_matmul_readvariableop_resource6ipeigywbwc_bdgxhkvdqy_matmul_1_readvariableop_resource5ipeigywbwc_bdgxhkvdqy_biasadd_readvariableop_resource-ipeigywbwc_bdgxhkvdqy_readvariableop_resource/ipeigywbwc_bdgxhkvdqy_readvariableop_1_resource/ipeigywbwc_bdgxhkvdqy_readvariableop_2_resource*
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
ipeigywbwc_while_body_364594*(
cond R
ipeigywbwc_while_cond_364593*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
ipeigywbwc/whileË
;ipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;ipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shape
-ipeigywbwc/TensorArrayV2Stack/TensorListStackTensorListStackipeigywbwc/while:output:3Dipeigywbwc/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-ipeigywbwc/TensorArrayV2Stack/TensorListStack
 ipeigywbwc/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 ipeigywbwc/strided_slice_3/stack
"ipeigywbwc/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"ipeigywbwc/strided_slice_3/stack_1
"ipeigywbwc/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"ipeigywbwc/strided_slice_3/stack_2Ü
ipeigywbwc/strided_slice_3StridedSlice6ipeigywbwc/TensorArrayV2Stack/TensorListStack:tensor:0)ipeigywbwc/strided_slice_3/stack:output:0+ipeigywbwc/strided_slice_3/stack_1:output:0+ipeigywbwc/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
ipeigywbwc/strided_slice_3
ipeigywbwc/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
ipeigywbwc/transpose_1/permÑ
ipeigywbwc/transpose_1	Transpose6ipeigywbwc/TensorArrayV2Stack/TensorListStack:tensor:0$ipeigywbwc/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/transpose_1®
 pemnqztknd/MatMul/ReadVariableOpReadVariableOp)pemnqztknd_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 pemnqztknd/MatMul/ReadVariableOp±
pemnqztknd/MatMulMatMul#ipeigywbwc/strided_slice_3:output:0(pemnqztknd/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pemnqztknd/MatMul­
!pemnqztknd/BiasAdd/ReadVariableOpReadVariableOp*pemnqztknd_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!pemnqztknd/BiasAdd/ReadVariableOp­
pemnqztknd/BiasAddBiasAddpemnqztknd/MatMul:product:0)pemnqztknd/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
pemnqztknd/BiasAddÏ
IdentityIdentitypemnqztknd/BiasAdd:output:0.^ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp5^ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp-^ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp,^ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp.^ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp%^ipeigywbwc/bdgxhkvdqy/ReadVariableOp'^ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1'^ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2^ipeigywbwc/while"^pemnqztknd/BiasAdd/ReadVariableOp!^pemnqztknd/MatMul/ReadVariableOp-^zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp,^zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp.^zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp%^zeuewnmlut/kfdgklwsil/ReadVariableOp'^zeuewnmlut/kfdgklwsil/ReadVariableOp_1'^zeuewnmlut/kfdgklwsil/ReadVariableOp_2^zeuewnmlut/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2^
-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp-ekzorghjta/conv1d/ExpandDims_1/ReadVariableOp2l
4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp4ekzorghjta/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp,ipeigywbwc/bdgxhkvdqy/BiasAdd/ReadVariableOp2Z
+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp+ipeigywbwc/bdgxhkvdqy/MatMul/ReadVariableOp2^
-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp-ipeigywbwc/bdgxhkvdqy/MatMul_1/ReadVariableOp2L
$ipeigywbwc/bdgxhkvdqy/ReadVariableOp$ipeigywbwc/bdgxhkvdqy/ReadVariableOp2P
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_1&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_12P
&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_2&ipeigywbwc/bdgxhkvdqy/ReadVariableOp_22$
ipeigywbwc/whileipeigywbwc/while2F
!pemnqztknd/BiasAdd/ReadVariableOp!pemnqztknd/BiasAdd/ReadVariableOp2D
 pemnqztknd/MatMul/ReadVariableOp pemnqztknd/MatMul/ReadVariableOp2\
,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp,zeuewnmlut/kfdgklwsil/BiasAdd/ReadVariableOp2Z
+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp+zeuewnmlut/kfdgklwsil/MatMul/ReadVariableOp2^
-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp-zeuewnmlut/kfdgklwsil/MatMul_1/ReadVariableOp2L
$zeuewnmlut/kfdgklwsil/ReadVariableOp$zeuewnmlut/kfdgklwsil/ReadVariableOp2P
&zeuewnmlut/kfdgklwsil/ReadVariableOp_1&zeuewnmlut/kfdgklwsil/ReadVariableOp_12P
&zeuewnmlut/kfdgklwsil/ReadVariableOp_2&zeuewnmlut/kfdgklwsil/ReadVariableOp_22$
zeuewnmlut/whilezeuewnmlut/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


å
while_cond_362799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362799___redundant_placeholder04
0while_while_cond_362799___redundant_placeholder14
0while_while_cond_362799___redundant_placeholder24
0while_while_cond_362799___redundant_placeholder34
0while_while_cond_362799___redundant_placeholder44
0while_while_cond_362799___redundant_placeholder54
0while_while_cond_362799___redundant_placeholder6
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
while_cond_365097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365097___redundant_placeholder04
0while_while_cond_365097___redundant_placeholder14
0while_while_cond_365097___redundant_placeholder24
0while_while_cond_365097___redundant_placeholder34
0while_while_cond_365097___redundant_placeholder44
0while_while_cond_365097___redundant_placeholder54
0while_while_cond_365097___redundant_placeholder6
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
p
É
ipeigywbwc_while_body_3641902
.ipeigywbwc_while_ipeigywbwc_while_loop_counter8
4ipeigywbwc_while_ipeigywbwc_while_maximum_iterations 
ipeigywbwc_while_placeholder"
ipeigywbwc_while_placeholder_1"
ipeigywbwc_while_placeholder_2"
ipeigywbwc_while_placeholder_31
-ipeigywbwc_while_ipeigywbwc_strided_slice_1_0m
iipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0O
<ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 Q
>ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 L
=ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	C
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0: E
7ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0: E
7ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0: 
ipeigywbwc_while_identity
ipeigywbwc_while_identity_1
ipeigywbwc_while_identity_2
ipeigywbwc_while_identity_3
ipeigywbwc_while_identity_4
ipeigywbwc_while_identity_5/
+ipeigywbwc_while_ipeigywbwc_strided_slice_1k
gipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensorM
:ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource:	 O
<ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 J
;ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource:	A
3ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource: C
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource: C
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource: ¢2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp¢3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp¢,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1¢,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2Ù
Bipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shape
4ipeigywbwc/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0ipeigywbwc_while_placeholderKipeigywbwc/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4ipeigywbwc/while/TensorArrayV2Read/TensorListGetItemä
1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp<ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOpý
"ipeigywbwc/while/bdgxhkvdqy/MatMulMatMul;ipeigywbwc/while/TensorArrayV2Read/TensorListGetItem:item:09ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"ipeigywbwc/while/bdgxhkvdqy/MatMulê
3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp>ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOpæ
$ipeigywbwc/while/bdgxhkvdqy/MatMul_1MatMulipeigywbwc_while_placeholder_2;ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$ipeigywbwc/while/bdgxhkvdqy/MatMul_1Ü
ipeigywbwc/while/bdgxhkvdqy/addAddV2,ipeigywbwc/while/bdgxhkvdqy/MatMul:product:0.ipeigywbwc/while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
ipeigywbwc/while/bdgxhkvdqy/addã
2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp=ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOpé
#ipeigywbwc/while/bdgxhkvdqy/BiasAddBiasAdd#ipeigywbwc/while/bdgxhkvdqy/add:z:0:ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#ipeigywbwc/while/bdgxhkvdqy/BiasAdd
+ipeigywbwc/while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+ipeigywbwc/while/bdgxhkvdqy/split/split_dim¯
!ipeigywbwc/while/bdgxhkvdqy/splitSplit4ipeigywbwc/while/bdgxhkvdqy/split/split_dim:output:0,ipeigywbwc/while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!ipeigywbwc/while/bdgxhkvdqy/splitÊ
*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOpReadVariableOp5ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOpÏ
ipeigywbwc/while/bdgxhkvdqy/mulMul2ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp:value:0ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
ipeigywbwc/while/bdgxhkvdqy/mulÒ
!ipeigywbwc/while/bdgxhkvdqy/add_1AddV2*ipeigywbwc/while/bdgxhkvdqy/split:output:0#ipeigywbwc/while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_1®
#ipeigywbwc/while/bdgxhkvdqy/SigmoidSigmoid%ipeigywbwc/while/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#ipeigywbwc/while/bdgxhkvdqy/SigmoidÐ
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp7ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1Õ
!ipeigywbwc/while/bdgxhkvdqy/mul_1Mul4ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1:value:0ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_1Ô
!ipeigywbwc/while/bdgxhkvdqy/add_2AddV2*ipeigywbwc/while/bdgxhkvdqy/split:output:1%ipeigywbwc/while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_2²
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1Sigmoid%ipeigywbwc/while/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1Ê
!ipeigywbwc/while/bdgxhkvdqy/mul_2Mul)ipeigywbwc/while/bdgxhkvdqy/Sigmoid_1:y:0ipeigywbwc_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_2ª
 ipeigywbwc/while/bdgxhkvdqy/TanhTanh*ipeigywbwc/while/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 ipeigywbwc/while/bdgxhkvdqy/TanhÎ
!ipeigywbwc/while/bdgxhkvdqy/mul_3Mul'ipeigywbwc/while/bdgxhkvdqy/Sigmoid:y:0$ipeigywbwc/while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_3Ï
!ipeigywbwc/while/bdgxhkvdqy/add_3AddV2%ipeigywbwc/while/bdgxhkvdqy/mul_2:z:0%ipeigywbwc/while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_3Ð
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp7ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2Ü
!ipeigywbwc/while/bdgxhkvdqy/mul_4Mul4ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2:value:0%ipeigywbwc/while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_4Ô
!ipeigywbwc/while/bdgxhkvdqy/add_4AddV2*ipeigywbwc/while/bdgxhkvdqy/split:output:3%ipeigywbwc/while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/add_4²
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2Sigmoid%ipeigywbwc/while/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2©
"ipeigywbwc/while/bdgxhkvdqy/Tanh_1Tanh%ipeigywbwc/while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"ipeigywbwc/while/bdgxhkvdqy/Tanh_1Ò
!ipeigywbwc/while/bdgxhkvdqy/mul_5Mul)ipeigywbwc/while/bdgxhkvdqy/Sigmoid_2:y:0&ipeigywbwc/while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!ipeigywbwc/while/bdgxhkvdqy/mul_5
5ipeigywbwc/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemipeigywbwc_while_placeholder_1ipeigywbwc_while_placeholder%ipeigywbwc/while/bdgxhkvdqy/mul_5:z:0*
_output_shapes
: *
element_dtype027
5ipeigywbwc/while/TensorArrayV2Write/TensorListSetItemr
ipeigywbwc/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
ipeigywbwc/while/add/y
ipeigywbwc/while/addAddV2ipeigywbwc_while_placeholderipeigywbwc/while/add/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/while/addv
ipeigywbwc/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
ipeigywbwc/while/add_1/y­
ipeigywbwc/while/add_1AddV2.ipeigywbwc_while_ipeigywbwc_while_loop_counter!ipeigywbwc/while/add_1/y:output:0*
T0*
_output_shapes
: 2
ipeigywbwc/while/add_1©
ipeigywbwc/while/IdentityIdentityipeigywbwc/while/add_1:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/IdentityÇ
ipeigywbwc/while/Identity_1Identity4ipeigywbwc_while_ipeigywbwc_while_maximum_iterations3^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/Identity_1«
ipeigywbwc/while/Identity_2Identityipeigywbwc/while/add:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/Identity_2Ø
ipeigywbwc/while/Identity_3IdentityEipeigywbwc/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
ipeigywbwc/while/Identity_3É
ipeigywbwc/while/Identity_4Identity%ipeigywbwc/while/bdgxhkvdqy/mul_5:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/while/Identity_4É
ipeigywbwc/while/Identity_5Identity%ipeigywbwc/while/bdgxhkvdqy/add_3:z:03^ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2^ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp4^ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp+^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1-^ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
ipeigywbwc/while/Identity_5"|
;ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource=ipeigywbwc_while_bdgxhkvdqy_biasadd_readvariableop_resource_0"~
<ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource>ipeigywbwc_while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"z
:ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource<ipeigywbwc_while_bdgxhkvdqy_matmul_readvariableop_resource_0"p
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource7ipeigywbwc_while_bdgxhkvdqy_readvariableop_1_resource_0"p
5ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource7ipeigywbwc_while_bdgxhkvdqy_readvariableop_2_resource_0"l
3ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource5ipeigywbwc_while_bdgxhkvdqy_readvariableop_resource_0"?
ipeigywbwc_while_identity"ipeigywbwc/while/Identity:output:0"C
ipeigywbwc_while_identity_1$ipeigywbwc/while/Identity_1:output:0"C
ipeigywbwc_while_identity_2$ipeigywbwc/while/Identity_2:output:0"C
ipeigywbwc_while_identity_3$ipeigywbwc/while/Identity_3:output:0"C
ipeigywbwc_while_identity_4$ipeigywbwc/while/Identity_4:output:0"C
ipeigywbwc_while_identity_5$ipeigywbwc/while/Identity_5:output:0"\
+ipeigywbwc_while_ipeigywbwc_strided_slice_1-ipeigywbwc_while_ipeigywbwc_strided_slice_1_0"Ô
gipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensoriipeigywbwc_while_tensorarrayv2read_tensorlistgetitem_ipeigywbwc_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2ipeigywbwc/while/bdgxhkvdqy/BiasAdd/ReadVariableOp2f
1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp1ipeigywbwc/while/bdgxhkvdqy/MatMul/ReadVariableOp2j
3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp3ipeigywbwc/while/bdgxhkvdqy/MatMul_1/ReadVariableOp2X
*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp*ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp2\
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_1,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_12\
,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2,ipeigywbwc/while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Æ

+__inference_ekzorghjta_layer_call_fn_364821

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
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_3627012
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


å
while_cond_366245
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_366245___redundant_placeholder04
0while_while_cond_366245___redundant_placeholder14
0while_while_cond_366245___redundant_placeholder24
0while_while_cond_366245___redundant_placeholder34
0while_while_cond_366245___redundant_placeholder44
0while_while_cond_366245___redundant_placeholder54
0while_while_cond_366245___redundant_placeholder6
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
é

+__inference_ipeigywbwc_layer_call_fn_366364
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3620862
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
¸'
´
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_366522

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


å
while_cond_363267
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_363267___redundant_placeholder04
0while_while_cond_363267___redundant_placeholder14
0while_while_cond_363267___redundant_placeholder24
0while_while_cond_363267___redundant_placeholder34
0while_while_cond_363267___redundant_placeholder44
0while_while_cond_363267___redundant_placeholder54
0while_while_cond_363267___redundant_placeholder6
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
while_body_363482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_kfdgklwsil_matmul_readvariableop_resource_0:	F
3while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 A
2while_kfdgklwsil_biasadd_readvariableop_resource_0:	8
*while_kfdgklwsil_readvariableop_resource_0: :
,while_kfdgklwsil_readvariableop_1_resource_0: :
,while_kfdgklwsil_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_kfdgklwsil_matmul_readvariableop_resource:	D
1while_kfdgklwsil_matmul_1_readvariableop_resource:	 ?
0while_kfdgklwsil_biasadd_readvariableop_resource:	6
(while_kfdgklwsil_readvariableop_resource: 8
*while_kfdgklwsil_readvariableop_1_resource: 8
*while_kfdgklwsil_readvariableop_2_resource: ¢'while/kfdgklwsil/BiasAdd/ReadVariableOp¢&while/kfdgklwsil/MatMul/ReadVariableOp¢(while/kfdgklwsil/MatMul_1/ReadVariableOp¢while/kfdgklwsil/ReadVariableOp¢!while/kfdgklwsil/ReadVariableOp_1¢!while/kfdgklwsil/ReadVariableOp_2Ã
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
&while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOp1while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/kfdgklwsil/MatMul/ReadVariableOpÑ
while/kfdgklwsil/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMulÉ
(while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp3while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/kfdgklwsil/MatMul_1/ReadVariableOpº
while/kfdgklwsil/MatMul_1MatMulwhile_placeholder_20while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/MatMul_1°
while/kfdgklwsil/addAddV2!while/kfdgklwsil/MatMul:product:0#while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/addÂ
'while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp2while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/kfdgklwsil/BiasAdd/ReadVariableOp½
while/kfdgklwsil/BiasAddBiasAddwhile/kfdgklwsil/add:z:0/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/kfdgklwsil/BiasAdd
 while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/kfdgklwsil/split/split_dim
while/kfdgklwsil/splitSplit)while/kfdgklwsil/split/split_dim:output:0!while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/kfdgklwsil/split©
while/kfdgklwsil/ReadVariableOpReadVariableOp*while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/kfdgklwsil/ReadVariableOp£
while/kfdgklwsil/mulMul'while/kfdgklwsil/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul¦
while/kfdgklwsil/add_1AddV2while/kfdgklwsil/split:output:0while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_1
while/kfdgklwsil/SigmoidSigmoidwhile/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid¯
!while/kfdgklwsil/ReadVariableOp_1ReadVariableOp,while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_1©
while/kfdgklwsil/mul_1Mul)while/kfdgklwsil/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_1¨
while/kfdgklwsil/add_2AddV2while/kfdgklwsil/split:output:1while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_2
while/kfdgklwsil/Sigmoid_1Sigmoidwhile/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_1
while/kfdgklwsil/mul_2Mulwhile/kfdgklwsil/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_2
while/kfdgklwsil/TanhTanhwhile/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh¢
while/kfdgklwsil/mul_3Mulwhile/kfdgklwsil/Sigmoid:y:0while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_3£
while/kfdgklwsil/add_3AddV2while/kfdgklwsil/mul_2:z:0while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_3¯
!while/kfdgklwsil/ReadVariableOp_2ReadVariableOp,while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/kfdgklwsil/ReadVariableOp_2°
while/kfdgklwsil/mul_4Mul)while/kfdgklwsil/ReadVariableOp_2:value:0while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_4¨
while/kfdgklwsil/add_4AddV2while/kfdgklwsil/split:output:3while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/add_4
while/kfdgklwsil/Sigmoid_2Sigmoidwhile/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Sigmoid_2
while/kfdgklwsil/Tanh_1Tanhwhile/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/Tanh_1¦
while/kfdgklwsil/mul_5Mulwhile/kfdgklwsil/Sigmoid_2:y:0while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/kfdgklwsil/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/kfdgklwsil/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/kfdgklwsil/mul_5:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/kfdgklwsil/add_3:z:0(^while/kfdgklwsil/BiasAdd/ReadVariableOp'^while/kfdgklwsil/MatMul/ReadVariableOp)^while/kfdgklwsil/MatMul_1/ReadVariableOp ^while/kfdgklwsil/ReadVariableOp"^while/kfdgklwsil/ReadVariableOp_1"^while/kfdgklwsil/ReadVariableOp_2*
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
0while_kfdgklwsil_biasadd_readvariableop_resource2while_kfdgklwsil_biasadd_readvariableop_resource_0"h
1while_kfdgklwsil_matmul_1_readvariableop_resource3while_kfdgklwsil_matmul_1_readvariableop_resource_0"d
/while_kfdgklwsil_matmul_readvariableop_resource1while_kfdgklwsil_matmul_readvariableop_resource_0"Z
*while_kfdgklwsil_readvariableop_1_resource,while_kfdgklwsil_readvariableop_1_resource_0"Z
*while_kfdgklwsil_readvariableop_2_resource,while_kfdgklwsil_readvariableop_2_resource_0"V
(while_kfdgklwsil_readvariableop_resource*while_kfdgklwsil_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/kfdgklwsil/BiasAdd/ReadVariableOp'while/kfdgklwsil/BiasAdd/ReadVariableOp2P
&while/kfdgklwsil/MatMul/ReadVariableOp&while/kfdgklwsil/MatMul/ReadVariableOp2T
(while/kfdgklwsil/MatMul_1/ReadVariableOp(while/kfdgklwsil/MatMul_1/ReadVariableOp2B
while/kfdgklwsil/ReadVariableOpwhile/kfdgklwsil/ReadVariableOp2F
!while/kfdgklwsil/ReadVariableOp_1!while/kfdgklwsil/ReadVariableOp_12F
!while/kfdgklwsil/ReadVariableOp_2!while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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


+__inference_sequential_layer_call_fn_363766

kjggqknufb
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
kjggqknufbunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
F__inference_sequential_layer_call_and_return_conditional_losses_3636942
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
kjggqknufb
Ùh

F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_365807
inputs_0<
)bdgxhkvdqy_matmul_readvariableop_resource:	 >
+bdgxhkvdqy_matmul_1_readvariableop_resource:	 9
*bdgxhkvdqy_biasadd_readvariableop_resource:	0
"bdgxhkvdqy_readvariableop_resource: 2
$bdgxhkvdqy_readvariableop_1_resource: 2
$bdgxhkvdqy_readvariableop_2_resource: 
identity¢!bdgxhkvdqy/BiasAdd/ReadVariableOp¢ bdgxhkvdqy/MatMul/ReadVariableOp¢"bdgxhkvdqy/MatMul_1/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp_1¢bdgxhkvdqy/ReadVariableOp_2¢whileF
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
 bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp)bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 bdgxhkvdqy/MatMul/ReadVariableOp§
bdgxhkvdqy/MatMulMatMulstrided_slice_2:output:0(bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMulµ
"bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp+bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"bdgxhkvdqy/MatMul_1/ReadVariableOp£
bdgxhkvdqy/MatMul_1MatMulzeros:output:0*bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMul_1
bdgxhkvdqy/addAddV2bdgxhkvdqy/MatMul:product:0bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/add®
!bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp*bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!bdgxhkvdqy/BiasAdd/ReadVariableOp¥
bdgxhkvdqy/BiasAddBiasAddbdgxhkvdqy/add:z:0)bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/BiasAddz
bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
bdgxhkvdqy/split/split_dimë
bdgxhkvdqy/splitSplit#bdgxhkvdqy/split/split_dim:output:0bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
bdgxhkvdqy/split
bdgxhkvdqy/ReadVariableOpReadVariableOp"bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp
bdgxhkvdqy/mulMul!bdgxhkvdqy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul
bdgxhkvdqy/add_1AddV2bdgxhkvdqy/split:output:0bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_1{
bdgxhkvdqy/SigmoidSigmoidbdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid
bdgxhkvdqy/ReadVariableOp_1ReadVariableOp$bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_1
bdgxhkvdqy/mul_1Mul#bdgxhkvdqy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_1
bdgxhkvdqy/add_2AddV2bdgxhkvdqy/split:output:1bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_2
bdgxhkvdqy/Sigmoid_1Sigmoidbdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_1
bdgxhkvdqy/mul_2Mulbdgxhkvdqy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_2w
bdgxhkvdqy/TanhTanhbdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh
bdgxhkvdqy/mul_3Mulbdgxhkvdqy/Sigmoid:y:0bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_3
bdgxhkvdqy/add_3AddV2bdgxhkvdqy/mul_2:z:0bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_3
bdgxhkvdqy/ReadVariableOp_2ReadVariableOp$bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_2
bdgxhkvdqy/mul_4Mul#bdgxhkvdqy/ReadVariableOp_2:value:0bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_4
bdgxhkvdqy/add_4AddV2bdgxhkvdqy/split:output:3bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_4
bdgxhkvdqy/Sigmoid_2Sigmoidbdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_2v
bdgxhkvdqy/Tanh_1Tanhbdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh_1
bdgxhkvdqy/mul_5Mulbdgxhkvdqy/Sigmoid_2:y:0bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)bdgxhkvdqy_matmul_readvariableop_resource+bdgxhkvdqy_matmul_1_readvariableop_resource*bdgxhkvdqy_biasadd_readvariableop_resource"bdgxhkvdqy_readvariableop_resource$bdgxhkvdqy_readvariableop_1_resource$bdgxhkvdqy_readvariableop_2_resource*
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
while_body_365706*
condR
while_cond_365705*Q
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
IdentityIdentitystrided_slice_3:output:0"^bdgxhkvdqy/BiasAdd/ReadVariableOp!^bdgxhkvdqy/MatMul/ReadVariableOp#^bdgxhkvdqy/MatMul_1/ReadVariableOp^bdgxhkvdqy/ReadVariableOp^bdgxhkvdqy/ReadVariableOp_1^bdgxhkvdqy/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!bdgxhkvdqy/BiasAdd/ReadVariableOp!bdgxhkvdqy/BiasAdd/ReadVariableOp2D
 bdgxhkvdqy/MatMul/ReadVariableOp bdgxhkvdqy/MatMul/ReadVariableOp2H
"bdgxhkvdqy/MatMul_1/ReadVariableOp"bdgxhkvdqy/MatMul_1/ReadVariableOp26
bdgxhkvdqy/ReadVariableOpbdgxhkvdqy/ReadVariableOp2:
bdgxhkvdqy/ReadVariableOp_1bdgxhkvdqy/ReadVariableOp_12:
bdgxhkvdqy/ReadVariableOp_2bdgxhkvdqy/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
¨0
»
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_362701

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
Ù

+__inference_zeuewnmlut_layer_call_fn_365627

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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3635832
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
h

F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_362901

inputs<
)kfdgklwsil_matmul_readvariableop_resource:	>
+kfdgklwsil_matmul_1_readvariableop_resource:	 9
*kfdgklwsil_biasadd_readvariableop_resource:	0
"kfdgklwsil_readvariableop_resource: 2
$kfdgklwsil_readvariableop_1_resource: 2
$kfdgklwsil_readvariableop_2_resource: 
identity¢!kfdgklwsil/BiasAdd/ReadVariableOp¢ kfdgklwsil/MatMul/ReadVariableOp¢"kfdgklwsil/MatMul_1/ReadVariableOp¢kfdgklwsil/ReadVariableOp¢kfdgklwsil/ReadVariableOp_1¢kfdgklwsil/ReadVariableOp_2¢whileD
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
 kfdgklwsil/MatMul/ReadVariableOpReadVariableOp)kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 kfdgklwsil/MatMul/ReadVariableOp§
kfdgklwsil/MatMulMatMulstrided_slice_2:output:0(kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMulµ
"kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp+kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kfdgklwsil/MatMul_1/ReadVariableOp£
kfdgklwsil/MatMul_1MatMulzeros:output:0*kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMul_1
kfdgklwsil/addAddV2kfdgklwsil/MatMul:product:0kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/add®
!kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp*kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kfdgklwsil/BiasAdd/ReadVariableOp¥
kfdgklwsil/BiasAddBiasAddkfdgklwsil/add:z:0)kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/BiasAddz
kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kfdgklwsil/split/split_dimë
kfdgklwsil/splitSplit#kfdgklwsil/split/split_dim:output:0kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kfdgklwsil/split
kfdgklwsil/ReadVariableOpReadVariableOp"kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp
kfdgklwsil/mulMul!kfdgklwsil/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul
kfdgklwsil/add_1AddV2kfdgklwsil/split:output:0kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_1{
kfdgklwsil/SigmoidSigmoidkfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid
kfdgklwsil/ReadVariableOp_1ReadVariableOp$kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_1
kfdgklwsil/mul_1Mul#kfdgklwsil/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_1
kfdgklwsil/add_2AddV2kfdgklwsil/split:output:1kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_2
kfdgklwsil/Sigmoid_1Sigmoidkfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_1
kfdgklwsil/mul_2Mulkfdgklwsil/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_2w
kfdgklwsil/TanhTanhkfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh
kfdgklwsil/mul_3Mulkfdgklwsil/Sigmoid:y:0kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_3
kfdgklwsil/add_3AddV2kfdgklwsil/mul_2:z:0kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_3
kfdgklwsil/ReadVariableOp_2ReadVariableOp$kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_2
kfdgklwsil/mul_4Mul#kfdgklwsil/ReadVariableOp_2:value:0kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_4
kfdgklwsil/add_4AddV2kfdgklwsil/split:output:3kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_4
kfdgklwsil/Sigmoid_2Sigmoidkfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_2v
kfdgklwsil/Tanh_1Tanhkfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh_1
kfdgklwsil/mul_5Mulkfdgklwsil/Sigmoid_2:y:0kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kfdgklwsil_matmul_readvariableop_resource+kfdgklwsil_matmul_1_readvariableop_resource*kfdgklwsil_biasadd_readvariableop_resource"kfdgklwsil_readvariableop_resource$kfdgklwsil_readvariableop_1_resource$kfdgklwsil_readvariableop_2_resource*
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
while_body_362800*
condR
while_cond_362799*Q
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
IdentityIdentitytranspose_1:y:0"^kfdgklwsil/BiasAdd/ReadVariableOp!^kfdgklwsil/MatMul/ReadVariableOp#^kfdgklwsil/MatMul_1/ReadVariableOp^kfdgklwsil/ReadVariableOp^kfdgklwsil/ReadVariableOp_1^kfdgklwsil/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!kfdgklwsil/BiasAdd/ReadVariableOp!kfdgklwsil/BiasAdd/ReadVariableOp2D
 kfdgklwsil/MatMul/ReadVariableOp kfdgklwsil/MatMul/ReadVariableOp2H
"kfdgklwsil/MatMul_1/ReadVariableOp"kfdgklwsil/MatMul_1/ReadVariableOp26
kfdgklwsil/ReadVariableOpkfdgklwsil/ReadVariableOp2:
kfdgklwsil/ReadVariableOp_1kfdgklwsil/ReadVariableOp_12:
kfdgklwsil/ReadVariableOp_2kfdgklwsil/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯

"__inference__traced_restore_366969
file_prefix8
"assignvariableop_ekzorghjta_kernel:0
"assignvariableop_1_ekzorghjta_bias:6
$assignvariableop_2_pemnqztknd_kernel: 0
"assignvariableop_3_pemnqztknd_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_zeuewnmlut_kfdgklwsil_kernel:	M
:assignvariableop_10_zeuewnmlut_kfdgklwsil_recurrent_kernel:	 =
.assignvariableop_11_zeuewnmlut_kfdgklwsil_bias:	S
Eassignvariableop_12_zeuewnmlut_kfdgklwsil_input_gate_peephole_weights: T
Fassignvariableop_13_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weights: T
Fassignvariableop_14_zeuewnmlut_kfdgklwsil_output_gate_peephole_weights: C
0assignvariableop_15_ipeigywbwc_bdgxhkvdqy_kernel:	 M
:assignvariableop_16_ipeigywbwc_bdgxhkvdqy_recurrent_kernel:	 =
.assignvariableop_17_ipeigywbwc_bdgxhkvdqy_bias:	S
Eassignvariableop_18_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weights: T
Fassignvariableop_19_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weights: T
Fassignvariableop_20_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_ekzorghjta_kernel_rms:=
/assignvariableop_24_rmsprop_ekzorghjta_bias_rms:C
1assignvariableop_25_rmsprop_pemnqztknd_kernel_rms: =
/assignvariableop_26_rmsprop_pemnqztknd_bias_rms:O
<assignvariableop_27_rmsprop_zeuewnmlut_kfdgklwsil_kernel_rms:	Y
Fassignvariableop_28_rmsprop_zeuewnmlut_kfdgklwsil_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_zeuewnmlut_kfdgklwsil_bias_rms:	_
Qassignvariableop_30_rmsprop_zeuewnmlut_kfdgklwsil_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_zeuewnmlut_kfdgklwsil_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_ipeigywbwc_bdgxhkvdqy_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_ipeigywbwc_bdgxhkvdqy_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_ipeigywbwc_bdgxhkvdqy_bias_rms:	_
Qassignvariableop_36_rmsprop_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_ekzorghjta_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_ekzorghjta_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_pemnqztknd_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_pemnqztknd_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_zeuewnmlut_kfdgklwsil_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_zeuewnmlut_kfdgklwsil_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_zeuewnmlut_kfdgklwsil_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_zeuewnmlut_kfdgklwsil_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_zeuewnmlut_kfdgklwsil_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_ipeigywbwc_bdgxhkvdqy_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_ipeigywbwc_bdgxhkvdqy_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_ipeigywbwc_bdgxhkvdqy_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_ekzorghjta_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_ekzorghjta_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_pemnqztknd_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_pemnqztknd_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_zeuewnmlut_kfdgklwsil_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_zeuewnmlut_kfdgklwsil_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_zeuewnmlut_kfdgklwsil_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_zeuewnmlut_kfdgklwsil_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_zeuewnmlut_kfdgklwsil_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_ipeigywbwc_bdgxhkvdqy_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_ipeigywbwc_bdgxhkvdqy_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_ipeigywbwc_bdgxhkvdqy_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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
£h

F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_366167

inputs<
)bdgxhkvdqy_matmul_readvariableop_resource:	 >
+bdgxhkvdqy_matmul_1_readvariableop_resource:	 9
*bdgxhkvdqy_biasadd_readvariableop_resource:	0
"bdgxhkvdqy_readvariableop_resource: 2
$bdgxhkvdqy_readvariableop_1_resource: 2
$bdgxhkvdqy_readvariableop_2_resource: 
identity¢!bdgxhkvdqy/BiasAdd/ReadVariableOp¢ bdgxhkvdqy/MatMul/ReadVariableOp¢"bdgxhkvdqy/MatMul_1/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp_1¢bdgxhkvdqy/ReadVariableOp_2¢whileD
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
 bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp)bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 bdgxhkvdqy/MatMul/ReadVariableOp§
bdgxhkvdqy/MatMulMatMulstrided_slice_2:output:0(bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMulµ
"bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp+bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"bdgxhkvdqy/MatMul_1/ReadVariableOp£
bdgxhkvdqy/MatMul_1MatMulzeros:output:0*bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMul_1
bdgxhkvdqy/addAddV2bdgxhkvdqy/MatMul:product:0bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/add®
!bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp*bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!bdgxhkvdqy/BiasAdd/ReadVariableOp¥
bdgxhkvdqy/BiasAddBiasAddbdgxhkvdqy/add:z:0)bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/BiasAddz
bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
bdgxhkvdqy/split/split_dimë
bdgxhkvdqy/splitSplit#bdgxhkvdqy/split/split_dim:output:0bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
bdgxhkvdqy/split
bdgxhkvdqy/ReadVariableOpReadVariableOp"bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp
bdgxhkvdqy/mulMul!bdgxhkvdqy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul
bdgxhkvdqy/add_1AddV2bdgxhkvdqy/split:output:0bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_1{
bdgxhkvdqy/SigmoidSigmoidbdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid
bdgxhkvdqy/ReadVariableOp_1ReadVariableOp$bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_1
bdgxhkvdqy/mul_1Mul#bdgxhkvdqy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_1
bdgxhkvdqy/add_2AddV2bdgxhkvdqy/split:output:1bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_2
bdgxhkvdqy/Sigmoid_1Sigmoidbdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_1
bdgxhkvdqy/mul_2Mulbdgxhkvdqy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_2w
bdgxhkvdqy/TanhTanhbdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh
bdgxhkvdqy/mul_3Mulbdgxhkvdqy/Sigmoid:y:0bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_3
bdgxhkvdqy/add_3AddV2bdgxhkvdqy/mul_2:z:0bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_3
bdgxhkvdqy/ReadVariableOp_2ReadVariableOp$bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_2
bdgxhkvdqy/mul_4Mul#bdgxhkvdqy/ReadVariableOp_2:value:0bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_4
bdgxhkvdqy/add_4AddV2bdgxhkvdqy/split:output:3bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_4
bdgxhkvdqy/Sigmoid_2Sigmoidbdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_2v
bdgxhkvdqy/Tanh_1Tanhbdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh_1
bdgxhkvdqy/mul_5Mulbdgxhkvdqy/Sigmoid_2:y:0bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)bdgxhkvdqy_matmul_readvariableop_resource+bdgxhkvdqy_matmul_1_readvariableop_resource*bdgxhkvdqy_biasadd_readvariableop_resource"bdgxhkvdqy_readvariableop_resource$bdgxhkvdqy_readvariableop_1_resource$bdgxhkvdqy_readvariableop_2_resource*
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
while_body_366066*
condR
while_cond_366065*Q
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
IdentityIdentitystrided_slice_3:output:0"^bdgxhkvdqy/BiasAdd/ReadVariableOp!^bdgxhkvdqy/MatMul/ReadVariableOp#^bdgxhkvdqy/MatMul_1/ReadVariableOp^bdgxhkvdqy/ReadVariableOp^bdgxhkvdqy/ReadVariableOp_1^bdgxhkvdqy/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!bdgxhkvdqy/BiasAdd/ReadVariableOp!bdgxhkvdqy/BiasAdd/ReadVariableOp2D
 bdgxhkvdqy/MatMul/ReadVariableOp bdgxhkvdqy/MatMul/ReadVariableOp2H
"bdgxhkvdqy/MatMul_1/ReadVariableOp"bdgxhkvdqy/MatMul_1/ReadVariableOp26
bdgxhkvdqy/ReadVariableOpbdgxhkvdqy/ReadVariableOp2:
bdgxhkvdqy/ReadVariableOp_1bdgxhkvdqy/ReadVariableOp_12:
bdgxhkvdqy/ReadVariableOp_2bdgxhkvdqy/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢

+__inference_pemnqztknd_layer_call_fn_366434

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
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_3631182
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
Ýh

F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365199
inputs_0<
)kfdgklwsil_matmul_readvariableop_resource:	>
+kfdgklwsil_matmul_1_readvariableop_resource:	 9
*kfdgklwsil_biasadd_readvariableop_resource:	0
"kfdgklwsil_readvariableop_resource: 2
$kfdgklwsil_readvariableop_1_resource: 2
$kfdgklwsil_readvariableop_2_resource: 
identity¢!kfdgklwsil/BiasAdd/ReadVariableOp¢ kfdgklwsil/MatMul/ReadVariableOp¢"kfdgklwsil/MatMul_1/ReadVariableOp¢kfdgklwsil/ReadVariableOp¢kfdgklwsil/ReadVariableOp_1¢kfdgklwsil/ReadVariableOp_2¢whileF
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
 kfdgklwsil/MatMul/ReadVariableOpReadVariableOp)kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 kfdgklwsil/MatMul/ReadVariableOp§
kfdgklwsil/MatMulMatMulstrided_slice_2:output:0(kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMulµ
"kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp+kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kfdgklwsil/MatMul_1/ReadVariableOp£
kfdgklwsil/MatMul_1MatMulzeros:output:0*kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMul_1
kfdgklwsil/addAddV2kfdgklwsil/MatMul:product:0kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/add®
!kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp*kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kfdgklwsil/BiasAdd/ReadVariableOp¥
kfdgklwsil/BiasAddBiasAddkfdgklwsil/add:z:0)kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/BiasAddz
kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kfdgklwsil/split/split_dimë
kfdgklwsil/splitSplit#kfdgklwsil/split/split_dim:output:0kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kfdgklwsil/split
kfdgklwsil/ReadVariableOpReadVariableOp"kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp
kfdgklwsil/mulMul!kfdgklwsil/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul
kfdgklwsil/add_1AddV2kfdgklwsil/split:output:0kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_1{
kfdgklwsil/SigmoidSigmoidkfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid
kfdgklwsil/ReadVariableOp_1ReadVariableOp$kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_1
kfdgklwsil/mul_1Mul#kfdgklwsil/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_1
kfdgklwsil/add_2AddV2kfdgklwsil/split:output:1kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_2
kfdgklwsil/Sigmoid_1Sigmoidkfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_1
kfdgklwsil/mul_2Mulkfdgklwsil/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_2w
kfdgklwsil/TanhTanhkfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh
kfdgklwsil/mul_3Mulkfdgklwsil/Sigmoid:y:0kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_3
kfdgklwsil/add_3AddV2kfdgklwsil/mul_2:z:0kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_3
kfdgklwsil/ReadVariableOp_2ReadVariableOp$kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_2
kfdgklwsil/mul_4Mul#kfdgklwsil/ReadVariableOp_2:value:0kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_4
kfdgklwsil/add_4AddV2kfdgklwsil/split:output:3kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_4
kfdgklwsil/Sigmoid_2Sigmoidkfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_2v
kfdgklwsil/Tanh_1Tanhkfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh_1
kfdgklwsil/mul_5Mulkfdgklwsil/Sigmoid_2:y:0kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kfdgklwsil_matmul_readvariableop_resource+kfdgklwsil_matmul_1_readvariableop_resource*kfdgklwsil_biasadd_readvariableop_resource"kfdgklwsil_readvariableop_resource$kfdgklwsil_readvariableop_1_resource$kfdgklwsil_readvariableop_2_resource*
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
while_body_365098*
condR
while_cond_365097*Q
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
IdentityIdentitytranspose_1:y:0"^kfdgklwsil/BiasAdd/ReadVariableOp!^kfdgklwsil/MatMul/ReadVariableOp#^kfdgklwsil/MatMul_1/ReadVariableOp^kfdgklwsil/ReadVariableOp^kfdgklwsil/ReadVariableOp_1^kfdgklwsil/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!kfdgklwsil/BiasAdd/ReadVariableOp!kfdgklwsil/BiasAdd/ReadVariableOp2D
 kfdgklwsil/MatMul/ReadVariableOp kfdgklwsil/MatMul/ReadVariableOp2H
"kfdgklwsil/MatMul_1/ReadVariableOp"kfdgklwsil/MatMul_1/ReadVariableOp26
kfdgklwsil/ReadVariableOpkfdgklwsil/ReadVariableOp2:
kfdgklwsil/ReadVariableOp_1kfdgklwsil/ReadVariableOp_12:
kfdgklwsil/ReadVariableOp_2kfdgklwsil/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÿ
¿
+__inference_bdgxhkvdqy_layer_call_fn_366679

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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_3619862
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


ipeigywbwc_while_cond_3641892
.ipeigywbwc_while_ipeigywbwc_while_loop_counter8
4ipeigywbwc_while_ipeigywbwc_while_maximum_iterations 
ipeigywbwc_while_placeholder"
ipeigywbwc_while_placeholder_1"
ipeigywbwc_while_placeholder_2"
ipeigywbwc_while_placeholder_34
0ipeigywbwc_while_less_ipeigywbwc_strided_slice_1J
Fipeigywbwc_while_ipeigywbwc_while_cond_364189___redundant_placeholder0J
Fipeigywbwc_while_ipeigywbwc_while_cond_364189___redundant_placeholder1J
Fipeigywbwc_while_ipeigywbwc_while_cond_364189___redundant_placeholder2J
Fipeigywbwc_while_ipeigywbwc_while_cond_364189___redundant_placeholder3J
Fipeigywbwc_while_ipeigywbwc_while_cond_364189___redundant_placeholder4J
Fipeigywbwc_while_ipeigywbwc_while_cond_364189___redundant_placeholder5J
Fipeigywbwc_while_ipeigywbwc_while_cond_364189___redundant_placeholder6
ipeigywbwc_while_identity
§
ipeigywbwc/while/LessLessipeigywbwc_while_placeholder0ipeigywbwc_while_less_ipeigywbwc_strided_slice_1*
T0*
_output_shapes
: 2
ipeigywbwc/while/Less~
ipeigywbwc/while/IdentityIdentityipeigywbwc/while/Less:z:0*
T0
*
_output_shapes
: 2
ipeigywbwc/while/Identity"?
ipeigywbwc_while_identity"ipeigywbwc/while/Identity:output:0*(
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
while_cond_364917
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_364917___redundant_placeholder04
0while_while_cond_364917___redundant_placeholder14
0while_while_cond_364917___redundant_placeholder24
0while_while_cond_364917___redundant_placeholder34
0while_while_cond_364917___redundant_placeholder44
0while_while_cond_364917___redundant_placeholder54
0while_while_cond_364917___redundant_placeholder6
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
+__inference_zeuewnmlut_layer_call_fn_365593
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3615912
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


å
while_cond_365277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365277___redundant_placeholder04
0while_while_cond_365277___redundant_placeholder14
0while_while_cond_365277___redundant_placeholder24
0while_while_cond_365277___redundant_placeholder34
0while_while_cond_365277___redundant_placeholder44
0while_while_cond_365277___redundant_placeholder54
0while_while_cond_365277___redundant_placeholder6
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

b
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_362720

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
Û

'sequential_zeuewnmlut_while_body_360858H
Dsequential_zeuewnmlut_while_sequential_zeuewnmlut_while_loop_counterN
Jsequential_zeuewnmlut_while_sequential_zeuewnmlut_while_maximum_iterations+
'sequential_zeuewnmlut_while_placeholder-
)sequential_zeuewnmlut_while_placeholder_1-
)sequential_zeuewnmlut_while_placeholder_2-
)sequential_zeuewnmlut_while_placeholder_3G
Csequential_zeuewnmlut_while_sequential_zeuewnmlut_strided_slice_1_0
sequential_zeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_sequential_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0:	\
Isequential_zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0:	 W
Hsequential_zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0:	N
@sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0: P
Bsequential_zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0: P
Bsequential_zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0: (
$sequential_zeuewnmlut_while_identity*
&sequential_zeuewnmlut_while_identity_1*
&sequential_zeuewnmlut_while_identity_2*
&sequential_zeuewnmlut_while_identity_3*
&sequential_zeuewnmlut_while_identity_4*
&sequential_zeuewnmlut_while_identity_5E
Asequential_zeuewnmlut_while_sequential_zeuewnmlut_strided_slice_1
}sequential_zeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_sequential_zeuewnmlut_tensorarrayunstack_tensorlistfromtensorX
Esequential_zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource:	Z
Gsequential_zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource:	 U
Fsequential_zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource:	L
>sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_resource: N
@sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource: N
@sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource: ¢=sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp¢<sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp¢>sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp¢5sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp¢7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1¢7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2ï
Msequential/zeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/zeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/zeuewnmlut/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_zeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_sequential_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0'sequential_zeuewnmlut_while_placeholderVsequential/zeuewnmlut/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/zeuewnmlut/while/TensorArrayV2Read/TensorListGetItem
<sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOpReadVariableOpGsequential_zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp©
-sequential/zeuewnmlut/while/kfdgklwsil/MatMulMatMulFsequential/zeuewnmlut/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/zeuewnmlut/while/kfdgklwsil/MatMul
>sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOpIsequential_zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp
/sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1MatMul)sequential_zeuewnmlut_while_placeholder_2Fsequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1
*sequential/zeuewnmlut/while/kfdgklwsil/addAddV27sequential/zeuewnmlut/while/kfdgklwsil/MatMul:product:09sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/zeuewnmlut/while/kfdgklwsil/add
=sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOpHsequential_zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp
.sequential/zeuewnmlut/while/kfdgklwsil/BiasAddBiasAdd.sequential/zeuewnmlut/while/kfdgklwsil/add:z:0Esequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd²
6sequential/zeuewnmlut/while/kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/zeuewnmlut/while/kfdgklwsil/split/split_dimÛ
,sequential/zeuewnmlut/while/kfdgklwsil/splitSplit?sequential/zeuewnmlut/while/kfdgklwsil/split/split_dim:output:07sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/zeuewnmlut/while/kfdgklwsil/splitë
5sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOpReadVariableOp@sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOpû
*sequential/zeuewnmlut/while/kfdgklwsil/mulMul=sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp:value:0)sequential_zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/zeuewnmlut/while/kfdgklwsil/mulþ
,sequential/zeuewnmlut/while/kfdgklwsil/add_1AddV25sequential/zeuewnmlut/while/kfdgklwsil/split:output:0.sequential/zeuewnmlut/while/kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/add_1Ï
.sequential/zeuewnmlut/while/kfdgklwsil/SigmoidSigmoid0sequential/zeuewnmlut/while/kfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/zeuewnmlut/while/kfdgklwsil/Sigmoidñ
7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1ReadVariableOpBsequential_zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1
,sequential/zeuewnmlut/while/kfdgklwsil/mul_1Mul?sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_1:value:0)sequential_zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/mul_1
,sequential/zeuewnmlut/while/kfdgklwsil/add_2AddV25sequential/zeuewnmlut/while/kfdgklwsil/split:output:10sequential/zeuewnmlut/while/kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/add_2Ó
0sequential/zeuewnmlut/while/kfdgklwsil/Sigmoid_1Sigmoid0sequential/zeuewnmlut/while/kfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/zeuewnmlut/while/kfdgklwsil/Sigmoid_1ö
,sequential/zeuewnmlut/while/kfdgklwsil/mul_2Mul4sequential/zeuewnmlut/while/kfdgklwsil/Sigmoid_1:y:0)sequential_zeuewnmlut_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/mul_2Ë
+sequential/zeuewnmlut/while/kfdgklwsil/TanhTanh5sequential/zeuewnmlut/while/kfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/zeuewnmlut/while/kfdgklwsil/Tanhú
,sequential/zeuewnmlut/while/kfdgklwsil/mul_3Mul2sequential/zeuewnmlut/while/kfdgklwsil/Sigmoid:y:0/sequential/zeuewnmlut/while/kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/mul_3û
,sequential/zeuewnmlut/while/kfdgklwsil/add_3AddV20sequential/zeuewnmlut/while/kfdgklwsil/mul_2:z:00sequential/zeuewnmlut/while/kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/add_3ñ
7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2ReadVariableOpBsequential_zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2
,sequential/zeuewnmlut/while/kfdgklwsil/mul_4Mul?sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2:value:00sequential/zeuewnmlut/while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/mul_4
,sequential/zeuewnmlut/while/kfdgklwsil/add_4AddV25sequential/zeuewnmlut/while/kfdgklwsil/split:output:30sequential/zeuewnmlut/while/kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/add_4Ó
0sequential/zeuewnmlut/while/kfdgklwsil/Sigmoid_2Sigmoid0sequential/zeuewnmlut/while/kfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/zeuewnmlut/while/kfdgklwsil/Sigmoid_2Ê
-sequential/zeuewnmlut/while/kfdgklwsil/Tanh_1Tanh0sequential/zeuewnmlut/while/kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/zeuewnmlut/while/kfdgklwsil/Tanh_1þ
,sequential/zeuewnmlut/while/kfdgklwsil/mul_5Mul4sequential/zeuewnmlut/while/kfdgklwsil/Sigmoid_2:y:01sequential/zeuewnmlut/while/kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/zeuewnmlut/while/kfdgklwsil/mul_5Ì
@sequential/zeuewnmlut/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_zeuewnmlut_while_placeholder_1'sequential_zeuewnmlut_while_placeholder0sequential/zeuewnmlut/while/kfdgklwsil/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/zeuewnmlut/while/TensorArrayV2Write/TensorListSetItem
!sequential/zeuewnmlut/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/zeuewnmlut/while/add/yÁ
sequential/zeuewnmlut/while/addAddV2'sequential_zeuewnmlut_while_placeholder*sequential/zeuewnmlut/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/zeuewnmlut/while/add
#sequential/zeuewnmlut/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/zeuewnmlut/while/add_1/yä
!sequential/zeuewnmlut/while/add_1AddV2Dsequential_zeuewnmlut_while_sequential_zeuewnmlut_while_loop_counter,sequential/zeuewnmlut/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/zeuewnmlut/while/add_1
$sequential/zeuewnmlut/while/IdentityIdentity%sequential/zeuewnmlut/while/add_1:z:0>^sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp=^sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp?^sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp6^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp8^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_18^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/zeuewnmlut/while/Identityµ
&sequential/zeuewnmlut/while/Identity_1IdentityJsequential_zeuewnmlut_while_sequential_zeuewnmlut_while_maximum_iterations>^sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp=^sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp?^sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp6^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp8^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_18^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/zeuewnmlut/while/Identity_1
&sequential/zeuewnmlut/while/Identity_2Identity#sequential/zeuewnmlut/while/add:z:0>^sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp=^sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp?^sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp6^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp8^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_18^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/zeuewnmlut/while/Identity_2»
&sequential/zeuewnmlut/while/Identity_3IdentityPsequential/zeuewnmlut/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp=^sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp?^sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp6^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp8^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_18^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/zeuewnmlut/while/Identity_3¬
&sequential/zeuewnmlut/while/Identity_4Identity0sequential/zeuewnmlut/while/kfdgklwsil/mul_5:z:0>^sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp=^sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp?^sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp6^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp8^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_18^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/while/Identity_4¬
&sequential/zeuewnmlut/while/Identity_5Identity0sequential/zeuewnmlut/while/kfdgklwsil/add_3:z:0>^sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp=^sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp?^sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp6^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp8^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_18^sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/zeuewnmlut/while/Identity_5"U
$sequential_zeuewnmlut_while_identity-sequential/zeuewnmlut/while/Identity:output:0"Y
&sequential_zeuewnmlut_while_identity_1/sequential/zeuewnmlut/while/Identity_1:output:0"Y
&sequential_zeuewnmlut_while_identity_2/sequential/zeuewnmlut/while/Identity_2:output:0"Y
&sequential_zeuewnmlut_while_identity_3/sequential/zeuewnmlut/while/Identity_3:output:0"Y
&sequential_zeuewnmlut_while_identity_4/sequential/zeuewnmlut/while/Identity_4:output:0"Y
&sequential_zeuewnmlut_while_identity_5/sequential/zeuewnmlut/while/Identity_5:output:0"
Fsequential_zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resourceHsequential_zeuewnmlut_while_kfdgklwsil_biasadd_readvariableop_resource_0"
Gsequential_zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resourceIsequential_zeuewnmlut_while_kfdgklwsil_matmul_1_readvariableop_resource_0"
Esequential_zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resourceGsequential_zeuewnmlut_while_kfdgklwsil_matmul_readvariableop_resource_0"
@sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_1_resourceBsequential_zeuewnmlut_while_kfdgklwsil_readvariableop_1_resource_0"
@sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_2_resourceBsequential_zeuewnmlut_while_kfdgklwsil_readvariableop_2_resource_0"
>sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_resource@sequential_zeuewnmlut_while_kfdgklwsil_readvariableop_resource_0"
Asequential_zeuewnmlut_while_sequential_zeuewnmlut_strided_slice_1Csequential_zeuewnmlut_while_sequential_zeuewnmlut_strided_slice_1_0"
}sequential_zeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_sequential_zeuewnmlut_tensorarrayunstack_tensorlistfromtensorsequential_zeuewnmlut_while_tensorarrayv2read_tensorlistgetitem_sequential_zeuewnmlut_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp=sequential/zeuewnmlut/while/kfdgklwsil/BiasAdd/ReadVariableOp2|
<sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp<sequential/zeuewnmlut/while/kfdgklwsil/MatMul/ReadVariableOp2
>sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp>sequential/zeuewnmlut/while/kfdgklwsil/MatMul_1/ReadVariableOp2n
5sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp5sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp2r
7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_17sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_12r
7sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_27sequential/zeuewnmlut/while/kfdgklwsil/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_363369

inputs<
)bdgxhkvdqy_matmul_readvariableop_resource:	 >
+bdgxhkvdqy_matmul_1_readvariableop_resource:	 9
*bdgxhkvdqy_biasadd_readvariableop_resource:	0
"bdgxhkvdqy_readvariableop_resource: 2
$bdgxhkvdqy_readvariableop_1_resource: 2
$bdgxhkvdqy_readvariableop_2_resource: 
identity¢!bdgxhkvdqy/BiasAdd/ReadVariableOp¢ bdgxhkvdqy/MatMul/ReadVariableOp¢"bdgxhkvdqy/MatMul_1/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp_1¢bdgxhkvdqy/ReadVariableOp_2¢whileD
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
 bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp)bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 bdgxhkvdqy/MatMul/ReadVariableOp§
bdgxhkvdqy/MatMulMatMulstrided_slice_2:output:0(bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMulµ
"bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp+bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"bdgxhkvdqy/MatMul_1/ReadVariableOp£
bdgxhkvdqy/MatMul_1MatMulzeros:output:0*bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMul_1
bdgxhkvdqy/addAddV2bdgxhkvdqy/MatMul:product:0bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/add®
!bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp*bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!bdgxhkvdqy/BiasAdd/ReadVariableOp¥
bdgxhkvdqy/BiasAddBiasAddbdgxhkvdqy/add:z:0)bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/BiasAddz
bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
bdgxhkvdqy/split/split_dimë
bdgxhkvdqy/splitSplit#bdgxhkvdqy/split/split_dim:output:0bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
bdgxhkvdqy/split
bdgxhkvdqy/ReadVariableOpReadVariableOp"bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp
bdgxhkvdqy/mulMul!bdgxhkvdqy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul
bdgxhkvdqy/add_1AddV2bdgxhkvdqy/split:output:0bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_1{
bdgxhkvdqy/SigmoidSigmoidbdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid
bdgxhkvdqy/ReadVariableOp_1ReadVariableOp$bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_1
bdgxhkvdqy/mul_1Mul#bdgxhkvdqy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_1
bdgxhkvdqy/add_2AddV2bdgxhkvdqy/split:output:1bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_2
bdgxhkvdqy/Sigmoid_1Sigmoidbdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_1
bdgxhkvdqy/mul_2Mulbdgxhkvdqy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_2w
bdgxhkvdqy/TanhTanhbdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh
bdgxhkvdqy/mul_3Mulbdgxhkvdqy/Sigmoid:y:0bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_3
bdgxhkvdqy/add_3AddV2bdgxhkvdqy/mul_2:z:0bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_3
bdgxhkvdqy/ReadVariableOp_2ReadVariableOp$bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_2
bdgxhkvdqy/mul_4Mul#bdgxhkvdqy/ReadVariableOp_2:value:0bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_4
bdgxhkvdqy/add_4AddV2bdgxhkvdqy/split:output:3bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_4
bdgxhkvdqy/Sigmoid_2Sigmoidbdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_2v
bdgxhkvdqy/Tanh_1Tanhbdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh_1
bdgxhkvdqy/mul_5Mulbdgxhkvdqy/Sigmoid_2:y:0bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)bdgxhkvdqy_matmul_readvariableop_resource+bdgxhkvdqy_matmul_1_readvariableop_resource*bdgxhkvdqy_biasadd_readvariableop_resource"bdgxhkvdqy_readvariableop_resource$bdgxhkvdqy_readvariableop_1_resource$bdgxhkvdqy_readvariableop_2_resource*
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
while_body_363268*
condR
while_cond_363267*Q
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
IdentityIdentitystrided_slice_3:output:0"^bdgxhkvdqy/BiasAdd/ReadVariableOp!^bdgxhkvdqy/MatMul/ReadVariableOp#^bdgxhkvdqy/MatMul_1/ReadVariableOp^bdgxhkvdqy/ReadVariableOp^bdgxhkvdqy/ReadVariableOp_1^bdgxhkvdqy/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!bdgxhkvdqy/BiasAdd/ReadVariableOp!bdgxhkvdqy/BiasAdd/ReadVariableOp2D
 bdgxhkvdqy/MatMul/ReadVariableOp bdgxhkvdqy/MatMul/ReadVariableOp2H
"bdgxhkvdqy/MatMul_1/ReadVariableOp"bdgxhkvdqy/MatMul_1/ReadVariableOp26
bdgxhkvdqy/ReadVariableOpbdgxhkvdqy/ReadVariableOp2:
bdgxhkvdqy/ReadVariableOp_1bdgxhkvdqy/ReadVariableOp_12:
bdgxhkvdqy/ReadVariableOp_2bdgxhkvdqy/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ùh

F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_365987
inputs_0<
)bdgxhkvdqy_matmul_readvariableop_resource:	 >
+bdgxhkvdqy_matmul_1_readvariableop_resource:	 9
*bdgxhkvdqy_biasadd_readvariableop_resource:	0
"bdgxhkvdqy_readvariableop_resource: 2
$bdgxhkvdqy_readvariableop_1_resource: 2
$bdgxhkvdqy_readvariableop_2_resource: 
identity¢!bdgxhkvdqy/BiasAdd/ReadVariableOp¢ bdgxhkvdqy/MatMul/ReadVariableOp¢"bdgxhkvdqy/MatMul_1/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp_1¢bdgxhkvdqy/ReadVariableOp_2¢whileF
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
 bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp)bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 bdgxhkvdqy/MatMul/ReadVariableOp§
bdgxhkvdqy/MatMulMatMulstrided_slice_2:output:0(bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMulµ
"bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp+bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"bdgxhkvdqy/MatMul_1/ReadVariableOp£
bdgxhkvdqy/MatMul_1MatMulzeros:output:0*bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMul_1
bdgxhkvdqy/addAddV2bdgxhkvdqy/MatMul:product:0bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/add®
!bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp*bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!bdgxhkvdqy/BiasAdd/ReadVariableOp¥
bdgxhkvdqy/BiasAddBiasAddbdgxhkvdqy/add:z:0)bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/BiasAddz
bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
bdgxhkvdqy/split/split_dimë
bdgxhkvdqy/splitSplit#bdgxhkvdqy/split/split_dim:output:0bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
bdgxhkvdqy/split
bdgxhkvdqy/ReadVariableOpReadVariableOp"bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp
bdgxhkvdqy/mulMul!bdgxhkvdqy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul
bdgxhkvdqy/add_1AddV2bdgxhkvdqy/split:output:0bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_1{
bdgxhkvdqy/SigmoidSigmoidbdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid
bdgxhkvdqy/ReadVariableOp_1ReadVariableOp$bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_1
bdgxhkvdqy/mul_1Mul#bdgxhkvdqy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_1
bdgxhkvdqy/add_2AddV2bdgxhkvdqy/split:output:1bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_2
bdgxhkvdqy/Sigmoid_1Sigmoidbdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_1
bdgxhkvdqy/mul_2Mulbdgxhkvdqy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_2w
bdgxhkvdqy/TanhTanhbdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh
bdgxhkvdqy/mul_3Mulbdgxhkvdqy/Sigmoid:y:0bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_3
bdgxhkvdqy/add_3AddV2bdgxhkvdqy/mul_2:z:0bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_3
bdgxhkvdqy/ReadVariableOp_2ReadVariableOp$bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_2
bdgxhkvdqy/mul_4Mul#bdgxhkvdqy/ReadVariableOp_2:value:0bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_4
bdgxhkvdqy/add_4AddV2bdgxhkvdqy/split:output:3bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_4
bdgxhkvdqy/Sigmoid_2Sigmoidbdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_2v
bdgxhkvdqy/Tanh_1Tanhbdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh_1
bdgxhkvdqy/mul_5Mulbdgxhkvdqy/Sigmoid_2:y:0bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)bdgxhkvdqy_matmul_readvariableop_resource+bdgxhkvdqy_matmul_1_readvariableop_resource*bdgxhkvdqy_biasadd_readvariableop_resource"bdgxhkvdqy_readvariableop_resource$bdgxhkvdqy_readvariableop_1_resource$bdgxhkvdqy_readvariableop_2_resource*
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
while_body_365886*
condR
while_cond_365885*Q
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
IdentityIdentitystrided_slice_3:output:0"^bdgxhkvdqy/BiasAdd/ReadVariableOp!^bdgxhkvdqy/MatMul/ReadVariableOp#^bdgxhkvdqy/MatMul_1/ReadVariableOp^bdgxhkvdqy/ReadVariableOp^bdgxhkvdqy/ReadVariableOp_1^bdgxhkvdqy/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!bdgxhkvdqy/BiasAdd/ReadVariableOp!bdgxhkvdqy/BiasAdd/ReadVariableOp2D
 bdgxhkvdqy/MatMul/ReadVariableOp bdgxhkvdqy/MatMul/ReadVariableOp2H
"bdgxhkvdqy/MatMul_1/ReadVariableOp"bdgxhkvdqy/MatMul_1/ReadVariableOp26
bdgxhkvdqy/ReadVariableOpbdgxhkvdqy/ReadVariableOp2:
bdgxhkvdqy/ReadVariableOp_1bdgxhkvdqy/ReadVariableOp_12:
bdgxhkvdqy/ReadVariableOp_2bdgxhkvdqy/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


å
while_cond_362268
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362268___redundant_placeholder04
0while_while_cond_362268___redundant_placeholder14
0while_while_cond_362268___redundant_placeholder24
0while_while_cond_362268___redundant_placeholder34
0while_while_cond_362268___redundant_placeholder44
0while_while_cond_362268___redundant_placeholder54
0while_while_cond_362268___redundant_placeholder6
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
+__inference_xsvuntduhq_layer_call_fn_364839

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
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_3627202
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
ÿ
¿
+__inference_kfdgklwsil_layer_call_fn_366545

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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_3612282
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
while_cond_362005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_362005___redundant_placeholder04
0while_while_cond_362005___redundant_placeholder14
0while_while_cond_362005___redundant_placeholder24
0while_while_cond_362005___redundant_placeholder34
0while_while_cond_362005___redundant_placeholder44
0while_while_cond_362005___redundant_placeholder54
0while_while_cond_362005___redundant_placeholder6
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
+__inference_bdgxhkvdqy_layer_call_fn_366702

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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_3621732
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
Ç)
Å
while_body_362006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_bdgxhkvdqy_362030_0:	 ,
while_bdgxhkvdqy_362032_0:	 (
while_bdgxhkvdqy_362034_0:	'
while_bdgxhkvdqy_362036_0: '
while_bdgxhkvdqy_362038_0: '
while_bdgxhkvdqy_362040_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_bdgxhkvdqy_362030:	 *
while_bdgxhkvdqy_362032:	 &
while_bdgxhkvdqy_362034:	%
while_bdgxhkvdqy_362036: %
while_bdgxhkvdqy_362038: %
while_bdgxhkvdqy_362040: ¢(while/bdgxhkvdqy/StatefulPartitionedCallÃ
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
(while/bdgxhkvdqy/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_bdgxhkvdqy_362030_0while_bdgxhkvdqy_362032_0while_bdgxhkvdqy_362034_0while_bdgxhkvdqy_362036_0while_bdgxhkvdqy_362038_0while_bdgxhkvdqy_362040_0*
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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_3619862*
(while/bdgxhkvdqy/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/bdgxhkvdqy/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/bdgxhkvdqy/StatefulPartitionedCall:output:1)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/bdgxhkvdqy/StatefulPartitionedCall:output:2)^while/bdgxhkvdqy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"4
while_bdgxhkvdqy_362030while_bdgxhkvdqy_362030_0"4
while_bdgxhkvdqy_362032while_bdgxhkvdqy_362032_0"4
while_bdgxhkvdqy_362034while_bdgxhkvdqy_362034_0"4
while_bdgxhkvdqy_362036while_bdgxhkvdqy_362036_0"4
while_bdgxhkvdqy_362038while_bdgxhkvdqy_362038_0"4
while_bdgxhkvdqy_362040while_bdgxhkvdqy_362040_0")
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
(while/bdgxhkvdqy/StatefulPartitionedCall(while/bdgxhkvdqy/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_361510
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_361510___redundant_placeholder04
0while_while_cond_361510___redundant_placeholder14
0while_while_cond_361510___redundant_placeholder24
0while_while_cond_361510___redundant_placeholder34
0while_while_cond_361510___redundant_placeholder44
0while_while_cond_361510___redundant_placeholder54
0while_while_cond_361510___redundant_placeholder6
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
+__inference_ipeigywbwc_layer_call_fn_366415

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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3633692
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
Ýh

F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365019
inputs_0<
)kfdgklwsil_matmul_readvariableop_resource:	>
+kfdgklwsil_matmul_1_readvariableop_resource:	 9
*kfdgklwsil_biasadd_readvariableop_resource:	0
"kfdgklwsil_readvariableop_resource: 2
$kfdgklwsil_readvariableop_1_resource: 2
$kfdgklwsil_readvariableop_2_resource: 
identity¢!kfdgklwsil/BiasAdd/ReadVariableOp¢ kfdgklwsil/MatMul/ReadVariableOp¢"kfdgklwsil/MatMul_1/ReadVariableOp¢kfdgklwsil/ReadVariableOp¢kfdgklwsil/ReadVariableOp_1¢kfdgklwsil/ReadVariableOp_2¢whileF
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
 kfdgklwsil/MatMul/ReadVariableOpReadVariableOp)kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 kfdgklwsil/MatMul/ReadVariableOp§
kfdgklwsil/MatMulMatMulstrided_slice_2:output:0(kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMulµ
"kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp+kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kfdgklwsil/MatMul_1/ReadVariableOp£
kfdgklwsil/MatMul_1MatMulzeros:output:0*kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMul_1
kfdgklwsil/addAddV2kfdgklwsil/MatMul:product:0kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/add®
!kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp*kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kfdgklwsil/BiasAdd/ReadVariableOp¥
kfdgklwsil/BiasAddBiasAddkfdgklwsil/add:z:0)kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/BiasAddz
kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kfdgklwsil/split/split_dimë
kfdgklwsil/splitSplit#kfdgklwsil/split/split_dim:output:0kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kfdgklwsil/split
kfdgklwsil/ReadVariableOpReadVariableOp"kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp
kfdgklwsil/mulMul!kfdgklwsil/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul
kfdgklwsil/add_1AddV2kfdgklwsil/split:output:0kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_1{
kfdgklwsil/SigmoidSigmoidkfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid
kfdgklwsil/ReadVariableOp_1ReadVariableOp$kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_1
kfdgklwsil/mul_1Mul#kfdgklwsil/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_1
kfdgklwsil/add_2AddV2kfdgklwsil/split:output:1kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_2
kfdgklwsil/Sigmoid_1Sigmoidkfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_1
kfdgklwsil/mul_2Mulkfdgklwsil/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_2w
kfdgklwsil/TanhTanhkfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh
kfdgklwsil/mul_3Mulkfdgklwsil/Sigmoid:y:0kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_3
kfdgklwsil/add_3AddV2kfdgklwsil/mul_2:z:0kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_3
kfdgklwsil/ReadVariableOp_2ReadVariableOp$kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_2
kfdgklwsil/mul_4Mul#kfdgklwsil/ReadVariableOp_2:value:0kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_4
kfdgklwsil/add_4AddV2kfdgklwsil/split:output:3kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_4
kfdgklwsil/Sigmoid_2Sigmoidkfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_2v
kfdgklwsil/Tanh_1Tanhkfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh_1
kfdgklwsil/mul_5Mulkfdgklwsil/Sigmoid_2:y:0kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kfdgklwsil_matmul_readvariableop_resource+kfdgklwsil_matmul_1_readvariableop_resource*kfdgklwsil_biasadd_readvariableop_resource"kfdgklwsil_readvariableop_resource$kfdgklwsil_readvariableop_1_resource$kfdgklwsil_readvariableop_2_resource*
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
while_body_364918*
condR
while_cond_364917*Q
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
IdentityIdentitytranspose_1:y:0"^kfdgklwsil/BiasAdd/ReadVariableOp!^kfdgklwsil/MatMul/ReadVariableOp#^kfdgklwsil/MatMul_1/ReadVariableOp^kfdgklwsil/ReadVariableOp^kfdgklwsil/ReadVariableOp_1^kfdgklwsil/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!kfdgklwsil/BiasAdd/ReadVariableOp!kfdgklwsil/BiasAdd/ReadVariableOp2D
 kfdgklwsil/MatMul/ReadVariableOp kfdgklwsil/MatMul/ReadVariableOp2H
"kfdgklwsil/MatMul_1/ReadVariableOp"kfdgklwsil/MatMul_1/ReadVariableOp26
kfdgklwsil/ReadVariableOpkfdgklwsil/ReadVariableOp2:
kfdgklwsil/ReadVariableOp_1kfdgklwsil/ReadVariableOp_12:
kfdgklwsil/ReadVariableOp_2kfdgklwsil/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
°'
²
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_361986

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


ipeigywbwc_while_cond_3645932
.ipeigywbwc_while_ipeigywbwc_while_loop_counter8
4ipeigywbwc_while_ipeigywbwc_while_maximum_iterations 
ipeigywbwc_while_placeholder"
ipeigywbwc_while_placeholder_1"
ipeigywbwc_while_placeholder_2"
ipeigywbwc_while_placeholder_34
0ipeigywbwc_while_less_ipeigywbwc_strided_slice_1J
Fipeigywbwc_while_ipeigywbwc_while_cond_364593___redundant_placeholder0J
Fipeigywbwc_while_ipeigywbwc_while_cond_364593___redundant_placeholder1J
Fipeigywbwc_while_ipeigywbwc_while_cond_364593___redundant_placeholder2J
Fipeigywbwc_while_ipeigywbwc_while_cond_364593___redundant_placeholder3J
Fipeigywbwc_while_ipeigywbwc_while_cond_364593___redundant_placeholder4J
Fipeigywbwc_while_ipeigywbwc_while_cond_364593___redundant_placeholder5J
Fipeigywbwc_while_ipeigywbwc_while_cond_364593___redundant_placeholder6
ipeigywbwc_while_identity
§
ipeigywbwc/while/LessLessipeigywbwc_while_placeholder0ipeigywbwc_while_less_ipeigywbwc_strided_slice_1*
T0*
_output_shapes
: 2
ipeigywbwc/while/Less~
ipeigywbwc/while/IdentityIdentityipeigywbwc/while/Less:z:0*
T0
*
_output_shapes
: 2
ipeigywbwc/while/Identity"?
ipeigywbwc_while_identity"ipeigywbwc/while/Identity:output:0*(
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_361328

inputs$
kfdgklwsil_361229:	$
kfdgklwsil_361231:	  
kfdgklwsil_361233:	
kfdgklwsil_361235: 
kfdgklwsil_361237: 
kfdgklwsil_361239: 
identity¢"kfdgklwsil/StatefulPartitionedCall¢whileD
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
"kfdgklwsil/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0kfdgklwsil_361229kfdgklwsil_361231kfdgklwsil_361233kfdgklwsil_361235kfdgklwsil_361237kfdgklwsil_361239*
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_3612282$
"kfdgklwsil/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kfdgklwsil_361229kfdgklwsil_361231kfdgklwsil_361233kfdgklwsil_361235kfdgklwsil_361237kfdgklwsil_361239*
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
while_body_361248*
condR
while_cond_361247*Q
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
IdentityIdentitytranspose_1:y:0#^kfdgklwsil/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"kfdgklwsil/StatefulPartitionedCall"kfdgklwsil/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

«
F__inference_sequential_layer_call_and_return_conditional_losses_363125

inputs'
ekzorghjta_362702:
ekzorghjta_362704:$
zeuewnmlut_362902:	$
zeuewnmlut_362904:	  
zeuewnmlut_362906:	
zeuewnmlut_362908: 
zeuewnmlut_362910: 
zeuewnmlut_362912: $
ipeigywbwc_363095:	 $
ipeigywbwc_363097:	  
ipeigywbwc_363099:	
ipeigywbwc_363101: 
ipeigywbwc_363103: 
ipeigywbwc_363105: #
pemnqztknd_363119: 
pemnqztknd_363121:
identity¢"ekzorghjta/StatefulPartitionedCall¢"ipeigywbwc/StatefulPartitionedCall¢"pemnqztknd/StatefulPartitionedCall¢"zeuewnmlut/StatefulPartitionedCall©
"ekzorghjta/StatefulPartitionedCallStatefulPartitionedCallinputsekzorghjta_362702ekzorghjta_362704*
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
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_3627012$
"ekzorghjta/StatefulPartitionedCall
xsvuntduhq/PartitionedCallPartitionedCall+ekzorghjta/StatefulPartitionedCall:output:0*
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
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_3627202
xsvuntduhq/PartitionedCall
"zeuewnmlut/StatefulPartitionedCallStatefulPartitionedCall#xsvuntduhq/PartitionedCall:output:0zeuewnmlut_362902zeuewnmlut_362904zeuewnmlut_362906zeuewnmlut_362908zeuewnmlut_362910zeuewnmlut_362912*
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3629012$
"zeuewnmlut/StatefulPartitionedCall
"ipeigywbwc/StatefulPartitionedCallStatefulPartitionedCall+zeuewnmlut/StatefulPartitionedCall:output:0ipeigywbwc_363095ipeigywbwc_363097ipeigywbwc_363099ipeigywbwc_363101ipeigywbwc_363103ipeigywbwc_363105*
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3630942$
"ipeigywbwc/StatefulPartitionedCallÆ
"pemnqztknd/StatefulPartitionedCallStatefulPartitionedCall+ipeigywbwc/StatefulPartitionedCall:output:0pemnqztknd_363119pemnqztknd_363121*
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
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_3631182$
"pemnqztknd/StatefulPartitionedCall
IdentityIdentity+pemnqztknd/StatefulPartitionedCall:output:0#^ekzorghjta/StatefulPartitionedCall#^ipeigywbwc/StatefulPartitionedCall#^pemnqztknd/StatefulPartitionedCall#^zeuewnmlut/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ekzorghjta/StatefulPartitionedCall"ekzorghjta/StatefulPartitionedCall2H
"ipeigywbwc/StatefulPartitionedCall"ipeigywbwc/StatefulPartitionedCall2H
"pemnqztknd/StatefulPartitionedCall"pemnqztknd/StatefulPartitionedCall2H
"zeuewnmlut/StatefulPartitionedCall"zeuewnmlut/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
F
ã
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_361591

inputs$
kfdgklwsil_361492:	$
kfdgklwsil_361494:	  
kfdgklwsil_361496:	
kfdgklwsil_361498: 
kfdgklwsil_361500: 
kfdgklwsil_361502: 
identity¢"kfdgklwsil/StatefulPartitionedCall¢whileD
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
"kfdgklwsil/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0kfdgklwsil_361492kfdgklwsil_361494kfdgklwsil_361496kfdgklwsil_361498kfdgklwsil_361500kfdgklwsil_361502*
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_3614152$
"kfdgklwsil/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kfdgklwsil_361492kfdgklwsil_361494kfdgklwsil_361496kfdgklwsil_361498kfdgklwsil_361500kfdgklwsil_361502*
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
while_body_361511*
condR
while_cond_361510*Q
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
IdentityIdentitytranspose_1:y:0#^kfdgklwsil/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"kfdgklwsil/StatefulPartitionedCall"kfdgklwsil/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

+__inference_zeuewnmlut_layer_call_fn_365576
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3613282
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
é

+__inference_ipeigywbwc_layer_call_fn_366381
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3623492
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


å
while_cond_366065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_366065___redundant_placeholder04
0while_while_cond_366065___redundant_placeholder14
0while_while_cond_366065___redundant_placeholder24
0while_while_cond_366065___redundant_placeholder34
0while_while_cond_366065___redundant_placeholder44
0while_while_cond_366065___redundant_placeholder54
0while_while_cond_366065___redundant_placeholder6
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
while_body_365886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 F
3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 A
2while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	8
*while_bdgxhkvdqy_readvariableop_resource_0: :
,while_bdgxhkvdqy_readvariableop_1_resource_0: :
,while_bdgxhkvdqy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_bdgxhkvdqy_matmul_readvariableop_resource:	 D
1while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 ?
0while_bdgxhkvdqy_biasadd_readvariableop_resource:	6
(while_bdgxhkvdqy_readvariableop_resource: 8
*while_bdgxhkvdqy_readvariableop_1_resource: 8
*while_bdgxhkvdqy_readvariableop_2_resource: ¢'while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢&while/bdgxhkvdqy/MatMul/ReadVariableOp¢(while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢while/bdgxhkvdqy/ReadVariableOp¢!while/bdgxhkvdqy/ReadVariableOp_1¢!while/bdgxhkvdqy/ReadVariableOp_2Ã
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
&while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp1while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/bdgxhkvdqy/MatMul/ReadVariableOpÑ
while/bdgxhkvdqy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMulÉ
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpº
while/bdgxhkvdqy/MatMul_1MatMulwhile_placeholder_20while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMul_1°
while/bdgxhkvdqy/addAddV2!while/bdgxhkvdqy/MatMul:product:0#while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/addÂ
'while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp2while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp½
while/bdgxhkvdqy/BiasAddBiasAddwhile/bdgxhkvdqy/add:z:0/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/BiasAdd
 while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/bdgxhkvdqy/split/split_dim
while/bdgxhkvdqy/splitSplit)while/bdgxhkvdqy/split/split_dim:output:0!while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/bdgxhkvdqy/split©
while/bdgxhkvdqy/ReadVariableOpReadVariableOp*while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/bdgxhkvdqy/ReadVariableOp£
while/bdgxhkvdqy/mulMul'while/bdgxhkvdqy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul¦
while/bdgxhkvdqy/add_1AddV2while/bdgxhkvdqy/split:output:0while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_1
while/bdgxhkvdqy/SigmoidSigmoidwhile/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid¯
!while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp,while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_1©
while/bdgxhkvdqy/mul_1Mul)while/bdgxhkvdqy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_1¨
while/bdgxhkvdqy/add_2AddV2while/bdgxhkvdqy/split:output:1while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_2
while/bdgxhkvdqy/Sigmoid_1Sigmoidwhile/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_1
while/bdgxhkvdqy/mul_2Mulwhile/bdgxhkvdqy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_2
while/bdgxhkvdqy/TanhTanhwhile/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh¢
while/bdgxhkvdqy/mul_3Mulwhile/bdgxhkvdqy/Sigmoid:y:0while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_3£
while/bdgxhkvdqy/add_3AddV2while/bdgxhkvdqy/mul_2:z:0while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_3¯
!while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp,while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_2°
while/bdgxhkvdqy/mul_4Mul)while/bdgxhkvdqy/ReadVariableOp_2:value:0while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_4¨
while/bdgxhkvdqy/add_4AddV2while/bdgxhkvdqy/split:output:3while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_4
while/bdgxhkvdqy/Sigmoid_2Sigmoidwhile/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_2
while/bdgxhkvdqy/Tanh_1Tanhwhile/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh_1¦
while/bdgxhkvdqy/mul_5Mulwhile/bdgxhkvdqy/Sigmoid_2:y:0while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/bdgxhkvdqy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/bdgxhkvdqy/mul_5:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/bdgxhkvdqy/add_3:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_bdgxhkvdqy_biasadd_readvariableop_resource2while_bdgxhkvdqy_biasadd_readvariableop_resource_0"h
1while_bdgxhkvdqy_matmul_1_readvariableop_resource3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"d
/while_bdgxhkvdqy_matmul_readvariableop_resource1while_bdgxhkvdqy_matmul_readvariableop_resource_0"Z
*while_bdgxhkvdqy_readvariableop_1_resource,while_bdgxhkvdqy_readvariableop_1_resource_0"Z
*while_bdgxhkvdqy_readvariableop_2_resource,while_bdgxhkvdqy_readvariableop_2_resource_0"V
(while_bdgxhkvdqy_readvariableop_resource*while_bdgxhkvdqy_readvariableop_resource_0")
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
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp'while/bdgxhkvdqy/BiasAdd/ReadVariableOp2P
&while/bdgxhkvdqy/MatMul/ReadVariableOp&while/bdgxhkvdqy/MatMul/ReadVariableOp2T
(while/bdgxhkvdqy/MatMul_1/ReadVariableOp(while/bdgxhkvdqy/MatMul_1/ReadVariableOp2B
while/bdgxhkvdqy/ReadVariableOpwhile/bdgxhkvdqy/ReadVariableOp2F
!while/bdgxhkvdqy/ReadVariableOp_1!while/bdgxhkvdqy/ReadVariableOp_12F
!while/bdgxhkvdqy/ReadVariableOp_2!while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_365457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_365457___redundant_placeholder04
0while_while_cond_365457___redundant_placeholder14
0while_while_cond_365457___redundant_placeholder24
0while_while_cond_365457___redundant_placeholder34
0while_while_cond_365457___redundant_placeholder44
0while_while_cond_365457___redundant_placeholder54
0while_while_cond_365457___redundant_placeholder6
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
+__inference_kfdgklwsil_layer_call_fn_366568

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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_3614152
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
£h

F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_366347

inputs<
)bdgxhkvdqy_matmul_readvariableop_resource:	 >
+bdgxhkvdqy_matmul_1_readvariableop_resource:	 9
*bdgxhkvdqy_biasadd_readvariableop_resource:	0
"bdgxhkvdqy_readvariableop_resource: 2
$bdgxhkvdqy_readvariableop_1_resource: 2
$bdgxhkvdqy_readvariableop_2_resource: 
identity¢!bdgxhkvdqy/BiasAdd/ReadVariableOp¢ bdgxhkvdqy/MatMul/ReadVariableOp¢"bdgxhkvdqy/MatMul_1/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp¢bdgxhkvdqy/ReadVariableOp_1¢bdgxhkvdqy/ReadVariableOp_2¢whileD
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
 bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp)bdgxhkvdqy_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 bdgxhkvdqy/MatMul/ReadVariableOp§
bdgxhkvdqy/MatMulMatMulstrided_slice_2:output:0(bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMulµ
"bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp+bdgxhkvdqy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"bdgxhkvdqy/MatMul_1/ReadVariableOp£
bdgxhkvdqy/MatMul_1MatMulzeros:output:0*bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/MatMul_1
bdgxhkvdqy/addAddV2bdgxhkvdqy/MatMul:product:0bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/add®
!bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp*bdgxhkvdqy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!bdgxhkvdqy/BiasAdd/ReadVariableOp¥
bdgxhkvdqy/BiasAddBiasAddbdgxhkvdqy/add:z:0)bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bdgxhkvdqy/BiasAddz
bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
bdgxhkvdqy/split/split_dimë
bdgxhkvdqy/splitSplit#bdgxhkvdqy/split/split_dim:output:0bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
bdgxhkvdqy/split
bdgxhkvdqy/ReadVariableOpReadVariableOp"bdgxhkvdqy_readvariableop_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp
bdgxhkvdqy/mulMul!bdgxhkvdqy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul
bdgxhkvdqy/add_1AddV2bdgxhkvdqy/split:output:0bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_1{
bdgxhkvdqy/SigmoidSigmoidbdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid
bdgxhkvdqy/ReadVariableOp_1ReadVariableOp$bdgxhkvdqy_readvariableop_1_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_1
bdgxhkvdqy/mul_1Mul#bdgxhkvdqy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_1
bdgxhkvdqy/add_2AddV2bdgxhkvdqy/split:output:1bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_2
bdgxhkvdqy/Sigmoid_1Sigmoidbdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_1
bdgxhkvdqy/mul_2Mulbdgxhkvdqy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_2w
bdgxhkvdqy/TanhTanhbdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh
bdgxhkvdqy/mul_3Mulbdgxhkvdqy/Sigmoid:y:0bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_3
bdgxhkvdqy/add_3AddV2bdgxhkvdqy/mul_2:z:0bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_3
bdgxhkvdqy/ReadVariableOp_2ReadVariableOp$bdgxhkvdqy_readvariableop_2_resource*
_output_shapes
: *
dtype02
bdgxhkvdqy/ReadVariableOp_2
bdgxhkvdqy/mul_4Mul#bdgxhkvdqy/ReadVariableOp_2:value:0bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_4
bdgxhkvdqy/add_4AddV2bdgxhkvdqy/split:output:3bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/add_4
bdgxhkvdqy/Sigmoid_2Sigmoidbdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Sigmoid_2v
bdgxhkvdqy/Tanh_1Tanhbdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/Tanh_1
bdgxhkvdqy/mul_5Mulbdgxhkvdqy/Sigmoid_2:y:0bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
bdgxhkvdqy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)bdgxhkvdqy_matmul_readvariableop_resource+bdgxhkvdqy_matmul_1_readvariableop_resource*bdgxhkvdqy_biasadd_readvariableop_resource"bdgxhkvdqy_readvariableop_resource$bdgxhkvdqy_readvariableop_1_resource$bdgxhkvdqy_readvariableop_2_resource*
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
while_body_366246*
condR
while_cond_366245*Q
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
IdentityIdentitystrided_slice_3:output:0"^bdgxhkvdqy/BiasAdd/ReadVariableOp!^bdgxhkvdqy/MatMul/ReadVariableOp#^bdgxhkvdqy/MatMul_1/ReadVariableOp^bdgxhkvdqy/ReadVariableOp^bdgxhkvdqy/ReadVariableOp_1^bdgxhkvdqy/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!bdgxhkvdqy/BiasAdd/ReadVariableOp!bdgxhkvdqy/BiasAdd/ReadVariableOp2D
 bdgxhkvdqy/MatMul/ReadVariableOp bdgxhkvdqy/MatMul/ReadVariableOp2H
"bdgxhkvdqy/MatMul_1/ReadVariableOp"bdgxhkvdqy/MatMul_1/ReadVariableOp26
bdgxhkvdqy/ReadVariableOpbdgxhkvdqy/ReadVariableOp2:
bdgxhkvdqy/ReadVariableOp_1bdgxhkvdqy/ReadVariableOp_12:
bdgxhkvdqy/ReadVariableOp_2bdgxhkvdqy/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


å
while_cond_363481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_363481___redundant_placeholder04
0while_while_cond_363481___redundant_placeholder14
0while_while_cond_363481___redundant_placeholder24
0while_while_cond_363481___redundant_placeholder34
0while_while_cond_363481___redundant_placeholder44
0while_while_cond_363481___redundant_placeholder54
0while_while_cond_363481___redundant_placeholder6
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_366478

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
ê

$__inference_signature_wrapper_363893

kjggqknufb
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
kjggqknufbunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_3611412
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
kjggqknufb

¡	
'sequential_ipeigywbwc_while_cond_361033H
Dsequential_ipeigywbwc_while_sequential_ipeigywbwc_while_loop_counterN
Jsequential_ipeigywbwc_while_sequential_ipeigywbwc_while_maximum_iterations+
'sequential_ipeigywbwc_while_placeholder-
)sequential_ipeigywbwc_while_placeholder_1-
)sequential_ipeigywbwc_while_placeholder_2-
)sequential_ipeigywbwc_while_placeholder_3J
Fsequential_ipeigywbwc_while_less_sequential_ipeigywbwc_strided_slice_1`
\sequential_ipeigywbwc_while_sequential_ipeigywbwc_while_cond_361033___redundant_placeholder0`
\sequential_ipeigywbwc_while_sequential_ipeigywbwc_while_cond_361033___redundant_placeholder1`
\sequential_ipeigywbwc_while_sequential_ipeigywbwc_while_cond_361033___redundant_placeholder2`
\sequential_ipeigywbwc_while_sequential_ipeigywbwc_while_cond_361033___redundant_placeholder3`
\sequential_ipeigywbwc_while_sequential_ipeigywbwc_while_cond_361033___redundant_placeholder4`
\sequential_ipeigywbwc_while_sequential_ipeigywbwc_while_cond_361033___redundant_placeholder5`
\sequential_ipeigywbwc_while_sequential_ipeigywbwc_while_cond_361033___redundant_placeholder6(
$sequential_ipeigywbwc_while_identity
Þ
 sequential/ipeigywbwc/while/LessLess'sequential_ipeigywbwc_while_placeholderFsequential_ipeigywbwc_while_less_sequential_ipeigywbwc_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/ipeigywbwc/while/Less
$sequential/ipeigywbwc/while/IdentityIdentity$sequential/ipeigywbwc/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/ipeigywbwc/while/Identity"U
$sequential_ipeigywbwc_while_identity-sequential/ipeigywbwc/while/Identity:output:0*(
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
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_364812

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
ßY

while_body_365706
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 F
3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 A
2while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	8
*while_bdgxhkvdqy_readvariableop_resource_0: :
,while_bdgxhkvdqy_readvariableop_1_resource_0: :
,while_bdgxhkvdqy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_bdgxhkvdqy_matmul_readvariableop_resource:	 D
1while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 ?
0while_bdgxhkvdqy_biasadd_readvariableop_resource:	6
(while_bdgxhkvdqy_readvariableop_resource: 8
*while_bdgxhkvdqy_readvariableop_1_resource: 8
*while_bdgxhkvdqy_readvariableop_2_resource: ¢'while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢&while/bdgxhkvdqy/MatMul/ReadVariableOp¢(while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢while/bdgxhkvdqy/ReadVariableOp¢!while/bdgxhkvdqy/ReadVariableOp_1¢!while/bdgxhkvdqy/ReadVariableOp_2Ã
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
&while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp1while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/bdgxhkvdqy/MatMul/ReadVariableOpÑ
while/bdgxhkvdqy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMulÉ
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpº
while/bdgxhkvdqy/MatMul_1MatMulwhile_placeholder_20while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMul_1°
while/bdgxhkvdqy/addAddV2!while/bdgxhkvdqy/MatMul:product:0#while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/addÂ
'while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp2while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp½
while/bdgxhkvdqy/BiasAddBiasAddwhile/bdgxhkvdqy/add:z:0/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/BiasAdd
 while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/bdgxhkvdqy/split/split_dim
while/bdgxhkvdqy/splitSplit)while/bdgxhkvdqy/split/split_dim:output:0!while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/bdgxhkvdqy/split©
while/bdgxhkvdqy/ReadVariableOpReadVariableOp*while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/bdgxhkvdqy/ReadVariableOp£
while/bdgxhkvdqy/mulMul'while/bdgxhkvdqy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul¦
while/bdgxhkvdqy/add_1AddV2while/bdgxhkvdqy/split:output:0while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_1
while/bdgxhkvdqy/SigmoidSigmoidwhile/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid¯
!while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp,while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_1©
while/bdgxhkvdqy/mul_1Mul)while/bdgxhkvdqy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_1¨
while/bdgxhkvdqy/add_2AddV2while/bdgxhkvdqy/split:output:1while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_2
while/bdgxhkvdqy/Sigmoid_1Sigmoidwhile/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_1
while/bdgxhkvdqy/mul_2Mulwhile/bdgxhkvdqy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_2
while/bdgxhkvdqy/TanhTanhwhile/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh¢
while/bdgxhkvdqy/mul_3Mulwhile/bdgxhkvdqy/Sigmoid:y:0while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_3£
while/bdgxhkvdqy/add_3AddV2while/bdgxhkvdqy/mul_2:z:0while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_3¯
!while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp,while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_2°
while/bdgxhkvdqy/mul_4Mul)while/bdgxhkvdqy/ReadVariableOp_2:value:0while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_4¨
while/bdgxhkvdqy/add_4AddV2while/bdgxhkvdqy/split:output:3while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_4
while/bdgxhkvdqy/Sigmoid_2Sigmoidwhile/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_2
while/bdgxhkvdqy/Tanh_1Tanhwhile/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh_1¦
while/bdgxhkvdqy/mul_5Mulwhile/bdgxhkvdqy/Sigmoid_2:y:0while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/bdgxhkvdqy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/bdgxhkvdqy/mul_5:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/bdgxhkvdqy/add_3:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_bdgxhkvdqy_biasadd_readvariableop_resource2while_bdgxhkvdqy_biasadd_readvariableop_resource_0"h
1while_bdgxhkvdqy_matmul_1_readvariableop_resource3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"d
/while_bdgxhkvdqy_matmul_readvariableop_resource1while_bdgxhkvdqy_matmul_readvariableop_resource_0"Z
*while_bdgxhkvdqy_readvariableop_1_resource,while_bdgxhkvdqy_readvariableop_1_resource_0"Z
*while_bdgxhkvdqy_readvariableop_2_resource,while_bdgxhkvdqy_readvariableop_2_resource_0"V
(while_bdgxhkvdqy_readvariableop_resource*while_bdgxhkvdqy_readvariableop_resource_0")
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
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp'while/bdgxhkvdqy/BiasAdd/ReadVariableOp2P
&while/bdgxhkvdqy/MatMul/ReadVariableOp&while/bdgxhkvdqy/MatMul/ReadVariableOp2T
(while/bdgxhkvdqy/MatMul_1/ReadVariableOp(while/bdgxhkvdqy/MatMul_1/ReadVariableOp2B
while/bdgxhkvdqy/ReadVariableOpwhile/bdgxhkvdqy/ReadVariableOp2F
!while/bdgxhkvdqy/ReadVariableOp_1!while/bdgxhkvdqy/ReadVariableOp_12F
!while/bdgxhkvdqy/ReadVariableOp_2!while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_364834

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

«
F__inference_sequential_layer_call_and_return_conditional_losses_363694

inputs'
ekzorghjta_363656:
ekzorghjta_363658:$
zeuewnmlut_363662:	$
zeuewnmlut_363664:	  
zeuewnmlut_363666:	
zeuewnmlut_363668: 
zeuewnmlut_363670: 
zeuewnmlut_363672: $
ipeigywbwc_363675:	 $
ipeigywbwc_363677:	  
ipeigywbwc_363679:	
ipeigywbwc_363681: 
ipeigywbwc_363683: 
ipeigywbwc_363685: #
pemnqztknd_363688: 
pemnqztknd_363690:
identity¢"ekzorghjta/StatefulPartitionedCall¢"ipeigywbwc/StatefulPartitionedCall¢"pemnqztknd/StatefulPartitionedCall¢"zeuewnmlut/StatefulPartitionedCall©
"ekzorghjta/StatefulPartitionedCallStatefulPartitionedCallinputsekzorghjta_363656ekzorghjta_363658*
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
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_3627012$
"ekzorghjta/StatefulPartitionedCall
xsvuntduhq/PartitionedCallPartitionedCall+ekzorghjta/StatefulPartitionedCall:output:0*
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
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_3627202
xsvuntduhq/PartitionedCall
"zeuewnmlut/StatefulPartitionedCallStatefulPartitionedCall#xsvuntduhq/PartitionedCall:output:0zeuewnmlut_363662zeuewnmlut_363664zeuewnmlut_363666zeuewnmlut_363668zeuewnmlut_363670zeuewnmlut_363672*
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3635832$
"zeuewnmlut/StatefulPartitionedCall
"ipeigywbwc/StatefulPartitionedCallStatefulPartitionedCall+zeuewnmlut/StatefulPartitionedCall:output:0ipeigywbwc_363675ipeigywbwc_363677ipeigywbwc_363679ipeigywbwc_363681ipeigywbwc_363683ipeigywbwc_363685*
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_3633692$
"ipeigywbwc/StatefulPartitionedCallÆ
"pemnqztknd/StatefulPartitionedCallStatefulPartitionedCall+ipeigywbwc/StatefulPartitionedCall:output:0pemnqztknd_363688pemnqztknd_363690*
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
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_3631182$
"pemnqztknd/StatefulPartitionedCall
IdentityIdentity+pemnqztknd/StatefulPartitionedCall:output:0#^ekzorghjta/StatefulPartitionedCall#^ipeigywbwc/StatefulPartitionedCall#^pemnqztknd/StatefulPartitionedCall#^zeuewnmlut/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"ekzorghjta/StatefulPartitionedCall"ekzorghjta/StatefulPartitionedCall2H
"ipeigywbwc/StatefulPartitionedCall"ipeigywbwc/StatefulPartitionedCall2H
"pemnqztknd/StatefulPartitionedCall"pemnqztknd/StatefulPartitionedCall2H
"zeuewnmlut/StatefulPartitionedCall"zeuewnmlut/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_sequential_layer_call_fn_364775

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
F__inference_sequential_layer_call_and_return_conditional_losses_3636942
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
while_body_362993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_bdgxhkvdqy_matmul_readvariableop_resource_0:	 F
3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0:	 A
2while_bdgxhkvdqy_biasadd_readvariableop_resource_0:	8
*while_bdgxhkvdqy_readvariableop_resource_0: :
,while_bdgxhkvdqy_readvariableop_1_resource_0: :
,while_bdgxhkvdqy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_bdgxhkvdqy_matmul_readvariableop_resource:	 D
1while_bdgxhkvdqy_matmul_1_readvariableop_resource:	 ?
0while_bdgxhkvdqy_biasadd_readvariableop_resource:	6
(while_bdgxhkvdqy_readvariableop_resource: 8
*while_bdgxhkvdqy_readvariableop_1_resource: 8
*while_bdgxhkvdqy_readvariableop_2_resource: ¢'while/bdgxhkvdqy/BiasAdd/ReadVariableOp¢&while/bdgxhkvdqy/MatMul/ReadVariableOp¢(while/bdgxhkvdqy/MatMul_1/ReadVariableOp¢while/bdgxhkvdqy/ReadVariableOp¢!while/bdgxhkvdqy/ReadVariableOp_1¢!while/bdgxhkvdqy/ReadVariableOp_2Ã
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
&while/bdgxhkvdqy/MatMul/ReadVariableOpReadVariableOp1while_bdgxhkvdqy_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/bdgxhkvdqy/MatMul/ReadVariableOpÑ
while/bdgxhkvdqy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/bdgxhkvdqy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMulÉ
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpReadVariableOp3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/bdgxhkvdqy/MatMul_1/ReadVariableOpº
while/bdgxhkvdqy/MatMul_1MatMulwhile_placeholder_20while/bdgxhkvdqy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/MatMul_1°
while/bdgxhkvdqy/addAddV2!while/bdgxhkvdqy/MatMul:product:0#while/bdgxhkvdqy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/addÂ
'while/bdgxhkvdqy/BiasAdd/ReadVariableOpReadVariableOp2while_bdgxhkvdqy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp½
while/bdgxhkvdqy/BiasAddBiasAddwhile/bdgxhkvdqy/add:z:0/while/bdgxhkvdqy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/bdgxhkvdqy/BiasAdd
 while/bdgxhkvdqy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/bdgxhkvdqy/split/split_dim
while/bdgxhkvdqy/splitSplit)while/bdgxhkvdqy/split/split_dim:output:0!while/bdgxhkvdqy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/bdgxhkvdqy/split©
while/bdgxhkvdqy/ReadVariableOpReadVariableOp*while_bdgxhkvdqy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/bdgxhkvdqy/ReadVariableOp£
while/bdgxhkvdqy/mulMul'while/bdgxhkvdqy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul¦
while/bdgxhkvdqy/add_1AddV2while/bdgxhkvdqy/split:output:0while/bdgxhkvdqy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_1
while/bdgxhkvdqy/SigmoidSigmoidwhile/bdgxhkvdqy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid¯
!while/bdgxhkvdqy/ReadVariableOp_1ReadVariableOp,while_bdgxhkvdqy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_1©
while/bdgxhkvdqy/mul_1Mul)while/bdgxhkvdqy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_1¨
while/bdgxhkvdqy/add_2AddV2while/bdgxhkvdqy/split:output:1while/bdgxhkvdqy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_2
while/bdgxhkvdqy/Sigmoid_1Sigmoidwhile/bdgxhkvdqy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_1
while/bdgxhkvdqy/mul_2Mulwhile/bdgxhkvdqy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_2
while/bdgxhkvdqy/TanhTanhwhile/bdgxhkvdqy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh¢
while/bdgxhkvdqy/mul_3Mulwhile/bdgxhkvdqy/Sigmoid:y:0while/bdgxhkvdqy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_3£
while/bdgxhkvdqy/add_3AddV2while/bdgxhkvdqy/mul_2:z:0while/bdgxhkvdqy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_3¯
!while/bdgxhkvdqy/ReadVariableOp_2ReadVariableOp,while_bdgxhkvdqy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/bdgxhkvdqy/ReadVariableOp_2°
while/bdgxhkvdqy/mul_4Mul)while/bdgxhkvdqy/ReadVariableOp_2:value:0while/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_4¨
while/bdgxhkvdqy/add_4AddV2while/bdgxhkvdqy/split:output:3while/bdgxhkvdqy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/add_4
while/bdgxhkvdqy/Sigmoid_2Sigmoidwhile/bdgxhkvdqy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Sigmoid_2
while/bdgxhkvdqy/Tanh_1Tanhwhile/bdgxhkvdqy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/Tanh_1¦
while/bdgxhkvdqy/mul_5Mulwhile/bdgxhkvdqy/Sigmoid_2:y:0while/bdgxhkvdqy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/bdgxhkvdqy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/bdgxhkvdqy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/bdgxhkvdqy/mul_5:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/bdgxhkvdqy/add_3:z:0(^while/bdgxhkvdqy/BiasAdd/ReadVariableOp'^while/bdgxhkvdqy/MatMul/ReadVariableOp)^while/bdgxhkvdqy/MatMul_1/ReadVariableOp ^while/bdgxhkvdqy/ReadVariableOp"^while/bdgxhkvdqy/ReadVariableOp_1"^while/bdgxhkvdqy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_bdgxhkvdqy_biasadd_readvariableop_resource2while_bdgxhkvdqy_biasadd_readvariableop_resource_0"h
1while_bdgxhkvdqy_matmul_1_readvariableop_resource3while_bdgxhkvdqy_matmul_1_readvariableop_resource_0"d
/while_bdgxhkvdqy_matmul_readvariableop_resource1while_bdgxhkvdqy_matmul_readvariableop_resource_0"Z
*while_bdgxhkvdqy_readvariableop_1_resource,while_bdgxhkvdqy_readvariableop_1_resource_0"Z
*while_bdgxhkvdqy_readvariableop_2_resource,while_bdgxhkvdqy_readvariableop_2_resource_0"V
(while_bdgxhkvdqy_readvariableop_resource*while_bdgxhkvdqy_readvariableop_resource_0")
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
'while/bdgxhkvdqy/BiasAdd/ReadVariableOp'while/bdgxhkvdqy/BiasAdd/ReadVariableOp2P
&while/bdgxhkvdqy/MatMul/ReadVariableOp&while/bdgxhkvdqy/MatMul/ReadVariableOp2T
(while/bdgxhkvdqy/MatMul_1/ReadVariableOp(while/bdgxhkvdqy/MatMul_1/ReadVariableOp2B
while/bdgxhkvdqy/ReadVariableOpwhile/bdgxhkvdqy/ReadVariableOp2F
!while/bdgxhkvdqy/ReadVariableOp_1!while/bdgxhkvdqy/ReadVariableOp_12F
!while/bdgxhkvdqy/ReadVariableOp_2!while/bdgxhkvdqy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_361511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_kfdgklwsil_361535_0:	,
while_kfdgklwsil_361537_0:	 (
while_kfdgklwsil_361539_0:	'
while_kfdgklwsil_361541_0: '
while_kfdgklwsil_361543_0: '
while_kfdgklwsil_361545_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_kfdgklwsil_361535:	*
while_kfdgklwsil_361537:	 &
while_kfdgklwsil_361539:	%
while_kfdgklwsil_361541: %
while_kfdgklwsil_361543: %
while_kfdgklwsil_361545: ¢(while/kfdgklwsil/StatefulPartitionedCallÃ
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
(while/kfdgklwsil/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_kfdgklwsil_361535_0while_kfdgklwsil_361537_0while_kfdgklwsil_361539_0while_kfdgklwsil_361541_0while_kfdgklwsil_361543_0while_kfdgklwsil_361545_0*
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_3614152*
(while/kfdgklwsil/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/kfdgklwsil/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/kfdgklwsil/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/kfdgklwsil/StatefulPartitionedCall:output:1)^while/kfdgklwsil/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/kfdgklwsil/StatefulPartitionedCall:output:2)^while/kfdgklwsil/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_kfdgklwsil_361535while_kfdgklwsil_361535_0"4
while_kfdgklwsil_361537while_kfdgklwsil_361537_0"4
while_kfdgklwsil_361539while_kfdgklwsil_361539_0"4
while_kfdgklwsil_361541while_kfdgklwsil_361541_0"4
while_kfdgklwsil_361543while_kfdgklwsil_361543_0"4
while_kfdgklwsil_361545while_kfdgklwsil_361545_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/kfdgklwsil/StatefulPartitionedCall(while/kfdgklwsil/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
Z
Ê
__inference__traced_save_366842
file_prefix0
,savev2_ekzorghjta_kernel_read_readvariableop.
*savev2_ekzorghjta_bias_read_readvariableop0
,savev2_pemnqztknd_kernel_read_readvariableop.
*savev2_pemnqztknd_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_zeuewnmlut_kfdgklwsil_kernel_read_readvariableopE
Asavev2_zeuewnmlut_kfdgklwsil_recurrent_kernel_read_readvariableop9
5savev2_zeuewnmlut_kfdgklwsil_bias_read_readvariableopP
Lsavev2_zeuewnmlut_kfdgklwsil_input_gate_peephole_weights_read_readvariableopQ
Msavev2_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_zeuewnmlut_kfdgklwsil_output_gate_peephole_weights_read_readvariableop;
7savev2_ipeigywbwc_bdgxhkvdqy_kernel_read_readvariableopE
Asavev2_ipeigywbwc_bdgxhkvdqy_recurrent_kernel_read_readvariableop9
5savev2_ipeigywbwc_bdgxhkvdqy_bias_read_readvariableopP
Lsavev2_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weights_read_readvariableopQ
Msavev2_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_ekzorghjta_kernel_rms_read_readvariableop:
6savev2_rmsprop_ekzorghjta_bias_rms_read_readvariableop<
8savev2_rmsprop_pemnqztknd_kernel_rms_read_readvariableop:
6savev2_rmsprop_pemnqztknd_bias_rms_read_readvariableopG
Csavev2_rmsprop_zeuewnmlut_kfdgklwsil_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_zeuewnmlut_kfdgklwsil_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_zeuewnmlut_kfdgklwsil_bias_rms_read_readvariableop\
Xsavev2_rmsprop_zeuewnmlut_kfdgklwsil_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_zeuewnmlut_kfdgklwsil_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_bias_rms_read_readvariableop\
Xsavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_ekzorghjta_kernel_read_readvariableop*savev2_ekzorghjta_bias_read_readvariableop,savev2_pemnqztknd_kernel_read_readvariableop*savev2_pemnqztknd_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_zeuewnmlut_kfdgklwsil_kernel_read_readvariableopAsavev2_zeuewnmlut_kfdgklwsil_recurrent_kernel_read_readvariableop5savev2_zeuewnmlut_kfdgklwsil_bias_read_readvariableopLsavev2_zeuewnmlut_kfdgklwsil_input_gate_peephole_weights_read_readvariableopMsavev2_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weights_read_readvariableopMsavev2_zeuewnmlut_kfdgklwsil_output_gate_peephole_weights_read_readvariableop7savev2_ipeigywbwc_bdgxhkvdqy_kernel_read_readvariableopAsavev2_ipeigywbwc_bdgxhkvdqy_recurrent_kernel_read_readvariableop5savev2_ipeigywbwc_bdgxhkvdqy_bias_read_readvariableopLsavev2_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weights_read_readvariableopMsavev2_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weights_read_readvariableopMsavev2_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_ekzorghjta_kernel_rms_read_readvariableop6savev2_rmsprop_ekzorghjta_bias_rms_read_readvariableop8savev2_rmsprop_pemnqztknd_kernel_rms_read_readvariableop6savev2_rmsprop_pemnqztknd_bias_rms_read_readvariableopCsavev2_rmsprop_zeuewnmlut_kfdgklwsil_kernel_rms_read_readvariableopMsavev2_rmsprop_zeuewnmlut_kfdgklwsil_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_zeuewnmlut_kfdgklwsil_bias_rms_read_readvariableopXsavev2_rmsprop_zeuewnmlut_kfdgklwsil_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_zeuewnmlut_kfdgklwsil_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_zeuewnmlut_kfdgklwsil_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_kernel_rms_read_readvariableopMsavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_bias_rms_read_readvariableopXsavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_ipeigywbwc_bdgxhkvdqy_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
h

F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365379

inputs<
)kfdgklwsil_matmul_readvariableop_resource:	>
+kfdgklwsil_matmul_1_readvariableop_resource:	 9
*kfdgklwsil_biasadd_readvariableop_resource:	0
"kfdgklwsil_readvariableop_resource: 2
$kfdgklwsil_readvariableop_1_resource: 2
$kfdgklwsil_readvariableop_2_resource: 
identity¢!kfdgklwsil/BiasAdd/ReadVariableOp¢ kfdgklwsil/MatMul/ReadVariableOp¢"kfdgklwsil/MatMul_1/ReadVariableOp¢kfdgklwsil/ReadVariableOp¢kfdgklwsil/ReadVariableOp_1¢kfdgklwsil/ReadVariableOp_2¢whileD
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
 kfdgklwsil/MatMul/ReadVariableOpReadVariableOp)kfdgklwsil_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 kfdgklwsil/MatMul/ReadVariableOp§
kfdgklwsil/MatMulMatMulstrided_slice_2:output:0(kfdgklwsil/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMulµ
"kfdgklwsil/MatMul_1/ReadVariableOpReadVariableOp+kfdgklwsil_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"kfdgklwsil/MatMul_1/ReadVariableOp£
kfdgklwsil/MatMul_1MatMulzeros:output:0*kfdgklwsil/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/MatMul_1
kfdgklwsil/addAddV2kfdgklwsil/MatMul:product:0kfdgklwsil/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/add®
!kfdgklwsil/BiasAdd/ReadVariableOpReadVariableOp*kfdgklwsil_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!kfdgklwsil/BiasAdd/ReadVariableOp¥
kfdgklwsil/BiasAddBiasAddkfdgklwsil/add:z:0)kfdgklwsil/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
kfdgklwsil/BiasAddz
kfdgklwsil/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
kfdgklwsil/split/split_dimë
kfdgklwsil/splitSplit#kfdgklwsil/split/split_dim:output:0kfdgklwsil/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
kfdgklwsil/split
kfdgklwsil/ReadVariableOpReadVariableOp"kfdgklwsil_readvariableop_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp
kfdgklwsil/mulMul!kfdgklwsil/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul
kfdgklwsil/add_1AddV2kfdgklwsil/split:output:0kfdgklwsil/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_1{
kfdgklwsil/SigmoidSigmoidkfdgklwsil/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid
kfdgklwsil/ReadVariableOp_1ReadVariableOp$kfdgklwsil_readvariableop_1_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_1
kfdgklwsil/mul_1Mul#kfdgklwsil/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_1
kfdgklwsil/add_2AddV2kfdgklwsil/split:output:1kfdgklwsil/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_2
kfdgklwsil/Sigmoid_1Sigmoidkfdgklwsil/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_1
kfdgklwsil/mul_2Mulkfdgklwsil/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_2w
kfdgklwsil/TanhTanhkfdgklwsil/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh
kfdgklwsil/mul_3Mulkfdgklwsil/Sigmoid:y:0kfdgklwsil/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_3
kfdgklwsil/add_3AddV2kfdgklwsil/mul_2:z:0kfdgklwsil/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_3
kfdgklwsil/ReadVariableOp_2ReadVariableOp$kfdgklwsil_readvariableop_2_resource*
_output_shapes
: *
dtype02
kfdgklwsil/ReadVariableOp_2
kfdgklwsil/mul_4Mul#kfdgklwsil/ReadVariableOp_2:value:0kfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_4
kfdgklwsil/add_4AddV2kfdgklwsil/split:output:3kfdgklwsil/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/add_4
kfdgklwsil/Sigmoid_2Sigmoidkfdgklwsil/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Sigmoid_2v
kfdgklwsil/Tanh_1Tanhkfdgklwsil/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/Tanh_1
kfdgklwsil/mul_5Mulkfdgklwsil/Sigmoid_2:y:0kfdgklwsil/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
kfdgklwsil/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)kfdgklwsil_matmul_readvariableop_resource+kfdgklwsil_matmul_1_readvariableop_resource*kfdgklwsil_biasadd_readvariableop_resource"kfdgklwsil_readvariableop_resource$kfdgklwsil_readvariableop_1_resource$kfdgklwsil_readvariableop_2_resource*
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
while_body_365278*
condR
while_cond_365277*Q
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
IdentityIdentitytranspose_1:y:0"^kfdgklwsil/BiasAdd/ReadVariableOp!^kfdgklwsil/MatMul/ReadVariableOp#^kfdgklwsil/MatMul_1/ReadVariableOp^kfdgklwsil/ReadVariableOp^kfdgklwsil/ReadVariableOp_1^kfdgklwsil/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!kfdgklwsil/BiasAdd/ReadVariableOp!kfdgklwsil/BiasAdd/ReadVariableOp2D
 kfdgklwsil/MatMul/ReadVariableOp kfdgklwsil/MatMul/ReadVariableOp2H
"kfdgklwsil/MatMul_1/ReadVariableOp"kfdgklwsil/MatMul_1/ReadVariableOp26
kfdgklwsil/ReadVariableOpkfdgklwsil/ReadVariableOp2:
kfdgklwsil/ReadVariableOp_1kfdgklwsil/ReadVariableOp_12:
kfdgklwsil/ReadVariableOp_2kfdgklwsil/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù

+__inference_zeuewnmlut_layer_call_fn_365610

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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_3629012
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

kjggqknufb;
serving_default_kjggqknufb:0ÿÿÿÿÿÿÿÿÿ>

pemnqztknd0
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
+&call_and_return_all_conditional_losses
__call__"ÂA
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "kjggqknufb"}}, {"class_name": "Conv1D", "config": {"name": "ekzorghjta", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "xsvuntduhq", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "zeuewnmlut", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kfdgklwsil", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "ipeigywbwc", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "bdgxhkvdqy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "pemnqztknd", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "kjggqknufb"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "kjggqknufb"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "ekzorghjta", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "xsvuntduhq", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "zeuewnmlut", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kfdgklwsil", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "ipeigywbwc", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "bdgxhkvdqy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "pemnqztknd", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
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
{"name": "ekzorghjta", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "ekzorghjta", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"name": "xsvuntduhq", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "xsvuntduhq", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
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
_tf_keras_rnn_layerä{"name": "zeuewnmlut", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "zeuewnmlut", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kfdgklwsil", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
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
_tf_keras_rnn_layerê{"name": "ipeigywbwc", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "ipeigywbwc", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "bdgxhkvdqy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "pemnqztknd", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "pemnqztknd", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
':%2ekzorghjta/kernel
:2ekzorghjta/bias
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
_tf_keras_layer¼{"name": "kfdgklwsil", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "kfdgklwsil", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
_tf_keras_layerÀ{"name": "bdgxhkvdqy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "bdgxhkvdqy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
#:! 2pemnqztknd/kernel
:2pemnqztknd/bias
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
/:-	2zeuewnmlut/kfdgklwsil/kernel
9:7	 2&zeuewnmlut/kfdgklwsil/recurrent_kernel
):'2zeuewnmlut/kfdgklwsil/bias
?:= 21zeuewnmlut/kfdgklwsil/input_gate_peephole_weights
@:> 22zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights
@:> 22zeuewnmlut/kfdgklwsil/output_gate_peephole_weights
/:-	 2ipeigywbwc/bdgxhkvdqy/kernel
9:7	 2&ipeigywbwc/bdgxhkvdqy/recurrent_kernel
):'2ipeigywbwc/bdgxhkvdqy/bias
?:= 21ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights
@:> 22ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights
@:> 22ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights
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
1:/2RMSprop/ekzorghjta/kernel/rms
':%2RMSprop/ekzorghjta/bias/rms
-:+ 2RMSprop/pemnqztknd/kernel/rms
':%2RMSprop/pemnqztknd/bias/rms
9:7	2(RMSprop/zeuewnmlut/kfdgklwsil/kernel/rms
C:A	 22RMSprop/zeuewnmlut/kfdgklwsil/recurrent_kernel/rms
3:12&RMSprop/zeuewnmlut/kfdgklwsil/bias/rms
I:G 2=RMSprop/zeuewnmlut/kfdgklwsil/input_gate_peephole_weights/rms
J:H 2>RMSprop/zeuewnmlut/kfdgklwsil/forget_gate_peephole_weights/rms
J:H 2>RMSprop/zeuewnmlut/kfdgklwsil/output_gate_peephole_weights/rms
9:7	 2(RMSprop/ipeigywbwc/bdgxhkvdqy/kernel/rms
C:A	 22RMSprop/ipeigywbwc/bdgxhkvdqy/recurrent_kernel/rms
3:12&RMSprop/ipeigywbwc/bdgxhkvdqy/bias/rms
I:G 2=RMSprop/ipeigywbwc/bdgxhkvdqy/input_gate_peephole_weights/rms
J:H 2>RMSprop/ipeigywbwc/bdgxhkvdqy/forget_gate_peephole_weights/rms
J:H 2>RMSprop/ipeigywbwc/bdgxhkvdqy/output_gate_peephole_weights/rms
ê2ç
!__inference__wrapped_model_361141Á
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

kjggqknufbÿÿÿÿÿÿÿÿÿ
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_364297
F__inference_sequential_layer_call_and_return_conditional_losses_364701
F__inference_sequential_layer_call_and_return_conditional_losses_363807
F__inference_sequential_layer_call_and_return_conditional_losses_363848À
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
+__inference_sequential_layer_call_fn_363160
+__inference_sequential_layer_call_fn_364738
+__inference_sequential_layer_call_fn_364775
+__inference_sequential_layer_call_fn_363766À
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
ð2í
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_364812¢
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
+__inference_ekzorghjta_layer_call_fn_364821¢
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
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_364834¢
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
+__inference_xsvuntduhq_layer_call_fn_364839¢
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365019
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365199
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365379
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365559æ
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
+__inference_zeuewnmlut_layer_call_fn_365576
+__inference_zeuewnmlut_layer_call_fn_365593
+__inference_zeuewnmlut_layer_call_fn_365610
+__inference_zeuewnmlut_layer_call_fn_365627æ
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_365807
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_365987
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_366167
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_366347æ
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
+__inference_ipeigywbwc_layer_call_fn_366364
+__inference_ipeigywbwc_layer_call_fn_366381
+__inference_ipeigywbwc_layer_call_fn_366398
+__inference_ipeigywbwc_layer_call_fn_366415æ
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
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_366425¢
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
+__inference_pemnqztknd_layer_call_fn_366434¢
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
$__inference_signature_wrapper_363893
kjggqknufb"
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_366478
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_366522¾
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
+__inference_kfdgklwsil_layer_call_fn_366545
+__inference_kfdgklwsil_layer_call_fn_366568¾
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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_366612
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_366656¾
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
+__inference_bdgxhkvdqy_layer_call_fn_366679
+__inference_bdgxhkvdqy_layer_call_fn_366702¾
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
!__inference__wrapped_model_361141-./012345678"#;¢8
1¢.
,)

kjggqknufbÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

pemnqztknd$!

pemnqztkndÿÿÿÿÿÿÿÿÿË
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_366612345678¢}
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
F__inference_bdgxhkvdqy_layer_call_and_return_conditional_losses_366656345678¢}
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
+__inference_bdgxhkvdqy_layer_call_fn_366679ð345678¢}
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
+__inference_bdgxhkvdqy_layer_call_fn_366702ð345678¢}
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
1/1ÿÿÿÿÿÿÿÿÿ ¶
F__inference_ekzorghjta_layer_call_and_return_conditional_losses_364812l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_ekzorghjta_layer_call_fn_364821_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÏ
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_365807345678S¢P
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_365987345678S¢P
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_366167t345678C¢@
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
F__inference_ipeigywbwc_layer_call_and_return_conditional_losses_366347t345678C¢@
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
+__inference_ipeigywbwc_layer_call_fn_366364w345678S¢P
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
+__inference_ipeigywbwc_layer_call_fn_366381w345678S¢P
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
+__inference_ipeigywbwc_layer_call_fn_366398g345678C¢@
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
+__inference_ipeigywbwc_layer_call_fn_366415g345678C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ Ë
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_366478-./012¢}
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
F__inference_kfdgklwsil_layer_call_and_return_conditional_losses_366522-./012¢}
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
+__inference_kfdgklwsil_layer_call_fn_366545ð-./012¢}
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
+__inference_kfdgklwsil_layer_call_fn_366568ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ ¦
F__inference_pemnqztknd_layer_call_and_return_conditional_losses_366425\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_pemnqztknd_layer_call_fn_366434O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÈ
F__inference_sequential_layer_call_and_return_conditional_losses_363807~-./012345678"#C¢@
9¢6
,)

kjggqknufbÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
F__inference_sequential_layer_call_and_return_conditional_losses_363848~-./012345678"#C¢@
9¢6
,)

kjggqknufbÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_364297z-./012345678"#?¢<
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
F__inference_sequential_layer_call_and_return_conditional_losses_364701z-./012345678"#?¢<
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
+__inference_sequential_layer_call_fn_363160q-./012345678"#C¢@
9¢6
,)

kjggqknufbÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_sequential_layer_call_fn_363766q-./012345678"#C¢@
9¢6
,)

kjggqknufbÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_364738m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_364775m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
$__inference_signature_wrapper_363893-./012345678"#I¢F
¢ 
?ª<
:

kjggqknufb,)

kjggqknufbÿÿÿÿÿÿÿÿÿ"7ª4
2

pemnqztknd$!

pemnqztkndÿÿÿÿÿÿÿÿÿ®
F__inference_xsvuntduhq_layer_call_and_return_conditional_losses_364834d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_xsvuntduhq_layer_call_fn_364839W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÜ
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365019-./012S¢P
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365199-./012S¢P
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365379x-./012C¢@
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
F__inference_zeuewnmlut_layer_call_and_return_conditional_losses_365559x-./012C¢@
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
+__inference_zeuewnmlut_layer_call_fn_365576-./012S¢P
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
+__inference_zeuewnmlut_layer_call_fn_365593-./012S¢P
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
+__inference_zeuewnmlut_layer_call_fn_365610k-./012C¢@
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
+__inference_zeuewnmlut_layer_call_fn_365627k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ 