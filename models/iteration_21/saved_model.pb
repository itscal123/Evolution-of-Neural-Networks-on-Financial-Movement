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
gtjikcltwy/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namegtjikcltwy/kernel
{
%gtjikcltwy/kernel/Read/ReadVariableOpReadVariableOpgtjikcltwy/kernel*"
_output_shapes
:*
dtype0
v
gtjikcltwy/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namegtjikcltwy/bias
o
#gtjikcltwy/bias/Read/ReadVariableOpReadVariableOpgtjikcltwy/bias*
_output_shapes
:*
dtype0
~
chsgvefspq/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_namechsgvefspq/kernel
w
%chsgvefspq/kernel/Read/ReadVariableOpReadVariableOpchsgvefspq/kernel*
_output_shapes

: *
dtype0
v
chsgvefspq/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namechsgvefspq/bias
o
#chsgvefspq/bias/Read/ReadVariableOpReadVariableOpchsgvefspq/bias*
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
dnzlhpjizj/hswofenhiy/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_namednzlhpjizj/hswofenhiy/kernel

0dnzlhpjizj/hswofenhiy/kernel/Read/ReadVariableOpReadVariableOpdnzlhpjizj/hswofenhiy/kernel*
_output_shapes
:	*
dtype0
©
&dnzlhpjizj/hswofenhiy/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&dnzlhpjizj/hswofenhiy/recurrent_kernel
¢
:dnzlhpjizj/hswofenhiy/recurrent_kernel/Read/ReadVariableOpReadVariableOp&dnzlhpjizj/hswofenhiy/recurrent_kernel*
_output_shapes
:	 *
dtype0

dnzlhpjizj/hswofenhiy/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namednzlhpjizj/hswofenhiy/bias

.dnzlhpjizj/hswofenhiy/bias/Read/ReadVariableOpReadVariableOpdnzlhpjizj/hswofenhiy/bias*
_output_shapes	
:*
dtype0
º
1dnzlhpjizj/hswofenhiy/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31dnzlhpjizj/hswofenhiy/input_gate_peephole_weights
³
Ednzlhpjizj/hswofenhiy/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1dnzlhpjizj/hswofenhiy/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2dnzlhpjizj/hswofenhiy/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights
µ
Fdnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2dnzlhpjizj/hswofenhiy/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42dnzlhpjizj/hswofenhiy/output_gate_peephole_weights
µ
Fdnzlhpjizj/hswofenhiy/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2dnzlhpjizj/hswofenhiy/output_gate_peephole_weights*
_output_shapes
: *
dtype0

nyosplwtfa/lwptfvtmlx/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *-
shared_namenyosplwtfa/lwptfvtmlx/kernel

0nyosplwtfa/lwptfvtmlx/kernel/Read/ReadVariableOpReadVariableOpnyosplwtfa/lwptfvtmlx/kernel*
_output_shapes
:	 *
dtype0
©
&nyosplwtfa/lwptfvtmlx/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *7
shared_name(&nyosplwtfa/lwptfvtmlx/recurrent_kernel
¢
:nyosplwtfa/lwptfvtmlx/recurrent_kernel/Read/ReadVariableOpReadVariableOp&nyosplwtfa/lwptfvtmlx/recurrent_kernel*
_output_shapes
:	 *
dtype0

nyosplwtfa/lwptfvtmlx/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namenyosplwtfa/lwptfvtmlx/bias

.nyosplwtfa/lwptfvtmlx/bias/Read/ReadVariableOpReadVariableOpnyosplwtfa/lwptfvtmlx/bias*
_output_shapes	
:*
dtype0
º
1nyosplwtfa/lwptfvtmlx/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights
³
Enyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp1nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights
µ
Fnyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
¼
2nyosplwtfa/lwptfvtmlx/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights
µ
Fnyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights*
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
RMSprop/gtjikcltwy/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/gtjikcltwy/kernel/rms

1RMSprop/gtjikcltwy/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/gtjikcltwy/kernel/rms*"
_output_shapes
:*
dtype0

RMSprop/gtjikcltwy/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/gtjikcltwy/bias/rms

/RMSprop/gtjikcltwy/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/gtjikcltwy/bias/rms*
_output_shapes
:*
dtype0

RMSprop/chsgvefspq/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameRMSprop/chsgvefspq/kernel/rms

1RMSprop/chsgvefspq/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/chsgvefspq/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/chsgvefspq/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/chsgvefspq/bias/rms

/RMSprop/chsgvefspq/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/chsgvefspq/bias/rms*
_output_shapes
:*
dtype0
­
(RMSprop/dnzlhpjizj/hswofenhiy/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*9
shared_name*(RMSprop/dnzlhpjizj/hswofenhiy/kernel/rms
¦
<RMSprop/dnzlhpjizj/hswofenhiy/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/dnzlhpjizj/hswofenhiy/kernel/rms*
_output_shapes
:	*
dtype0
Á
2RMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rms
º
FRMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/dnzlhpjizj/hswofenhiy/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/dnzlhpjizj/hswofenhiy/bias/rms

:RMSprop/dnzlhpjizj/hswofenhiy/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/dnzlhpjizj/hswofenhiy/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rms
Ë
QRMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rms
Í
RRMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rms
Í
RRMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
­
(RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *9
shared_name*(RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rms
¦
<RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rms*
_output_shapes
:	 *
dtype0
Á
2RMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *C
shared_name42RMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rms
º
FRMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp2RMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rms*
_output_shapes
:	 *
dtype0
¥
&RMSprop/nyosplwtfa/lwptfvtmlx/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/nyosplwtfa/lwptfvtmlx/bias/rms

:RMSprop/nyosplwtfa/lwptfvtmlx/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/nyosplwtfa/lwptfvtmlx/bias/rms*
_output_shapes	
:*
dtype0
Ò
=RMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=RMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rms
Ë
QRMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp=RMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rms
Í
RRMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rms*
_output_shapes
: *
dtype0
Ô
>RMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *O
shared_name@>RMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rms
Í
RRMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rms/Read/ReadVariableOpReadVariableOp>RMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rms*
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
VARIABLE_VALUEgtjikcltwy/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgtjikcltwy/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEchsgvefspq/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEchsgvefspq/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdnzlhpjizj/hswofenhiy/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&dnzlhpjizj/hswofenhiy/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdnzlhpjizj/hswofenhiy/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1dnzlhpjizj/hswofenhiy/input_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2dnzlhpjizj/hswofenhiy/output_gate_peephole_weights&variables/7/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEnyosplwtfa/lwptfvtmlx/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&nyosplwtfa/lwptfvtmlx/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEnyosplwtfa/lwptfvtmlx/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights'variables/11/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights'variables/12/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights'variables/13/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUERMSprop/gtjikcltwy/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/gtjikcltwy/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/chsgvefspq/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/chsgvefspq/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/dnzlhpjizj/hswofenhiy/kernel/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&RMSprop/dnzlhpjizj/hswofenhiy/bias/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rmsDvariables/7/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rmsDvariables/8/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2RMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rmsDvariables/9/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&RMSprop/nyosplwtfa/lwptfvtmlx/bias/rmsEvariables/10/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=RMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rmsEvariables/11/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rmsEvariables/12/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>RMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rmsEvariables/13/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_bdeyofgzkqPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_bdeyofgzkqgtjikcltwy/kernelgtjikcltwy/biasdnzlhpjizj/hswofenhiy/kernel&dnzlhpjizj/hswofenhiy/recurrent_kerneldnzlhpjizj/hswofenhiy/bias1dnzlhpjizj/hswofenhiy/input_gate_peephole_weights2dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights2dnzlhpjizj/hswofenhiy/output_gate_peephole_weightsnyosplwtfa/lwptfvtmlx/kernel&nyosplwtfa/lwptfvtmlx/recurrent_kernelnyosplwtfa/lwptfvtmlx/bias1nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights2nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights2nyosplwtfa/lwptfvtmlx/output_gate_peephole_weightschsgvefspq/kernelchsgvefspq/bias*
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
%__inference_signature_wrapper_2596439
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%gtjikcltwy/kernel/Read/ReadVariableOp#gtjikcltwy/bias/Read/ReadVariableOp%chsgvefspq/kernel/Read/ReadVariableOp#chsgvefspq/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp0dnzlhpjizj/hswofenhiy/kernel/Read/ReadVariableOp:dnzlhpjizj/hswofenhiy/recurrent_kernel/Read/ReadVariableOp.dnzlhpjizj/hswofenhiy/bias/Read/ReadVariableOpEdnzlhpjizj/hswofenhiy/input_gate_peephole_weights/Read/ReadVariableOpFdnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/Read/ReadVariableOpFdnzlhpjizj/hswofenhiy/output_gate_peephole_weights/Read/ReadVariableOp0nyosplwtfa/lwptfvtmlx/kernel/Read/ReadVariableOp:nyosplwtfa/lwptfvtmlx/recurrent_kernel/Read/ReadVariableOp.nyosplwtfa/lwptfvtmlx/bias/Read/ReadVariableOpEnyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/Read/ReadVariableOpFnyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/Read/ReadVariableOpFnyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1RMSprop/gtjikcltwy/kernel/rms/Read/ReadVariableOp/RMSprop/gtjikcltwy/bias/rms/Read/ReadVariableOp1RMSprop/chsgvefspq/kernel/rms/Read/ReadVariableOp/RMSprop/chsgvefspq/bias/rms/Read/ReadVariableOp<RMSprop/dnzlhpjizj/hswofenhiy/kernel/rms/Read/ReadVariableOpFRMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/dnzlhpjizj/hswofenhiy/bias/rms/Read/ReadVariableOpQRMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rms/Read/ReadVariableOp<RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rms/Read/ReadVariableOpFRMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rms/Read/ReadVariableOp:RMSprop/nyosplwtfa/lwptfvtmlx/bias/rms/Read/ReadVariableOpQRMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rms/Read/ReadVariableOpRRMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rms/Read/ReadVariableOpConst*4
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
 __inference__traced_save_2599388
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegtjikcltwy/kernelgtjikcltwy/biaschsgvefspq/kernelchsgvefspq/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhodnzlhpjizj/hswofenhiy/kernel&dnzlhpjizj/hswofenhiy/recurrent_kerneldnzlhpjizj/hswofenhiy/bias1dnzlhpjizj/hswofenhiy/input_gate_peephole_weights2dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights2dnzlhpjizj/hswofenhiy/output_gate_peephole_weightsnyosplwtfa/lwptfvtmlx/kernel&nyosplwtfa/lwptfvtmlx/recurrent_kernelnyosplwtfa/lwptfvtmlx/bias1nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights2nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights2nyosplwtfa/lwptfvtmlx/output_gate_peephole_weightstotalcountRMSprop/gtjikcltwy/kernel/rmsRMSprop/gtjikcltwy/bias/rmsRMSprop/chsgvefspq/kernel/rmsRMSprop/chsgvefspq/bias/rms(RMSprop/dnzlhpjizj/hswofenhiy/kernel/rms2RMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rms&RMSprop/dnzlhpjizj/hswofenhiy/bias/rms=RMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rms>RMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rms>RMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rms(RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rms2RMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rms&RMSprop/nyosplwtfa/lwptfvtmlx/bias/rms=RMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rms>RMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rms>RMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rms*3
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
#__inference__traced_restore_2599515Âà-
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_2596843

inputsL
6gtjikcltwy_conv1d_expanddims_1_readvariableop_resource:K
=gtjikcltwy_squeeze_batch_dims_biasadd_readvariableop_resource:G
4dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource:	I
6dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource:	 D
5dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource:	;
-dnzlhpjizj_hswofenhiy_readvariableop_resource: =
/dnzlhpjizj_hswofenhiy_readvariableop_1_resource: =
/dnzlhpjizj_hswofenhiy_readvariableop_2_resource: G
4nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource:	 I
6nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource:	 D
5nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource:	;
-nyosplwtfa_lwptfvtmlx_readvariableop_resource: =
/nyosplwtfa_lwptfvtmlx_readvariableop_1_resource: =
/nyosplwtfa_lwptfvtmlx_readvariableop_2_resource: ;
)chsgvefspq_matmul_readvariableop_resource: 8
*chsgvefspq_biasadd_readvariableop_resource:
identity¢!chsgvefspq/BiasAdd/ReadVariableOp¢ chsgvefspq/MatMul/ReadVariableOp¢,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp¢+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp¢-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp¢$dnzlhpjizj/hswofenhiy/ReadVariableOp¢&dnzlhpjizj/hswofenhiy/ReadVariableOp_1¢&dnzlhpjizj/hswofenhiy/ReadVariableOp_2¢dnzlhpjizj/while¢-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp¢4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp¢+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp¢-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp¢$nyosplwtfa/lwptfvtmlx/ReadVariableOp¢&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1¢&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2¢nyosplwtfa/while
 gtjikcltwy/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 gtjikcltwy/conv1d/ExpandDims/dim»
gtjikcltwy/conv1d/ExpandDims
ExpandDimsinputs)gtjikcltwy/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/ExpandDimsÙ
-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6gtjikcltwy_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp
"gtjikcltwy/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"gtjikcltwy/conv1d/ExpandDims_1/dimã
gtjikcltwy/conv1d/ExpandDims_1
ExpandDims5gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp:value:0+gtjikcltwy/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
gtjikcltwy/conv1d/ExpandDims_1
gtjikcltwy/conv1d/ShapeShape%gtjikcltwy/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
gtjikcltwy/conv1d/Shape
%gtjikcltwy/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%gtjikcltwy/conv1d/strided_slice/stack¥
'gtjikcltwy/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'gtjikcltwy/conv1d/strided_slice/stack_1
'gtjikcltwy/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'gtjikcltwy/conv1d/strided_slice/stack_2Ì
gtjikcltwy/conv1d/strided_sliceStridedSlice gtjikcltwy/conv1d/Shape:output:0.gtjikcltwy/conv1d/strided_slice/stack:output:00gtjikcltwy/conv1d/strided_slice/stack_1:output:00gtjikcltwy/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
gtjikcltwy/conv1d/strided_slice
gtjikcltwy/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
gtjikcltwy/conv1d/Reshape/shapeÌ
gtjikcltwy/conv1d/ReshapeReshape%gtjikcltwy/conv1d/ExpandDims:output:0(gtjikcltwy/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/Reshapeî
gtjikcltwy/conv1d/Conv2DConv2D"gtjikcltwy/conv1d/Reshape:output:0'gtjikcltwy/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
gtjikcltwy/conv1d/Conv2D
!gtjikcltwy/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!gtjikcltwy/conv1d/concat/values_1
gtjikcltwy/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/concat/axisì
gtjikcltwy/conv1d/concatConcatV2(gtjikcltwy/conv1d/strided_slice:output:0*gtjikcltwy/conv1d/concat/values_1:output:0&gtjikcltwy/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
gtjikcltwy/conv1d/concatÉ
gtjikcltwy/conv1d/Reshape_1Reshape!gtjikcltwy/conv1d/Conv2D:output:0!gtjikcltwy/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/Reshape_1Á
gtjikcltwy/conv1d/SqueezeSqueeze$gtjikcltwy/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/Squeeze
#gtjikcltwy/squeeze_batch_dims/ShapeShape"gtjikcltwy/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#gtjikcltwy/squeeze_batch_dims/Shape°
1gtjikcltwy/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1gtjikcltwy/squeeze_batch_dims/strided_slice/stack½
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1´
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2
+gtjikcltwy/squeeze_batch_dims/strided_sliceStridedSlice,gtjikcltwy/squeeze_batch_dims/Shape:output:0:gtjikcltwy/squeeze_batch_dims/strided_slice/stack:output:0<gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1:output:0<gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+gtjikcltwy/squeeze_batch_dims/strided_slice¯
+gtjikcltwy/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+gtjikcltwy/squeeze_batch_dims/Reshape/shapeé
%gtjikcltwy/squeeze_batch_dims/ReshapeReshape"gtjikcltwy/conv1d/Squeeze:output:04gtjikcltwy/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%gtjikcltwy/squeeze_batch_dims/Reshapeæ
4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=gtjikcltwy_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%gtjikcltwy/squeeze_batch_dims/BiasAddBiasAdd.gtjikcltwy/squeeze_batch_dims/Reshape:output:0<gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%gtjikcltwy/squeeze_batch_dims/BiasAdd¯
-gtjikcltwy/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-gtjikcltwy/squeeze_batch_dims/concat/values_1¡
)gtjikcltwy/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)gtjikcltwy/squeeze_batch_dims/concat/axis¨
$gtjikcltwy/squeeze_batch_dims/concatConcatV24gtjikcltwy/squeeze_batch_dims/strided_slice:output:06gtjikcltwy/squeeze_batch_dims/concat/values_1:output:02gtjikcltwy/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$gtjikcltwy/squeeze_batch_dims/concatö
'gtjikcltwy/squeeze_batch_dims/Reshape_1Reshape.gtjikcltwy/squeeze_batch_dims/BiasAdd:output:0-gtjikcltwy/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'gtjikcltwy/squeeze_batch_dims/Reshape_1
ezubtmdnwx/ShapeShape0gtjikcltwy/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ezubtmdnwx/Shape
ezubtmdnwx/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ezubtmdnwx/strided_slice/stack
 ezubtmdnwx/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ezubtmdnwx/strided_slice/stack_1
 ezubtmdnwx/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ezubtmdnwx/strided_slice/stack_2¤
ezubtmdnwx/strided_sliceStridedSliceezubtmdnwx/Shape:output:0'ezubtmdnwx/strided_slice/stack:output:0)ezubtmdnwx/strided_slice/stack_1:output:0)ezubtmdnwx/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ezubtmdnwx/strided_slicez
ezubtmdnwx/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ezubtmdnwx/Reshape/shape/1z
ezubtmdnwx/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ezubtmdnwx/Reshape/shape/2×
ezubtmdnwx/Reshape/shapePack!ezubtmdnwx/strided_slice:output:0#ezubtmdnwx/Reshape/shape/1:output:0#ezubtmdnwx/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ezubtmdnwx/Reshape/shape¾
ezubtmdnwx/ReshapeReshape0gtjikcltwy/squeeze_batch_dims/Reshape_1:output:0!ezubtmdnwx/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ezubtmdnwx/Reshapeo
dnzlhpjizj/ShapeShapeezubtmdnwx/Reshape:output:0*
T0*
_output_shapes
:2
dnzlhpjizj/Shape
dnzlhpjizj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
dnzlhpjizj/strided_slice/stack
 dnzlhpjizj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 dnzlhpjizj/strided_slice/stack_1
 dnzlhpjizj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 dnzlhpjizj/strided_slice/stack_2¤
dnzlhpjizj/strided_sliceStridedSlicednzlhpjizj/Shape:output:0'dnzlhpjizj/strided_slice/stack:output:0)dnzlhpjizj/strided_slice/stack_1:output:0)dnzlhpjizj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dnzlhpjizj/strided_slicer
dnzlhpjizj/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros/mul/y
dnzlhpjizj/zeros/mulMul!dnzlhpjizj/strided_slice:output:0dnzlhpjizj/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros/mulu
dnzlhpjizj/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
dnzlhpjizj/zeros/Less/y
dnzlhpjizj/zeros/LessLessdnzlhpjizj/zeros/mul:z:0 dnzlhpjizj/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros/Lessx
dnzlhpjizj/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros/packed/1¯
dnzlhpjizj/zeros/packedPack!dnzlhpjizj/strided_slice:output:0"dnzlhpjizj/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
dnzlhpjizj/zeros/packedu
dnzlhpjizj/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dnzlhpjizj/zeros/Const¡
dnzlhpjizj/zerosFill dnzlhpjizj/zeros/packed:output:0dnzlhpjizj/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/zerosv
dnzlhpjizj/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros_1/mul/y
dnzlhpjizj/zeros_1/mulMul!dnzlhpjizj/strided_slice:output:0!dnzlhpjizj/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros_1/muly
dnzlhpjizj/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
dnzlhpjizj/zeros_1/Less/y
dnzlhpjizj/zeros_1/LessLessdnzlhpjizj/zeros_1/mul:z:0"dnzlhpjizj/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros_1/Less|
dnzlhpjizj/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros_1/packed/1µ
dnzlhpjizj/zeros_1/packedPack!dnzlhpjizj/strided_slice:output:0$dnzlhpjizj/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
dnzlhpjizj/zeros_1/packedy
dnzlhpjizj/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dnzlhpjizj/zeros_1/Const©
dnzlhpjizj/zeros_1Fill"dnzlhpjizj/zeros_1/packed:output:0!dnzlhpjizj/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/zeros_1
dnzlhpjizj/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dnzlhpjizj/transpose/perm°
dnzlhpjizj/transpose	Transposeezubtmdnwx/Reshape:output:0"dnzlhpjizj/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/transposep
dnzlhpjizj/Shape_1Shapednzlhpjizj/transpose:y:0*
T0*
_output_shapes
:2
dnzlhpjizj/Shape_1
 dnzlhpjizj/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dnzlhpjizj/strided_slice_1/stack
"dnzlhpjizj/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_1/stack_1
"dnzlhpjizj/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_1/stack_2°
dnzlhpjizj/strided_slice_1StridedSlicednzlhpjizj/Shape_1:output:0)dnzlhpjizj/strided_slice_1/stack:output:0+dnzlhpjizj/strided_slice_1/stack_1:output:0+dnzlhpjizj/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dnzlhpjizj/strided_slice_1
&dnzlhpjizj/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&dnzlhpjizj/TensorArrayV2/element_shapeÞ
dnzlhpjizj/TensorArrayV2TensorListReserve/dnzlhpjizj/TensorArrayV2/element_shape:output:0#dnzlhpjizj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dnzlhpjizj/TensorArrayV2Õ
@dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2dnzlhpjizj/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordnzlhpjizj/transpose:y:0Idnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor
 dnzlhpjizj/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dnzlhpjizj/strided_slice_2/stack
"dnzlhpjizj/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_2/stack_1
"dnzlhpjizj/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_2/stack_2¾
dnzlhpjizj/strided_slice_2StridedSlicednzlhpjizj/transpose:y:0)dnzlhpjizj/strided_slice_2/stack:output:0+dnzlhpjizj/strided_slice_2/stack_1:output:0+dnzlhpjizj/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
dnzlhpjizj/strided_slice_2Ð
+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOpReadVariableOp4dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOpÓ
dnzlhpjizj/hswofenhiy/MatMulMatMul#dnzlhpjizj/strided_slice_2:output:03dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/hswofenhiy/MatMulÖ
-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp6dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOpÏ
dnzlhpjizj/hswofenhiy/MatMul_1MatMuldnzlhpjizj/zeros:output:05dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dnzlhpjizj/hswofenhiy/MatMul_1Ä
dnzlhpjizj/hswofenhiy/addAddV2&dnzlhpjizj/hswofenhiy/MatMul:product:0(dnzlhpjizj/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/hswofenhiy/addÏ
,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp5dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOpÑ
dnzlhpjizj/hswofenhiy/BiasAddBiasAdddnzlhpjizj/hswofenhiy/add:z:04dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/hswofenhiy/BiasAdd
%dnzlhpjizj/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%dnzlhpjizj/hswofenhiy/split/split_dim
dnzlhpjizj/hswofenhiy/splitSplit.dnzlhpjizj/hswofenhiy/split/split_dim:output:0&dnzlhpjizj/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dnzlhpjizj/hswofenhiy/split¶
$dnzlhpjizj/hswofenhiy/ReadVariableOpReadVariableOp-dnzlhpjizj_hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02&
$dnzlhpjizj/hswofenhiy/ReadVariableOpº
dnzlhpjizj/hswofenhiy/mulMul,dnzlhpjizj/hswofenhiy/ReadVariableOp:value:0dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mulº
dnzlhpjizj/hswofenhiy/add_1AddV2$dnzlhpjizj/hswofenhiy/split:output:0dnzlhpjizj/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_1
dnzlhpjizj/hswofenhiy/SigmoidSigmoiddnzlhpjizj/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/Sigmoid¼
&dnzlhpjizj/hswofenhiy/ReadVariableOp_1ReadVariableOp/dnzlhpjizj_hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&dnzlhpjizj/hswofenhiy/ReadVariableOp_1À
dnzlhpjizj/hswofenhiy/mul_1Mul.dnzlhpjizj/hswofenhiy/ReadVariableOp_1:value:0dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_1¼
dnzlhpjizj/hswofenhiy/add_2AddV2$dnzlhpjizj/hswofenhiy/split:output:1dnzlhpjizj/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_2 
dnzlhpjizj/hswofenhiy/Sigmoid_1Sigmoiddnzlhpjizj/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dnzlhpjizj/hswofenhiy/Sigmoid_1µ
dnzlhpjizj/hswofenhiy/mul_2Mul#dnzlhpjizj/hswofenhiy/Sigmoid_1:y:0dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_2
dnzlhpjizj/hswofenhiy/TanhTanh$dnzlhpjizj/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/Tanh¶
dnzlhpjizj/hswofenhiy/mul_3Mul!dnzlhpjizj/hswofenhiy/Sigmoid:y:0dnzlhpjizj/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_3·
dnzlhpjizj/hswofenhiy/add_3AddV2dnzlhpjizj/hswofenhiy/mul_2:z:0dnzlhpjizj/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_3¼
&dnzlhpjizj/hswofenhiy/ReadVariableOp_2ReadVariableOp/dnzlhpjizj_hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&dnzlhpjizj/hswofenhiy/ReadVariableOp_2Ä
dnzlhpjizj/hswofenhiy/mul_4Mul.dnzlhpjizj/hswofenhiy/ReadVariableOp_2:value:0dnzlhpjizj/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_4¼
dnzlhpjizj/hswofenhiy/add_4AddV2$dnzlhpjizj/hswofenhiy/split:output:3dnzlhpjizj/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_4 
dnzlhpjizj/hswofenhiy/Sigmoid_2Sigmoiddnzlhpjizj/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dnzlhpjizj/hswofenhiy/Sigmoid_2
dnzlhpjizj/hswofenhiy/Tanh_1Tanhdnzlhpjizj/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/Tanh_1º
dnzlhpjizj/hswofenhiy/mul_5Mul#dnzlhpjizj/hswofenhiy/Sigmoid_2:y:0 dnzlhpjizj/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_5¥
(dnzlhpjizj/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(dnzlhpjizj/TensorArrayV2_1/element_shapeä
dnzlhpjizj/TensorArrayV2_1TensorListReserve1dnzlhpjizj/TensorArrayV2_1/element_shape:output:0#dnzlhpjizj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dnzlhpjizj/TensorArrayV2_1d
dnzlhpjizj/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/time
#dnzlhpjizj/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#dnzlhpjizj/while/maximum_iterations
dnzlhpjizj/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/while/loop_counter²
dnzlhpjizj/whileWhile&dnzlhpjizj/while/loop_counter:output:0,dnzlhpjizj/while/maximum_iterations:output:0dnzlhpjizj/time:output:0#dnzlhpjizj/TensorArrayV2_1:handle:0dnzlhpjizj/zeros:output:0dnzlhpjizj/zeros_1:output:0#dnzlhpjizj/strided_slice_1:output:0Bdnzlhpjizj/TensorArrayUnstack/TensorListFromTensor:output_handle:04dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource6dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource5dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource-dnzlhpjizj_hswofenhiy_readvariableop_resource/dnzlhpjizj_hswofenhiy_readvariableop_1_resource/dnzlhpjizj_hswofenhiy_readvariableop_2_resource*
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
dnzlhpjizj_while_body_2596560*)
cond!R
dnzlhpjizj_while_cond_2596559*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
dnzlhpjizj/whileË
;dnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;dnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shape
-dnzlhpjizj/TensorArrayV2Stack/TensorListStackTensorListStackdnzlhpjizj/while:output:3Ddnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-dnzlhpjizj/TensorArrayV2Stack/TensorListStack
 dnzlhpjizj/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 dnzlhpjizj/strided_slice_3/stack
"dnzlhpjizj/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"dnzlhpjizj/strided_slice_3/stack_1
"dnzlhpjizj/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_3/stack_2Ü
dnzlhpjizj/strided_slice_3StridedSlice6dnzlhpjizj/TensorArrayV2Stack/TensorListStack:tensor:0)dnzlhpjizj/strided_slice_3/stack:output:0+dnzlhpjizj/strided_slice_3/stack_1:output:0+dnzlhpjizj/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
dnzlhpjizj/strided_slice_3
dnzlhpjizj/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dnzlhpjizj/transpose_1/permÑ
dnzlhpjizj/transpose_1	Transpose6dnzlhpjizj/TensorArrayV2Stack/TensorListStack:tensor:0$dnzlhpjizj/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/transpose_1n
nyosplwtfa/ShapeShapednzlhpjizj/transpose_1:y:0*
T0*
_output_shapes
:2
nyosplwtfa/Shape
nyosplwtfa/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
nyosplwtfa/strided_slice/stack
 nyosplwtfa/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 nyosplwtfa/strided_slice/stack_1
 nyosplwtfa/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 nyosplwtfa/strided_slice/stack_2¤
nyosplwtfa/strided_sliceStridedSlicenyosplwtfa/Shape:output:0'nyosplwtfa/strided_slice/stack:output:0)nyosplwtfa/strided_slice/stack_1:output:0)nyosplwtfa/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
nyosplwtfa/strided_slicer
nyosplwtfa/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros/mul/y
nyosplwtfa/zeros/mulMul!nyosplwtfa/strided_slice:output:0nyosplwtfa/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros/mulu
nyosplwtfa/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
nyosplwtfa/zeros/Less/y
nyosplwtfa/zeros/LessLessnyosplwtfa/zeros/mul:z:0 nyosplwtfa/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros/Lessx
nyosplwtfa/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros/packed/1¯
nyosplwtfa/zeros/packedPack!nyosplwtfa/strided_slice:output:0"nyosplwtfa/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
nyosplwtfa/zeros/packedu
nyosplwtfa/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
nyosplwtfa/zeros/Const¡
nyosplwtfa/zerosFill nyosplwtfa/zeros/packed:output:0nyosplwtfa/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/zerosv
nyosplwtfa/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros_1/mul/y
nyosplwtfa/zeros_1/mulMul!nyosplwtfa/strided_slice:output:0!nyosplwtfa/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros_1/muly
nyosplwtfa/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
nyosplwtfa/zeros_1/Less/y
nyosplwtfa/zeros_1/LessLessnyosplwtfa/zeros_1/mul:z:0"nyosplwtfa/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros_1/Less|
nyosplwtfa/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros_1/packed/1µ
nyosplwtfa/zeros_1/packedPack!nyosplwtfa/strided_slice:output:0$nyosplwtfa/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
nyosplwtfa/zeros_1/packedy
nyosplwtfa/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
nyosplwtfa/zeros_1/Const©
nyosplwtfa/zeros_1Fill"nyosplwtfa/zeros_1/packed:output:0!nyosplwtfa/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/zeros_1
nyosplwtfa/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
nyosplwtfa/transpose/perm¯
nyosplwtfa/transpose	Transposednzlhpjizj/transpose_1:y:0"nyosplwtfa/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/transposep
nyosplwtfa/Shape_1Shapenyosplwtfa/transpose:y:0*
T0*
_output_shapes
:2
nyosplwtfa/Shape_1
 nyosplwtfa/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 nyosplwtfa/strided_slice_1/stack
"nyosplwtfa/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_1/stack_1
"nyosplwtfa/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_1/stack_2°
nyosplwtfa/strided_slice_1StridedSlicenyosplwtfa/Shape_1:output:0)nyosplwtfa/strided_slice_1/stack:output:0+nyosplwtfa/strided_slice_1/stack_1:output:0+nyosplwtfa/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
nyosplwtfa/strided_slice_1
&nyosplwtfa/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&nyosplwtfa/TensorArrayV2/element_shapeÞ
nyosplwtfa/TensorArrayV2TensorListReserve/nyosplwtfa/TensorArrayV2/element_shape:output:0#nyosplwtfa/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
nyosplwtfa/TensorArrayV2Õ
@nyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@nyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2nyosplwtfa/TensorArrayUnstack/TensorListFromTensorTensorListFromTensornyosplwtfa/transpose:y:0Inyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2nyosplwtfa/TensorArrayUnstack/TensorListFromTensor
 nyosplwtfa/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 nyosplwtfa/strided_slice_2/stack
"nyosplwtfa/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_2/stack_1
"nyosplwtfa/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_2/stack_2¾
nyosplwtfa/strided_slice_2StridedSlicenyosplwtfa/transpose:y:0)nyosplwtfa/strided_slice_2/stack:output:0+nyosplwtfa/strided_slice_2/stack_1:output:0+nyosplwtfa/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
nyosplwtfa/strided_slice_2Ð
+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp4nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOpÓ
nyosplwtfa/lwptfvtmlx/MatMulMatMul#nyosplwtfa/strided_slice_2:output:03nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nyosplwtfa/lwptfvtmlx/MatMulÖ
-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp6nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOpÏ
nyosplwtfa/lwptfvtmlx/MatMul_1MatMulnyosplwtfa/zeros:output:05nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
nyosplwtfa/lwptfvtmlx/MatMul_1Ä
nyosplwtfa/lwptfvtmlx/addAddV2&nyosplwtfa/lwptfvtmlx/MatMul:product:0(nyosplwtfa/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nyosplwtfa/lwptfvtmlx/addÏ
,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp5nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOpÑ
nyosplwtfa/lwptfvtmlx/BiasAddBiasAddnyosplwtfa/lwptfvtmlx/add:z:04nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nyosplwtfa/lwptfvtmlx/BiasAdd
%nyosplwtfa/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%nyosplwtfa/lwptfvtmlx/split/split_dim
nyosplwtfa/lwptfvtmlx/splitSplit.nyosplwtfa/lwptfvtmlx/split/split_dim:output:0&nyosplwtfa/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nyosplwtfa/lwptfvtmlx/split¶
$nyosplwtfa/lwptfvtmlx/ReadVariableOpReadVariableOp-nyosplwtfa_lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02&
$nyosplwtfa/lwptfvtmlx/ReadVariableOpº
nyosplwtfa/lwptfvtmlx/mulMul,nyosplwtfa/lwptfvtmlx/ReadVariableOp:value:0nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mulº
nyosplwtfa/lwptfvtmlx/add_1AddV2$nyosplwtfa/lwptfvtmlx/split:output:0nyosplwtfa/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_1
nyosplwtfa/lwptfvtmlx/SigmoidSigmoidnyosplwtfa/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/Sigmoid¼
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1ReadVariableOp/nyosplwtfa_lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1À
nyosplwtfa/lwptfvtmlx/mul_1Mul.nyosplwtfa/lwptfvtmlx/ReadVariableOp_1:value:0nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_1¼
nyosplwtfa/lwptfvtmlx/add_2AddV2$nyosplwtfa/lwptfvtmlx/split:output:1nyosplwtfa/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_2 
nyosplwtfa/lwptfvtmlx/Sigmoid_1Sigmoidnyosplwtfa/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
nyosplwtfa/lwptfvtmlx/Sigmoid_1µ
nyosplwtfa/lwptfvtmlx/mul_2Mul#nyosplwtfa/lwptfvtmlx/Sigmoid_1:y:0nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_2
nyosplwtfa/lwptfvtmlx/TanhTanh$nyosplwtfa/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/Tanh¶
nyosplwtfa/lwptfvtmlx/mul_3Mul!nyosplwtfa/lwptfvtmlx/Sigmoid:y:0nyosplwtfa/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_3·
nyosplwtfa/lwptfvtmlx/add_3AddV2nyosplwtfa/lwptfvtmlx/mul_2:z:0nyosplwtfa/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_3¼
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2ReadVariableOp/nyosplwtfa_lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2Ä
nyosplwtfa/lwptfvtmlx/mul_4Mul.nyosplwtfa/lwptfvtmlx/ReadVariableOp_2:value:0nyosplwtfa/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_4¼
nyosplwtfa/lwptfvtmlx/add_4AddV2$nyosplwtfa/lwptfvtmlx/split:output:3nyosplwtfa/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_4 
nyosplwtfa/lwptfvtmlx/Sigmoid_2Sigmoidnyosplwtfa/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
nyosplwtfa/lwptfvtmlx/Sigmoid_2
nyosplwtfa/lwptfvtmlx/Tanh_1Tanhnyosplwtfa/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/Tanh_1º
nyosplwtfa/lwptfvtmlx/mul_5Mul#nyosplwtfa/lwptfvtmlx/Sigmoid_2:y:0 nyosplwtfa/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_5¥
(nyosplwtfa/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(nyosplwtfa/TensorArrayV2_1/element_shapeä
nyosplwtfa/TensorArrayV2_1TensorListReserve1nyosplwtfa/TensorArrayV2_1/element_shape:output:0#nyosplwtfa/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
nyosplwtfa/TensorArrayV2_1d
nyosplwtfa/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/time
#nyosplwtfa/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#nyosplwtfa/while/maximum_iterations
nyosplwtfa/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/while/loop_counter²
nyosplwtfa/whileWhile&nyosplwtfa/while/loop_counter:output:0,nyosplwtfa/while/maximum_iterations:output:0nyosplwtfa/time:output:0#nyosplwtfa/TensorArrayV2_1:handle:0nyosplwtfa/zeros:output:0nyosplwtfa/zeros_1:output:0#nyosplwtfa/strided_slice_1:output:0Bnyosplwtfa/TensorArrayUnstack/TensorListFromTensor:output_handle:04nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource6nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource5nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource-nyosplwtfa_lwptfvtmlx_readvariableop_resource/nyosplwtfa_lwptfvtmlx_readvariableop_1_resource/nyosplwtfa_lwptfvtmlx_readvariableop_2_resource*
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
nyosplwtfa_while_body_2596736*)
cond!R
nyosplwtfa_while_cond_2596735*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
nyosplwtfa/whileË
;nyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;nyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shape
-nyosplwtfa/TensorArrayV2Stack/TensorListStackTensorListStacknyosplwtfa/while:output:3Dnyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-nyosplwtfa/TensorArrayV2Stack/TensorListStack
 nyosplwtfa/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 nyosplwtfa/strided_slice_3/stack
"nyosplwtfa/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"nyosplwtfa/strided_slice_3/stack_1
"nyosplwtfa/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_3/stack_2Ü
nyosplwtfa/strided_slice_3StridedSlice6nyosplwtfa/TensorArrayV2Stack/TensorListStack:tensor:0)nyosplwtfa/strided_slice_3/stack:output:0+nyosplwtfa/strided_slice_3/stack_1:output:0+nyosplwtfa/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
nyosplwtfa/strided_slice_3
nyosplwtfa/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
nyosplwtfa/transpose_1/permÑ
nyosplwtfa/transpose_1	Transpose6nyosplwtfa/TensorArrayV2Stack/TensorListStack:tensor:0$nyosplwtfa/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/transpose_1®
 chsgvefspq/MatMul/ReadVariableOpReadVariableOp)chsgvefspq_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 chsgvefspq/MatMul/ReadVariableOp±
chsgvefspq/MatMulMatMul#nyosplwtfa/strided_slice_3:output:0(chsgvefspq/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
chsgvefspq/MatMul­
!chsgvefspq/BiasAdd/ReadVariableOpReadVariableOp*chsgvefspq_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!chsgvefspq/BiasAdd/ReadVariableOp­
chsgvefspq/BiasAddBiasAddchsgvefspq/MatMul:product:0)chsgvefspq/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
chsgvefspq/BiasAddÏ
IdentityIdentitychsgvefspq/BiasAdd:output:0"^chsgvefspq/BiasAdd/ReadVariableOp!^chsgvefspq/MatMul/ReadVariableOp-^dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp,^dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp.^dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp%^dnzlhpjizj/hswofenhiy/ReadVariableOp'^dnzlhpjizj/hswofenhiy/ReadVariableOp_1'^dnzlhpjizj/hswofenhiy/ReadVariableOp_2^dnzlhpjizj/while.^gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp5^gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp-^nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp,^nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp.^nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp%^nyosplwtfa/lwptfvtmlx/ReadVariableOp'^nyosplwtfa/lwptfvtmlx/ReadVariableOp_1'^nyosplwtfa/lwptfvtmlx/ReadVariableOp_2^nyosplwtfa/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2F
!chsgvefspq/BiasAdd/ReadVariableOp!chsgvefspq/BiasAdd/ReadVariableOp2D
 chsgvefspq/MatMul/ReadVariableOp chsgvefspq/MatMul/ReadVariableOp2\
,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp2Z
+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp2^
-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp2L
$dnzlhpjizj/hswofenhiy/ReadVariableOp$dnzlhpjizj/hswofenhiy/ReadVariableOp2P
&dnzlhpjizj/hswofenhiy/ReadVariableOp_1&dnzlhpjizj/hswofenhiy/ReadVariableOp_12P
&dnzlhpjizj/hswofenhiy/ReadVariableOp_2&dnzlhpjizj/hswofenhiy/ReadVariableOp_22$
dnzlhpjizj/whilednzlhpjizj/while2^
-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp2l
4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp2Z
+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp2^
-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp2L
$nyosplwtfa/lwptfvtmlx/ReadVariableOp$nyosplwtfa/lwptfvtmlx/ReadVariableOp2P
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1&nyosplwtfa/lwptfvtmlx/ReadVariableOp_12P
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2&nyosplwtfa/lwptfvtmlx/ReadVariableOp_22$
nyosplwtfa/whilenyosplwtfa/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_2598431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2598431___redundant_placeholder05
1while_while_cond_2598431___redundant_placeholder15
1while_while_cond_2598431___redundant_placeholder25
1while_while_cond_2598431___redundant_placeholder35
1while_while_cond_2598431___redundant_placeholder45
1while_while_cond_2598431___redundant_placeholder55
1while_while_cond_2598431___redundant_placeholder6
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

,__inference_sequential_layer_call_fn_2596312

bdeyofgzkq
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
bdeyofgzkqunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_25962402
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
bdeyofgzkq
àY

while_body_2595539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lwptfvtmlx_matmul_readvariableop_resource_0:	 F
3while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 A
2while_lwptfvtmlx_biasadd_readvariableop_resource_0:	8
*while_lwptfvtmlx_readvariableop_resource_0: :
,while_lwptfvtmlx_readvariableop_1_resource_0: :
,while_lwptfvtmlx_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lwptfvtmlx_matmul_readvariableop_resource:	 D
1while_lwptfvtmlx_matmul_1_readvariableop_resource:	 ?
0while_lwptfvtmlx_biasadd_readvariableop_resource:	6
(while_lwptfvtmlx_readvariableop_resource: 8
*while_lwptfvtmlx_readvariableop_1_resource: 8
*while_lwptfvtmlx_readvariableop_2_resource: ¢'while/lwptfvtmlx/BiasAdd/ReadVariableOp¢&while/lwptfvtmlx/MatMul/ReadVariableOp¢(while/lwptfvtmlx/MatMul_1/ReadVariableOp¢while/lwptfvtmlx/ReadVariableOp¢!while/lwptfvtmlx/ReadVariableOp_1¢!while/lwptfvtmlx/ReadVariableOp_2Ã
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
&while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp1while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/lwptfvtmlx/MatMul/ReadVariableOpÑ
while/lwptfvtmlx/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMulÉ
(while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp3while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/lwptfvtmlx/MatMul_1/ReadVariableOpº
while/lwptfvtmlx/MatMul_1MatMulwhile_placeholder_20while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMul_1°
while/lwptfvtmlx/addAddV2!while/lwptfvtmlx/MatMul:product:0#while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/addÂ
'while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp2while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/lwptfvtmlx/BiasAdd/ReadVariableOp½
while/lwptfvtmlx/BiasAddBiasAddwhile/lwptfvtmlx/add:z:0/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/BiasAdd
 while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/lwptfvtmlx/split/split_dim
while/lwptfvtmlx/splitSplit)while/lwptfvtmlx/split/split_dim:output:0!while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/lwptfvtmlx/split©
while/lwptfvtmlx/ReadVariableOpReadVariableOp*while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/lwptfvtmlx/ReadVariableOp£
while/lwptfvtmlx/mulMul'while/lwptfvtmlx/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul¦
while/lwptfvtmlx/add_1AddV2while/lwptfvtmlx/split:output:0while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_1
while/lwptfvtmlx/SigmoidSigmoidwhile/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid¯
!while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp,while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_1©
while/lwptfvtmlx/mul_1Mul)while/lwptfvtmlx/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_1¨
while/lwptfvtmlx/add_2AddV2while/lwptfvtmlx/split:output:1while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_2
while/lwptfvtmlx/Sigmoid_1Sigmoidwhile/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_1
while/lwptfvtmlx/mul_2Mulwhile/lwptfvtmlx/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_2
while/lwptfvtmlx/TanhTanhwhile/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh¢
while/lwptfvtmlx/mul_3Mulwhile/lwptfvtmlx/Sigmoid:y:0while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_3£
while/lwptfvtmlx/add_3AddV2while/lwptfvtmlx/mul_2:z:0while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_3¯
!while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp,while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_2°
while/lwptfvtmlx/mul_4Mul)while/lwptfvtmlx/ReadVariableOp_2:value:0while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_4¨
while/lwptfvtmlx/add_4AddV2while/lwptfvtmlx/split:output:3while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_4
while/lwptfvtmlx/Sigmoid_2Sigmoidwhile/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_2
while/lwptfvtmlx/Tanh_1Tanhwhile/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh_1¦
while/lwptfvtmlx/mul_5Mulwhile/lwptfvtmlx/Sigmoid_2:y:0while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lwptfvtmlx/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lwptfvtmlx/mul_5:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/lwptfvtmlx/add_3:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
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
0while_lwptfvtmlx_biasadd_readvariableop_resource2while_lwptfvtmlx_biasadd_readvariableop_resource_0"h
1while_lwptfvtmlx_matmul_1_readvariableop_resource3while_lwptfvtmlx_matmul_1_readvariableop_resource_0"d
/while_lwptfvtmlx_matmul_readvariableop_resource1while_lwptfvtmlx_matmul_readvariableop_resource_0"Z
*while_lwptfvtmlx_readvariableop_1_resource,while_lwptfvtmlx_readvariableop_1_resource_0"Z
*while_lwptfvtmlx_readvariableop_2_resource,while_lwptfvtmlx_readvariableop_2_resource_0"V
(while_lwptfvtmlx_readvariableop_resource*while_lwptfvtmlx_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/lwptfvtmlx/BiasAdd/ReadVariableOp'while/lwptfvtmlx/BiasAdd/ReadVariableOp2P
&while/lwptfvtmlx/MatMul/ReadVariableOp&while/lwptfvtmlx/MatMul/ReadVariableOp2T
(while/lwptfvtmlx/MatMul_1/ReadVariableOp(while/lwptfvtmlx/MatMul_1/ReadVariableOp2B
while/lwptfvtmlx/ReadVariableOpwhile/lwptfvtmlx/ReadVariableOp2F
!while/lwptfvtmlx/ReadVariableOp_1!while/lwptfvtmlx/ReadVariableOp_12F
!while/lwptfvtmlx/ReadVariableOp_2!while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2595345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2595345___redundant_placeholder05
1while_while_cond_2595345___redundant_placeholder15
1while_while_cond_2595345___redundant_placeholder25
1while_while_cond_2595345___redundant_placeholder35
1while_while_cond_2595345___redundant_placeholder45
1while_while_cond_2595345___redundant_placeholder55
1while_while_cond_2595345___redundant_placeholder6
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
¤

,__inference_chsgvefspq_layer_call_fn_2598980

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
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_25956642
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
©0
¼
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_2595247

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
Üh

G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598353
inputs_0<
)lwptfvtmlx_matmul_readvariableop_resource:	 >
+lwptfvtmlx_matmul_1_readvariableop_resource:	 9
*lwptfvtmlx_biasadd_readvariableop_resource:	0
"lwptfvtmlx_readvariableop_resource: 2
$lwptfvtmlx_readvariableop_1_resource: 2
$lwptfvtmlx_readvariableop_2_resource: 
identity¢!lwptfvtmlx/BiasAdd/ReadVariableOp¢ lwptfvtmlx/MatMul/ReadVariableOp¢"lwptfvtmlx/MatMul_1/ReadVariableOp¢lwptfvtmlx/ReadVariableOp¢lwptfvtmlx/ReadVariableOp_1¢lwptfvtmlx/ReadVariableOp_2¢whileF
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
 lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp)lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 lwptfvtmlx/MatMul/ReadVariableOp§
lwptfvtmlx/MatMulMatMulstrided_slice_2:output:0(lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMulµ
"lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp+lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lwptfvtmlx/MatMul_1/ReadVariableOp£
lwptfvtmlx/MatMul_1MatMulzeros:output:0*lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMul_1
lwptfvtmlx/addAddV2lwptfvtmlx/MatMul:product:0lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/add®
!lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp*lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!lwptfvtmlx/BiasAdd/ReadVariableOp¥
lwptfvtmlx/BiasAddBiasAddlwptfvtmlx/add:z:0)lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/BiasAddz
lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lwptfvtmlx/split/split_dimë
lwptfvtmlx/splitSplit#lwptfvtmlx/split/split_dim:output:0lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lwptfvtmlx/split
lwptfvtmlx/ReadVariableOpReadVariableOp"lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp
lwptfvtmlx/mulMul!lwptfvtmlx/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul
lwptfvtmlx/add_1AddV2lwptfvtmlx/split:output:0lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_1{
lwptfvtmlx/SigmoidSigmoidlwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid
lwptfvtmlx/ReadVariableOp_1ReadVariableOp$lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_1
lwptfvtmlx/mul_1Mul#lwptfvtmlx/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_1
lwptfvtmlx/add_2AddV2lwptfvtmlx/split:output:1lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_2
lwptfvtmlx/Sigmoid_1Sigmoidlwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_1
lwptfvtmlx/mul_2Mullwptfvtmlx/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_2w
lwptfvtmlx/TanhTanhlwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh
lwptfvtmlx/mul_3Mullwptfvtmlx/Sigmoid:y:0lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_3
lwptfvtmlx/add_3AddV2lwptfvtmlx/mul_2:z:0lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_3
lwptfvtmlx/ReadVariableOp_2ReadVariableOp$lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_2
lwptfvtmlx/mul_4Mul#lwptfvtmlx/ReadVariableOp_2:value:0lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_4
lwptfvtmlx/add_4AddV2lwptfvtmlx/split:output:3lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_4
lwptfvtmlx/Sigmoid_2Sigmoidlwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_2v
lwptfvtmlx/Tanh_1Tanhlwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh_1
lwptfvtmlx/mul_5Mullwptfvtmlx/Sigmoid_2:y:0lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lwptfvtmlx_matmul_readvariableop_resource+lwptfvtmlx_matmul_1_readvariableop_resource*lwptfvtmlx_biasadd_readvariableop_resource"lwptfvtmlx_readvariableop_resource$lwptfvtmlx_readvariableop_1_resource$lwptfvtmlx_readvariableop_2_resource*
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
while_body_2598252*
condR
while_cond_2598251*Q
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
IdentityIdentitystrided_slice_3:output:0"^lwptfvtmlx/BiasAdd/ReadVariableOp!^lwptfvtmlx/MatMul/ReadVariableOp#^lwptfvtmlx/MatMul_1/ReadVariableOp^lwptfvtmlx/ReadVariableOp^lwptfvtmlx/ReadVariableOp_1^lwptfvtmlx/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!lwptfvtmlx/BiasAdd/ReadVariableOp!lwptfvtmlx/BiasAdd/ReadVariableOp2D
 lwptfvtmlx/MatMul/ReadVariableOp lwptfvtmlx/MatMul/ReadVariableOp2H
"lwptfvtmlx/MatMul_1/ReadVariableOp"lwptfvtmlx/MatMul_1/ReadVariableOp26
lwptfvtmlx/ReadVariableOplwptfvtmlx/ReadVariableOp2:
lwptfvtmlx/ReadVariableOp_1lwptfvtmlx/ReadVariableOp_12:
lwptfvtmlx/ReadVariableOp_2lwptfvtmlx/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0


dnzlhpjizj_while_cond_25965592
.dnzlhpjizj_while_dnzlhpjizj_while_loop_counter8
4dnzlhpjizj_while_dnzlhpjizj_while_maximum_iterations 
dnzlhpjizj_while_placeholder"
dnzlhpjizj_while_placeholder_1"
dnzlhpjizj_while_placeholder_2"
dnzlhpjizj_while_placeholder_34
0dnzlhpjizj_while_less_dnzlhpjizj_strided_slice_1K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596559___redundant_placeholder0K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596559___redundant_placeholder1K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596559___redundant_placeholder2K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596559___redundant_placeholder3K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596559___redundant_placeholder4K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596559___redundant_placeholder5K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596559___redundant_placeholder6
dnzlhpjizj_while_identity
§
dnzlhpjizj/while/LessLessdnzlhpjizj_while_placeholder0dnzlhpjizj_while_less_dnzlhpjizj_strided_slice_1*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Less~
dnzlhpjizj/while/IdentityIdentitydnzlhpjizj/while/Less:z:0*
T0
*
_output_shapes
: 2
dnzlhpjizj/while/Identity"?
dnzlhpjizj_while_identity"dnzlhpjizj/while/Identity:output:0*(
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
G__inference_sequential_layer_call_and_return_conditional_losses_2595671

inputs(
gtjikcltwy_2595248: 
gtjikcltwy_2595250:%
dnzlhpjizj_2595448:	%
dnzlhpjizj_2595450:	 !
dnzlhpjizj_2595452:	 
dnzlhpjizj_2595454:  
dnzlhpjizj_2595456:  
dnzlhpjizj_2595458: %
nyosplwtfa_2595641:	 %
nyosplwtfa_2595643:	 !
nyosplwtfa_2595645:	 
nyosplwtfa_2595647:  
nyosplwtfa_2595649:  
nyosplwtfa_2595651: $
chsgvefspq_2595665:  
chsgvefspq_2595667:
identity¢"chsgvefspq/StatefulPartitionedCall¢"dnzlhpjizj/StatefulPartitionedCall¢"gtjikcltwy/StatefulPartitionedCall¢"nyosplwtfa/StatefulPartitionedCall¬
"gtjikcltwy/StatefulPartitionedCallStatefulPartitionedCallinputsgtjikcltwy_2595248gtjikcltwy_2595250*
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
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_25952472$
"gtjikcltwy/StatefulPartitionedCall
ezubtmdnwx/PartitionedCallPartitionedCall+gtjikcltwy/StatefulPartitionedCall:output:0*
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
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_25952662
ezubtmdnwx/PartitionedCall
"dnzlhpjizj/StatefulPartitionedCallStatefulPartitionedCall#ezubtmdnwx/PartitionedCall:output:0dnzlhpjizj_2595448dnzlhpjizj_2595450dnzlhpjizj_2595452dnzlhpjizj_2595454dnzlhpjizj_2595456dnzlhpjizj_2595458*
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25954472$
"dnzlhpjizj/StatefulPartitionedCall¡
"nyosplwtfa/StatefulPartitionedCallStatefulPartitionedCall+dnzlhpjizj/StatefulPartitionedCall:output:0nyosplwtfa_2595641nyosplwtfa_2595643nyosplwtfa_2595645nyosplwtfa_2595647nyosplwtfa_2595649nyosplwtfa_2595651*
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25956402$
"nyosplwtfa/StatefulPartitionedCallÉ
"chsgvefspq/StatefulPartitionedCallStatefulPartitionedCall+nyosplwtfa/StatefulPartitionedCall:output:0chsgvefspq_2595665chsgvefspq_2595667*
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
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_25956642$
"chsgvefspq/StatefulPartitionedCall
IdentityIdentity+chsgvefspq/StatefulPartitionedCall:output:0#^chsgvefspq/StatefulPartitionedCall#^dnzlhpjizj/StatefulPartitionedCall#^gtjikcltwy/StatefulPartitionedCall#^nyosplwtfa/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"chsgvefspq/StatefulPartitionedCall"chsgvefspq/StatefulPartitionedCall2H
"dnzlhpjizj/StatefulPartitionedCall"dnzlhpjizj/StatefulPartitionedCall2H
"gtjikcltwy/StatefulPartitionedCall"gtjikcltwy/StatefulPartitionedCall2H
"nyosplwtfa/StatefulPartitionedCall"nyosplwtfa/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç)
Ò
while_body_2594815
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lwptfvtmlx_2594839_0:	 -
while_lwptfvtmlx_2594841_0:	 )
while_lwptfvtmlx_2594843_0:	(
while_lwptfvtmlx_2594845_0: (
while_lwptfvtmlx_2594847_0: (
while_lwptfvtmlx_2594849_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lwptfvtmlx_2594839:	 +
while_lwptfvtmlx_2594841:	 '
while_lwptfvtmlx_2594843:	&
while_lwptfvtmlx_2594845: &
while_lwptfvtmlx_2594847: &
while_lwptfvtmlx_2594849: ¢(while/lwptfvtmlx/StatefulPartitionedCallÃ
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
(while/lwptfvtmlx/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lwptfvtmlx_2594839_0while_lwptfvtmlx_2594841_0while_lwptfvtmlx_2594843_0while_lwptfvtmlx_2594845_0while_lwptfvtmlx_2594847_0while_lwptfvtmlx_2594849_0*
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_25947192*
(while/lwptfvtmlx/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/lwptfvtmlx/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/lwptfvtmlx/StatefulPartitionedCall:output:1)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/lwptfvtmlx/StatefulPartitionedCall:output:2)^while/lwptfvtmlx/StatefulPartitionedCall*
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
while_lwptfvtmlx_2594839while_lwptfvtmlx_2594839_0"6
while_lwptfvtmlx_2594841while_lwptfvtmlx_2594841_0"6
while_lwptfvtmlx_2594843while_lwptfvtmlx_2594843_0"6
while_lwptfvtmlx_2594845while_lwptfvtmlx_2594845_0"6
while_lwptfvtmlx_2594847while_lwptfvtmlx_2594847_0"6
while_lwptfvtmlx_2594849while_lwptfvtmlx_2594849_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/lwptfvtmlx/StatefulPartitionedCall(while/lwptfvtmlx/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
dnzlhpjizj_while_body_25969642
.dnzlhpjizj_while_dnzlhpjizj_while_loop_counter8
4dnzlhpjizj_while_dnzlhpjizj_while_maximum_iterations 
dnzlhpjizj_while_placeholder"
dnzlhpjizj_while_placeholder_1"
dnzlhpjizj_while_placeholder_2"
dnzlhpjizj_while_placeholder_31
-dnzlhpjizj_while_dnzlhpjizj_strided_slice_1_0m
idnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0O
<dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0:	Q
>dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0:	 L
=dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0:	C
5dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0: E
7dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0: E
7dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0: 
dnzlhpjizj_while_identity
dnzlhpjizj_while_identity_1
dnzlhpjizj_while_identity_2
dnzlhpjizj_while_identity_3
dnzlhpjizj_while_identity_4
dnzlhpjizj_while_identity_5/
+dnzlhpjizj_while_dnzlhpjizj_strided_slice_1k
gdnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensorM
:dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource:	O
<dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource:	 J
;dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource:	A
3dnzlhpjizj_while_hswofenhiy_readvariableop_resource: C
5dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource: C
5dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource: ¢2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp¢1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp¢3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp¢*dnzlhpjizj/while/hswofenhiy/ReadVariableOp¢,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1¢,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2Ù
Bdnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bdnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shape
4dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemidnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0dnzlhpjizj_while_placeholderKdnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItemä
1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp<dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOpý
"dnzlhpjizj/while/hswofenhiy/MatMulMatMul;dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem:item:09dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"dnzlhpjizj/while/hswofenhiy/MatMulê
3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp>dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOpæ
$dnzlhpjizj/while/hswofenhiy/MatMul_1MatMuldnzlhpjizj_while_placeholder_2;dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$dnzlhpjizj/while/hswofenhiy/MatMul_1Ü
dnzlhpjizj/while/hswofenhiy/addAddV2,dnzlhpjizj/while/hswofenhiy/MatMul:product:0.dnzlhpjizj/while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dnzlhpjizj/while/hswofenhiy/addã
2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp=dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOpé
#dnzlhpjizj/while/hswofenhiy/BiasAddBiasAdd#dnzlhpjizj/while/hswofenhiy/add:z:0:dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#dnzlhpjizj/while/hswofenhiy/BiasAdd
+dnzlhpjizj/while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+dnzlhpjizj/while/hswofenhiy/split/split_dim¯
!dnzlhpjizj/while/hswofenhiy/splitSplit4dnzlhpjizj/while/hswofenhiy/split/split_dim:output:0,dnzlhpjizj/while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!dnzlhpjizj/while/hswofenhiy/splitÊ
*dnzlhpjizj/while/hswofenhiy/ReadVariableOpReadVariableOp5dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*dnzlhpjizj/while/hswofenhiy/ReadVariableOpÏ
dnzlhpjizj/while/hswofenhiy/mulMul2dnzlhpjizj/while/hswofenhiy/ReadVariableOp:value:0dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dnzlhpjizj/while/hswofenhiy/mulÒ
!dnzlhpjizj/while/hswofenhiy/add_1AddV2*dnzlhpjizj/while/hswofenhiy/split:output:0#dnzlhpjizj/while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_1®
#dnzlhpjizj/while/hswofenhiy/SigmoidSigmoid%dnzlhpjizj/while/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#dnzlhpjizj/while/hswofenhiy/SigmoidÐ
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1ReadVariableOp7dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1Õ
!dnzlhpjizj/while/hswofenhiy/mul_1Mul4dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1:value:0dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_1Ô
!dnzlhpjizj/while/hswofenhiy/add_2AddV2*dnzlhpjizj/while/hswofenhiy/split:output:1%dnzlhpjizj/while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_2²
%dnzlhpjizj/while/hswofenhiy/Sigmoid_1Sigmoid%dnzlhpjizj/while/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%dnzlhpjizj/while/hswofenhiy/Sigmoid_1Ê
!dnzlhpjizj/while/hswofenhiy/mul_2Mul)dnzlhpjizj/while/hswofenhiy/Sigmoid_1:y:0dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_2ª
 dnzlhpjizj/while/hswofenhiy/TanhTanh*dnzlhpjizj/while/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 dnzlhpjizj/while/hswofenhiy/TanhÎ
!dnzlhpjizj/while/hswofenhiy/mul_3Mul'dnzlhpjizj/while/hswofenhiy/Sigmoid:y:0$dnzlhpjizj/while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_3Ï
!dnzlhpjizj/while/hswofenhiy/add_3AddV2%dnzlhpjizj/while/hswofenhiy/mul_2:z:0%dnzlhpjizj/while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_3Ð
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2ReadVariableOp7dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2Ü
!dnzlhpjizj/while/hswofenhiy/mul_4Mul4dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2:value:0%dnzlhpjizj/while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_4Ô
!dnzlhpjizj/while/hswofenhiy/add_4AddV2*dnzlhpjizj/while/hswofenhiy/split:output:3%dnzlhpjizj/while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_4²
%dnzlhpjizj/while/hswofenhiy/Sigmoid_2Sigmoid%dnzlhpjizj/while/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%dnzlhpjizj/while/hswofenhiy/Sigmoid_2©
"dnzlhpjizj/while/hswofenhiy/Tanh_1Tanh%dnzlhpjizj/while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"dnzlhpjizj/while/hswofenhiy/Tanh_1Ò
!dnzlhpjizj/while/hswofenhiy/mul_5Mul)dnzlhpjizj/while/hswofenhiy/Sigmoid_2:y:0&dnzlhpjizj/while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_5
5dnzlhpjizj/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemdnzlhpjizj_while_placeholder_1dnzlhpjizj_while_placeholder%dnzlhpjizj/while/hswofenhiy/mul_5:z:0*
_output_shapes
: *
element_dtype027
5dnzlhpjizj/while/TensorArrayV2Write/TensorListSetItemr
dnzlhpjizj/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dnzlhpjizj/while/add/y
dnzlhpjizj/while/addAddV2dnzlhpjizj_while_placeholderdnzlhpjizj/while/add/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/while/addv
dnzlhpjizj/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dnzlhpjizj/while/add_1/y­
dnzlhpjizj/while/add_1AddV2.dnzlhpjizj_while_dnzlhpjizj_while_loop_counter!dnzlhpjizj/while/add_1/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/while/add_1©
dnzlhpjizj/while/IdentityIdentitydnzlhpjizj/while/add_1:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/IdentityÇ
dnzlhpjizj/while/Identity_1Identity4dnzlhpjizj_while_dnzlhpjizj_while_maximum_iterations3^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Identity_1«
dnzlhpjizj/while/Identity_2Identitydnzlhpjizj/while/add:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Identity_2Ø
dnzlhpjizj/while/Identity_3IdentityEdnzlhpjizj/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Identity_3É
dnzlhpjizj/while/Identity_4Identity%dnzlhpjizj/while/hswofenhiy/mul_5:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/while/Identity_4É
dnzlhpjizj/while/Identity_5Identity%dnzlhpjizj/while/hswofenhiy/add_3:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/while/Identity_5"\
+dnzlhpjizj_while_dnzlhpjizj_strided_slice_1-dnzlhpjizj_while_dnzlhpjizj_strided_slice_1_0"|
;dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource=dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0"~
<dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource>dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0"z
:dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource<dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0"p
5dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource7dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0"p
5dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource7dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0"l
3dnzlhpjizj_while_hswofenhiy_readvariableop_resource5dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0"?
dnzlhpjizj_while_identity"dnzlhpjizj/while/Identity:output:0"C
dnzlhpjizj_while_identity_1$dnzlhpjizj/while/Identity_1:output:0"C
dnzlhpjizj_while_identity_2$dnzlhpjizj/while/Identity_2:output:0"C
dnzlhpjizj_while_identity_3$dnzlhpjizj/while/Identity_3:output:0"C
dnzlhpjizj_while_identity_4$dnzlhpjizj/while/Identity_4:output:0"C
dnzlhpjizj_while_identity_5$dnzlhpjizj/while/Identity_5:output:0"Ô
gdnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensoridnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2f
1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp2j
3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp2X
*dnzlhpjizj/while/hswofenhiy/ReadVariableOp*dnzlhpjizj/while/hswofenhiy/ReadVariableOp2\
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_12\
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598893

inputs<
)lwptfvtmlx_matmul_readvariableop_resource:	 >
+lwptfvtmlx_matmul_1_readvariableop_resource:	 9
*lwptfvtmlx_biasadd_readvariableop_resource:	0
"lwptfvtmlx_readvariableop_resource: 2
$lwptfvtmlx_readvariableop_1_resource: 2
$lwptfvtmlx_readvariableop_2_resource: 
identity¢!lwptfvtmlx/BiasAdd/ReadVariableOp¢ lwptfvtmlx/MatMul/ReadVariableOp¢"lwptfvtmlx/MatMul_1/ReadVariableOp¢lwptfvtmlx/ReadVariableOp¢lwptfvtmlx/ReadVariableOp_1¢lwptfvtmlx/ReadVariableOp_2¢whileD
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
 lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp)lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 lwptfvtmlx/MatMul/ReadVariableOp§
lwptfvtmlx/MatMulMatMulstrided_slice_2:output:0(lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMulµ
"lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp+lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lwptfvtmlx/MatMul_1/ReadVariableOp£
lwptfvtmlx/MatMul_1MatMulzeros:output:0*lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMul_1
lwptfvtmlx/addAddV2lwptfvtmlx/MatMul:product:0lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/add®
!lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp*lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!lwptfvtmlx/BiasAdd/ReadVariableOp¥
lwptfvtmlx/BiasAddBiasAddlwptfvtmlx/add:z:0)lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/BiasAddz
lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lwptfvtmlx/split/split_dimë
lwptfvtmlx/splitSplit#lwptfvtmlx/split/split_dim:output:0lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lwptfvtmlx/split
lwptfvtmlx/ReadVariableOpReadVariableOp"lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp
lwptfvtmlx/mulMul!lwptfvtmlx/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul
lwptfvtmlx/add_1AddV2lwptfvtmlx/split:output:0lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_1{
lwptfvtmlx/SigmoidSigmoidlwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid
lwptfvtmlx/ReadVariableOp_1ReadVariableOp$lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_1
lwptfvtmlx/mul_1Mul#lwptfvtmlx/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_1
lwptfvtmlx/add_2AddV2lwptfvtmlx/split:output:1lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_2
lwptfvtmlx/Sigmoid_1Sigmoidlwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_1
lwptfvtmlx/mul_2Mullwptfvtmlx/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_2w
lwptfvtmlx/TanhTanhlwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh
lwptfvtmlx/mul_3Mullwptfvtmlx/Sigmoid:y:0lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_3
lwptfvtmlx/add_3AddV2lwptfvtmlx/mul_2:z:0lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_3
lwptfvtmlx/ReadVariableOp_2ReadVariableOp$lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_2
lwptfvtmlx/mul_4Mul#lwptfvtmlx/ReadVariableOp_2:value:0lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_4
lwptfvtmlx/add_4AddV2lwptfvtmlx/split:output:3lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_4
lwptfvtmlx/Sigmoid_2Sigmoidlwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_2v
lwptfvtmlx/Tanh_1Tanhlwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh_1
lwptfvtmlx/mul_5Mullwptfvtmlx/Sigmoid_2:y:0lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lwptfvtmlx_matmul_readvariableop_resource+lwptfvtmlx_matmul_1_readvariableop_resource*lwptfvtmlx_biasadd_readvariableop_resource"lwptfvtmlx_readvariableop_resource$lwptfvtmlx_readvariableop_1_resource$lwptfvtmlx_readvariableop_2_resource*
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
while_body_2598792*
condR
while_cond_2598791*Q
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
IdentityIdentitystrided_slice_3:output:0"^lwptfvtmlx/BiasAdd/ReadVariableOp!^lwptfvtmlx/MatMul/ReadVariableOp#^lwptfvtmlx/MatMul_1/ReadVariableOp^lwptfvtmlx/ReadVariableOp^lwptfvtmlx/ReadVariableOp_1^lwptfvtmlx/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!lwptfvtmlx/BiasAdd/ReadVariableOp!lwptfvtmlx/BiasAdd/ReadVariableOp2D
 lwptfvtmlx/MatMul/ReadVariableOp lwptfvtmlx/MatMul/ReadVariableOp2H
"lwptfvtmlx/MatMul_1/ReadVariableOp"lwptfvtmlx/MatMul_1/ReadVariableOp26
lwptfvtmlx/ReadVariableOplwptfvtmlx/ReadVariableOp2:
lwptfvtmlx/ReadVariableOp_1lwptfvtmlx/ReadVariableOp_12:
lwptfvtmlx/ReadVariableOp_2lwptfvtmlx/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç)
Ò
while_body_2594552
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lwptfvtmlx_2594576_0:	 -
while_lwptfvtmlx_2594578_0:	 )
while_lwptfvtmlx_2594580_0:	(
while_lwptfvtmlx_2594582_0: (
while_lwptfvtmlx_2594584_0: (
while_lwptfvtmlx_2594586_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lwptfvtmlx_2594576:	 +
while_lwptfvtmlx_2594578:	 '
while_lwptfvtmlx_2594580:	&
while_lwptfvtmlx_2594582: &
while_lwptfvtmlx_2594584: &
while_lwptfvtmlx_2594586: ¢(while/lwptfvtmlx/StatefulPartitionedCallÃ
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
(while/lwptfvtmlx/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lwptfvtmlx_2594576_0while_lwptfvtmlx_2594578_0while_lwptfvtmlx_2594580_0while_lwptfvtmlx_2594582_0while_lwptfvtmlx_2594584_0while_lwptfvtmlx_2594586_0*
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_25945322*
(while/lwptfvtmlx/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/lwptfvtmlx/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/lwptfvtmlx/StatefulPartitionedCall:output:1)^while/lwptfvtmlx/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/lwptfvtmlx/StatefulPartitionedCall:output:2)^while/lwptfvtmlx/StatefulPartitionedCall*
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
while_lwptfvtmlx_2594576while_lwptfvtmlx_2594576_0"6
while_lwptfvtmlx_2594578while_lwptfvtmlx_2594578_0"6
while_lwptfvtmlx_2594580while_lwptfvtmlx_2594580_0"6
while_lwptfvtmlx_2594582while_lwptfvtmlx_2594582_0"6
while_lwptfvtmlx_2594584while_lwptfvtmlx_2594584_0"6
while_lwptfvtmlx_2594586while_lwptfvtmlx_2594586_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2T
(while/lwptfvtmlx/StatefulPartitionedCall(while/lwptfvtmlx/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
"__inference__wrapped_model_2593687

bdeyofgzkqW
Asequential_gtjikcltwy_conv1d_expanddims_1_readvariableop_resource:V
Hsequential_gtjikcltwy_squeeze_batch_dims_biasadd_readvariableop_resource:R
?sequential_dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource:	T
Asequential_dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource:	 O
@sequential_dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource:	F
8sequential_dnzlhpjizj_hswofenhiy_readvariableop_resource: H
:sequential_dnzlhpjizj_hswofenhiy_readvariableop_1_resource: H
:sequential_dnzlhpjizj_hswofenhiy_readvariableop_2_resource: R
?sequential_nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource:	 T
Asequential_nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource:	 O
@sequential_nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource:	F
8sequential_nyosplwtfa_lwptfvtmlx_readvariableop_resource: H
:sequential_nyosplwtfa_lwptfvtmlx_readvariableop_1_resource: H
:sequential_nyosplwtfa_lwptfvtmlx_readvariableop_2_resource: F
4sequential_chsgvefspq_matmul_readvariableop_resource: C
5sequential_chsgvefspq_biasadd_readvariableop_resource:
identity¢,sequential/chsgvefspq/BiasAdd/ReadVariableOp¢+sequential/chsgvefspq/MatMul/ReadVariableOp¢7sequential/dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp¢6sequential/dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp¢8sequential/dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp¢/sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp¢1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_1¢1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_2¢sequential/dnzlhpjizj/while¢8sequential/gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp¢?sequential/gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7sequential/nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp¢6sequential/nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp¢8sequential/nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp¢/sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp¢1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_1¢1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_2¢sequential/nyosplwtfa/while¥
+sequential/gtjikcltwy/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2-
+sequential/gtjikcltwy/conv1d/ExpandDims/dimà
'sequential/gtjikcltwy/conv1d/ExpandDims
ExpandDims
bdeyofgzkq4sequential/gtjikcltwy/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/gtjikcltwy/conv1d/ExpandDimsú
8sequential/gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_gtjikcltwy_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02:
8sequential/gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp 
-sequential/gtjikcltwy/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential/gtjikcltwy/conv1d/ExpandDims_1/dim
)sequential/gtjikcltwy/conv1d/ExpandDims_1
ExpandDims@sequential/gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential/gtjikcltwy/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2+
)sequential/gtjikcltwy/conv1d/ExpandDims_1¨
"sequential/gtjikcltwy/conv1d/ShapeShape0sequential/gtjikcltwy/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2$
"sequential/gtjikcltwy/conv1d/Shape®
0sequential/gtjikcltwy/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/gtjikcltwy/conv1d/strided_slice/stack»
2sequential/gtjikcltwy/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential/gtjikcltwy/conv1d/strided_slice/stack_1²
2sequential/gtjikcltwy/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential/gtjikcltwy/conv1d/strided_slice/stack_2
*sequential/gtjikcltwy/conv1d/strided_sliceStridedSlice+sequential/gtjikcltwy/conv1d/Shape:output:09sequential/gtjikcltwy/conv1d/strided_slice/stack:output:0;sequential/gtjikcltwy/conv1d/strided_slice/stack_1:output:0;sequential/gtjikcltwy/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2,
*sequential/gtjikcltwy/conv1d/strided_slice±
*sequential/gtjikcltwy/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2,
*sequential/gtjikcltwy/conv1d/Reshape/shapeø
$sequential/gtjikcltwy/conv1d/ReshapeReshape0sequential/gtjikcltwy/conv1d/ExpandDims:output:03sequential/gtjikcltwy/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/gtjikcltwy/conv1d/Reshape
#sequential/gtjikcltwy/conv1d/Conv2DConv2D-sequential/gtjikcltwy/conv1d/Reshape:output:02sequential/gtjikcltwy/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2%
#sequential/gtjikcltwy/conv1d/Conv2D±
,sequential/gtjikcltwy/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2.
,sequential/gtjikcltwy/conv1d/concat/values_1
(sequential/gtjikcltwy/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/gtjikcltwy/conv1d/concat/axis£
#sequential/gtjikcltwy/conv1d/concatConcatV23sequential/gtjikcltwy/conv1d/strided_slice:output:05sequential/gtjikcltwy/conv1d/concat/values_1:output:01sequential/gtjikcltwy/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/gtjikcltwy/conv1d/concatõ
&sequential/gtjikcltwy/conv1d/Reshape_1Reshape,sequential/gtjikcltwy/conv1d/Conv2D:output:0,sequential/gtjikcltwy/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/gtjikcltwy/conv1d/Reshape_1â
$sequential/gtjikcltwy/conv1d/SqueezeSqueeze/sequential/gtjikcltwy/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2&
$sequential/gtjikcltwy/conv1d/Squeeze½
.sequential/gtjikcltwy/squeeze_batch_dims/ShapeShape-sequential/gtjikcltwy/conv1d/Squeeze:output:0*
T0*
_output_shapes
:20
.sequential/gtjikcltwy/squeeze_batch_dims/ShapeÆ
<sequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stackÓ
>sequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2@
>sequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1Ê
>sequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2Ö
6sequential/gtjikcltwy/squeeze_batch_dims/strided_sliceStridedSlice7sequential/gtjikcltwy/squeeze_batch_dims/Shape:output:0Esequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stack:output:0Gsequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1:output:0Gsequential/gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask28
6sequential/gtjikcltwy/squeeze_batch_dims/strided_sliceÅ
6sequential/gtjikcltwy/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      28
6sequential/gtjikcltwy/squeeze_batch_dims/Reshape/shape
0sequential/gtjikcltwy/squeeze_batch_dims/ReshapeReshape-sequential/gtjikcltwy/conv1d/Squeeze:output:0?sequential/gtjikcltwy/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/gtjikcltwy/squeeze_batch_dims/Reshape
?sequential/gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpHsequential_gtjikcltwy_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?sequential/gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp©
0sequential/gtjikcltwy/squeeze_batch_dims/BiasAddBiasAdd9sequential/gtjikcltwy/squeeze_batch_dims/Reshape:output:0Gsequential/gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential/gtjikcltwy/squeeze_batch_dims/BiasAddÅ
8sequential/gtjikcltwy/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2:
8sequential/gtjikcltwy/squeeze_batch_dims/concat/values_1·
4sequential/gtjikcltwy/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4sequential/gtjikcltwy/squeeze_batch_dims/concat/axisß
/sequential/gtjikcltwy/squeeze_batch_dims/concatConcatV2?sequential/gtjikcltwy/squeeze_batch_dims/strided_slice:output:0Asequential/gtjikcltwy/squeeze_batch_dims/concat/values_1:output:0=sequential/gtjikcltwy/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:21
/sequential/gtjikcltwy/squeeze_batch_dims/concat¢
2sequential/gtjikcltwy/squeeze_batch_dims/Reshape_1Reshape9sequential/gtjikcltwy/squeeze_batch_dims/BiasAdd:output:08sequential/gtjikcltwy/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential/gtjikcltwy/squeeze_batch_dims/Reshape_1¥
sequential/ezubtmdnwx/ShapeShape;sequential/gtjikcltwy/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
sequential/ezubtmdnwx/Shape 
)sequential/ezubtmdnwx/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/ezubtmdnwx/strided_slice/stack¤
+sequential/ezubtmdnwx/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ezubtmdnwx/strided_slice/stack_1¤
+sequential/ezubtmdnwx/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/ezubtmdnwx/strided_slice/stack_2æ
#sequential/ezubtmdnwx/strided_sliceStridedSlice$sequential/ezubtmdnwx/Shape:output:02sequential/ezubtmdnwx/strided_slice/stack:output:04sequential/ezubtmdnwx/strided_slice/stack_1:output:04sequential/ezubtmdnwx/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/ezubtmdnwx/strided_slice
%sequential/ezubtmdnwx/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ezubtmdnwx/Reshape/shape/1
%sequential/ezubtmdnwx/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/ezubtmdnwx/Reshape/shape/2
#sequential/ezubtmdnwx/Reshape/shapePack,sequential/ezubtmdnwx/strided_slice:output:0.sequential/ezubtmdnwx/Reshape/shape/1:output:0.sequential/ezubtmdnwx/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2%
#sequential/ezubtmdnwx/Reshape/shapeê
sequential/ezubtmdnwx/ReshapeReshape;sequential/gtjikcltwy/squeeze_batch_dims/Reshape_1:output:0,sequential/ezubtmdnwx/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/ezubtmdnwx/Reshape
sequential/dnzlhpjizj/ShapeShape&sequential/ezubtmdnwx/Reshape:output:0*
T0*
_output_shapes
:2
sequential/dnzlhpjizj/Shape 
)sequential/dnzlhpjizj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/dnzlhpjizj/strided_slice/stack¤
+sequential/dnzlhpjizj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/dnzlhpjizj/strided_slice/stack_1¤
+sequential/dnzlhpjizj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/dnzlhpjizj/strided_slice/stack_2æ
#sequential/dnzlhpjizj/strided_sliceStridedSlice$sequential/dnzlhpjizj/Shape:output:02sequential/dnzlhpjizj/strided_slice/stack:output:04sequential/dnzlhpjizj/strided_slice/stack_1:output:04sequential/dnzlhpjizj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/dnzlhpjizj/strided_slice
!sequential/dnzlhpjizj/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/dnzlhpjizj/zeros/mul/yÄ
sequential/dnzlhpjizj/zeros/mulMul,sequential/dnzlhpjizj/strided_slice:output:0*sequential/dnzlhpjizj/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/dnzlhpjizj/zeros/mul
"sequential/dnzlhpjizj/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/dnzlhpjizj/zeros/Less/y¿
 sequential/dnzlhpjizj/zeros/LessLess#sequential/dnzlhpjizj/zeros/mul:z:0+sequential/dnzlhpjizj/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/dnzlhpjizj/zeros/Less
$sequential/dnzlhpjizj/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/dnzlhpjizj/zeros/packed/1Û
"sequential/dnzlhpjizj/zeros/packedPack,sequential/dnzlhpjizj/strided_slice:output:0-sequential/dnzlhpjizj/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dnzlhpjizj/zeros/packed
!sequential/dnzlhpjizj/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/dnzlhpjizj/zeros/ConstÍ
sequential/dnzlhpjizj/zerosFill+sequential/dnzlhpjizj/zeros/packed:output:0*sequential/dnzlhpjizj/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/dnzlhpjizj/zeros
#sequential/dnzlhpjizj/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/dnzlhpjizj/zeros_1/mul/yÊ
!sequential/dnzlhpjizj/zeros_1/mulMul,sequential/dnzlhpjizj/strided_slice:output:0,sequential/dnzlhpjizj/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/dnzlhpjizj/zeros_1/mul
$sequential/dnzlhpjizj/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/dnzlhpjizj/zeros_1/Less/yÇ
"sequential/dnzlhpjizj/zeros_1/LessLess%sequential/dnzlhpjizj/zeros_1/mul:z:0-sequential/dnzlhpjizj/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/dnzlhpjizj/zeros_1/Less
&sequential/dnzlhpjizj/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/dnzlhpjizj/zeros_1/packed/1á
$sequential/dnzlhpjizj/zeros_1/packedPack,sequential/dnzlhpjizj/strided_slice:output:0/sequential/dnzlhpjizj/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dnzlhpjizj/zeros_1/packed
#sequential/dnzlhpjizj/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/dnzlhpjizj/zeros_1/ConstÕ
sequential/dnzlhpjizj/zeros_1Fill-sequential/dnzlhpjizj/zeros_1/packed:output:0,sequential/dnzlhpjizj/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/dnzlhpjizj/zeros_1¡
$sequential/dnzlhpjizj/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/dnzlhpjizj/transpose/permÜ
sequential/dnzlhpjizj/transpose	Transpose&sequential/ezubtmdnwx/Reshape:output:0-sequential/dnzlhpjizj/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/dnzlhpjizj/transpose
sequential/dnzlhpjizj/Shape_1Shape#sequential/dnzlhpjizj/transpose:y:0*
T0*
_output_shapes
:2
sequential/dnzlhpjizj/Shape_1¤
+sequential/dnzlhpjizj/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/dnzlhpjizj/strided_slice_1/stack¨
-sequential/dnzlhpjizj/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/dnzlhpjizj/strided_slice_1/stack_1¨
-sequential/dnzlhpjizj/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/dnzlhpjizj/strided_slice_1/stack_2ò
%sequential/dnzlhpjizj/strided_slice_1StridedSlice&sequential/dnzlhpjizj/Shape_1:output:04sequential/dnzlhpjizj/strided_slice_1/stack:output:06sequential/dnzlhpjizj/strided_slice_1/stack_1:output:06sequential/dnzlhpjizj/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/dnzlhpjizj/strided_slice_1±
1sequential/dnzlhpjizj/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/dnzlhpjizj/TensorArrayV2/element_shape
#sequential/dnzlhpjizj/TensorArrayV2TensorListReserve:sequential/dnzlhpjizj/TensorArrayV2/element_shape:output:0.sequential/dnzlhpjizj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/dnzlhpjizj/TensorArrayV2ë
Ksequential/dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential/dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/dnzlhpjizj/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/dnzlhpjizj/transpose:y:0Tsequential/dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor¤
+sequential/dnzlhpjizj/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/dnzlhpjizj/strided_slice_2/stack¨
-sequential/dnzlhpjizj/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/dnzlhpjizj/strided_slice_2/stack_1¨
-sequential/dnzlhpjizj/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/dnzlhpjizj/strided_slice_2/stack_2
%sequential/dnzlhpjizj/strided_slice_2StridedSlice#sequential/dnzlhpjizj/transpose:y:04sequential/dnzlhpjizj/strided_slice_2/stack:output:06sequential/dnzlhpjizj/strided_slice_2/stack_1:output:06sequential/dnzlhpjizj/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential/dnzlhpjizj/strided_slice_2ñ
6sequential/dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOpReadVariableOp?sequential_dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6sequential/dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOpÿ
'sequential/dnzlhpjizj/hswofenhiy/MatMulMatMul.sequential/dnzlhpjizj/strided_slice_2:output:0>sequential/dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/dnzlhpjizj/hswofenhiy/MatMul÷
8sequential/dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOpAsequential_dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOpû
)sequential/dnzlhpjizj/hswofenhiy/MatMul_1MatMul$sequential/dnzlhpjizj/zeros:output:0@sequential/dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/dnzlhpjizj/hswofenhiy/MatMul_1ð
$sequential/dnzlhpjizj/hswofenhiy/addAddV21sequential/dnzlhpjizj/hswofenhiy/MatMul:product:03sequential/dnzlhpjizj/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/dnzlhpjizj/hswofenhiy/addð
7sequential/dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp@sequential_dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOpý
(sequential/dnzlhpjizj/hswofenhiy/BiasAddBiasAdd(sequential/dnzlhpjizj/hswofenhiy/add:z:0?sequential/dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/dnzlhpjizj/hswofenhiy/BiasAdd¦
0sequential/dnzlhpjizj/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/dnzlhpjizj/hswofenhiy/split/split_dimÃ
&sequential/dnzlhpjizj/hswofenhiy/splitSplit9sequential/dnzlhpjizj/hswofenhiy/split/split_dim:output:01sequential/dnzlhpjizj/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/dnzlhpjizj/hswofenhiy/split×
/sequential/dnzlhpjizj/hswofenhiy/ReadVariableOpReadVariableOp8sequential_dnzlhpjizj_hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/dnzlhpjizj/hswofenhiy/ReadVariableOpæ
$sequential/dnzlhpjizj/hswofenhiy/mulMul7sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp:value:0&sequential/dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/dnzlhpjizj/hswofenhiy/mulæ
&sequential/dnzlhpjizj/hswofenhiy/add_1AddV2/sequential/dnzlhpjizj/hswofenhiy/split:output:0(sequential/dnzlhpjizj/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/add_1½
(sequential/dnzlhpjizj/hswofenhiy/SigmoidSigmoid*sequential/dnzlhpjizj/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/dnzlhpjizj/hswofenhiy/SigmoidÝ
1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_1ReadVariableOp:sequential_dnzlhpjizj_hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_1ì
&sequential/dnzlhpjizj/hswofenhiy/mul_1Mul9sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_1:value:0&sequential/dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/mul_1è
&sequential/dnzlhpjizj/hswofenhiy/add_2AddV2/sequential/dnzlhpjizj/hswofenhiy/split:output:1*sequential/dnzlhpjizj/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/add_2Á
*sequential/dnzlhpjizj/hswofenhiy/Sigmoid_1Sigmoid*sequential/dnzlhpjizj/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/dnzlhpjizj/hswofenhiy/Sigmoid_1á
&sequential/dnzlhpjizj/hswofenhiy/mul_2Mul.sequential/dnzlhpjizj/hswofenhiy/Sigmoid_1:y:0&sequential/dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/mul_2¹
%sequential/dnzlhpjizj/hswofenhiy/TanhTanh/sequential/dnzlhpjizj/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/dnzlhpjizj/hswofenhiy/Tanhâ
&sequential/dnzlhpjizj/hswofenhiy/mul_3Mul,sequential/dnzlhpjizj/hswofenhiy/Sigmoid:y:0)sequential/dnzlhpjizj/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/mul_3ã
&sequential/dnzlhpjizj/hswofenhiy/add_3AddV2*sequential/dnzlhpjizj/hswofenhiy/mul_2:z:0*sequential/dnzlhpjizj/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/add_3Ý
1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_2ReadVariableOp:sequential_dnzlhpjizj_hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_2ð
&sequential/dnzlhpjizj/hswofenhiy/mul_4Mul9sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_2:value:0*sequential/dnzlhpjizj/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/mul_4è
&sequential/dnzlhpjizj/hswofenhiy/add_4AddV2/sequential/dnzlhpjizj/hswofenhiy/split:output:3*sequential/dnzlhpjizj/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/add_4Á
*sequential/dnzlhpjizj/hswofenhiy/Sigmoid_2Sigmoid*sequential/dnzlhpjizj/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/dnzlhpjizj/hswofenhiy/Sigmoid_2¸
'sequential/dnzlhpjizj/hswofenhiy/Tanh_1Tanh*sequential/dnzlhpjizj/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/dnzlhpjizj/hswofenhiy/Tanh_1æ
&sequential/dnzlhpjizj/hswofenhiy/mul_5Mul.sequential/dnzlhpjizj/hswofenhiy/Sigmoid_2:y:0+sequential/dnzlhpjizj/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/hswofenhiy/mul_5»
3sequential/dnzlhpjizj/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/dnzlhpjizj/TensorArrayV2_1/element_shape
%sequential/dnzlhpjizj/TensorArrayV2_1TensorListReserve<sequential/dnzlhpjizj/TensorArrayV2_1/element_shape:output:0.sequential/dnzlhpjizj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/dnzlhpjizj/TensorArrayV2_1z
sequential/dnzlhpjizj/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/dnzlhpjizj/time«
.sequential/dnzlhpjizj/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/dnzlhpjizj/while/maximum_iterations
(sequential/dnzlhpjizj/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dnzlhpjizj/while/loop_counterø	
sequential/dnzlhpjizj/whileWhile1sequential/dnzlhpjizj/while/loop_counter:output:07sequential/dnzlhpjizj/while/maximum_iterations:output:0#sequential/dnzlhpjizj/time:output:0.sequential/dnzlhpjizj/TensorArrayV2_1:handle:0$sequential/dnzlhpjizj/zeros:output:0&sequential/dnzlhpjizj/zeros_1:output:0.sequential/dnzlhpjizj/strided_slice_1:output:0Msequential/dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_dnzlhpjizj_hswofenhiy_matmul_readvariableop_resourceAsequential_dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource@sequential_dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource8sequential_dnzlhpjizj_hswofenhiy_readvariableop_resource:sequential_dnzlhpjizj_hswofenhiy_readvariableop_1_resource:sequential_dnzlhpjizj_hswofenhiy_readvariableop_2_resource*
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
(sequential_dnzlhpjizj_while_body_2593404*4
cond,R*
(sequential_dnzlhpjizj_while_cond_2593403*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/dnzlhpjizj/whileá
Fsequential/dnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/dnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/dnzlhpjizj/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/dnzlhpjizj/while:output:3Osequential/dnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/dnzlhpjizj/TensorArrayV2Stack/TensorListStack­
+sequential/dnzlhpjizj/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/dnzlhpjizj/strided_slice_3/stack¨
-sequential/dnzlhpjizj/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/dnzlhpjizj/strided_slice_3/stack_1¨
-sequential/dnzlhpjizj/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/dnzlhpjizj/strided_slice_3/stack_2
%sequential/dnzlhpjizj/strided_slice_3StridedSliceAsequential/dnzlhpjizj/TensorArrayV2Stack/TensorListStack:tensor:04sequential/dnzlhpjizj/strided_slice_3/stack:output:06sequential/dnzlhpjizj/strided_slice_3/stack_1:output:06sequential/dnzlhpjizj/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/dnzlhpjizj/strided_slice_3¥
&sequential/dnzlhpjizj/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/dnzlhpjizj/transpose_1/permý
!sequential/dnzlhpjizj/transpose_1	TransposeAsequential/dnzlhpjizj/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/dnzlhpjizj/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/dnzlhpjizj/transpose_1
sequential/nyosplwtfa/ShapeShape%sequential/dnzlhpjizj/transpose_1:y:0*
T0*
_output_shapes
:2
sequential/nyosplwtfa/Shape 
)sequential/nyosplwtfa/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/nyosplwtfa/strided_slice/stack¤
+sequential/nyosplwtfa/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/nyosplwtfa/strided_slice/stack_1¤
+sequential/nyosplwtfa/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/nyosplwtfa/strided_slice/stack_2æ
#sequential/nyosplwtfa/strided_sliceStridedSlice$sequential/nyosplwtfa/Shape:output:02sequential/nyosplwtfa/strided_slice/stack:output:04sequential/nyosplwtfa/strided_slice/stack_1:output:04sequential/nyosplwtfa/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/nyosplwtfa/strided_slice
!sequential/nyosplwtfa/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/nyosplwtfa/zeros/mul/yÄ
sequential/nyosplwtfa/zeros/mulMul,sequential/nyosplwtfa/strided_slice:output:0*sequential/nyosplwtfa/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/nyosplwtfa/zeros/mul
"sequential/nyosplwtfa/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential/nyosplwtfa/zeros/Less/y¿
 sequential/nyosplwtfa/zeros/LessLess#sequential/nyosplwtfa/zeros/mul:z:0+sequential/nyosplwtfa/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/nyosplwtfa/zeros/Less
$sequential/nyosplwtfa/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential/nyosplwtfa/zeros/packed/1Û
"sequential/nyosplwtfa/zeros/packedPack,sequential/nyosplwtfa/strided_slice:output:0-sequential/nyosplwtfa/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/nyosplwtfa/zeros/packed
!sequential/nyosplwtfa/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/nyosplwtfa/zeros/ConstÍ
sequential/nyosplwtfa/zerosFill+sequential/nyosplwtfa/zeros/packed:output:0*sequential/nyosplwtfa/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/nyosplwtfa/zeros
#sequential/nyosplwtfa/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/nyosplwtfa/zeros_1/mul/yÊ
!sequential/nyosplwtfa/zeros_1/mulMul,sequential/nyosplwtfa/strided_slice:output:0,sequential/nyosplwtfa/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/nyosplwtfa/zeros_1/mul
$sequential/nyosplwtfa/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential/nyosplwtfa/zeros_1/Less/yÇ
"sequential/nyosplwtfa/zeros_1/LessLess%sequential/nyosplwtfa/zeros_1/mul:z:0-sequential/nyosplwtfa/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/nyosplwtfa/zeros_1/Less
&sequential/nyosplwtfa/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/nyosplwtfa/zeros_1/packed/1á
$sequential/nyosplwtfa/zeros_1/packedPack,sequential/nyosplwtfa/strided_slice:output:0/sequential/nyosplwtfa/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/nyosplwtfa/zeros_1/packed
#sequential/nyosplwtfa/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/nyosplwtfa/zeros_1/ConstÕ
sequential/nyosplwtfa/zeros_1Fill-sequential/nyosplwtfa/zeros_1/packed:output:0,sequential/nyosplwtfa/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/nyosplwtfa/zeros_1¡
$sequential/nyosplwtfa/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/nyosplwtfa/transpose/permÛ
sequential/nyosplwtfa/transpose	Transpose%sequential/dnzlhpjizj/transpose_1:y:0-sequential/nyosplwtfa/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/nyosplwtfa/transpose
sequential/nyosplwtfa/Shape_1Shape#sequential/nyosplwtfa/transpose:y:0*
T0*
_output_shapes
:2
sequential/nyosplwtfa/Shape_1¤
+sequential/nyosplwtfa/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/nyosplwtfa/strided_slice_1/stack¨
-sequential/nyosplwtfa/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/nyosplwtfa/strided_slice_1/stack_1¨
-sequential/nyosplwtfa/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/nyosplwtfa/strided_slice_1/stack_2ò
%sequential/nyosplwtfa/strided_slice_1StridedSlice&sequential/nyosplwtfa/Shape_1:output:04sequential/nyosplwtfa/strided_slice_1/stack:output:06sequential/nyosplwtfa/strided_slice_1/stack_1:output:06sequential/nyosplwtfa/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/nyosplwtfa/strided_slice_1±
1sequential/nyosplwtfa/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/nyosplwtfa/TensorArrayV2/element_shape
#sequential/nyosplwtfa/TensorArrayV2TensorListReserve:sequential/nyosplwtfa/TensorArrayV2/element_shape:output:0.sequential/nyosplwtfa/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/nyosplwtfa/TensorArrayV2ë
Ksequential/nyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2M
Ksequential/nyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential/nyosplwtfa/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/nyosplwtfa/transpose:y:0Tsequential/nyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/nyosplwtfa/TensorArrayUnstack/TensorListFromTensor¤
+sequential/nyosplwtfa/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/nyosplwtfa/strided_slice_2/stack¨
-sequential/nyosplwtfa/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/nyosplwtfa/strided_slice_2/stack_1¨
-sequential/nyosplwtfa/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/nyosplwtfa/strided_slice_2/stack_2
%sequential/nyosplwtfa/strided_slice_2StridedSlice#sequential/nyosplwtfa/transpose:y:04sequential/nyosplwtfa/strided_slice_2/stack:output:06sequential/nyosplwtfa/strided_slice_2/stack_1:output:06sequential/nyosplwtfa/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/nyosplwtfa/strided_slice_2ñ
6sequential/nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp?sequential_nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype028
6sequential/nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOpÿ
'sequential/nyosplwtfa/lwptfvtmlx/MatMulMatMul.sequential/nyosplwtfa/strided_slice_2:output:0>sequential/nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/nyosplwtfa/lwptfvtmlx/MatMul÷
8sequential/nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOpAsequential_nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02:
8sequential/nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOpû
)sequential/nyosplwtfa/lwptfvtmlx/MatMul_1MatMul$sequential/nyosplwtfa/zeros:output:0@sequential/nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/nyosplwtfa/lwptfvtmlx/MatMul_1ð
$sequential/nyosplwtfa/lwptfvtmlx/addAddV21sequential/nyosplwtfa/lwptfvtmlx/MatMul:product:03sequential/nyosplwtfa/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$sequential/nyosplwtfa/lwptfvtmlx/addð
7sequential/nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp@sequential_nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7sequential/nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOpý
(sequential/nyosplwtfa/lwptfvtmlx/BiasAddBiasAdd(sequential/nyosplwtfa/lwptfvtmlx/add:z:0?sequential/nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential/nyosplwtfa/lwptfvtmlx/BiasAdd¦
0sequential/nyosplwtfa/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential/nyosplwtfa/lwptfvtmlx/split/split_dimÃ
&sequential/nyosplwtfa/lwptfvtmlx/splitSplit9sequential/nyosplwtfa/lwptfvtmlx/split/split_dim:output:01sequential/nyosplwtfa/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2(
&sequential/nyosplwtfa/lwptfvtmlx/split×
/sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOpReadVariableOp8sequential_nyosplwtfa_lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOpæ
$sequential/nyosplwtfa/lwptfvtmlx/mulMul7sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp:value:0&sequential/nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$sequential/nyosplwtfa/lwptfvtmlx/mulæ
&sequential/nyosplwtfa/lwptfvtmlx/add_1AddV2/sequential/nyosplwtfa/lwptfvtmlx/split:output:0(sequential/nyosplwtfa/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/add_1½
(sequential/nyosplwtfa/lwptfvtmlx/SigmoidSigmoid*sequential/nyosplwtfa/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential/nyosplwtfa/lwptfvtmlx/SigmoidÝ
1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_1ReadVariableOp:sequential_nyosplwtfa_lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_1ì
&sequential/nyosplwtfa/lwptfvtmlx/mul_1Mul9sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_1:value:0&sequential/nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/mul_1è
&sequential/nyosplwtfa/lwptfvtmlx/add_2AddV2/sequential/nyosplwtfa/lwptfvtmlx/split:output:1*sequential/nyosplwtfa/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/add_2Á
*sequential/nyosplwtfa/lwptfvtmlx/Sigmoid_1Sigmoid*sequential/nyosplwtfa/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/nyosplwtfa/lwptfvtmlx/Sigmoid_1á
&sequential/nyosplwtfa/lwptfvtmlx/mul_2Mul.sequential/nyosplwtfa/lwptfvtmlx/Sigmoid_1:y:0&sequential/nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/mul_2¹
%sequential/nyosplwtfa/lwptfvtmlx/TanhTanh/sequential/nyosplwtfa/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/nyosplwtfa/lwptfvtmlx/Tanhâ
&sequential/nyosplwtfa/lwptfvtmlx/mul_3Mul,sequential/nyosplwtfa/lwptfvtmlx/Sigmoid:y:0)sequential/nyosplwtfa/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/mul_3ã
&sequential/nyosplwtfa/lwptfvtmlx/add_3AddV2*sequential/nyosplwtfa/lwptfvtmlx/mul_2:z:0*sequential/nyosplwtfa/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/add_3Ý
1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_2ReadVariableOp:sequential_nyosplwtfa_lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype023
1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_2ð
&sequential/nyosplwtfa/lwptfvtmlx/mul_4Mul9sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_2:value:0*sequential/nyosplwtfa/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/mul_4è
&sequential/nyosplwtfa/lwptfvtmlx/add_4AddV2/sequential/nyosplwtfa/lwptfvtmlx/split:output:3*sequential/nyosplwtfa/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/add_4Á
*sequential/nyosplwtfa/lwptfvtmlx/Sigmoid_2Sigmoid*sequential/nyosplwtfa/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/nyosplwtfa/lwptfvtmlx/Sigmoid_2¸
'sequential/nyosplwtfa/lwptfvtmlx/Tanh_1Tanh*sequential/nyosplwtfa/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/nyosplwtfa/lwptfvtmlx/Tanh_1æ
&sequential/nyosplwtfa/lwptfvtmlx/mul_5Mul.sequential/nyosplwtfa/lwptfvtmlx/Sigmoid_2:y:0+sequential/nyosplwtfa/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/lwptfvtmlx/mul_5»
3sequential/nyosplwtfa/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    25
3sequential/nyosplwtfa/TensorArrayV2_1/element_shape
%sequential/nyosplwtfa/TensorArrayV2_1TensorListReserve<sequential/nyosplwtfa/TensorArrayV2_1/element_shape:output:0.sequential/nyosplwtfa/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/nyosplwtfa/TensorArrayV2_1z
sequential/nyosplwtfa/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/nyosplwtfa/time«
.sequential/nyosplwtfa/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential/nyosplwtfa/while/maximum_iterations
(sequential/nyosplwtfa/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/nyosplwtfa/while/loop_counterø	
sequential/nyosplwtfa/whileWhile1sequential/nyosplwtfa/while/loop_counter:output:07sequential/nyosplwtfa/while/maximum_iterations:output:0#sequential/nyosplwtfa/time:output:0.sequential/nyosplwtfa/TensorArrayV2_1:handle:0$sequential/nyosplwtfa/zeros:output:0&sequential/nyosplwtfa/zeros_1:output:0.sequential/nyosplwtfa/strided_slice_1:output:0Msequential/nyosplwtfa/TensorArrayUnstack/TensorListFromTensor:output_handle:0?sequential_nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resourceAsequential_nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource@sequential_nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource8sequential_nyosplwtfa_lwptfvtmlx_readvariableop_resource:sequential_nyosplwtfa_lwptfvtmlx_readvariableop_1_resource:sequential_nyosplwtfa_lwptfvtmlx_readvariableop_2_resource*
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
(sequential_nyosplwtfa_while_body_2593580*4
cond,R*
(sequential_nyosplwtfa_while_cond_2593579*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
sequential/nyosplwtfa/whileá
Fsequential/nyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/nyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential/nyosplwtfa/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/nyosplwtfa/while:output:3Osequential/nyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02:
8sequential/nyosplwtfa/TensorArrayV2Stack/TensorListStack­
+sequential/nyosplwtfa/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential/nyosplwtfa/strided_slice_3/stack¨
-sequential/nyosplwtfa/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/nyosplwtfa/strided_slice_3/stack_1¨
-sequential/nyosplwtfa/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/nyosplwtfa/strided_slice_3/stack_2
%sequential/nyosplwtfa/strided_slice_3StridedSliceAsequential/nyosplwtfa/TensorArrayV2Stack/TensorListStack:tensor:04sequential/nyosplwtfa/strided_slice_3/stack:output:06sequential/nyosplwtfa/strided_slice_3/stack_1:output:06sequential/nyosplwtfa/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2'
%sequential/nyosplwtfa/strided_slice_3¥
&sequential/nyosplwtfa/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/nyosplwtfa/transpose_1/permý
!sequential/nyosplwtfa/transpose_1	TransposeAsequential/nyosplwtfa/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/nyosplwtfa/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/nyosplwtfa/transpose_1Ï
+sequential/chsgvefspq/MatMul/ReadVariableOpReadVariableOp4sequential_chsgvefspq_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential/chsgvefspq/MatMul/ReadVariableOpÝ
sequential/chsgvefspq/MatMulMatMul.sequential/nyosplwtfa/strided_slice_3:output:03sequential/chsgvefspq/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/chsgvefspq/MatMulÎ
,sequential/chsgvefspq/BiasAdd/ReadVariableOpReadVariableOp5sequential_chsgvefspq_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential/chsgvefspq/BiasAdd/ReadVariableOpÙ
sequential/chsgvefspq/BiasAddBiasAdd&sequential/chsgvefspq/MatMul:product:04sequential/chsgvefspq/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/chsgvefspq/BiasAdd 
IdentityIdentity&sequential/chsgvefspq/BiasAdd:output:0-^sequential/chsgvefspq/BiasAdd/ReadVariableOp,^sequential/chsgvefspq/MatMul/ReadVariableOp8^sequential/dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp7^sequential/dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp9^sequential/dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp0^sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp2^sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_12^sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_2^sequential/dnzlhpjizj/while9^sequential/gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp@^sequential/gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp8^sequential/nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp7^sequential/nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp9^sequential/nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp0^sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp2^sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_12^sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_2^sequential/nyosplwtfa/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2\
,sequential/chsgvefspq/BiasAdd/ReadVariableOp,sequential/chsgvefspq/BiasAdd/ReadVariableOp2Z
+sequential/chsgvefspq/MatMul/ReadVariableOp+sequential/chsgvefspq/MatMul/ReadVariableOp2r
7sequential/dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp7sequential/dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp2p
6sequential/dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp6sequential/dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp2t
8sequential/dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp8sequential/dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp2b
/sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp/sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp2f
1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_11sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_12f
1sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_21sequential/dnzlhpjizj/hswofenhiy/ReadVariableOp_22:
sequential/dnzlhpjizj/whilesequential/dnzlhpjizj/while2t
8sequential/gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp8sequential/gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp2
?sequential/gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp?sequential/gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7sequential/nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp7sequential/nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp2p
6sequential/nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp6sequential/nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp2t
8sequential/nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp8sequential/nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp2b
/sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp/sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp2f
1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_11sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_12f
1sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_21sequential/nyosplwtfa/lwptfvtmlx/ReadVariableOp_22:
sequential/nyosplwtfa/whilesequential/nyosplwtfa/while:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
bdeyofgzkq
¹'
µ
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2599158

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
Ü

(sequential_dnzlhpjizj_while_body_2593404H
Dsequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_loop_counterN
Jsequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_maximum_iterations+
'sequential_dnzlhpjizj_while_placeholder-
)sequential_dnzlhpjizj_while_placeholder_1-
)sequential_dnzlhpjizj_while_placeholder_2-
)sequential_dnzlhpjizj_while_placeholder_3G
Csequential_dnzlhpjizj_while_sequential_dnzlhpjizj_strided_slice_1_0
sequential_dnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_sequential_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0:	\
Isequential_dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0:	 W
Hsequential_dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0:	N
@sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0: P
Bsequential_dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0: P
Bsequential_dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0: (
$sequential_dnzlhpjizj_while_identity*
&sequential_dnzlhpjizj_while_identity_1*
&sequential_dnzlhpjizj_while_identity_2*
&sequential_dnzlhpjizj_while_identity_3*
&sequential_dnzlhpjizj_while_identity_4*
&sequential_dnzlhpjizj_while_identity_5E
Asequential_dnzlhpjizj_while_sequential_dnzlhpjizj_strided_slice_1
}sequential_dnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_sequential_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensorX
Esequential_dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource:	Z
Gsequential_dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource:	 U
Fsequential_dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource:	L
>sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_resource: N
@sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource: N
@sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource: ¢=sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp¢<sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp¢>sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp¢5sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp¢7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1¢7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2ï
Msequential/dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential/dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_dnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_sequential_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0'sequential_dnzlhpjizj_while_placeholderVsequential/dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential/dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem
<sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOpReadVariableOpGsequential_dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02>
<sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp©
-sequential/dnzlhpjizj/while/hswofenhiy/MatMulMatMulFsequential/dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/dnzlhpjizj/while/hswofenhiy/MatMul
>sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOpIsequential_dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp
/sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1MatMul)sequential_dnzlhpjizj_while_placeholder_2Fsequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1
*sequential/dnzlhpjizj/while/hswofenhiy/addAddV27sequential/dnzlhpjizj/while/hswofenhiy/MatMul:product:09sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/dnzlhpjizj/while/hswofenhiy/add
=sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOpHsequential_dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp
.sequential/dnzlhpjizj/while/hswofenhiy/BiasAddBiasAdd.sequential/dnzlhpjizj/while/hswofenhiy/add:z:0Esequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd²
6sequential/dnzlhpjizj/while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/dnzlhpjizj/while/hswofenhiy/split/split_dimÛ
,sequential/dnzlhpjizj/while/hswofenhiy/splitSplit?sequential/dnzlhpjizj/while/hswofenhiy/split/split_dim:output:07sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/dnzlhpjizj/while/hswofenhiy/splitë
5sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOpReadVariableOp@sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOpû
*sequential/dnzlhpjizj/while/hswofenhiy/mulMul=sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp:value:0)sequential_dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/dnzlhpjizj/while/hswofenhiy/mulþ
,sequential/dnzlhpjizj/while/hswofenhiy/add_1AddV25sequential/dnzlhpjizj/while/hswofenhiy/split:output:0.sequential/dnzlhpjizj/while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/add_1Ï
.sequential/dnzlhpjizj/while/hswofenhiy/SigmoidSigmoid0sequential/dnzlhpjizj/while/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/dnzlhpjizj/while/hswofenhiy/Sigmoidñ
7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1ReadVariableOpBsequential_dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1
,sequential/dnzlhpjizj/while/hswofenhiy/mul_1Mul?sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1:value:0)sequential_dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/mul_1
,sequential/dnzlhpjizj/while/hswofenhiy/add_2AddV25sequential/dnzlhpjizj/while/hswofenhiy/split:output:10sequential/dnzlhpjizj/while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/add_2Ó
0sequential/dnzlhpjizj/while/hswofenhiy/Sigmoid_1Sigmoid0sequential/dnzlhpjizj/while/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/dnzlhpjizj/while/hswofenhiy/Sigmoid_1ö
,sequential/dnzlhpjizj/while/hswofenhiy/mul_2Mul4sequential/dnzlhpjizj/while/hswofenhiy/Sigmoid_1:y:0)sequential_dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/mul_2Ë
+sequential/dnzlhpjizj/while/hswofenhiy/TanhTanh5sequential/dnzlhpjizj/while/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/dnzlhpjizj/while/hswofenhiy/Tanhú
,sequential/dnzlhpjizj/while/hswofenhiy/mul_3Mul2sequential/dnzlhpjizj/while/hswofenhiy/Sigmoid:y:0/sequential/dnzlhpjizj/while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/mul_3û
,sequential/dnzlhpjizj/while/hswofenhiy/add_3AddV20sequential/dnzlhpjizj/while/hswofenhiy/mul_2:z:00sequential/dnzlhpjizj/while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/add_3ñ
7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2ReadVariableOpBsequential_dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2
,sequential/dnzlhpjizj/while/hswofenhiy/mul_4Mul?sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2:value:00sequential/dnzlhpjizj/while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/mul_4
,sequential/dnzlhpjizj/while/hswofenhiy/add_4AddV25sequential/dnzlhpjizj/while/hswofenhiy/split:output:30sequential/dnzlhpjizj/while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/add_4Ó
0sequential/dnzlhpjizj/while/hswofenhiy/Sigmoid_2Sigmoid0sequential/dnzlhpjizj/while/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/dnzlhpjizj/while/hswofenhiy/Sigmoid_2Ê
-sequential/dnzlhpjizj/while/hswofenhiy/Tanh_1Tanh0sequential/dnzlhpjizj/while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/dnzlhpjizj/while/hswofenhiy/Tanh_1þ
,sequential/dnzlhpjizj/while/hswofenhiy/mul_5Mul4sequential/dnzlhpjizj/while/hswofenhiy/Sigmoid_2:y:01sequential/dnzlhpjizj/while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/dnzlhpjizj/while/hswofenhiy/mul_5Ì
@sequential/dnzlhpjizj/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_dnzlhpjizj_while_placeholder_1'sequential_dnzlhpjizj_while_placeholder0sequential/dnzlhpjizj/while/hswofenhiy/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/dnzlhpjizj/while/TensorArrayV2Write/TensorListSetItem
!sequential/dnzlhpjizj/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/dnzlhpjizj/while/add/yÁ
sequential/dnzlhpjizj/while/addAddV2'sequential_dnzlhpjizj_while_placeholder*sequential/dnzlhpjizj/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/dnzlhpjizj/while/add
#sequential/dnzlhpjizj/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/dnzlhpjizj/while/add_1/yä
!sequential/dnzlhpjizj/while/add_1AddV2Dsequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_loop_counter,sequential/dnzlhpjizj/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/dnzlhpjizj/while/add_1
$sequential/dnzlhpjizj/while/IdentityIdentity%sequential/dnzlhpjizj/while/add_1:z:0>^sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp=^sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp?^sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp6^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp8^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_18^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/dnzlhpjizj/while/Identityµ
&sequential/dnzlhpjizj/while/Identity_1IdentityJsequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_maximum_iterations>^sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp=^sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp?^sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp6^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp8^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_18^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/dnzlhpjizj/while/Identity_1
&sequential/dnzlhpjizj/while/Identity_2Identity#sequential/dnzlhpjizj/while/add:z:0>^sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp=^sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp?^sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp6^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp8^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_18^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/dnzlhpjizj/while/Identity_2»
&sequential/dnzlhpjizj/while/Identity_3IdentityPsequential/dnzlhpjizj/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp=^sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp?^sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp6^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp8^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_18^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/dnzlhpjizj/while/Identity_3¬
&sequential/dnzlhpjizj/while/Identity_4Identity0sequential/dnzlhpjizj/while/hswofenhiy/mul_5:z:0>^sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp=^sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp?^sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp6^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp8^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_18^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/while/Identity_4¬
&sequential/dnzlhpjizj/while/Identity_5Identity0sequential/dnzlhpjizj/while/hswofenhiy/add_3:z:0>^sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp=^sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp?^sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp6^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp8^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_18^sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/dnzlhpjizj/while/Identity_5"
Fsequential_dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resourceHsequential_dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0"
Gsequential_dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resourceIsequential_dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0"
Esequential_dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resourceGsequential_dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0"
@sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_1_resourceBsequential_dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0"
@sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_2_resourceBsequential_dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0"
>sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_resource@sequential_dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0"U
$sequential_dnzlhpjizj_while_identity-sequential/dnzlhpjizj/while/Identity:output:0"Y
&sequential_dnzlhpjizj_while_identity_1/sequential/dnzlhpjizj/while/Identity_1:output:0"Y
&sequential_dnzlhpjizj_while_identity_2/sequential/dnzlhpjizj/while/Identity_2:output:0"Y
&sequential_dnzlhpjizj_while_identity_3/sequential/dnzlhpjizj/while/Identity_3:output:0"Y
&sequential_dnzlhpjizj_while_identity_4/sequential/dnzlhpjizj/while/Identity_4:output:0"Y
&sequential_dnzlhpjizj_while_identity_5/sequential/dnzlhpjizj/while/Identity_5:output:0"
Asequential_dnzlhpjizj_while_sequential_dnzlhpjizj_strided_slice_1Csequential_dnzlhpjizj_while_sequential_dnzlhpjizj_strided_slice_1_0"
}sequential_dnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_sequential_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensorsequential_dnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_sequential_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp=sequential/dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2|
<sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp<sequential/dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp2
>sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp>sequential/dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp2n
5sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp5sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp2r
7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_17sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_12r
7sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_27sequential/dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_2595814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lwptfvtmlx_matmul_readvariableop_resource_0:	 F
3while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 A
2while_lwptfvtmlx_biasadd_readvariableop_resource_0:	8
*while_lwptfvtmlx_readvariableop_resource_0: :
,while_lwptfvtmlx_readvariableop_1_resource_0: :
,while_lwptfvtmlx_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lwptfvtmlx_matmul_readvariableop_resource:	 D
1while_lwptfvtmlx_matmul_1_readvariableop_resource:	 ?
0while_lwptfvtmlx_biasadd_readvariableop_resource:	6
(while_lwptfvtmlx_readvariableop_resource: 8
*while_lwptfvtmlx_readvariableop_1_resource: 8
*while_lwptfvtmlx_readvariableop_2_resource: ¢'while/lwptfvtmlx/BiasAdd/ReadVariableOp¢&while/lwptfvtmlx/MatMul/ReadVariableOp¢(while/lwptfvtmlx/MatMul_1/ReadVariableOp¢while/lwptfvtmlx/ReadVariableOp¢!while/lwptfvtmlx/ReadVariableOp_1¢!while/lwptfvtmlx/ReadVariableOp_2Ã
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
&while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp1while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/lwptfvtmlx/MatMul/ReadVariableOpÑ
while/lwptfvtmlx/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMulÉ
(while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp3while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/lwptfvtmlx/MatMul_1/ReadVariableOpº
while/lwptfvtmlx/MatMul_1MatMulwhile_placeholder_20while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMul_1°
while/lwptfvtmlx/addAddV2!while/lwptfvtmlx/MatMul:product:0#while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/addÂ
'while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp2while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/lwptfvtmlx/BiasAdd/ReadVariableOp½
while/lwptfvtmlx/BiasAddBiasAddwhile/lwptfvtmlx/add:z:0/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/BiasAdd
 while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/lwptfvtmlx/split/split_dim
while/lwptfvtmlx/splitSplit)while/lwptfvtmlx/split/split_dim:output:0!while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/lwptfvtmlx/split©
while/lwptfvtmlx/ReadVariableOpReadVariableOp*while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/lwptfvtmlx/ReadVariableOp£
while/lwptfvtmlx/mulMul'while/lwptfvtmlx/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul¦
while/lwptfvtmlx/add_1AddV2while/lwptfvtmlx/split:output:0while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_1
while/lwptfvtmlx/SigmoidSigmoidwhile/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid¯
!while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp,while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_1©
while/lwptfvtmlx/mul_1Mul)while/lwptfvtmlx/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_1¨
while/lwptfvtmlx/add_2AddV2while/lwptfvtmlx/split:output:1while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_2
while/lwptfvtmlx/Sigmoid_1Sigmoidwhile/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_1
while/lwptfvtmlx/mul_2Mulwhile/lwptfvtmlx/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_2
while/lwptfvtmlx/TanhTanhwhile/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh¢
while/lwptfvtmlx/mul_3Mulwhile/lwptfvtmlx/Sigmoid:y:0while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_3£
while/lwptfvtmlx/add_3AddV2while/lwptfvtmlx/mul_2:z:0while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_3¯
!while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp,while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_2°
while/lwptfvtmlx/mul_4Mul)while/lwptfvtmlx/ReadVariableOp_2:value:0while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_4¨
while/lwptfvtmlx/add_4AddV2while/lwptfvtmlx/split:output:3while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_4
while/lwptfvtmlx/Sigmoid_2Sigmoidwhile/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_2
while/lwptfvtmlx/Tanh_1Tanhwhile/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh_1¦
while/lwptfvtmlx/mul_5Mulwhile/lwptfvtmlx/Sigmoid_2:y:0while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lwptfvtmlx/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lwptfvtmlx/mul_5:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/lwptfvtmlx/add_3:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
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
0while_lwptfvtmlx_biasadd_readvariableop_resource2while_lwptfvtmlx_biasadd_readvariableop_resource_0"h
1while_lwptfvtmlx_matmul_1_readvariableop_resource3while_lwptfvtmlx_matmul_1_readvariableop_resource_0"d
/while_lwptfvtmlx_matmul_readvariableop_resource1while_lwptfvtmlx_matmul_readvariableop_resource_0"Z
*while_lwptfvtmlx_readvariableop_1_resource,while_lwptfvtmlx_readvariableop_1_resource_0"Z
*while_lwptfvtmlx_readvariableop_2_resource,while_lwptfvtmlx_readvariableop_2_resource_0"V
(while_lwptfvtmlx_readvariableop_resource*while_lwptfvtmlx_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/lwptfvtmlx/BiasAdd/ReadVariableOp'while/lwptfvtmlx/BiasAdd/ReadVariableOp2P
&while/lwptfvtmlx/MatMul/ReadVariableOp&while/lwptfvtmlx/MatMul/ReadVariableOp2T
(while/lwptfvtmlx/MatMul_1/ReadVariableOp(while/lwptfvtmlx/MatMul_1/ReadVariableOp2B
while/lwptfvtmlx/ReadVariableOpwhile/lwptfvtmlx/ReadVariableOp2F
!while/lwptfvtmlx/ReadVariableOp_1!while/lwptfvtmlx/ReadVariableOp_12F
!while/lwptfvtmlx/ReadVariableOp_2!while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_sequential_layer_call_and_return_conditional_losses_2596394

bdeyofgzkq(
gtjikcltwy_2596356: 
gtjikcltwy_2596358:%
dnzlhpjizj_2596362:	%
dnzlhpjizj_2596364:	 !
dnzlhpjizj_2596366:	 
dnzlhpjizj_2596368:  
dnzlhpjizj_2596370:  
dnzlhpjizj_2596372: %
nyosplwtfa_2596375:	 %
nyosplwtfa_2596377:	 !
nyosplwtfa_2596379:	 
nyosplwtfa_2596381:  
nyosplwtfa_2596383:  
nyosplwtfa_2596385: $
chsgvefspq_2596388:  
chsgvefspq_2596390:
identity¢"chsgvefspq/StatefulPartitionedCall¢"dnzlhpjizj/StatefulPartitionedCall¢"gtjikcltwy/StatefulPartitionedCall¢"nyosplwtfa/StatefulPartitionedCall°
"gtjikcltwy/StatefulPartitionedCallStatefulPartitionedCall
bdeyofgzkqgtjikcltwy_2596356gtjikcltwy_2596358*
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
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_25952472$
"gtjikcltwy/StatefulPartitionedCall
ezubtmdnwx/PartitionedCallPartitionedCall+gtjikcltwy/StatefulPartitionedCall:output:0*
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
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_25952662
ezubtmdnwx/PartitionedCall
"dnzlhpjizj/StatefulPartitionedCallStatefulPartitionedCall#ezubtmdnwx/PartitionedCall:output:0dnzlhpjizj_2596362dnzlhpjizj_2596364dnzlhpjizj_2596366dnzlhpjizj_2596368dnzlhpjizj_2596370dnzlhpjizj_2596372*
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25961292$
"dnzlhpjizj/StatefulPartitionedCall¡
"nyosplwtfa/StatefulPartitionedCallStatefulPartitionedCall+dnzlhpjizj/StatefulPartitionedCall:output:0nyosplwtfa_2596375nyosplwtfa_2596377nyosplwtfa_2596379nyosplwtfa_2596381nyosplwtfa_2596383nyosplwtfa_2596385*
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25959152$
"nyosplwtfa/StatefulPartitionedCallÉ
"chsgvefspq/StatefulPartitionedCallStatefulPartitionedCall+nyosplwtfa/StatefulPartitionedCall:output:0chsgvefspq_2596388chsgvefspq_2596390*
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
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_25956642$
"chsgvefspq/StatefulPartitionedCall
IdentityIdentity+chsgvefspq/StatefulPartitionedCall:output:0#^chsgvefspq/StatefulPartitionedCall#^dnzlhpjizj/StatefulPartitionedCall#^gtjikcltwy/StatefulPartitionedCall#^nyosplwtfa/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"chsgvefspq/StatefulPartitionedCall"chsgvefspq/StatefulPartitionedCall2H
"dnzlhpjizj/StatefulPartitionedCall"dnzlhpjizj/StatefulPartitionedCall2H
"gtjikcltwy/StatefulPartitionedCall"gtjikcltwy/StatefulPartitionedCall2H
"nyosplwtfa/StatefulPartitionedCall"nyosplwtfa/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
bdeyofgzkq


nyosplwtfa_while_cond_25971392
.nyosplwtfa_while_nyosplwtfa_while_loop_counter8
4nyosplwtfa_while_nyosplwtfa_while_maximum_iterations 
nyosplwtfa_while_placeholder"
nyosplwtfa_while_placeholder_1"
nyosplwtfa_while_placeholder_2"
nyosplwtfa_while_placeholder_34
0nyosplwtfa_while_less_nyosplwtfa_strided_slice_1K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2597139___redundant_placeholder0K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2597139___redundant_placeholder1K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2597139___redundant_placeholder2K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2597139___redundant_placeholder3K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2597139___redundant_placeholder4K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2597139___redundant_placeholder5K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2597139___redundant_placeholder6
nyosplwtfa_while_identity
§
nyosplwtfa/while/LessLessnyosplwtfa_while_placeholder0nyosplwtfa_while_less_nyosplwtfa_strided_slice_1*
T0*
_output_shapes
: 2
nyosplwtfa/while/Less~
nyosplwtfa/while/IdentityIdentitynyosplwtfa/while/Less:z:0*
T0
*
_output_shapes
: 2
nyosplwtfa/while/Identity"?
nyosplwtfa_while_identity"nyosplwtfa/while/Identity:output:0*(
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598533
inputs_0<
)lwptfvtmlx_matmul_readvariableop_resource:	 >
+lwptfvtmlx_matmul_1_readvariableop_resource:	 9
*lwptfvtmlx_biasadd_readvariableop_resource:	0
"lwptfvtmlx_readvariableop_resource: 2
$lwptfvtmlx_readvariableop_1_resource: 2
$lwptfvtmlx_readvariableop_2_resource: 
identity¢!lwptfvtmlx/BiasAdd/ReadVariableOp¢ lwptfvtmlx/MatMul/ReadVariableOp¢"lwptfvtmlx/MatMul_1/ReadVariableOp¢lwptfvtmlx/ReadVariableOp¢lwptfvtmlx/ReadVariableOp_1¢lwptfvtmlx/ReadVariableOp_2¢whileF
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
 lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp)lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 lwptfvtmlx/MatMul/ReadVariableOp§
lwptfvtmlx/MatMulMatMulstrided_slice_2:output:0(lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMulµ
"lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp+lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lwptfvtmlx/MatMul_1/ReadVariableOp£
lwptfvtmlx/MatMul_1MatMulzeros:output:0*lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMul_1
lwptfvtmlx/addAddV2lwptfvtmlx/MatMul:product:0lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/add®
!lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp*lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!lwptfvtmlx/BiasAdd/ReadVariableOp¥
lwptfvtmlx/BiasAddBiasAddlwptfvtmlx/add:z:0)lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/BiasAddz
lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lwptfvtmlx/split/split_dimë
lwptfvtmlx/splitSplit#lwptfvtmlx/split/split_dim:output:0lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lwptfvtmlx/split
lwptfvtmlx/ReadVariableOpReadVariableOp"lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp
lwptfvtmlx/mulMul!lwptfvtmlx/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul
lwptfvtmlx/add_1AddV2lwptfvtmlx/split:output:0lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_1{
lwptfvtmlx/SigmoidSigmoidlwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid
lwptfvtmlx/ReadVariableOp_1ReadVariableOp$lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_1
lwptfvtmlx/mul_1Mul#lwptfvtmlx/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_1
lwptfvtmlx/add_2AddV2lwptfvtmlx/split:output:1lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_2
lwptfvtmlx/Sigmoid_1Sigmoidlwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_1
lwptfvtmlx/mul_2Mullwptfvtmlx/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_2w
lwptfvtmlx/TanhTanhlwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh
lwptfvtmlx/mul_3Mullwptfvtmlx/Sigmoid:y:0lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_3
lwptfvtmlx/add_3AddV2lwptfvtmlx/mul_2:z:0lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_3
lwptfvtmlx/ReadVariableOp_2ReadVariableOp$lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_2
lwptfvtmlx/mul_4Mul#lwptfvtmlx/ReadVariableOp_2:value:0lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_4
lwptfvtmlx/add_4AddV2lwptfvtmlx/split:output:3lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_4
lwptfvtmlx/Sigmoid_2Sigmoidlwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_2v
lwptfvtmlx/Tanh_1Tanhlwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh_1
lwptfvtmlx/mul_5Mullwptfvtmlx/Sigmoid_2:y:0lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lwptfvtmlx_matmul_readvariableop_resource+lwptfvtmlx_matmul_1_readvariableop_resource*lwptfvtmlx_biasadd_readvariableop_resource"lwptfvtmlx_readvariableop_resource$lwptfvtmlx_readvariableop_1_resource$lwptfvtmlx_readvariableop_2_resource*
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
while_body_2598432*
condR
while_cond_2598431*Q
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
IdentityIdentitystrided_slice_3:output:0"^lwptfvtmlx/BiasAdd/ReadVariableOp!^lwptfvtmlx/MatMul/ReadVariableOp#^lwptfvtmlx/MatMul_1/ReadVariableOp^lwptfvtmlx/ReadVariableOp^lwptfvtmlx/ReadVariableOp_1^lwptfvtmlx/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!lwptfvtmlx/BiasAdd/ReadVariableOp!lwptfvtmlx/BiasAdd/ReadVariableOp2D
 lwptfvtmlx/MatMul/ReadVariableOp lwptfvtmlx/MatMul/ReadVariableOp2H
"lwptfvtmlx/MatMul_1/ReadVariableOp"lwptfvtmlx/MatMul_1/ReadVariableOp26
lwptfvtmlx/ReadVariableOplwptfvtmlx/ReadVariableOp2:
lwptfvtmlx/ReadVariableOp_1lwptfvtmlx/ReadVariableOp_12:
lwptfvtmlx/ReadVariableOp_2lwptfvtmlx/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0

À
,__inference_hswofenhiy_layer_call_fn_2599091

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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_25937742
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2599024

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
àY

while_body_2598612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lwptfvtmlx_matmul_readvariableop_resource_0:	 F
3while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 A
2while_lwptfvtmlx_biasadd_readvariableop_resource_0:	8
*while_lwptfvtmlx_readvariableop_resource_0: :
,while_lwptfvtmlx_readvariableop_1_resource_0: :
,while_lwptfvtmlx_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lwptfvtmlx_matmul_readvariableop_resource:	 D
1while_lwptfvtmlx_matmul_1_readvariableop_resource:	 ?
0while_lwptfvtmlx_biasadd_readvariableop_resource:	6
(while_lwptfvtmlx_readvariableop_resource: 8
*while_lwptfvtmlx_readvariableop_1_resource: 8
*while_lwptfvtmlx_readvariableop_2_resource: ¢'while/lwptfvtmlx/BiasAdd/ReadVariableOp¢&while/lwptfvtmlx/MatMul/ReadVariableOp¢(while/lwptfvtmlx/MatMul_1/ReadVariableOp¢while/lwptfvtmlx/ReadVariableOp¢!while/lwptfvtmlx/ReadVariableOp_1¢!while/lwptfvtmlx/ReadVariableOp_2Ã
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
&while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp1while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/lwptfvtmlx/MatMul/ReadVariableOpÑ
while/lwptfvtmlx/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMulÉ
(while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp3while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/lwptfvtmlx/MatMul_1/ReadVariableOpº
while/lwptfvtmlx/MatMul_1MatMulwhile_placeholder_20while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMul_1°
while/lwptfvtmlx/addAddV2!while/lwptfvtmlx/MatMul:product:0#while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/addÂ
'while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp2while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/lwptfvtmlx/BiasAdd/ReadVariableOp½
while/lwptfvtmlx/BiasAddBiasAddwhile/lwptfvtmlx/add:z:0/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/BiasAdd
 while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/lwptfvtmlx/split/split_dim
while/lwptfvtmlx/splitSplit)while/lwptfvtmlx/split/split_dim:output:0!while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/lwptfvtmlx/split©
while/lwptfvtmlx/ReadVariableOpReadVariableOp*while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/lwptfvtmlx/ReadVariableOp£
while/lwptfvtmlx/mulMul'while/lwptfvtmlx/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul¦
while/lwptfvtmlx/add_1AddV2while/lwptfvtmlx/split:output:0while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_1
while/lwptfvtmlx/SigmoidSigmoidwhile/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid¯
!while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp,while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_1©
while/lwptfvtmlx/mul_1Mul)while/lwptfvtmlx/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_1¨
while/lwptfvtmlx/add_2AddV2while/lwptfvtmlx/split:output:1while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_2
while/lwptfvtmlx/Sigmoid_1Sigmoidwhile/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_1
while/lwptfvtmlx/mul_2Mulwhile/lwptfvtmlx/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_2
while/lwptfvtmlx/TanhTanhwhile/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh¢
while/lwptfvtmlx/mul_3Mulwhile/lwptfvtmlx/Sigmoid:y:0while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_3£
while/lwptfvtmlx/add_3AddV2while/lwptfvtmlx/mul_2:z:0while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_3¯
!while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp,while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_2°
while/lwptfvtmlx/mul_4Mul)while/lwptfvtmlx/ReadVariableOp_2:value:0while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_4¨
while/lwptfvtmlx/add_4AddV2while/lwptfvtmlx/split:output:3while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_4
while/lwptfvtmlx/Sigmoid_2Sigmoidwhile/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_2
while/lwptfvtmlx/Tanh_1Tanhwhile/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh_1¦
while/lwptfvtmlx/mul_5Mulwhile/lwptfvtmlx/Sigmoid_2:y:0while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lwptfvtmlx/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lwptfvtmlx/mul_5:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/lwptfvtmlx/add_3:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
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
0while_lwptfvtmlx_biasadd_readvariableop_resource2while_lwptfvtmlx_biasadd_readvariableop_resource_0"h
1while_lwptfvtmlx_matmul_1_readvariableop_resource3while_lwptfvtmlx_matmul_1_readvariableop_resource_0"d
/while_lwptfvtmlx_matmul_readvariableop_resource1while_lwptfvtmlx_matmul_readvariableop_resource_0"Z
*while_lwptfvtmlx_readvariableop_1_resource,while_lwptfvtmlx_readvariableop_1_resource_0"Z
*while_lwptfvtmlx_readvariableop_2_resource,while_lwptfvtmlx_readvariableop_2_resource_0"V
(while_lwptfvtmlx_readvariableop_resource*while_lwptfvtmlx_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/lwptfvtmlx/BiasAdd/ReadVariableOp'while/lwptfvtmlx/BiasAdd/ReadVariableOp2P
&while/lwptfvtmlx/MatMul/ReadVariableOp&while/lwptfvtmlx/MatMul/ReadVariableOp2T
(while/lwptfvtmlx/MatMul_1/ReadVariableOp(while/lwptfvtmlx/MatMul_1/ReadVariableOp2B
while/lwptfvtmlx/ReadVariableOpwhile/lwptfvtmlx/ReadVariableOp2F
!while/lwptfvtmlx/ReadVariableOp_1!while/lwptfvtmlx/ReadVariableOp_12F
!while/lwptfvtmlx/ReadVariableOp_2!while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_dnzlhpjizj_layer_call_fn_2598139
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25941372
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
while_cond_2593793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2593793___redundant_placeholder05
1while_while_cond_2593793___redundant_placeholder15
1while_while_cond_2593793___redundant_placeholder25
1while_while_cond_2593793___redundant_placeholder35
1while_while_cond_2593793___redundant_placeholder45
1while_while_cond_2593793___redundant_placeholder55
1while_while_cond_2593793___redundant_placeholder6
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
while_cond_2598611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2598611___redundant_placeholder05
1while_while_cond_2598611___redundant_placeholder15
1while_while_cond_2598611___redundant_placeholder25
1while_while_cond_2598611___redundant_placeholder35
1while_while_cond_2598611___redundant_placeholder45
1while_while_cond_2598611___redundant_placeholder55
1while_while_cond_2598611___redundant_placeholder6
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
dnzlhpjizj_while_body_25965602
.dnzlhpjizj_while_dnzlhpjizj_while_loop_counter8
4dnzlhpjizj_while_dnzlhpjizj_while_maximum_iterations 
dnzlhpjizj_while_placeholder"
dnzlhpjizj_while_placeholder_1"
dnzlhpjizj_while_placeholder_2"
dnzlhpjizj_while_placeholder_31
-dnzlhpjizj_while_dnzlhpjizj_strided_slice_1_0m
idnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0O
<dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0:	Q
>dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0:	 L
=dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0:	C
5dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0: E
7dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0: E
7dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0: 
dnzlhpjizj_while_identity
dnzlhpjizj_while_identity_1
dnzlhpjizj_while_identity_2
dnzlhpjizj_while_identity_3
dnzlhpjizj_while_identity_4
dnzlhpjizj_while_identity_5/
+dnzlhpjizj_while_dnzlhpjizj_strided_slice_1k
gdnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensorM
:dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource:	O
<dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource:	 J
;dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource:	A
3dnzlhpjizj_while_hswofenhiy_readvariableop_resource: C
5dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource: C
5dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource: ¢2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp¢1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp¢3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp¢*dnzlhpjizj/while/hswofenhiy/ReadVariableOp¢,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1¢,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2Ù
Bdnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2D
Bdnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shape
4dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemidnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0dnzlhpjizj_while_placeholderKdnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype026
4dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItemä
1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp<dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype023
1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOpý
"dnzlhpjizj/while/hswofenhiy/MatMulMatMul;dnzlhpjizj/while/TensorArrayV2Read/TensorListGetItem:item:09dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"dnzlhpjizj/while/hswofenhiy/MatMulê
3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp>dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOpæ
$dnzlhpjizj/while/hswofenhiy/MatMul_1MatMuldnzlhpjizj_while_placeholder_2;dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$dnzlhpjizj/while/hswofenhiy/MatMul_1Ü
dnzlhpjizj/while/hswofenhiy/addAddV2,dnzlhpjizj/while/hswofenhiy/MatMul:product:0.dnzlhpjizj/while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dnzlhpjizj/while/hswofenhiy/addã
2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp=dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOpé
#dnzlhpjizj/while/hswofenhiy/BiasAddBiasAdd#dnzlhpjizj/while/hswofenhiy/add:z:0:dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#dnzlhpjizj/while/hswofenhiy/BiasAdd
+dnzlhpjizj/while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+dnzlhpjizj/while/hswofenhiy/split/split_dim¯
!dnzlhpjizj/while/hswofenhiy/splitSplit4dnzlhpjizj/while/hswofenhiy/split/split_dim:output:0,dnzlhpjizj/while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!dnzlhpjizj/while/hswofenhiy/splitÊ
*dnzlhpjizj/while/hswofenhiy/ReadVariableOpReadVariableOp5dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*dnzlhpjizj/while/hswofenhiy/ReadVariableOpÏ
dnzlhpjizj/while/hswofenhiy/mulMul2dnzlhpjizj/while/hswofenhiy/ReadVariableOp:value:0dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dnzlhpjizj/while/hswofenhiy/mulÒ
!dnzlhpjizj/while/hswofenhiy/add_1AddV2*dnzlhpjizj/while/hswofenhiy/split:output:0#dnzlhpjizj/while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_1®
#dnzlhpjizj/while/hswofenhiy/SigmoidSigmoid%dnzlhpjizj/while/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#dnzlhpjizj/while/hswofenhiy/SigmoidÐ
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1ReadVariableOp7dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1Õ
!dnzlhpjizj/while/hswofenhiy/mul_1Mul4dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1:value:0dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_1Ô
!dnzlhpjizj/while/hswofenhiy/add_2AddV2*dnzlhpjizj/while/hswofenhiy/split:output:1%dnzlhpjizj/while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_2²
%dnzlhpjizj/while/hswofenhiy/Sigmoid_1Sigmoid%dnzlhpjizj/while/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%dnzlhpjizj/while/hswofenhiy/Sigmoid_1Ê
!dnzlhpjizj/while/hswofenhiy/mul_2Mul)dnzlhpjizj/while/hswofenhiy/Sigmoid_1:y:0dnzlhpjizj_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_2ª
 dnzlhpjizj/while/hswofenhiy/TanhTanh*dnzlhpjizj/while/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 dnzlhpjizj/while/hswofenhiy/TanhÎ
!dnzlhpjizj/while/hswofenhiy/mul_3Mul'dnzlhpjizj/while/hswofenhiy/Sigmoid:y:0$dnzlhpjizj/while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_3Ï
!dnzlhpjizj/while/hswofenhiy/add_3AddV2%dnzlhpjizj/while/hswofenhiy/mul_2:z:0%dnzlhpjizj/while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_3Ð
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2ReadVariableOp7dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2Ü
!dnzlhpjizj/while/hswofenhiy/mul_4Mul4dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2:value:0%dnzlhpjizj/while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_4Ô
!dnzlhpjizj/while/hswofenhiy/add_4AddV2*dnzlhpjizj/while/hswofenhiy/split:output:3%dnzlhpjizj/while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/add_4²
%dnzlhpjizj/while/hswofenhiy/Sigmoid_2Sigmoid%dnzlhpjizj/while/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%dnzlhpjizj/while/hswofenhiy/Sigmoid_2©
"dnzlhpjizj/while/hswofenhiy/Tanh_1Tanh%dnzlhpjizj/while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"dnzlhpjizj/while/hswofenhiy/Tanh_1Ò
!dnzlhpjizj/while/hswofenhiy/mul_5Mul)dnzlhpjizj/while/hswofenhiy/Sigmoid_2:y:0&dnzlhpjizj/while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!dnzlhpjizj/while/hswofenhiy/mul_5
5dnzlhpjizj/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemdnzlhpjizj_while_placeholder_1dnzlhpjizj_while_placeholder%dnzlhpjizj/while/hswofenhiy/mul_5:z:0*
_output_shapes
: *
element_dtype027
5dnzlhpjizj/while/TensorArrayV2Write/TensorListSetItemr
dnzlhpjizj/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
dnzlhpjizj/while/add/y
dnzlhpjizj/while/addAddV2dnzlhpjizj_while_placeholderdnzlhpjizj/while/add/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/while/addv
dnzlhpjizj/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
dnzlhpjizj/while/add_1/y­
dnzlhpjizj/while/add_1AddV2.dnzlhpjizj_while_dnzlhpjizj_while_loop_counter!dnzlhpjizj/while/add_1/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/while/add_1©
dnzlhpjizj/while/IdentityIdentitydnzlhpjizj/while/add_1:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/IdentityÇ
dnzlhpjizj/while/Identity_1Identity4dnzlhpjizj_while_dnzlhpjizj_while_maximum_iterations3^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Identity_1«
dnzlhpjizj/while/Identity_2Identitydnzlhpjizj/while/add:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Identity_2Ø
dnzlhpjizj/while/Identity_3IdentityEdnzlhpjizj/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Identity_3É
dnzlhpjizj/while/Identity_4Identity%dnzlhpjizj/while/hswofenhiy/mul_5:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/while/Identity_4É
dnzlhpjizj/while/Identity_5Identity%dnzlhpjizj/while/hswofenhiy/add_3:z:03^dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2^dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp4^dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp+^dnzlhpjizj/while/hswofenhiy/ReadVariableOp-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1-^dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/while/Identity_5"\
+dnzlhpjizj_while_dnzlhpjizj_strided_slice_1-dnzlhpjizj_while_dnzlhpjizj_strided_slice_1_0"|
;dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource=dnzlhpjizj_while_hswofenhiy_biasadd_readvariableop_resource_0"~
<dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource>dnzlhpjizj_while_hswofenhiy_matmul_1_readvariableop_resource_0"z
:dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource<dnzlhpjizj_while_hswofenhiy_matmul_readvariableop_resource_0"p
5dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource7dnzlhpjizj_while_hswofenhiy_readvariableop_1_resource_0"p
5dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource7dnzlhpjizj_while_hswofenhiy_readvariableop_2_resource_0"l
3dnzlhpjizj_while_hswofenhiy_readvariableop_resource5dnzlhpjizj_while_hswofenhiy_readvariableop_resource_0"?
dnzlhpjizj_while_identity"dnzlhpjizj/while/Identity:output:0"C
dnzlhpjizj_while_identity_1$dnzlhpjizj/while/Identity_1:output:0"C
dnzlhpjizj_while_identity_2$dnzlhpjizj/while/Identity_2:output:0"C
dnzlhpjizj_while_identity_3$dnzlhpjizj/while/Identity_3:output:0"C
dnzlhpjizj_while_identity_4$dnzlhpjizj/while/Identity_4:output:0"C
dnzlhpjizj_while_identity_5$dnzlhpjizj/while/Identity_5:output:0"Ô
gdnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensoridnzlhpjizj_while_tensorarrayv2read_tensorlistgetitem_dnzlhpjizj_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2dnzlhpjizj/while/hswofenhiy/BiasAdd/ReadVariableOp2f
1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp1dnzlhpjizj/while/hswofenhiy/MatMul/ReadVariableOp2j
3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp3dnzlhpjizj/while/hswofenhiy/MatMul_1/ReadVariableOp2X
*dnzlhpjizj/while/hswofenhiy/ReadVariableOp*dnzlhpjizj/while/hswofenhiy/ReadVariableOp2\
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_1,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_12\
,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2,dnzlhpjizj/while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2594814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2594814___redundant_placeholder05
1while_while_cond_2594814___redundant_placeholder15
1while_while_cond_2594814___redundant_placeholder25
1while_while_cond_2594814___redundant_placeholder35
1while_while_cond_2594814___redundant_placeholder45
1while_while_cond_2594814___redundant_placeholder55
1while_while_cond_2594814___redundant_placeholder6
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
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_2595664

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
ç)
Ò
while_body_2594057
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_hswofenhiy_2594081_0:	-
while_hswofenhiy_2594083_0:	 )
while_hswofenhiy_2594085_0:	(
while_hswofenhiy_2594087_0: (
while_hswofenhiy_2594089_0: (
while_hswofenhiy_2594091_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_hswofenhiy_2594081:	+
while_hswofenhiy_2594083:	 '
while_hswofenhiy_2594085:	&
while_hswofenhiy_2594087: &
while_hswofenhiy_2594089: &
while_hswofenhiy_2594091: ¢(while/hswofenhiy/StatefulPartitionedCallÃ
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
(while/hswofenhiy/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_hswofenhiy_2594081_0while_hswofenhiy_2594083_0while_hswofenhiy_2594085_0while_hswofenhiy_2594087_0while_hswofenhiy_2594089_0while_hswofenhiy_2594091_0*
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_25939612*
(while/hswofenhiy/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/hswofenhiy/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/hswofenhiy/StatefulPartitionedCall:output:1)^while/hswofenhiy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/hswofenhiy/StatefulPartitionedCall:output:2)^while/hswofenhiy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_hswofenhiy_2594081while_hswofenhiy_2594081_0"6
while_hswofenhiy_2594083while_hswofenhiy_2594083_0"6
while_hswofenhiy_2594085while_hswofenhiy_2594085_0"6
while_hswofenhiy_2594087while_hswofenhiy_2594087_0"6
while_hswofenhiy_2594089while_hswofenhiy_2594089_0"6
while_hswofenhiy_2594091while_hswofenhiy_2594091_0")
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
(while/hswofenhiy/StatefulPartitionedCall(while/hswofenhiy/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
nyosplwtfa_while_body_25971402
.nyosplwtfa_while_nyosplwtfa_while_loop_counter8
4nyosplwtfa_while_nyosplwtfa_while_maximum_iterations 
nyosplwtfa_while_placeholder"
nyosplwtfa_while_placeholder_1"
nyosplwtfa_while_placeholder_2"
nyosplwtfa_while_placeholder_31
-nyosplwtfa_while_nyosplwtfa_strided_slice_1_0m
inyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0O
<nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0:	 Q
>nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 L
=nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0:	C
5nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0: E
7nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0: E
7nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0: 
nyosplwtfa_while_identity
nyosplwtfa_while_identity_1
nyosplwtfa_while_identity_2
nyosplwtfa_while_identity_3
nyosplwtfa_while_identity_4
nyosplwtfa_while_identity_5/
+nyosplwtfa_while_nyosplwtfa_strided_slice_1k
gnyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensorM
:nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource:	 O
<nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource:	 J
;nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource:	A
3nyosplwtfa_while_lwptfvtmlx_readvariableop_resource: C
5nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource: C
5nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource: ¢2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp¢1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp¢3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp¢*nyosplwtfa/while/lwptfvtmlx/ReadVariableOp¢,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1¢,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2Ù
Bnyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bnyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shape
4nyosplwtfa/while/TensorArrayV2Read/TensorListGetItemTensorListGetIteminyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0nyosplwtfa_while_placeholderKnyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4nyosplwtfa/while/TensorArrayV2Read/TensorListGetItemä
1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp<nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOpý
"nyosplwtfa/while/lwptfvtmlx/MatMulMatMul;nyosplwtfa/while/TensorArrayV2Read/TensorListGetItem:item:09nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"nyosplwtfa/while/lwptfvtmlx/MatMulê
3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp>nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOpæ
$nyosplwtfa/while/lwptfvtmlx/MatMul_1MatMulnyosplwtfa_while_placeholder_2;nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$nyosplwtfa/while/lwptfvtmlx/MatMul_1Ü
nyosplwtfa/while/lwptfvtmlx/addAddV2,nyosplwtfa/while/lwptfvtmlx/MatMul:product:0.nyosplwtfa/while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
nyosplwtfa/while/lwptfvtmlx/addã
2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp=nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOpé
#nyosplwtfa/while/lwptfvtmlx/BiasAddBiasAdd#nyosplwtfa/while/lwptfvtmlx/add:z:0:nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#nyosplwtfa/while/lwptfvtmlx/BiasAdd
+nyosplwtfa/while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+nyosplwtfa/while/lwptfvtmlx/split/split_dim¯
!nyosplwtfa/while/lwptfvtmlx/splitSplit4nyosplwtfa/while/lwptfvtmlx/split/split_dim:output:0,nyosplwtfa/while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!nyosplwtfa/while/lwptfvtmlx/splitÊ
*nyosplwtfa/while/lwptfvtmlx/ReadVariableOpReadVariableOp5nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*nyosplwtfa/while/lwptfvtmlx/ReadVariableOpÏ
nyosplwtfa/while/lwptfvtmlx/mulMul2nyosplwtfa/while/lwptfvtmlx/ReadVariableOp:value:0nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
nyosplwtfa/while/lwptfvtmlx/mulÒ
!nyosplwtfa/while/lwptfvtmlx/add_1AddV2*nyosplwtfa/while/lwptfvtmlx/split:output:0#nyosplwtfa/while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_1®
#nyosplwtfa/while/lwptfvtmlx/SigmoidSigmoid%nyosplwtfa/while/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#nyosplwtfa/while/lwptfvtmlx/SigmoidÐ
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp7nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1Õ
!nyosplwtfa/while/lwptfvtmlx/mul_1Mul4nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1:value:0nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_1Ô
!nyosplwtfa/while/lwptfvtmlx/add_2AddV2*nyosplwtfa/while/lwptfvtmlx/split:output:1%nyosplwtfa/while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_2²
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_1Sigmoid%nyosplwtfa/while/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_1Ê
!nyosplwtfa/while/lwptfvtmlx/mul_2Mul)nyosplwtfa/while/lwptfvtmlx/Sigmoid_1:y:0nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_2ª
 nyosplwtfa/while/lwptfvtmlx/TanhTanh*nyosplwtfa/while/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 nyosplwtfa/while/lwptfvtmlx/TanhÎ
!nyosplwtfa/while/lwptfvtmlx/mul_3Mul'nyosplwtfa/while/lwptfvtmlx/Sigmoid:y:0$nyosplwtfa/while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_3Ï
!nyosplwtfa/while/lwptfvtmlx/add_3AddV2%nyosplwtfa/while/lwptfvtmlx/mul_2:z:0%nyosplwtfa/while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_3Ð
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp7nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2Ü
!nyosplwtfa/while/lwptfvtmlx/mul_4Mul4nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2:value:0%nyosplwtfa/while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_4Ô
!nyosplwtfa/while/lwptfvtmlx/add_4AddV2*nyosplwtfa/while/lwptfvtmlx/split:output:3%nyosplwtfa/while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_4²
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_2Sigmoid%nyosplwtfa/while/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_2©
"nyosplwtfa/while/lwptfvtmlx/Tanh_1Tanh%nyosplwtfa/while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"nyosplwtfa/while/lwptfvtmlx/Tanh_1Ò
!nyosplwtfa/while/lwptfvtmlx/mul_5Mul)nyosplwtfa/while/lwptfvtmlx/Sigmoid_2:y:0&nyosplwtfa/while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_5
5nyosplwtfa/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemnyosplwtfa_while_placeholder_1nyosplwtfa_while_placeholder%nyosplwtfa/while/lwptfvtmlx/mul_5:z:0*
_output_shapes
: *
element_dtype027
5nyosplwtfa/while/TensorArrayV2Write/TensorListSetItemr
nyosplwtfa/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
nyosplwtfa/while/add/y
nyosplwtfa/while/addAddV2nyosplwtfa_while_placeholdernyosplwtfa/while/add/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/while/addv
nyosplwtfa/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
nyosplwtfa/while/add_1/y­
nyosplwtfa/while/add_1AddV2.nyosplwtfa_while_nyosplwtfa_while_loop_counter!nyosplwtfa/while/add_1/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/while/add_1©
nyosplwtfa/while/IdentityIdentitynyosplwtfa/while/add_1:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/IdentityÇ
nyosplwtfa/while/Identity_1Identity4nyosplwtfa_while_nyosplwtfa_while_maximum_iterations3^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/Identity_1«
nyosplwtfa/while/Identity_2Identitynyosplwtfa/while/add:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/Identity_2Ø
nyosplwtfa/while/Identity_3IdentityEnyosplwtfa/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/Identity_3É
nyosplwtfa/while/Identity_4Identity%nyosplwtfa/while/lwptfvtmlx/mul_5:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/while/Identity_4É
nyosplwtfa/while/Identity_5Identity%nyosplwtfa/while/lwptfvtmlx/add_3:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/while/Identity_5"?
nyosplwtfa_while_identity"nyosplwtfa/while/Identity:output:0"C
nyosplwtfa_while_identity_1$nyosplwtfa/while/Identity_1:output:0"C
nyosplwtfa_while_identity_2$nyosplwtfa/while/Identity_2:output:0"C
nyosplwtfa_while_identity_3$nyosplwtfa/while/Identity_3:output:0"C
nyosplwtfa_while_identity_4$nyosplwtfa/while/Identity_4:output:0"C
nyosplwtfa_while_identity_5$nyosplwtfa/while/Identity_5:output:0"|
;nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource=nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0"~
<nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource>nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0"z
:nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource<nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0"p
5nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource7nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0"p
5nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource7nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0"l
3nyosplwtfa_while_lwptfvtmlx_readvariableop_resource5nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0"\
+nyosplwtfa_while_nyosplwtfa_strided_slice_1-nyosplwtfa_while_nyosplwtfa_strided_slice_1_0"Ô
gnyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensorinyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2f
1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp2j
3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp2X
*nyosplwtfa/while/lwptfvtmlx/ReadVariableOp*nyosplwtfa/while/lwptfvtmlx/ReadVariableOp2\
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_12\
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_cond_2597643
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2597643___redundant_placeholder05
1while_while_cond_2597643___redundant_placeholder15
1while_while_cond_2597643___redundant_placeholder25
1while_while_cond_2597643___redundant_placeholder35
1while_while_cond_2597643___redundant_placeholder45
1while_while_cond_2597643___redundant_placeholder55
1while_while_cond_2597643___redundant_placeholder6
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2593774

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
while_body_2596028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_hswofenhiy_matmul_readvariableop_resource_0:	F
3while_hswofenhiy_matmul_1_readvariableop_resource_0:	 A
2while_hswofenhiy_biasadd_readvariableop_resource_0:	8
*while_hswofenhiy_readvariableop_resource_0: :
,while_hswofenhiy_readvariableop_1_resource_0: :
,while_hswofenhiy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_hswofenhiy_matmul_readvariableop_resource:	D
1while_hswofenhiy_matmul_1_readvariableop_resource:	 ?
0while_hswofenhiy_biasadd_readvariableop_resource:	6
(while_hswofenhiy_readvariableop_resource: 8
*while_hswofenhiy_readvariableop_1_resource: 8
*while_hswofenhiy_readvariableop_2_resource: ¢'while/hswofenhiy/BiasAdd/ReadVariableOp¢&while/hswofenhiy/MatMul/ReadVariableOp¢(while/hswofenhiy/MatMul_1/ReadVariableOp¢while/hswofenhiy/ReadVariableOp¢!while/hswofenhiy/ReadVariableOp_1¢!while/hswofenhiy/ReadVariableOp_2Ã
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
&while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp1while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/hswofenhiy/MatMul/ReadVariableOpÑ
while/hswofenhiy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMulÉ
(while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp3while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/hswofenhiy/MatMul_1/ReadVariableOpº
while/hswofenhiy/MatMul_1MatMulwhile_placeholder_20while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMul_1°
while/hswofenhiy/addAddV2!while/hswofenhiy/MatMul:product:0#while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/addÂ
'while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp2while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/hswofenhiy/BiasAdd/ReadVariableOp½
while/hswofenhiy/BiasAddBiasAddwhile/hswofenhiy/add:z:0/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/BiasAdd
 while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/hswofenhiy/split/split_dim
while/hswofenhiy/splitSplit)while/hswofenhiy/split/split_dim:output:0!while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/hswofenhiy/split©
while/hswofenhiy/ReadVariableOpReadVariableOp*while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/hswofenhiy/ReadVariableOp£
while/hswofenhiy/mulMul'while/hswofenhiy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul¦
while/hswofenhiy/add_1AddV2while/hswofenhiy/split:output:0while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_1
while/hswofenhiy/SigmoidSigmoidwhile/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid¯
!while/hswofenhiy/ReadVariableOp_1ReadVariableOp,while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_1©
while/hswofenhiy/mul_1Mul)while/hswofenhiy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_1¨
while/hswofenhiy/add_2AddV2while/hswofenhiy/split:output:1while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_2
while/hswofenhiy/Sigmoid_1Sigmoidwhile/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_1
while/hswofenhiy/mul_2Mulwhile/hswofenhiy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_2
while/hswofenhiy/TanhTanhwhile/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh¢
while/hswofenhiy/mul_3Mulwhile/hswofenhiy/Sigmoid:y:0while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_3£
while/hswofenhiy/add_3AddV2while/hswofenhiy/mul_2:z:0while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_3¯
!while/hswofenhiy/ReadVariableOp_2ReadVariableOp,while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_2°
while/hswofenhiy/mul_4Mul)while/hswofenhiy/ReadVariableOp_2:value:0while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_4¨
while/hswofenhiy/add_4AddV2while/hswofenhiy/split:output:3while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_4
while/hswofenhiy/Sigmoid_2Sigmoidwhile/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_2
while/hswofenhiy/Tanh_1Tanhwhile/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh_1¦
while/hswofenhiy/mul_5Mulwhile/hswofenhiy/Sigmoid_2:y:0while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/hswofenhiy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/hswofenhiy/mul_5:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/hswofenhiy/add_3:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_hswofenhiy_biasadd_readvariableop_resource2while_hswofenhiy_biasadd_readvariableop_resource_0"h
1while_hswofenhiy_matmul_1_readvariableop_resource3while_hswofenhiy_matmul_1_readvariableop_resource_0"d
/while_hswofenhiy_matmul_readvariableop_resource1while_hswofenhiy_matmul_readvariableop_resource_0"Z
*while_hswofenhiy_readvariableop_1_resource,while_hswofenhiy_readvariableop_1_resource_0"Z
*while_hswofenhiy_readvariableop_2_resource,while_hswofenhiy_readvariableop_2_resource_0"V
(while_hswofenhiy_readvariableop_resource*while_hswofenhiy_readvariableop_resource_0")
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
'while/hswofenhiy/BiasAdd/ReadVariableOp'while/hswofenhiy/BiasAdd/ReadVariableOp2P
&while/hswofenhiy/MatMul/ReadVariableOp&while/hswofenhiy/MatMul/ReadVariableOp2T
(while/hswofenhiy/MatMul_1/ReadVariableOp(while/hswofenhiy/MatMul_1/ReadVariableOp2B
while/hswofenhiy/ReadVariableOpwhile/hswofenhiy/ReadVariableOp2F
!while/hswofenhiy/ReadVariableOp_1!while/hswofenhiy/ReadVariableOp_12F
!while/hswofenhiy/ReadVariableOp_2!while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
,__inference_nyosplwtfa_layer_call_fn_2598961

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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25959152
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
ë

,__inference_nyosplwtfa_layer_call_fn_2598927
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25948952
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2595640

inputs<
)lwptfvtmlx_matmul_readvariableop_resource:	 >
+lwptfvtmlx_matmul_1_readvariableop_resource:	 9
*lwptfvtmlx_biasadd_readvariableop_resource:	0
"lwptfvtmlx_readvariableop_resource: 2
$lwptfvtmlx_readvariableop_1_resource: 2
$lwptfvtmlx_readvariableop_2_resource: 
identity¢!lwptfvtmlx/BiasAdd/ReadVariableOp¢ lwptfvtmlx/MatMul/ReadVariableOp¢"lwptfvtmlx/MatMul_1/ReadVariableOp¢lwptfvtmlx/ReadVariableOp¢lwptfvtmlx/ReadVariableOp_1¢lwptfvtmlx/ReadVariableOp_2¢whileD
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
 lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp)lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 lwptfvtmlx/MatMul/ReadVariableOp§
lwptfvtmlx/MatMulMatMulstrided_slice_2:output:0(lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMulµ
"lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp+lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lwptfvtmlx/MatMul_1/ReadVariableOp£
lwptfvtmlx/MatMul_1MatMulzeros:output:0*lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMul_1
lwptfvtmlx/addAddV2lwptfvtmlx/MatMul:product:0lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/add®
!lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp*lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!lwptfvtmlx/BiasAdd/ReadVariableOp¥
lwptfvtmlx/BiasAddBiasAddlwptfvtmlx/add:z:0)lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/BiasAddz
lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lwptfvtmlx/split/split_dimë
lwptfvtmlx/splitSplit#lwptfvtmlx/split/split_dim:output:0lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lwptfvtmlx/split
lwptfvtmlx/ReadVariableOpReadVariableOp"lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp
lwptfvtmlx/mulMul!lwptfvtmlx/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul
lwptfvtmlx/add_1AddV2lwptfvtmlx/split:output:0lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_1{
lwptfvtmlx/SigmoidSigmoidlwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid
lwptfvtmlx/ReadVariableOp_1ReadVariableOp$lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_1
lwptfvtmlx/mul_1Mul#lwptfvtmlx/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_1
lwptfvtmlx/add_2AddV2lwptfvtmlx/split:output:1lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_2
lwptfvtmlx/Sigmoid_1Sigmoidlwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_1
lwptfvtmlx/mul_2Mullwptfvtmlx/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_2w
lwptfvtmlx/TanhTanhlwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh
lwptfvtmlx/mul_3Mullwptfvtmlx/Sigmoid:y:0lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_3
lwptfvtmlx/add_3AddV2lwptfvtmlx/mul_2:z:0lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_3
lwptfvtmlx/ReadVariableOp_2ReadVariableOp$lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_2
lwptfvtmlx/mul_4Mul#lwptfvtmlx/ReadVariableOp_2:value:0lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_4
lwptfvtmlx/add_4AddV2lwptfvtmlx/split:output:3lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_4
lwptfvtmlx/Sigmoid_2Sigmoidlwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_2v
lwptfvtmlx/Tanh_1Tanhlwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh_1
lwptfvtmlx/mul_5Mullwptfvtmlx/Sigmoid_2:y:0lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lwptfvtmlx_matmul_readvariableop_resource+lwptfvtmlx_matmul_1_readvariableop_resource*lwptfvtmlx_biasadd_readvariableop_resource"lwptfvtmlx_readvariableop_resource$lwptfvtmlx_readvariableop_1_resource$lwptfvtmlx_readvariableop_2_resource*
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
while_body_2595539*
condR
while_cond_2595538*Q
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
IdentityIdentitystrided_slice_3:output:0"^lwptfvtmlx/BiasAdd/ReadVariableOp!^lwptfvtmlx/MatMul/ReadVariableOp#^lwptfvtmlx/MatMul_1/ReadVariableOp^lwptfvtmlx/ReadVariableOp^lwptfvtmlx/ReadVariableOp_1^lwptfvtmlx/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!lwptfvtmlx/BiasAdd/ReadVariableOp!lwptfvtmlx/BiasAdd/ReadVariableOp2D
 lwptfvtmlx/MatMul/ReadVariableOp lwptfvtmlx/MatMul/ReadVariableOp2H
"lwptfvtmlx/MatMul_1/ReadVariableOp"lwptfvtmlx/MatMul_1/ReadVariableOp26
lwptfvtmlx/ReadVariableOplwptfvtmlx/ReadVariableOp2:
lwptfvtmlx/ReadVariableOp_1lwptfvtmlx/ReadVariableOp_12:
lwptfvtmlx/ReadVariableOp_2lwptfvtmlx/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯

#__inference__traced_restore_2599515
file_prefix8
"assignvariableop_gtjikcltwy_kernel:0
"assignvariableop_1_gtjikcltwy_bias:6
$assignvariableop_2_chsgvefspq_kernel: 0
"assignvariableop_3_chsgvefspq_bias:)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: B
/assignvariableop_9_dnzlhpjizj_hswofenhiy_kernel:	M
:assignvariableop_10_dnzlhpjizj_hswofenhiy_recurrent_kernel:	 =
.assignvariableop_11_dnzlhpjizj_hswofenhiy_bias:	S
Eassignvariableop_12_dnzlhpjizj_hswofenhiy_input_gate_peephole_weights: T
Fassignvariableop_13_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weights: T
Fassignvariableop_14_dnzlhpjizj_hswofenhiy_output_gate_peephole_weights: C
0assignvariableop_15_nyosplwtfa_lwptfvtmlx_kernel:	 M
:assignvariableop_16_nyosplwtfa_lwptfvtmlx_recurrent_kernel:	 =
.assignvariableop_17_nyosplwtfa_lwptfvtmlx_bias:	S
Eassignvariableop_18_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weights: T
Fassignvariableop_19_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weights: T
Fassignvariableop_20_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weights: #
assignvariableop_21_total: #
assignvariableop_22_count: G
1assignvariableop_23_rmsprop_gtjikcltwy_kernel_rms:=
/assignvariableop_24_rmsprop_gtjikcltwy_bias_rms:C
1assignvariableop_25_rmsprop_chsgvefspq_kernel_rms: =
/assignvariableop_26_rmsprop_chsgvefspq_bias_rms:O
<assignvariableop_27_rmsprop_dnzlhpjizj_hswofenhiy_kernel_rms:	Y
Fassignvariableop_28_rmsprop_dnzlhpjizj_hswofenhiy_recurrent_kernel_rms:	 I
:assignvariableop_29_rmsprop_dnzlhpjizj_hswofenhiy_bias_rms:	_
Qassignvariableop_30_rmsprop_dnzlhpjizj_hswofenhiy_input_gate_peephole_weights_rms: `
Rassignvariableop_31_rmsprop_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weights_rms: `
Rassignvariableop_32_rmsprop_dnzlhpjizj_hswofenhiy_output_gate_peephole_weights_rms: O
<assignvariableop_33_rmsprop_nyosplwtfa_lwptfvtmlx_kernel_rms:	 Y
Fassignvariableop_34_rmsprop_nyosplwtfa_lwptfvtmlx_recurrent_kernel_rms:	 I
:assignvariableop_35_rmsprop_nyosplwtfa_lwptfvtmlx_bias_rms:	_
Qassignvariableop_36_rmsprop_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weights_rms: `
Rassignvariableop_37_rmsprop_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weights_rms: `
Rassignvariableop_38_rmsprop_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weights_rms: 
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
AssignVariableOpAssignVariableOp"assignvariableop_gtjikcltwy_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_gtjikcltwy_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_chsgvefspq_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_chsgvefspq_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp/assignvariableop_9_dnzlhpjizj_hswofenhiy_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Â
AssignVariableOp_10AssignVariableOp:assignvariableop_10_dnzlhpjizj_hswofenhiy_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOp.assignvariableop_11_dnzlhpjizj_hswofenhiy_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Í
AssignVariableOp_12AssignVariableOpEassignvariableop_12_dnzlhpjizj_hswofenhiy_input_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Î
AssignVariableOp_13AssignVariableOpFassignvariableop_13_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weightsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Î
AssignVariableOp_14AssignVariableOpFassignvariableop_14_dnzlhpjizj_hswofenhiy_output_gate_peephole_weightsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¸
AssignVariableOp_15AssignVariableOp0assignvariableop_15_nyosplwtfa_lwptfvtmlx_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Â
AssignVariableOp_16AssignVariableOp:assignvariableop_16_nyosplwtfa_lwptfvtmlx_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_nyosplwtfa_lwptfvtmlx_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Í
AssignVariableOp_18AssignVariableOpEassignvariableop_18_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weightsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Î
AssignVariableOp_19AssignVariableOpFassignvariableop_19_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weightsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Î
AssignVariableOp_20AssignVariableOpFassignvariableop_20_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weightsIdentity_20:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp1assignvariableop_23_rmsprop_gtjikcltwy_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_rmsprop_gtjikcltwy_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¹
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_chsgvefspq_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_rmsprop_chsgvefspq_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_dnzlhpjizj_hswofenhiy_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_rmsprop_dnzlhpjizj_hswofenhiy_recurrent_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_rmsprop_dnzlhpjizj_hswofenhiy_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ù
AssignVariableOp_30AssignVariableOpQassignvariableop_30_rmsprop_dnzlhpjizj_hswofenhiy_input_gate_peephole_weights_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ú
AssignVariableOp_31AssignVariableOpRassignvariableop_31_rmsprop_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weights_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ú
AssignVariableOp_32AssignVariableOpRassignvariableop_32_rmsprop_dnzlhpjizj_hswofenhiy_output_gate_peephole_weights_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ä
AssignVariableOp_33AssignVariableOp<assignvariableop_33_rmsprop_nyosplwtfa_lwptfvtmlx_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_rmsprop_nyosplwtfa_lwptfvtmlx_recurrent_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Â
AssignVariableOp_35AssignVariableOp:assignvariableop_35_rmsprop_nyosplwtfa_lwptfvtmlx_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ù
AssignVariableOp_36AssignVariableOpQassignvariableop_36_rmsprop_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weights_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ú
AssignVariableOp_37AssignVariableOpRassignvariableop_37_rmsprop_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weights_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ú
AssignVariableOp_38AssignVariableOpRassignvariableop_38_rmsprop_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weights_rmsIdentity_38:output:0"/device:CPU:0*
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

c
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_2595266

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
while_cond_2598003
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2598003___redundant_placeholder05
1while_while_cond_2598003___redundant_placeholder15
1while_while_cond_2598003___redundant_placeholder25
1while_while_cond_2598003___redundant_placeholder35
1while_while_cond_2598003___redundant_placeholder45
1while_while_cond_2598003___redundant_placeholder55
1while_while_cond_2598003___redundant_placeholder6
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2596129

inputs<
)hswofenhiy_matmul_readvariableop_resource:	>
+hswofenhiy_matmul_1_readvariableop_resource:	 9
*hswofenhiy_biasadd_readvariableop_resource:	0
"hswofenhiy_readvariableop_resource: 2
$hswofenhiy_readvariableop_1_resource: 2
$hswofenhiy_readvariableop_2_resource: 
identity¢!hswofenhiy/BiasAdd/ReadVariableOp¢ hswofenhiy/MatMul/ReadVariableOp¢"hswofenhiy/MatMul_1/ReadVariableOp¢hswofenhiy/ReadVariableOp¢hswofenhiy/ReadVariableOp_1¢hswofenhiy/ReadVariableOp_2¢whileD
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
 hswofenhiy/MatMul/ReadVariableOpReadVariableOp)hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 hswofenhiy/MatMul/ReadVariableOp§
hswofenhiy/MatMulMatMulstrided_slice_2:output:0(hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMulµ
"hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp+hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"hswofenhiy/MatMul_1/ReadVariableOp£
hswofenhiy/MatMul_1MatMulzeros:output:0*hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMul_1
hswofenhiy/addAddV2hswofenhiy/MatMul:product:0hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/add®
!hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp*hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!hswofenhiy/BiasAdd/ReadVariableOp¥
hswofenhiy/BiasAddBiasAddhswofenhiy/add:z:0)hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/BiasAddz
hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
hswofenhiy/split/split_dimë
hswofenhiy/splitSplit#hswofenhiy/split/split_dim:output:0hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
hswofenhiy/split
hswofenhiy/ReadVariableOpReadVariableOp"hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp
hswofenhiy/mulMul!hswofenhiy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul
hswofenhiy/add_1AddV2hswofenhiy/split:output:0hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_1{
hswofenhiy/SigmoidSigmoidhswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid
hswofenhiy/ReadVariableOp_1ReadVariableOp$hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_1
hswofenhiy/mul_1Mul#hswofenhiy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_1
hswofenhiy/add_2AddV2hswofenhiy/split:output:1hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_2
hswofenhiy/Sigmoid_1Sigmoidhswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_1
hswofenhiy/mul_2Mulhswofenhiy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_2w
hswofenhiy/TanhTanhhswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh
hswofenhiy/mul_3Mulhswofenhiy/Sigmoid:y:0hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_3
hswofenhiy/add_3AddV2hswofenhiy/mul_2:z:0hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_3
hswofenhiy/ReadVariableOp_2ReadVariableOp$hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_2
hswofenhiy/mul_4Mul#hswofenhiy/ReadVariableOp_2:value:0hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_4
hswofenhiy/add_4AddV2hswofenhiy/split:output:3hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_4
hswofenhiy/Sigmoid_2Sigmoidhswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_2v
hswofenhiy/Tanh_1Tanhhswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh_1
hswofenhiy/mul_5Mulhswofenhiy/Sigmoid_2:y:0hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)hswofenhiy_matmul_readvariableop_resource+hswofenhiy_matmul_1_readvariableop_resource*hswofenhiy_biasadd_readvariableop_resource"hswofenhiy_readvariableop_resource$hswofenhiy_readvariableop_1_resource$hswofenhiy_readvariableop_2_resource*
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
while_body_2596028*
condR
while_cond_2596027*Q
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
IdentityIdentitytranspose_1:y:0"^hswofenhiy/BiasAdd/ReadVariableOp!^hswofenhiy/MatMul/ReadVariableOp#^hswofenhiy/MatMul_1/ReadVariableOp^hswofenhiy/ReadVariableOp^hswofenhiy/ReadVariableOp_1^hswofenhiy/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!hswofenhiy/BiasAdd/ReadVariableOp!hswofenhiy/BiasAdd/ReadVariableOp2D
 hswofenhiy/MatMul/ReadVariableOp hswofenhiy/MatMul/ReadVariableOp2H
"hswofenhiy/MatMul_1/ReadVariableOp"hswofenhiy/MatMul_1/ReadVariableOp26
hswofenhiy/ReadVariableOphswofenhiy/ReadVariableOp2:
hswofenhiy/ReadVariableOp_1hswofenhiy/ReadVariableOp_12:
hswofenhiy/ReadVariableOp_2hswofenhiy/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡h

G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2595447

inputs<
)hswofenhiy_matmul_readvariableop_resource:	>
+hswofenhiy_matmul_1_readvariableop_resource:	 9
*hswofenhiy_biasadd_readvariableop_resource:	0
"hswofenhiy_readvariableop_resource: 2
$hswofenhiy_readvariableop_1_resource: 2
$hswofenhiy_readvariableop_2_resource: 
identity¢!hswofenhiy/BiasAdd/ReadVariableOp¢ hswofenhiy/MatMul/ReadVariableOp¢"hswofenhiy/MatMul_1/ReadVariableOp¢hswofenhiy/ReadVariableOp¢hswofenhiy/ReadVariableOp_1¢hswofenhiy/ReadVariableOp_2¢whileD
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
 hswofenhiy/MatMul/ReadVariableOpReadVariableOp)hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 hswofenhiy/MatMul/ReadVariableOp§
hswofenhiy/MatMulMatMulstrided_slice_2:output:0(hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMulµ
"hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp+hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"hswofenhiy/MatMul_1/ReadVariableOp£
hswofenhiy/MatMul_1MatMulzeros:output:0*hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMul_1
hswofenhiy/addAddV2hswofenhiy/MatMul:product:0hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/add®
!hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp*hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!hswofenhiy/BiasAdd/ReadVariableOp¥
hswofenhiy/BiasAddBiasAddhswofenhiy/add:z:0)hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/BiasAddz
hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
hswofenhiy/split/split_dimë
hswofenhiy/splitSplit#hswofenhiy/split/split_dim:output:0hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
hswofenhiy/split
hswofenhiy/ReadVariableOpReadVariableOp"hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp
hswofenhiy/mulMul!hswofenhiy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul
hswofenhiy/add_1AddV2hswofenhiy/split:output:0hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_1{
hswofenhiy/SigmoidSigmoidhswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid
hswofenhiy/ReadVariableOp_1ReadVariableOp$hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_1
hswofenhiy/mul_1Mul#hswofenhiy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_1
hswofenhiy/add_2AddV2hswofenhiy/split:output:1hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_2
hswofenhiy/Sigmoid_1Sigmoidhswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_1
hswofenhiy/mul_2Mulhswofenhiy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_2w
hswofenhiy/TanhTanhhswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh
hswofenhiy/mul_3Mulhswofenhiy/Sigmoid:y:0hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_3
hswofenhiy/add_3AddV2hswofenhiy/mul_2:z:0hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_3
hswofenhiy/ReadVariableOp_2ReadVariableOp$hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_2
hswofenhiy/mul_4Mul#hswofenhiy/ReadVariableOp_2:value:0hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_4
hswofenhiy/add_4AddV2hswofenhiy/split:output:3hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_4
hswofenhiy/Sigmoid_2Sigmoidhswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_2v
hswofenhiy/Tanh_1Tanhhswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh_1
hswofenhiy/mul_5Mulhswofenhiy/Sigmoid_2:y:0hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)hswofenhiy_matmul_readvariableop_resource+hswofenhiy_matmul_1_readvariableop_resource*hswofenhiy_biasadd_readvariableop_resource"hswofenhiy_readvariableop_resource$hswofenhiy_readvariableop_1_resource$hswofenhiy_readvariableop_2_resource*
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
while_body_2595346*
condR
while_cond_2595345*Q
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
IdentityIdentitytranspose_1:y:0"^hswofenhiy/BiasAdd/ReadVariableOp!^hswofenhiy/MatMul/ReadVariableOp#^hswofenhiy/MatMul_1/ReadVariableOp^hswofenhiy/ReadVariableOp^hswofenhiy/ReadVariableOp_1^hswofenhiy/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!hswofenhiy/BiasAdd/ReadVariableOp!hswofenhiy/BiasAdd/ReadVariableOp2D
 hswofenhiy/MatMul/ReadVariableOp hswofenhiy/MatMul/ReadVariableOp2H
"hswofenhiy/MatMul_1/ReadVariableOp"hswofenhiy/MatMul_1/ReadVariableOp26
hswofenhiy/ReadVariableOphswofenhiy/ReadVariableOp2:
hswofenhiy/ReadVariableOp_1hswofenhiy/ReadVariableOp_12:
hswofenhiy/ReadVariableOp_2hswofenhiy/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

,__inference_dnzlhpjizj_layer_call_fn_2598156

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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25954472
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
¯F
ê
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2594632

inputs%
lwptfvtmlx_2594533:	 %
lwptfvtmlx_2594535:	 !
lwptfvtmlx_2594537:	 
lwptfvtmlx_2594539:  
lwptfvtmlx_2594541:  
lwptfvtmlx_2594543: 
identity¢"lwptfvtmlx/StatefulPartitionedCall¢whileD
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
"lwptfvtmlx/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lwptfvtmlx_2594533lwptfvtmlx_2594535lwptfvtmlx_2594537lwptfvtmlx_2594539lwptfvtmlx_2594541lwptfvtmlx_2594543*
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_25945322$
"lwptfvtmlx/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lwptfvtmlx_2594533lwptfvtmlx_2594535lwptfvtmlx_2594537lwptfvtmlx_2594539lwptfvtmlx_2594541lwptfvtmlx_2594543*
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
while_body_2594552*
condR
while_cond_2594551*Q
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
IdentityIdentitystrided_slice_3:output:0#^lwptfvtmlx/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"lwptfvtmlx/StatefulPartitionedCall"lwptfvtmlx/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±'
³
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2593961

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
while_cond_2596027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2596027___redundant_placeholder05
1while_while_cond_2596027___redundant_placeholder15
1while_while_cond_2596027___redundant_placeholder25
1while_while_cond_2596027___redundant_placeholder35
1while_while_cond_2596027___redundant_placeholder45
1while_while_cond_2596027___redundant_placeholder55
1while_while_cond_2596027___redundant_placeholder6
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
while_cond_2595538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2595538___redundant_placeholder05
1while_while_cond_2595538___redundant_placeholder15
1while_while_cond_2595538___redundant_placeholder25
1while_while_cond_2595538___redundant_placeholder35
1while_while_cond_2595538___redundant_placeholder45
1while_while_cond_2595538___redundant_placeholder55
1while_while_cond_2595538___redundant_placeholder6
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
while_body_2598432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lwptfvtmlx_matmul_readvariableop_resource_0:	 F
3while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 A
2while_lwptfvtmlx_biasadd_readvariableop_resource_0:	8
*while_lwptfvtmlx_readvariableop_resource_0: :
,while_lwptfvtmlx_readvariableop_1_resource_0: :
,while_lwptfvtmlx_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lwptfvtmlx_matmul_readvariableop_resource:	 D
1while_lwptfvtmlx_matmul_1_readvariableop_resource:	 ?
0while_lwptfvtmlx_biasadd_readvariableop_resource:	6
(while_lwptfvtmlx_readvariableop_resource: 8
*while_lwptfvtmlx_readvariableop_1_resource: 8
*while_lwptfvtmlx_readvariableop_2_resource: ¢'while/lwptfvtmlx/BiasAdd/ReadVariableOp¢&while/lwptfvtmlx/MatMul/ReadVariableOp¢(while/lwptfvtmlx/MatMul_1/ReadVariableOp¢while/lwptfvtmlx/ReadVariableOp¢!while/lwptfvtmlx/ReadVariableOp_1¢!while/lwptfvtmlx/ReadVariableOp_2Ã
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
&while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp1while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/lwptfvtmlx/MatMul/ReadVariableOpÑ
while/lwptfvtmlx/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMulÉ
(while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp3while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/lwptfvtmlx/MatMul_1/ReadVariableOpº
while/lwptfvtmlx/MatMul_1MatMulwhile_placeholder_20while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMul_1°
while/lwptfvtmlx/addAddV2!while/lwptfvtmlx/MatMul:product:0#while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/addÂ
'while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp2while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/lwptfvtmlx/BiasAdd/ReadVariableOp½
while/lwptfvtmlx/BiasAddBiasAddwhile/lwptfvtmlx/add:z:0/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/BiasAdd
 while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/lwptfvtmlx/split/split_dim
while/lwptfvtmlx/splitSplit)while/lwptfvtmlx/split/split_dim:output:0!while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/lwptfvtmlx/split©
while/lwptfvtmlx/ReadVariableOpReadVariableOp*while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/lwptfvtmlx/ReadVariableOp£
while/lwptfvtmlx/mulMul'while/lwptfvtmlx/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul¦
while/lwptfvtmlx/add_1AddV2while/lwptfvtmlx/split:output:0while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_1
while/lwptfvtmlx/SigmoidSigmoidwhile/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid¯
!while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp,while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_1©
while/lwptfvtmlx/mul_1Mul)while/lwptfvtmlx/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_1¨
while/lwptfvtmlx/add_2AddV2while/lwptfvtmlx/split:output:1while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_2
while/lwptfvtmlx/Sigmoid_1Sigmoidwhile/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_1
while/lwptfvtmlx/mul_2Mulwhile/lwptfvtmlx/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_2
while/lwptfvtmlx/TanhTanhwhile/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh¢
while/lwptfvtmlx/mul_3Mulwhile/lwptfvtmlx/Sigmoid:y:0while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_3£
while/lwptfvtmlx/add_3AddV2while/lwptfvtmlx/mul_2:z:0while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_3¯
!while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp,while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_2°
while/lwptfvtmlx/mul_4Mul)while/lwptfvtmlx/ReadVariableOp_2:value:0while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_4¨
while/lwptfvtmlx/add_4AddV2while/lwptfvtmlx/split:output:3while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_4
while/lwptfvtmlx/Sigmoid_2Sigmoidwhile/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_2
while/lwptfvtmlx/Tanh_1Tanhwhile/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh_1¦
while/lwptfvtmlx/mul_5Mulwhile/lwptfvtmlx/Sigmoid_2:y:0while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lwptfvtmlx/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lwptfvtmlx/mul_5:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/lwptfvtmlx/add_3:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
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
0while_lwptfvtmlx_biasadd_readvariableop_resource2while_lwptfvtmlx_biasadd_readvariableop_resource_0"h
1while_lwptfvtmlx_matmul_1_readvariableop_resource3while_lwptfvtmlx_matmul_1_readvariableop_resource_0"d
/while_lwptfvtmlx_matmul_readvariableop_resource1while_lwptfvtmlx_matmul_readvariableop_resource_0"Z
*while_lwptfvtmlx_readvariableop_1_resource,while_lwptfvtmlx_readvariableop_1_resource_0"Z
*while_lwptfvtmlx_readvariableop_2_resource,while_lwptfvtmlx_readvariableop_2_resource_0"V
(while_lwptfvtmlx_readvariableop_resource*while_lwptfvtmlx_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/lwptfvtmlx/BiasAdd/ReadVariableOp'while/lwptfvtmlx/BiasAdd/ReadVariableOp2P
&while/lwptfvtmlx/MatMul/ReadVariableOp&while/lwptfvtmlx/MatMul/ReadVariableOp2T
(while/lwptfvtmlx/MatMul_1/ReadVariableOp(while/lwptfvtmlx/MatMul_1/ReadVariableOp2B
while/lwptfvtmlx/ReadVariableOpwhile/lwptfvtmlx/ReadVariableOp2F
!while/lwptfvtmlx/ReadVariableOp_1!while/lwptfvtmlx/ReadVariableOp_12F
!while/lwptfvtmlx/ReadVariableOp_2!while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
(sequential_nyosplwtfa_while_cond_2593579H
Dsequential_nyosplwtfa_while_sequential_nyosplwtfa_while_loop_counterN
Jsequential_nyosplwtfa_while_sequential_nyosplwtfa_while_maximum_iterations+
'sequential_nyosplwtfa_while_placeholder-
)sequential_nyosplwtfa_while_placeholder_1-
)sequential_nyosplwtfa_while_placeholder_2-
)sequential_nyosplwtfa_while_placeholder_3J
Fsequential_nyosplwtfa_while_less_sequential_nyosplwtfa_strided_slice_1a
]sequential_nyosplwtfa_while_sequential_nyosplwtfa_while_cond_2593579___redundant_placeholder0a
]sequential_nyosplwtfa_while_sequential_nyosplwtfa_while_cond_2593579___redundant_placeholder1a
]sequential_nyosplwtfa_while_sequential_nyosplwtfa_while_cond_2593579___redundant_placeholder2a
]sequential_nyosplwtfa_while_sequential_nyosplwtfa_while_cond_2593579___redundant_placeholder3a
]sequential_nyosplwtfa_while_sequential_nyosplwtfa_while_cond_2593579___redundant_placeholder4a
]sequential_nyosplwtfa_while_sequential_nyosplwtfa_while_cond_2593579___redundant_placeholder5a
]sequential_nyosplwtfa_while_sequential_nyosplwtfa_while_cond_2593579___redundant_placeholder6(
$sequential_nyosplwtfa_while_identity
Þ
 sequential/nyosplwtfa/while/LessLess'sequential_nyosplwtfa_while_placeholderFsequential_nyosplwtfa_while_less_sequential_nyosplwtfa_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/nyosplwtfa/while/Less
$sequential/nyosplwtfa/while/IdentityIdentity$sequential/nyosplwtfa/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/nyosplwtfa/while/Identity"U
$sequential_nyosplwtfa_while_identity-sequential/nyosplwtfa/while/Identity:output:0*(
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
while_cond_2597823
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2597823___redundant_placeholder05
1while_while_cond_2597823___redundant_placeholder15
1while_while_cond_2597823___redundant_placeholder25
1while_while_cond_2597823___redundant_placeholder35
1while_while_cond_2597823___redundant_placeholder45
1while_while_cond_2597823___redundant_placeholder55
1while_while_cond_2597823___redundant_placeholder6
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
Ó

,__inference_nyosplwtfa_layer_call_fn_2598944

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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25956402
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2594719

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
³F
ê
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2593874

inputs%
hswofenhiy_2593775:	%
hswofenhiy_2593777:	 !
hswofenhiy_2593779:	 
hswofenhiy_2593781:  
hswofenhiy_2593783:  
hswofenhiy_2593785: 
identity¢"hswofenhiy/StatefulPartitionedCall¢whileD
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
"hswofenhiy/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0hswofenhiy_2593775hswofenhiy_2593777hswofenhiy_2593779hswofenhiy_2593781hswofenhiy_2593783hswofenhiy_2593785*
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_25937742$
"hswofenhiy/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0hswofenhiy_2593775hswofenhiy_2593777hswofenhiy_2593779hswofenhiy_2593781hswofenhiy_2593783hswofenhiy_2593785*
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
while_body_2593794*
condR
while_cond_2593793*Q
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
IdentityIdentitytranspose_1:y:0#^hswofenhiy/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"hswofenhiy/StatefulPartitionedCall"hswofenhiy/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_sequential_layer_call_fn_2595706

bdeyofgzkq
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
bdeyofgzkqunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
G__inference_sequential_layer_call_and_return_conditional_losses_25956712
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
bdeyofgzkq
Ä
À
G__inference_sequential_layer_call_and_return_conditional_losses_2596353

bdeyofgzkq(
gtjikcltwy_2596315: 
gtjikcltwy_2596317:%
dnzlhpjizj_2596321:	%
dnzlhpjizj_2596323:	 !
dnzlhpjizj_2596325:	 
dnzlhpjizj_2596327:  
dnzlhpjizj_2596329:  
dnzlhpjizj_2596331: %
nyosplwtfa_2596334:	 %
nyosplwtfa_2596336:	 !
nyosplwtfa_2596338:	 
nyosplwtfa_2596340:  
nyosplwtfa_2596342:  
nyosplwtfa_2596344: $
chsgvefspq_2596347:  
chsgvefspq_2596349:
identity¢"chsgvefspq/StatefulPartitionedCall¢"dnzlhpjizj/StatefulPartitionedCall¢"gtjikcltwy/StatefulPartitionedCall¢"nyosplwtfa/StatefulPartitionedCall°
"gtjikcltwy/StatefulPartitionedCallStatefulPartitionedCall
bdeyofgzkqgtjikcltwy_2596315gtjikcltwy_2596317*
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
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_25952472$
"gtjikcltwy/StatefulPartitionedCall
ezubtmdnwx/PartitionedCallPartitionedCall+gtjikcltwy/StatefulPartitionedCall:output:0*
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
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_25952662
ezubtmdnwx/PartitionedCall
"dnzlhpjizj/StatefulPartitionedCallStatefulPartitionedCall#ezubtmdnwx/PartitionedCall:output:0dnzlhpjizj_2596321dnzlhpjizj_2596323dnzlhpjizj_2596325dnzlhpjizj_2596327dnzlhpjizj_2596329dnzlhpjizj_2596331*
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25954472$
"dnzlhpjizj/StatefulPartitionedCall¡
"nyosplwtfa/StatefulPartitionedCallStatefulPartitionedCall+dnzlhpjizj/StatefulPartitionedCall:output:0nyosplwtfa_2596334nyosplwtfa_2596336nyosplwtfa_2596338nyosplwtfa_2596340nyosplwtfa_2596342nyosplwtfa_2596344*
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25956402$
"nyosplwtfa/StatefulPartitionedCallÉ
"chsgvefspq/StatefulPartitionedCallStatefulPartitionedCall+nyosplwtfa/StatefulPartitionedCall:output:0chsgvefspq_2596347chsgvefspq_2596349*
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
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_25956642$
"chsgvefspq/StatefulPartitionedCall
IdentityIdentity+chsgvefspq/StatefulPartitionedCall:output:0#^chsgvefspq/StatefulPartitionedCall#^dnzlhpjizj/StatefulPartitionedCall#^gtjikcltwy/StatefulPartitionedCall#^nyosplwtfa/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"chsgvefspq/StatefulPartitionedCall"chsgvefspq/StatefulPartitionedCall2H
"dnzlhpjizj/StatefulPartitionedCall"dnzlhpjizj/StatefulPartitionedCall2H
"gtjikcltwy/StatefulPartitionedCall"gtjikcltwy/StatefulPartitionedCall2H
"nyosplwtfa/StatefulPartitionedCall"nyosplwtfa/StatefulPartitionedCall:[ W
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
bdeyofgzkq
àY

while_body_2597824
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_hswofenhiy_matmul_readvariableop_resource_0:	F
3while_hswofenhiy_matmul_1_readvariableop_resource_0:	 A
2while_hswofenhiy_biasadd_readvariableop_resource_0:	8
*while_hswofenhiy_readvariableop_resource_0: :
,while_hswofenhiy_readvariableop_1_resource_0: :
,while_hswofenhiy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_hswofenhiy_matmul_readvariableop_resource:	D
1while_hswofenhiy_matmul_1_readvariableop_resource:	 ?
0while_hswofenhiy_biasadd_readvariableop_resource:	6
(while_hswofenhiy_readvariableop_resource: 8
*while_hswofenhiy_readvariableop_1_resource: 8
*while_hswofenhiy_readvariableop_2_resource: ¢'while/hswofenhiy/BiasAdd/ReadVariableOp¢&while/hswofenhiy/MatMul/ReadVariableOp¢(while/hswofenhiy/MatMul_1/ReadVariableOp¢while/hswofenhiy/ReadVariableOp¢!while/hswofenhiy/ReadVariableOp_1¢!while/hswofenhiy/ReadVariableOp_2Ã
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
&while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp1while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/hswofenhiy/MatMul/ReadVariableOpÑ
while/hswofenhiy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMulÉ
(while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp3while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/hswofenhiy/MatMul_1/ReadVariableOpº
while/hswofenhiy/MatMul_1MatMulwhile_placeholder_20while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMul_1°
while/hswofenhiy/addAddV2!while/hswofenhiy/MatMul:product:0#while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/addÂ
'while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp2while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/hswofenhiy/BiasAdd/ReadVariableOp½
while/hswofenhiy/BiasAddBiasAddwhile/hswofenhiy/add:z:0/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/BiasAdd
 while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/hswofenhiy/split/split_dim
while/hswofenhiy/splitSplit)while/hswofenhiy/split/split_dim:output:0!while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/hswofenhiy/split©
while/hswofenhiy/ReadVariableOpReadVariableOp*while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/hswofenhiy/ReadVariableOp£
while/hswofenhiy/mulMul'while/hswofenhiy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul¦
while/hswofenhiy/add_1AddV2while/hswofenhiy/split:output:0while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_1
while/hswofenhiy/SigmoidSigmoidwhile/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid¯
!while/hswofenhiy/ReadVariableOp_1ReadVariableOp,while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_1©
while/hswofenhiy/mul_1Mul)while/hswofenhiy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_1¨
while/hswofenhiy/add_2AddV2while/hswofenhiy/split:output:1while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_2
while/hswofenhiy/Sigmoid_1Sigmoidwhile/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_1
while/hswofenhiy/mul_2Mulwhile/hswofenhiy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_2
while/hswofenhiy/TanhTanhwhile/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh¢
while/hswofenhiy/mul_3Mulwhile/hswofenhiy/Sigmoid:y:0while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_3£
while/hswofenhiy/add_3AddV2while/hswofenhiy/mul_2:z:0while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_3¯
!while/hswofenhiy/ReadVariableOp_2ReadVariableOp,while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_2°
while/hswofenhiy/mul_4Mul)while/hswofenhiy/ReadVariableOp_2:value:0while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_4¨
while/hswofenhiy/add_4AddV2while/hswofenhiy/split:output:3while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_4
while/hswofenhiy/Sigmoid_2Sigmoidwhile/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_2
while/hswofenhiy/Tanh_1Tanhwhile/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh_1¦
while/hswofenhiy/mul_5Mulwhile/hswofenhiy/Sigmoid_2:y:0while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/hswofenhiy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/hswofenhiy/mul_5:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/hswofenhiy/add_3:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_hswofenhiy_biasadd_readvariableop_resource2while_hswofenhiy_biasadd_readvariableop_resource_0"h
1while_hswofenhiy_matmul_1_readvariableop_resource3while_hswofenhiy_matmul_1_readvariableop_resource_0"d
/while_hswofenhiy_matmul_readvariableop_resource1while_hswofenhiy_matmul_readvariableop_resource_0"Z
*while_hswofenhiy_readvariableop_1_resource,while_hswofenhiy_readvariableop_1_resource_0"Z
*while_hswofenhiy_readvariableop_2_resource,while_hswofenhiy_readvariableop_2_resource_0"V
(while_hswofenhiy_readvariableop_resource*while_hswofenhiy_readvariableop_resource_0")
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
'while/hswofenhiy/BiasAdd/ReadVariableOp'while/hswofenhiy/BiasAdd/ReadVariableOp2P
&while/hswofenhiy/MatMul/ReadVariableOp&while/hswofenhiy/MatMul/ReadVariableOp2T
(while/hswofenhiy/MatMul_1/ReadVariableOp(while/hswofenhiy/MatMul_1/ReadVariableOp2B
while/hswofenhiy/ReadVariableOpwhile/hswofenhiy/ReadVariableOp2F
!while/hswofenhiy/ReadVariableOp_1!while/hswofenhiy/ReadVariableOp_12F
!while/hswofenhiy/ReadVariableOp_2!while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_sequential_layer_call_and_return_conditional_losses_2596240

inputs(
gtjikcltwy_2596202: 
gtjikcltwy_2596204:%
dnzlhpjizj_2596208:	%
dnzlhpjizj_2596210:	 !
dnzlhpjizj_2596212:	 
dnzlhpjizj_2596214:  
dnzlhpjizj_2596216:  
dnzlhpjizj_2596218: %
nyosplwtfa_2596221:	 %
nyosplwtfa_2596223:	 !
nyosplwtfa_2596225:	 
nyosplwtfa_2596227:  
nyosplwtfa_2596229:  
nyosplwtfa_2596231: $
chsgvefspq_2596234:  
chsgvefspq_2596236:
identity¢"chsgvefspq/StatefulPartitionedCall¢"dnzlhpjizj/StatefulPartitionedCall¢"gtjikcltwy/StatefulPartitionedCall¢"nyosplwtfa/StatefulPartitionedCall¬
"gtjikcltwy/StatefulPartitionedCallStatefulPartitionedCallinputsgtjikcltwy_2596202gtjikcltwy_2596204*
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
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_25952472$
"gtjikcltwy/StatefulPartitionedCall
ezubtmdnwx/PartitionedCallPartitionedCall+gtjikcltwy/StatefulPartitionedCall:output:0*
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
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_25952662
ezubtmdnwx/PartitionedCall
"dnzlhpjizj/StatefulPartitionedCallStatefulPartitionedCall#ezubtmdnwx/PartitionedCall:output:0dnzlhpjizj_2596208dnzlhpjizj_2596210dnzlhpjizj_2596212dnzlhpjizj_2596214dnzlhpjizj_2596216dnzlhpjizj_2596218*
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25961292$
"dnzlhpjizj/StatefulPartitionedCall¡
"nyosplwtfa/StatefulPartitionedCallStatefulPartitionedCall+dnzlhpjizj/StatefulPartitionedCall:output:0nyosplwtfa_2596221nyosplwtfa_2596223nyosplwtfa_2596225nyosplwtfa_2596227nyosplwtfa_2596229nyosplwtfa_2596231*
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25959152$
"nyosplwtfa/StatefulPartitionedCallÉ
"chsgvefspq/StatefulPartitionedCallStatefulPartitionedCall+nyosplwtfa/StatefulPartitionedCall:output:0chsgvefspq_2596234chsgvefspq_2596236*
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
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_25956642$
"chsgvefspq/StatefulPartitionedCall
IdentityIdentity+chsgvefspq/StatefulPartitionedCall:output:0#^chsgvefspq/StatefulPartitionedCall#^dnzlhpjizj/StatefulPartitionedCall#^gtjikcltwy/StatefulPartitionedCall#^nyosplwtfa/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2H
"chsgvefspq/StatefulPartitionedCall"chsgvefspq/StatefulPartitionedCall2H
"dnzlhpjizj/StatefulPartitionedCall"dnzlhpjizj/StatefulPartitionedCall2H
"gtjikcltwy/StatefulPartitionedCall"gtjikcltwy/StatefulPartitionedCall2H
"nyosplwtfa/StatefulPartitionedCall"nyosplwtfa/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


dnzlhpjizj_while_cond_25969632
.dnzlhpjizj_while_dnzlhpjizj_while_loop_counter8
4dnzlhpjizj_while_dnzlhpjizj_while_maximum_iterations 
dnzlhpjizj_while_placeholder"
dnzlhpjizj_while_placeholder_1"
dnzlhpjizj_while_placeholder_2"
dnzlhpjizj_while_placeholder_34
0dnzlhpjizj_while_less_dnzlhpjizj_strided_slice_1K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596963___redundant_placeholder0K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596963___redundant_placeholder1K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596963___redundant_placeholder2K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596963___redundant_placeholder3K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596963___redundant_placeholder4K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596963___redundant_placeholder5K
Gdnzlhpjizj_while_dnzlhpjizj_while_cond_2596963___redundant_placeholder6
dnzlhpjizj_while_identity
§
dnzlhpjizj/while/LessLessdnzlhpjizj_while_placeholder0dnzlhpjizj_while_less_dnzlhpjizj_strided_slice_1*
T0*
_output_shapes
: 2
dnzlhpjizj/while/Less~
dnzlhpjizj/while/IdentityIdentitydnzlhpjizj/while/Less:z:0*
T0
*
_output_shapes
: 2
dnzlhpjizj/while/Identity"?
dnzlhpjizj_while_identity"dnzlhpjizj/while/Identity:output:0*(
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
while_body_2595346
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_hswofenhiy_matmul_readvariableop_resource_0:	F
3while_hswofenhiy_matmul_1_readvariableop_resource_0:	 A
2while_hswofenhiy_biasadd_readvariableop_resource_0:	8
*while_hswofenhiy_readvariableop_resource_0: :
,while_hswofenhiy_readvariableop_1_resource_0: :
,while_hswofenhiy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_hswofenhiy_matmul_readvariableop_resource:	D
1while_hswofenhiy_matmul_1_readvariableop_resource:	 ?
0while_hswofenhiy_biasadd_readvariableop_resource:	6
(while_hswofenhiy_readvariableop_resource: 8
*while_hswofenhiy_readvariableop_1_resource: 8
*while_hswofenhiy_readvariableop_2_resource: ¢'while/hswofenhiy/BiasAdd/ReadVariableOp¢&while/hswofenhiy/MatMul/ReadVariableOp¢(while/hswofenhiy/MatMul_1/ReadVariableOp¢while/hswofenhiy/ReadVariableOp¢!while/hswofenhiy/ReadVariableOp_1¢!while/hswofenhiy/ReadVariableOp_2Ã
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
&while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp1while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/hswofenhiy/MatMul/ReadVariableOpÑ
while/hswofenhiy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMulÉ
(while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp3while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/hswofenhiy/MatMul_1/ReadVariableOpº
while/hswofenhiy/MatMul_1MatMulwhile_placeholder_20while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMul_1°
while/hswofenhiy/addAddV2!while/hswofenhiy/MatMul:product:0#while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/addÂ
'while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp2while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/hswofenhiy/BiasAdd/ReadVariableOp½
while/hswofenhiy/BiasAddBiasAddwhile/hswofenhiy/add:z:0/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/BiasAdd
 while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/hswofenhiy/split/split_dim
while/hswofenhiy/splitSplit)while/hswofenhiy/split/split_dim:output:0!while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/hswofenhiy/split©
while/hswofenhiy/ReadVariableOpReadVariableOp*while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/hswofenhiy/ReadVariableOp£
while/hswofenhiy/mulMul'while/hswofenhiy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul¦
while/hswofenhiy/add_1AddV2while/hswofenhiy/split:output:0while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_1
while/hswofenhiy/SigmoidSigmoidwhile/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid¯
!while/hswofenhiy/ReadVariableOp_1ReadVariableOp,while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_1©
while/hswofenhiy/mul_1Mul)while/hswofenhiy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_1¨
while/hswofenhiy/add_2AddV2while/hswofenhiy/split:output:1while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_2
while/hswofenhiy/Sigmoid_1Sigmoidwhile/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_1
while/hswofenhiy/mul_2Mulwhile/hswofenhiy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_2
while/hswofenhiy/TanhTanhwhile/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh¢
while/hswofenhiy/mul_3Mulwhile/hswofenhiy/Sigmoid:y:0while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_3£
while/hswofenhiy/add_3AddV2while/hswofenhiy/mul_2:z:0while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_3¯
!while/hswofenhiy/ReadVariableOp_2ReadVariableOp,while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_2°
while/hswofenhiy/mul_4Mul)while/hswofenhiy/ReadVariableOp_2:value:0while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_4¨
while/hswofenhiy/add_4AddV2while/hswofenhiy/split:output:3while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_4
while/hswofenhiy/Sigmoid_2Sigmoidwhile/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_2
while/hswofenhiy/Tanh_1Tanhwhile/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh_1¦
while/hswofenhiy/mul_5Mulwhile/hswofenhiy/Sigmoid_2:y:0while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/hswofenhiy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/hswofenhiy/mul_5:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/hswofenhiy/add_3:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_hswofenhiy_biasadd_readvariableop_resource2while_hswofenhiy_biasadd_readvariableop_resource_0"h
1while_hswofenhiy_matmul_1_readvariableop_resource3while_hswofenhiy_matmul_1_readvariableop_resource_0"d
/while_hswofenhiy_matmul_readvariableop_resource1while_hswofenhiy_matmul_readvariableop_resource_0"Z
*while_hswofenhiy_readvariableop_1_resource,while_hswofenhiy_readvariableop_1_resource_0"Z
*while_hswofenhiy_readvariableop_2_resource,while_hswofenhiy_readvariableop_2_resource_0"V
(while_hswofenhiy_readvariableop_resource*while_hswofenhiy_readvariableop_resource_0")
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
'while/hswofenhiy/BiasAdd/ReadVariableOp'while/hswofenhiy/BiasAdd/ReadVariableOp2P
&while/hswofenhiy/MatMul/ReadVariableOp&while/hswofenhiy/MatMul/ReadVariableOp2T
(while/hswofenhiy/MatMul_1/ReadVariableOp(while/hswofenhiy/MatMul_1/ReadVariableOp2B
while/hswofenhiy/ReadVariableOpwhile/hswofenhiy/ReadVariableOp2F
!while/hswofenhiy/ReadVariableOp_1!while/hswofenhiy/ReadVariableOp_12F
!while/hswofenhiy/ReadVariableOp_2!while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597745
inputs_0<
)hswofenhiy_matmul_readvariableop_resource:	>
+hswofenhiy_matmul_1_readvariableop_resource:	 9
*hswofenhiy_biasadd_readvariableop_resource:	0
"hswofenhiy_readvariableop_resource: 2
$hswofenhiy_readvariableop_1_resource: 2
$hswofenhiy_readvariableop_2_resource: 
identity¢!hswofenhiy/BiasAdd/ReadVariableOp¢ hswofenhiy/MatMul/ReadVariableOp¢"hswofenhiy/MatMul_1/ReadVariableOp¢hswofenhiy/ReadVariableOp¢hswofenhiy/ReadVariableOp_1¢hswofenhiy/ReadVariableOp_2¢whileF
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
 hswofenhiy/MatMul/ReadVariableOpReadVariableOp)hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 hswofenhiy/MatMul/ReadVariableOp§
hswofenhiy/MatMulMatMulstrided_slice_2:output:0(hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMulµ
"hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp+hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"hswofenhiy/MatMul_1/ReadVariableOp£
hswofenhiy/MatMul_1MatMulzeros:output:0*hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMul_1
hswofenhiy/addAddV2hswofenhiy/MatMul:product:0hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/add®
!hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp*hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!hswofenhiy/BiasAdd/ReadVariableOp¥
hswofenhiy/BiasAddBiasAddhswofenhiy/add:z:0)hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/BiasAddz
hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
hswofenhiy/split/split_dimë
hswofenhiy/splitSplit#hswofenhiy/split/split_dim:output:0hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
hswofenhiy/split
hswofenhiy/ReadVariableOpReadVariableOp"hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp
hswofenhiy/mulMul!hswofenhiy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul
hswofenhiy/add_1AddV2hswofenhiy/split:output:0hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_1{
hswofenhiy/SigmoidSigmoidhswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid
hswofenhiy/ReadVariableOp_1ReadVariableOp$hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_1
hswofenhiy/mul_1Mul#hswofenhiy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_1
hswofenhiy/add_2AddV2hswofenhiy/split:output:1hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_2
hswofenhiy/Sigmoid_1Sigmoidhswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_1
hswofenhiy/mul_2Mulhswofenhiy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_2w
hswofenhiy/TanhTanhhswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh
hswofenhiy/mul_3Mulhswofenhiy/Sigmoid:y:0hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_3
hswofenhiy/add_3AddV2hswofenhiy/mul_2:z:0hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_3
hswofenhiy/ReadVariableOp_2ReadVariableOp$hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_2
hswofenhiy/mul_4Mul#hswofenhiy/ReadVariableOp_2:value:0hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_4
hswofenhiy/add_4AddV2hswofenhiy/split:output:3hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_4
hswofenhiy/Sigmoid_2Sigmoidhswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_2v
hswofenhiy/Tanh_1Tanhhswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh_1
hswofenhiy/mul_5Mulhswofenhiy/Sigmoid_2:y:0hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)hswofenhiy_matmul_readvariableop_resource+hswofenhiy_matmul_1_readvariableop_resource*hswofenhiy_biasadd_readvariableop_resource"hswofenhiy_readvariableop_resource$hswofenhiy_readvariableop_1_resource$hswofenhiy_readvariableop_2_resource*
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
while_body_2597644*
condR
while_cond_2597643*Q
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
IdentityIdentitytranspose_1:y:0"^hswofenhiy/BiasAdd/ReadVariableOp!^hswofenhiy/MatMul/ReadVariableOp#^hswofenhiy/MatMul_1/ReadVariableOp^hswofenhiy/ReadVariableOp^hswofenhiy/ReadVariableOp_1^hswofenhiy/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!hswofenhiy/BiasAdd/ReadVariableOp!hswofenhiy/BiasAdd/ReadVariableOp2D
 hswofenhiy/MatMul/ReadVariableOp hswofenhiy/MatMul/ReadVariableOp2H
"hswofenhiy/MatMul_1/ReadVariableOp"hswofenhiy/MatMul_1/ReadVariableOp26
hswofenhiy/ReadVariableOphswofenhiy/ReadVariableOp2:
hswofenhiy/ReadVariableOp_1hswofenhiy/ReadVariableOp_12:
hswofenhiy/ReadVariableOp_2hswofenhiy/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ó	
ø
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_2598971

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
ÞÊ

G__inference_sequential_layer_call_and_return_conditional_losses_2597247

inputsL
6gtjikcltwy_conv1d_expanddims_1_readvariableop_resource:K
=gtjikcltwy_squeeze_batch_dims_biasadd_readvariableop_resource:G
4dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource:	I
6dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource:	 D
5dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource:	;
-dnzlhpjizj_hswofenhiy_readvariableop_resource: =
/dnzlhpjizj_hswofenhiy_readvariableop_1_resource: =
/dnzlhpjizj_hswofenhiy_readvariableop_2_resource: G
4nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource:	 I
6nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource:	 D
5nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource:	;
-nyosplwtfa_lwptfvtmlx_readvariableop_resource: =
/nyosplwtfa_lwptfvtmlx_readvariableop_1_resource: =
/nyosplwtfa_lwptfvtmlx_readvariableop_2_resource: ;
)chsgvefspq_matmul_readvariableop_resource: 8
*chsgvefspq_biasadd_readvariableop_resource:
identity¢!chsgvefspq/BiasAdd/ReadVariableOp¢ chsgvefspq/MatMul/ReadVariableOp¢,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp¢+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp¢-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp¢$dnzlhpjizj/hswofenhiy/ReadVariableOp¢&dnzlhpjizj/hswofenhiy/ReadVariableOp_1¢&dnzlhpjizj/hswofenhiy/ReadVariableOp_2¢dnzlhpjizj/while¢-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp¢4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp¢,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp¢+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp¢-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp¢$nyosplwtfa/lwptfvtmlx/ReadVariableOp¢&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1¢&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2¢nyosplwtfa/while
 gtjikcltwy/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2"
 gtjikcltwy/conv1d/ExpandDims/dim»
gtjikcltwy/conv1d/ExpandDims
ExpandDimsinputs)gtjikcltwy/conv1d/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/ExpandDimsÙ
-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp6gtjikcltwy_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02/
-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp
"gtjikcltwy/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2$
"gtjikcltwy/conv1d/ExpandDims_1/dimã
gtjikcltwy/conv1d/ExpandDims_1
ExpandDims5gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp:value:0+gtjikcltwy/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2 
gtjikcltwy/conv1d/ExpandDims_1
gtjikcltwy/conv1d/ShapeShape%gtjikcltwy/conv1d/ExpandDims:output:0*
T0*
_output_shapes
:2
gtjikcltwy/conv1d/Shape
%gtjikcltwy/conv1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%gtjikcltwy/conv1d/strided_slice/stack¥
'gtjikcltwy/conv1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2)
'gtjikcltwy/conv1d/strided_slice/stack_1
'gtjikcltwy/conv1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'gtjikcltwy/conv1d/strided_slice/stack_2Ì
gtjikcltwy/conv1d/strided_sliceStridedSlice gtjikcltwy/conv1d/Shape:output:0.gtjikcltwy/conv1d/strided_slice/stack:output:00gtjikcltwy/conv1d/strided_slice/stack_1:output:00gtjikcltwy/conv1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2!
gtjikcltwy/conv1d/strided_slice
gtjikcltwy/conv1d/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ         2!
gtjikcltwy/conv1d/Reshape/shapeÌ
gtjikcltwy/conv1d/ReshapeReshape%gtjikcltwy/conv1d/ExpandDims:output:0(gtjikcltwy/conv1d/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/Reshapeî
gtjikcltwy/conv1d/Conv2DConv2D"gtjikcltwy/conv1d/Reshape:output:0'gtjikcltwy/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
gtjikcltwy/conv1d/Conv2D
!gtjikcltwy/conv1d/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"         2#
!gtjikcltwy/conv1d/concat/values_1
gtjikcltwy/conv1d/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/concat/axisì
gtjikcltwy/conv1d/concatConcatV2(gtjikcltwy/conv1d/strided_slice:output:0*gtjikcltwy/conv1d/concat/values_1:output:0&gtjikcltwy/conv1d/concat/axis:output:0*
N*
T0*
_output_shapes
:2
gtjikcltwy/conv1d/concatÉ
gtjikcltwy/conv1d/Reshape_1Reshape!gtjikcltwy/conv1d/Conv2D:output:0!gtjikcltwy/conv1d/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/Reshape_1Á
gtjikcltwy/conv1d/SqueezeSqueeze$gtjikcltwy/conv1d/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
gtjikcltwy/conv1d/Squeeze
#gtjikcltwy/squeeze_batch_dims/ShapeShape"gtjikcltwy/conv1d/Squeeze:output:0*
T0*
_output_shapes
:2%
#gtjikcltwy/squeeze_batch_dims/Shape°
1gtjikcltwy/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1gtjikcltwy/squeeze_batch_dims/strided_slice/stack½
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ25
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1´
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2
+gtjikcltwy/squeeze_batch_dims/strided_sliceStridedSlice,gtjikcltwy/squeeze_batch_dims/Shape:output:0:gtjikcltwy/squeeze_batch_dims/strided_slice/stack:output:0<gtjikcltwy/squeeze_batch_dims/strided_slice/stack_1:output:0<gtjikcltwy/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+gtjikcltwy/squeeze_batch_dims/strided_slice¯
+gtjikcltwy/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ      2-
+gtjikcltwy/squeeze_batch_dims/Reshape/shapeé
%gtjikcltwy/squeeze_batch_dims/ReshapeReshape"gtjikcltwy/conv1d/Squeeze:output:04gtjikcltwy/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%gtjikcltwy/squeeze_batch_dims/Reshapeæ
4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp=gtjikcltwy_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOpý
%gtjikcltwy/squeeze_batch_dims/BiasAddBiasAdd.gtjikcltwy/squeeze_batch_dims/Reshape:output:0<gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%gtjikcltwy/squeeze_batch_dims/BiasAdd¯
-gtjikcltwy/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-gtjikcltwy/squeeze_batch_dims/concat/values_1¡
)gtjikcltwy/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)gtjikcltwy/squeeze_batch_dims/concat/axis¨
$gtjikcltwy/squeeze_batch_dims/concatConcatV24gtjikcltwy/squeeze_batch_dims/strided_slice:output:06gtjikcltwy/squeeze_batch_dims/concat/values_1:output:02gtjikcltwy/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$gtjikcltwy/squeeze_batch_dims/concatö
'gtjikcltwy/squeeze_batch_dims/Reshape_1Reshape.gtjikcltwy/squeeze_batch_dims/BiasAdd:output:0-gtjikcltwy/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'gtjikcltwy/squeeze_batch_dims/Reshape_1
ezubtmdnwx/ShapeShape0gtjikcltwy/squeeze_batch_dims/Reshape_1:output:0*
T0*
_output_shapes
:2
ezubtmdnwx/Shape
ezubtmdnwx/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
ezubtmdnwx/strided_slice/stack
 ezubtmdnwx/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 ezubtmdnwx/strided_slice/stack_1
 ezubtmdnwx/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 ezubtmdnwx/strided_slice/stack_2¤
ezubtmdnwx/strided_sliceStridedSliceezubtmdnwx/Shape:output:0'ezubtmdnwx/strided_slice/stack:output:0)ezubtmdnwx/strided_slice/stack_1:output:0)ezubtmdnwx/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
ezubtmdnwx/strided_slicez
ezubtmdnwx/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
ezubtmdnwx/Reshape/shape/1z
ezubtmdnwx/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
ezubtmdnwx/Reshape/shape/2×
ezubtmdnwx/Reshape/shapePack!ezubtmdnwx/strided_slice:output:0#ezubtmdnwx/Reshape/shape/1:output:0#ezubtmdnwx/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
ezubtmdnwx/Reshape/shape¾
ezubtmdnwx/ReshapeReshape0gtjikcltwy/squeeze_batch_dims/Reshape_1:output:0!ezubtmdnwx/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ezubtmdnwx/Reshapeo
dnzlhpjizj/ShapeShapeezubtmdnwx/Reshape:output:0*
T0*
_output_shapes
:2
dnzlhpjizj/Shape
dnzlhpjizj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
dnzlhpjizj/strided_slice/stack
 dnzlhpjizj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 dnzlhpjizj/strided_slice/stack_1
 dnzlhpjizj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 dnzlhpjizj/strided_slice/stack_2¤
dnzlhpjizj/strided_sliceStridedSlicednzlhpjizj/Shape:output:0'dnzlhpjizj/strided_slice/stack:output:0)dnzlhpjizj/strided_slice/stack_1:output:0)dnzlhpjizj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dnzlhpjizj/strided_slicer
dnzlhpjizj/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros/mul/y
dnzlhpjizj/zeros/mulMul!dnzlhpjizj/strided_slice:output:0dnzlhpjizj/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros/mulu
dnzlhpjizj/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
dnzlhpjizj/zeros/Less/y
dnzlhpjizj/zeros/LessLessdnzlhpjizj/zeros/mul:z:0 dnzlhpjizj/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros/Lessx
dnzlhpjizj/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros/packed/1¯
dnzlhpjizj/zeros/packedPack!dnzlhpjizj/strided_slice:output:0"dnzlhpjizj/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
dnzlhpjizj/zeros/packedu
dnzlhpjizj/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dnzlhpjizj/zeros/Const¡
dnzlhpjizj/zerosFill dnzlhpjizj/zeros/packed:output:0dnzlhpjizj/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/zerosv
dnzlhpjizj/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros_1/mul/y
dnzlhpjizj/zeros_1/mulMul!dnzlhpjizj/strided_slice:output:0!dnzlhpjizj/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros_1/muly
dnzlhpjizj/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
dnzlhpjizj/zeros_1/Less/y
dnzlhpjizj/zeros_1/LessLessdnzlhpjizj/zeros_1/mul:z:0"dnzlhpjizj/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
dnzlhpjizj/zeros_1/Less|
dnzlhpjizj/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/zeros_1/packed/1µ
dnzlhpjizj/zeros_1/packedPack!dnzlhpjizj/strided_slice:output:0$dnzlhpjizj/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
dnzlhpjizj/zeros_1/packedy
dnzlhpjizj/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dnzlhpjizj/zeros_1/Const©
dnzlhpjizj/zeros_1Fill"dnzlhpjizj/zeros_1/packed:output:0!dnzlhpjizj/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/zeros_1
dnzlhpjizj/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dnzlhpjizj/transpose/perm°
dnzlhpjizj/transpose	Transposeezubtmdnwx/Reshape:output:0"dnzlhpjizj/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/transposep
dnzlhpjizj/Shape_1Shapednzlhpjizj/transpose:y:0*
T0*
_output_shapes
:2
dnzlhpjizj/Shape_1
 dnzlhpjizj/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dnzlhpjizj/strided_slice_1/stack
"dnzlhpjizj/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_1/stack_1
"dnzlhpjizj/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_1/stack_2°
dnzlhpjizj/strided_slice_1StridedSlicednzlhpjizj/Shape_1:output:0)dnzlhpjizj/strided_slice_1/stack:output:0+dnzlhpjizj/strided_slice_1/stack_1:output:0+dnzlhpjizj/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dnzlhpjizj/strided_slice_1
&dnzlhpjizj/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&dnzlhpjizj/TensorArrayV2/element_shapeÞ
dnzlhpjizj/TensorArrayV2TensorListReserve/dnzlhpjizj/TensorArrayV2/element_shape:output:0#dnzlhpjizj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dnzlhpjizj/TensorArrayV2Õ
@dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2dnzlhpjizj/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordnzlhpjizj/transpose:y:0Idnzlhpjizj/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2dnzlhpjizj/TensorArrayUnstack/TensorListFromTensor
 dnzlhpjizj/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 dnzlhpjizj/strided_slice_2/stack
"dnzlhpjizj/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_2/stack_1
"dnzlhpjizj/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_2/stack_2¾
dnzlhpjizj/strided_slice_2StridedSlicednzlhpjizj/transpose:y:0)dnzlhpjizj/strided_slice_2/stack:output:0+dnzlhpjizj/strided_slice_2/stack_1:output:0+dnzlhpjizj/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
dnzlhpjizj/strided_slice_2Ð
+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOpReadVariableOp4dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02-
+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOpÓ
dnzlhpjizj/hswofenhiy/MatMulMatMul#dnzlhpjizj/strided_slice_2:output:03dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/hswofenhiy/MatMulÖ
-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp6dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOpÏ
dnzlhpjizj/hswofenhiy/MatMul_1MatMuldnzlhpjizj/zeros:output:05dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dnzlhpjizj/hswofenhiy/MatMul_1Ä
dnzlhpjizj/hswofenhiy/addAddV2&dnzlhpjizj/hswofenhiy/MatMul:product:0(dnzlhpjizj/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/hswofenhiy/addÏ
,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp5dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOpÑ
dnzlhpjizj/hswofenhiy/BiasAddBiasAdddnzlhpjizj/hswofenhiy/add:z:04dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dnzlhpjizj/hswofenhiy/BiasAdd
%dnzlhpjizj/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%dnzlhpjizj/hswofenhiy/split/split_dim
dnzlhpjizj/hswofenhiy/splitSplit.dnzlhpjizj/hswofenhiy/split/split_dim:output:0&dnzlhpjizj/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
dnzlhpjizj/hswofenhiy/split¶
$dnzlhpjizj/hswofenhiy/ReadVariableOpReadVariableOp-dnzlhpjizj_hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02&
$dnzlhpjizj/hswofenhiy/ReadVariableOpº
dnzlhpjizj/hswofenhiy/mulMul,dnzlhpjizj/hswofenhiy/ReadVariableOp:value:0dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mulº
dnzlhpjizj/hswofenhiy/add_1AddV2$dnzlhpjizj/hswofenhiy/split:output:0dnzlhpjizj/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_1
dnzlhpjizj/hswofenhiy/SigmoidSigmoiddnzlhpjizj/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/Sigmoid¼
&dnzlhpjizj/hswofenhiy/ReadVariableOp_1ReadVariableOp/dnzlhpjizj_hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&dnzlhpjizj/hswofenhiy/ReadVariableOp_1À
dnzlhpjizj/hswofenhiy/mul_1Mul.dnzlhpjizj/hswofenhiy/ReadVariableOp_1:value:0dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_1¼
dnzlhpjizj/hswofenhiy/add_2AddV2$dnzlhpjizj/hswofenhiy/split:output:1dnzlhpjizj/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_2 
dnzlhpjizj/hswofenhiy/Sigmoid_1Sigmoiddnzlhpjizj/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dnzlhpjizj/hswofenhiy/Sigmoid_1µ
dnzlhpjizj/hswofenhiy/mul_2Mul#dnzlhpjizj/hswofenhiy/Sigmoid_1:y:0dnzlhpjizj/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_2
dnzlhpjizj/hswofenhiy/TanhTanh$dnzlhpjizj/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/Tanh¶
dnzlhpjizj/hswofenhiy/mul_3Mul!dnzlhpjizj/hswofenhiy/Sigmoid:y:0dnzlhpjizj/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_3·
dnzlhpjizj/hswofenhiy/add_3AddV2dnzlhpjizj/hswofenhiy/mul_2:z:0dnzlhpjizj/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_3¼
&dnzlhpjizj/hswofenhiy/ReadVariableOp_2ReadVariableOp/dnzlhpjizj_hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&dnzlhpjizj/hswofenhiy/ReadVariableOp_2Ä
dnzlhpjizj/hswofenhiy/mul_4Mul.dnzlhpjizj/hswofenhiy/ReadVariableOp_2:value:0dnzlhpjizj/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_4¼
dnzlhpjizj/hswofenhiy/add_4AddV2$dnzlhpjizj/hswofenhiy/split:output:3dnzlhpjizj/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/add_4 
dnzlhpjizj/hswofenhiy/Sigmoid_2Sigmoiddnzlhpjizj/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
dnzlhpjizj/hswofenhiy/Sigmoid_2
dnzlhpjizj/hswofenhiy/Tanh_1Tanhdnzlhpjizj/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/Tanh_1º
dnzlhpjizj/hswofenhiy/mul_5Mul#dnzlhpjizj/hswofenhiy/Sigmoid_2:y:0 dnzlhpjizj/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/hswofenhiy/mul_5¥
(dnzlhpjizj/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(dnzlhpjizj/TensorArrayV2_1/element_shapeä
dnzlhpjizj/TensorArrayV2_1TensorListReserve1dnzlhpjizj/TensorArrayV2_1/element_shape:output:0#dnzlhpjizj/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
dnzlhpjizj/TensorArrayV2_1d
dnzlhpjizj/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/time
#dnzlhpjizj/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#dnzlhpjizj/while/maximum_iterations
dnzlhpjizj/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
dnzlhpjizj/while/loop_counter²
dnzlhpjizj/whileWhile&dnzlhpjizj/while/loop_counter:output:0,dnzlhpjizj/while/maximum_iterations:output:0dnzlhpjizj/time:output:0#dnzlhpjizj/TensorArrayV2_1:handle:0dnzlhpjizj/zeros:output:0dnzlhpjizj/zeros_1:output:0#dnzlhpjizj/strided_slice_1:output:0Bdnzlhpjizj/TensorArrayUnstack/TensorListFromTensor:output_handle:04dnzlhpjizj_hswofenhiy_matmul_readvariableop_resource6dnzlhpjizj_hswofenhiy_matmul_1_readvariableop_resource5dnzlhpjizj_hswofenhiy_biasadd_readvariableop_resource-dnzlhpjizj_hswofenhiy_readvariableop_resource/dnzlhpjizj_hswofenhiy_readvariableop_1_resource/dnzlhpjizj_hswofenhiy_readvariableop_2_resource*
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
dnzlhpjizj_while_body_2596964*)
cond!R
dnzlhpjizj_while_cond_2596963*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
dnzlhpjizj/whileË
;dnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;dnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shape
-dnzlhpjizj/TensorArrayV2Stack/TensorListStackTensorListStackdnzlhpjizj/while:output:3Ddnzlhpjizj/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-dnzlhpjizj/TensorArrayV2Stack/TensorListStack
 dnzlhpjizj/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 dnzlhpjizj/strided_slice_3/stack
"dnzlhpjizj/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"dnzlhpjizj/strided_slice_3/stack_1
"dnzlhpjizj/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"dnzlhpjizj/strided_slice_3/stack_2Ü
dnzlhpjizj/strided_slice_3StridedSlice6dnzlhpjizj/TensorArrayV2Stack/TensorListStack:tensor:0)dnzlhpjizj/strided_slice_3/stack:output:0+dnzlhpjizj/strided_slice_3/stack_1:output:0+dnzlhpjizj/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
dnzlhpjizj/strided_slice_3
dnzlhpjizj/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dnzlhpjizj/transpose_1/permÑ
dnzlhpjizj/transpose_1	Transpose6dnzlhpjizj/TensorArrayV2Stack/TensorListStack:tensor:0$dnzlhpjizj/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dnzlhpjizj/transpose_1n
nyosplwtfa/ShapeShapednzlhpjizj/transpose_1:y:0*
T0*
_output_shapes
:2
nyosplwtfa/Shape
nyosplwtfa/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
nyosplwtfa/strided_slice/stack
 nyosplwtfa/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 nyosplwtfa/strided_slice/stack_1
 nyosplwtfa/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 nyosplwtfa/strided_slice/stack_2¤
nyosplwtfa/strided_sliceStridedSlicenyosplwtfa/Shape:output:0'nyosplwtfa/strided_slice/stack:output:0)nyosplwtfa/strided_slice/stack_1:output:0)nyosplwtfa/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
nyosplwtfa/strided_slicer
nyosplwtfa/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros/mul/y
nyosplwtfa/zeros/mulMul!nyosplwtfa/strided_slice:output:0nyosplwtfa/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros/mulu
nyosplwtfa/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
nyosplwtfa/zeros/Less/y
nyosplwtfa/zeros/LessLessnyosplwtfa/zeros/mul:z:0 nyosplwtfa/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros/Lessx
nyosplwtfa/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros/packed/1¯
nyosplwtfa/zeros/packedPack!nyosplwtfa/strided_slice:output:0"nyosplwtfa/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
nyosplwtfa/zeros/packedu
nyosplwtfa/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
nyosplwtfa/zeros/Const¡
nyosplwtfa/zerosFill nyosplwtfa/zeros/packed:output:0nyosplwtfa/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/zerosv
nyosplwtfa/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros_1/mul/y
nyosplwtfa/zeros_1/mulMul!nyosplwtfa/strided_slice:output:0!nyosplwtfa/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros_1/muly
nyosplwtfa/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
nyosplwtfa/zeros_1/Less/y
nyosplwtfa/zeros_1/LessLessnyosplwtfa/zeros_1/mul:z:0"nyosplwtfa/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/zeros_1/Less|
nyosplwtfa/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/zeros_1/packed/1µ
nyosplwtfa/zeros_1/packedPack!nyosplwtfa/strided_slice:output:0$nyosplwtfa/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
nyosplwtfa/zeros_1/packedy
nyosplwtfa/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
nyosplwtfa/zeros_1/Const©
nyosplwtfa/zeros_1Fill"nyosplwtfa/zeros_1/packed:output:0!nyosplwtfa/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/zeros_1
nyosplwtfa/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
nyosplwtfa/transpose/perm¯
nyosplwtfa/transpose	Transposednzlhpjizj/transpose_1:y:0"nyosplwtfa/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/transposep
nyosplwtfa/Shape_1Shapenyosplwtfa/transpose:y:0*
T0*
_output_shapes
:2
nyosplwtfa/Shape_1
 nyosplwtfa/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 nyosplwtfa/strided_slice_1/stack
"nyosplwtfa/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_1/stack_1
"nyosplwtfa/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_1/stack_2°
nyosplwtfa/strided_slice_1StridedSlicenyosplwtfa/Shape_1:output:0)nyosplwtfa/strided_slice_1/stack:output:0+nyosplwtfa/strided_slice_1/stack_1:output:0+nyosplwtfa/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
nyosplwtfa/strided_slice_1
&nyosplwtfa/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&nyosplwtfa/TensorArrayV2/element_shapeÞ
nyosplwtfa/TensorArrayV2TensorListReserve/nyosplwtfa/TensorArrayV2/element_shape:output:0#nyosplwtfa/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
nyosplwtfa/TensorArrayV2Õ
@nyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2B
@nyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2nyosplwtfa/TensorArrayUnstack/TensorListFromTensorTensorListFromTensornyosplwtfa/transpose:y:0Inyosplwtfa/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2nyosplwtfa/TensorArrayUnstack/TensorListFromTensor
 nyosplwtfa/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 nyosplwtfa/strided_slice_2/stack
"nyosplwtfa/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_2/stack_1
"nyosplwtfa/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_2/stack_2¾
nyosplwtfa/strided_slice_2StridedSlicenyosplwtfa/transpose:y:0)nyosplwtfa/strided_slice_2/stack:output:0+nyosplwtfa/strided_slice_2/stack_1:output:0+nyosplwtfa/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
nyosplwtfa/strided_slice_2Ð
+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp4nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02-
+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOpÓ
nyosplwtfa/lwptfvtmlx/MatMulMatMul#nyosplwtfa/strided_slice_2:output:03nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nyosplwtfa/lwptfvtmlx/MatMulÖ
-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp6nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02/
-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOpÏ
nyosplwtfa/lwptfvtmlx/MatMul_1MatMulnyosplwtfa/zeros:output:05nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
nyosplwtfa/lwptfvtmlx/MatMul_1Ä
nyosplwtfa/lwptfvtmlx/addAddV2&nyosplwtfa/lwptfvtmlx/MatMul:product:0(nyosplwtfa/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nyosplwtfa/lwptfvtmlx/addÏ
,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp5nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOpÑ
nyosplwtfa/lwptfvtmlx/BiasAddBiasAddnyosplwtfa/lwptfvtmlx/add:z:04nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
nyosplwtfa/lwptfvtmlx/BiasAdd
%nyosplwtfa/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%nyosplwtfa/lwptfvtmlx/split/split_dim
nyosplwtfa/lwptfvtmlx/splitSplit.nyosplwtfa/lwptfvtmlx/split/split_dim:output:0&nyosplwtfa/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
nyosplwtfa/lwptfvtmlx/split¶
$nyosplwtfa/lwptfvtmlx/ReadVariableOpReadVariableOp-nyosplwtfa_lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02&
$nyosplwtfa/lwptfvtmlx/ReadVariableOpº
nyosplwtfa/lwptfvtmlx/mulMul,nyosplwtfa/lwptfvtmlx/ReadVariableOp:value:0nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mulº
nyosplwtfa/lwptfvtmlx/add_1AddV2$nyosplwtfa/lwptfvtmlx/split:output:0nyosplwtfa/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_1
nyosplwtfa/lwptfvtmlx/SigmoidSigmoidnyosplwtfa/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/Sigmoid¼
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1ReadVariableOp/nyosplwtfa_lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1À
nyosplwtfa/lwptfvtmlx/mul_1Mul.nyosplwtfa/lwptfvtmlx/ReadVariableOp_1:value:0nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_1¼
nyosplwtfa/lwptfvtmlx/add_2AddV2$nyosplwtfa/lwptfvtmlx/split:output:1nyosplwtfa/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_2 
nyosplwtfa/lwptfvtmlx/Sigmoid_1Sigmoidnyosplwtfa/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
nyosplwtfa/lwptfvtmlx/Sigmoid_1µ
nyosplwtfa/lwptfvtmlx/mul_2Mul#nyosplwtfa/lwptfvtmlx/Sigmoid_1:y:0nyosplwtfa/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_2
nyosplwtfa/lwptfvtmlx/TanhTanh$nyosplwtfa/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/Tanh¶
nyosplwtfa/lwptfvtmlx/mul_3Mul!nyosplwtfa/lwptfvtmlx/Sigmoid:y:0nyosplwtfa/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_3·
nyosplwtfa/lwptfvtmlx/add_3AddV2nyosplwtfa/lwptfvtmlx/mul_2:z:0nyosplwtfa/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_3¼
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2ReadVariableOp/nyosplwtfa_lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2Ä
nyosplwtfa/lwptfvtmlx/mul_4Mul.nyosplwtfa/lwptfvtmlx/ReadVariableOp_2:value:0nyosplwtfa/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_4¼
nyosplwtfa/lwptfvtmlx/add_4AddV2$nyosplwtfa/lwptfvtmlx/split:output:3nyosplwtfa/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/add_4 
nyosplwtfa/lwptfvtmlx/Sigmoid_2Sigmoidnyosplwtfa/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
nyosplwtfa/lwptfvtmlx/Sigmoid_2
nyosplwtfa/lwptfvtmlx/Tanh_1Tanhnyosplwtfa/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/Tanh_1º
nyosplwtfa/lwptfvtmlx/mul_5Mul#nyosplwtfa/lwptfvtmlx/Sigmoid_2:y:0 nyosplwtfa/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/lwptfvtmlx/mul_5¥
(nyosplwtfa/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2*
(nyosplwtfa/TensorArrayV2_1/element_shapeä
nyosplwtfa/TensorArrayV2_1TensorListReserve1nyosplwtfa/TensorArrayV2_1/element_shape:output:0#nyosplwtfa/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
nyosplwtfa/TensorArrayV2_1d
nyosplwtfa/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/time
#nyosplwtfa/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#nyosplwtfa/while/maximum_iterations
nyosplwtfa/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
nyosplwtfa/while/loop_counter²
nyosplwtfa/whileWhile&nyosplwtfa/while/loop_counter:output:0,nyosplwtfa/while/maximum_iterations:output:0nyosplwtfa/time:output:0#nyosplwtfa/TensorArrayV2_1:handle:0nyosplwtfa/zeros:output:0nyosplwtfa/zeros_1:output:0#nyosplwtfa/strided_slice_1:output:0Bnyosplwtfa/TensorArrayUnstack/TensorListFromTensor:output_handle:04nyosplwtfa_lwptfvtmlx_matmul_readvariableop_resource6nyosplwtfa_lwptfvtmlx_matmul_1_readvariableop_resource5nyosplwtfa_lwptfvtmlx_biasadd_readvariableop_resource-nyosplwtfa_lwptfvtmlx_readvariableop_resource/nyosplwtfa_lwptfvtmlx_readvariableop_1_resource/nyosplwtfa_lwptfvtmlx_readvariableop_2_resource*
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
nyosplwtfa_while_body_2597140*)
cond!R
nyosplwtfa_while_cond_2597139*Q
output_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : *
parallel_iterations 2
nyosplwtfa/whileË
;nyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;nyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shape
-nyosplwtfa/TensorArrayV2Stack/TensorListStackTensorListStacknyosplwtfa/while:output:3Dnyosplwtfa/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02/
-nyosplwtfa/TensorArrayV2Stack/TensorListStack
 nyosplwtfa/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 nyosplwtfa/strided_slice_3/stack
"nyosplwtfa/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"nyosplwtfa/strided_slice_3/stack_1
"nyosplwtfa/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"nyosplwtfa/strided_slice_3/stack_2Ü
nyosplwtfa/strided_slice_3StridedSlice6nyosplwtfa/TensorArrayV2Stack/TensorListStack:tensor:0)nyosplwtfa/strided_slice_3/stack:output:0+nyosplwtfa/strided_slice_3/stack_1:output:0+nyosplwtfa/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
nyosplwtfa/strided_slice_3
nyosplwtfa/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
nyosplwtfa/transpose_1/permÑ
nyosplwtfa/transpose_1	Transpose6nyosplwtfa/TensorArrayV2Stack/TensorListStack:tensor:0$nyosplwtfa/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/transpose_1®
 chsgvefspq/MatMul/ReadVariableOpReadVariableOp)chsgvefspq_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 chsgvefspq/MatMul/ReadVariableOp±
chsgvefspq/MatMulMatMul#nyosplwtfa/strided_slice_3:output:0(chsgvefspq/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
chsgvefspq/MatMul­
!chsgvefspq/BiasAdd/ReadVariableOpReadVariableOp*chsgvefspq_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!chsgvefspq/BiasAdd/ReadVariableOp­
chsgvefspq/BiasAddBiasAddchsgvefspq/MatMul:product:0)chsgvefspq/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
chsgvefspq/BiasAddÏ
IdentityIdentitychsgvefspq/BiasAdd:output:0"^chsgvefspq/BiasAdd/ReadVariableOp!^chsgvefspq/MatMul/ReadVariableOp-^dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp,^dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp.^dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp%^dnzlhpjizj/hswofenhiy/ReadVariableOp'^dnzlhpjizj/hswofenhiy/ReadVariableOp_1'^dnzlhpjizj/hswofenhiy/ReadVariableOp_2^dnzlhpjizj/while.^gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp5^gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp-^nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp,^nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp.^nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp%^nyosplwtfa/lwptfvtmlx/ReadVariableOp'^nyosplwtfa/lwptfvtmlx/ReadVariableOp_1'^nyosplwtfa/lwptfvtmlx/ReadVariableOp_2^nyosplwtfa/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2F
!chsgvefspq/BiasAdd/ReadVariableOp!chsgvefspq/BiasAdd/ReadVariableOp2D
 chsgvefspq/MatMul/ReadVariableOp chsgvefspq/MatMul/ReadVariableOp2\
,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp,dnzlhpjizj/hswofenhiy/BiasAdd/ReadVariableOp2Z
+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp+dnzlhpjizj/hswofenhiy/MatMul/ReadVariableOp2^
-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp-dnzlhpjizj/hswofenhiy/MatMul_1/ReadVariableOp2L
$dnzlhpjizj/hswofenhiy/ReadVariableOp$dnzlhpjizj/hswofenhiy/ReadVariableOp2P
&dnzlhpjizj/hswofenhiy/ReadVariableOp_1&dnzlhpjizj/hswofenhiy/ReadVariableOp_12P
&dnzlhpjizj/hswofenhiy/ReadVariableOp_2&dnzlhpjizj/hswofenhiy/ReadVariableOp_22$
dnzlhpjizj/whilednzlhpjizj/while2^
-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp-gtjikcltwy/conv1d/ExpandDims_1/ReadVariableOp2l
4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp4gtjikcltwy/squeeze_batch_dims/BiasAdd/ReadVariableOp2\
,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp,nyosplwtfa/lwptfvtmlx/BiasAdd/ReadVariableOp2Z
+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp+nyosplwtfa/lwptfvtmlx/MatMul/ReadVariableOp2^
-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp-nyosplwtfa/lwptfvtmlx/MatMul_1/ReadVariableOp2L
$nyosplwtfa/lwptfvtmlx/ReadVariableOp$nyosplwtfa/lwptfvtmlx/ReadVariableOp2P
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_1&nyosplwtfa/lwptfvtmlx/ReadVariableOp_12P
&nyosplwtfa/lwptfvtmlx/ReadVariableOp_2&nyosplwtfa/lwptfvtmlx/ReadVariableOp_22$
nyosplwtfa/whilenyosplwtfa/while:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
,__inference_hswofenhiy_layer_call_fn_2599114

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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_25939612
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
while_cond_2597463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2597463___redundant_placeholder05
1while_while_cond_2597463___redundant_placeholder15
1while_while_cond_2597463___redundant_placeholder25
1while_while_cond_2597463___redundant_placeholder35
1while_while_cond_2597463___redundant_placeholder45
1while_while_cond_2597463___redundant_placeholder55
1while_while_cond_2597463___redundant_placeholder6
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2594895

inputs%
lwptfvtmlx_2594796:	 %
lwptfvtmlx_2594798:	 !
lwptfvtmlx_2594800:	 
lwptfvtmlx_2594802:  
lwptfvtmlx_2594804:  
lwptfvtmlx_2594806: 
identity¢"lwptfvtmlx/StatefulPartitionedCall¢whileD
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
"lwptfvtmlx/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lwptfvtmlx_2594796lwptfvtmlx_2594798lwptfvtmlx_2594800lwptfvtmlx_2594802lwptfvtmlx_2594804lwptfvtmlx_2594806*
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_25947192$
"lwptfvtmlx/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lwptfvtmlx_2594796lwptfvtmlx_2594798lwptfvtmlx_2594800lwptfvtmlx_2594802lwptfvtmlx_2594804lwptfvtmlx_2594806*
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
while_body_2594815*
condR
while_cond_2594814*Q
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
IdentityIdentitystrided_slice_3:output:0#^lwptfvtmlx/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : : : 2H
"lwptfvtmlx/StatefulPartitionedCall"lwptfvtmlx/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
p
Ê
nyosplwtfa_while_body_25967362
.nyosplwtfa_while_nyosplwtfa_while_loop_counter8
4nyosplwtfa_while_nyosplwtfa_while_maximum_iterations 
nyosplwtfa_while_placeholder"
nyosplwtfa_while_placeholder_1"
nyosplwtfa_while_placeholder_2"
nyosplwtfa_while_placeholder_31
-nyosplwtfa_while_nyosplwtfa_strided_slice_1_0m
inyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0O
<nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0:	 Q
>nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 L
=nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0:	C
5nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0: E
7nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0: E
7nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0: 
nyosplwtfa_while_identity
nyosplwtfa_while_identity_1
nyosplwtfa_while_identity_2
nyosplwtfa_while_identity_3
nyosplwtfa_while_identity_4
nyosplwtfa_while_identity_5/
+nyosplwtfa_while_nyosplwtfa_strided_slice_1k
gnyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensorM
:nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource:	 O
<nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource:	 J
;nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource:	A
3nyosplwtfa_while_lwptfvtmlx_readvariableop_resource: C
5nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource: C
5nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource: ¢2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp¢1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp¢3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp¢*nyosplwtfa/while/lwptfvtmlx/ReadVariableOp¢,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1¢,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2Ù
Bnyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2D
Bnyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shape
4nyosplwtfa/while/TensorArrayV2Read/TensorListGetItemTensorListGetIteminyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0nyosplwtfa_while_placeholderKnyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype026
4nyosplwtfa/while/TensorArrayV2Read/TensorListGetItemä
1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp<nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype023
1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOpý
"nyosplwtfa/while/lwptfvtmlx/MatMulMatMul;nyosplwtfa/while/TensorArrayV2Read/TensorListGetItem:item:09nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"nyosplwtfa/while/lwptfvtmlx/MatMulê
3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp>nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype025
3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOpæ
$nyosplwtfa/while/lwptfvtmlx/MatMul_1MatMulnyosplwtfa_while_placeholder_2;nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$nyosplwtfa/while/lwptfvtmlx/MatMul_1Ü
nyosplwtfa/while/lwptfvtmlx/addAddV2,nyosplwtfa/while/lwptfvtmlx/MatMul:product:0.nyosplwtfa/while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
nyosplwtfa/while/lwptfvtmlx/addã
2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp=nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype024
2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOpé
#nyosplwtfa/while/lwptfvtmlx/BiasAddBiasAdd#nyosplwtfa/while/lwptfvtmlx/add:z:0:nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#nyosplwtfa/while/lwptfvtmlx/BiasAdd
+nyosplwtfa/while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+nyosplwtfa/while/lwptfvtmlx/split/split_dim¯
!nyosplwtfa/while/lwptfvtmlx/splitSplit4nyosplwtfa/while/lwptfvtmlx/split/split_dim:output:0,nyosplwtfa/while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2#
!nyosplwtfa/while/lwptfvtmlx/splitÊ
*nyosplwtfa/while/lwptfvtmlx/ReadVariableOpReadVariableOp5nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02,
*nyosplwtfa/while/lwptfvtmlx/ReadVariableOpÏ
nyosplwtfa/while/lwptfvtmlx/mulMul2nyosplwtfa/while/lwptfvtmlx/ReadVariableOp:value:0nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
nyosplwtfa/while/lwptfvtmlx/mulÒ
!nyosplwtfa/while/lwptfvtmlx/add_1AddV2*nyosplwtfa/while/lwptfvtmlx/split:output:0#nyosplwtfa/while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_1®
#nyosplwtfa/while/lwptfvtmlx/SigmoidSigmoid%nyosplwtfa/while/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#nyosplwtfa/while/lwptfvtmlx/SigmoidÐ
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp7nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02.
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1Õ
!nyosplwtfa/while/lwptfvtmlx/mul_1Mul4nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1:value:0nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_1Ô
!nyosplwtfa/while/lwptfvtmlx/add_2AddV2*nyosplwtfa/while/lwptfvtmlx/split:output:1%nyosplwtfa/while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_2²
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_1Sigmoid%nyosplwtfa/while/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_1Ê
!nyosplwtfa/while/lwptfvtmlx/mul_2Mul)nyosplwtfa/while/lwptfvtmlx/Sigmoid_1:y:0nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_2ª
 nyosplwtfa/while/lwptfvtmlx/TanhTanh*nyosplwtfa/while/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2"
 nyosplwtfa/while/lwptfvtmlx/TanhÎ
!nyosplwtfa/while/lwptfvtmlx/mul_3Mul'nyosplwtfa/while/lwptfvtmlx/Sigmoid:y:0$nyosplwtfa/while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_3Ï
!nyosplwtfa/while/lwptfvtmlx/add_3AddV2%nyosplwtfa/while/lwptfvtmlx/mul_2:z:0%nyosplwtfa/while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_3Ð
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp7nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02.
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2Ü
!nyosplwtfa/while/lwptfvtmlx/mul_4Mul4nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2:value:0%nyosplwtfa/while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_4Ô
!nyosplwtfa/while/lwptfvtmlx/add_4AddV2*nyosplwtfa/while/lwptfvtmlx/split:output:3%nyosplwtfa/while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/add_4²
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_2Sigmoid%nyosplwtfa/while/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%nyosplwtfa/while/lwptfvtmlx/Sigmoid_2©
"nyosplwtfa/while/lwptfvtmlx/Tanh_1Tanh%nyosplwtfa/while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"nyosplwtfa/while/lwptfvtmlx/Tanh_1Ò
!nyosplwtfa/while/lwptfvtmlx/mul_5Mul)nyosplwtfa/while/lwptfvtmlx/Sigmoid_2:y:0&nyosplwtfa/while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!nyosplwtfa/while/lwptfvtmlx/mul_5
5nyosplwtfa/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemnyosplwtfa_while_placeholder_1nyosplwtfa_while_placeholder%nyosplwtfa/while/lwptfvtmlx/mul_5:z:0*
_output_shapes
: *
element_dtype027
5nyosplwtfa/while/TensorArrayV2Write/TensorListSetItemr
nyosplwtfa/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
nyosplwtfa/while/add/y
nyosplwtfa/while/addAddV2nyosplwtfa_while_placeholdernyosplwtfa/while/add/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/while/addv
nyosplwtfa/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
nyosplwtfa/while/add_1/y­
nyosplwtfa/while/add_1AddV2.nyosplwtfa_while_nyosplwtfa_while_loop_counter!nyosplwtfa/while/add_1/y:output:0*
T0*
_output_shapes
: 2
nyosplwtfa/while/add_1©
nyosplwtfa/while/IdentityIdentitynyosplwtfa/while/add_1:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/IdentityÇ
nyosplwtfa/while/Identity_1Identity4nyosplwtfa_while_nyosplwtfa_while_maximum_iterations3^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/Identity_1«
nyosplwtfa/while/Identity_2Identitynyosplwtfa/while/add:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/Identity_2Ø
nyosplwtfa/while/Identity_3IdentityEnyosplwtfa/while/TensorArrayV2Write/TensorListSetItem:output_handle:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
nyosplwtfa/while/Identity_3É
nyosplwtfa/while/Identity_4Identity%nyosplwtfa/while/lwptfvtmlx/mul_5:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/while/Identity_4É
nyosplwtfa/while/Identity_5Identity%nyosplwtfa/while/lwptfvtmlx/add_3:z:03^nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2^nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp4^nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp+^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1-^nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
nyosplwtfa/while/Identity_5"?
nyosplwtfa_while_identity"nyosplwtfa/while/Identity:output:0"C
nyosplwtfa_while_identity_1$nyosplwtfa/while/Identity_1:output:0"C
nyosplwtfa_while_identity_2$nyosplwtfa/while/Identity_2:output:0"C
nyosplwtfa_while_identity_3$nyosplwtfa/while/Identity_3:output:0"C
nyosplwtfa_while_identity_4$nyosplwtfa/while/Identity_4:output:0"C
nyosplwtfa_while_identity_5$nyosplwtfa/while/Identity_5:output:0"|
;nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource=nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0"~
<nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource>nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0"z
:nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource<nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0"p
5nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource7nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0"p
5nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource7nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0"l
3nyosplwtfa_while_lwptfvtmlx_readvariableop_resource5nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0"\
+nyosplwtfa_while_nyosplwtfa_strided_slice_1-nyosplwtfa_while_nyosplwtfa_strided_slice_1_0"Ô
gnyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensorinyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2h
2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2f
1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp1nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp2j
3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp3nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp2X
*nyosplwtfa/while/lwptfvtmlx/ReadVariableOp*nyosplwtfa/while/lwptfvtmlx/ReadVariableOp2\
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_12\
,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2,nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2595915

inputs<
)lwptfvtmlx_matmul_readvariableop_resource:	 >
+lwptfvtmlx_matmul_1_readvariableop_resource:	 9
*lwptfvtmlx_biasadd_readvariableop_resource:	0
"lwptfvtmlx_readvariableop_resource: 2
$lwptfvtmlx_readvariableop_1_resource: 2
$lwptfvtmlx_readvariableop_2_resource: 
identity¢!lwptfvtmlx/BiasAdd/ReadVariableOp¢ lwptfvtmlx/MatMul/ReadVariableOp¢"lwptfvtmlx/MatMul_1/ReadVariableOp¢lwptfvtmlx/ReadVariableOp¢lwptfvtmlx/ReadVariableOp_1¢lwptfvtmlx/ReadVariableOp_2¢whileD
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
 lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp)lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 lwptfvtmlx/MatMul/ReadVariableOp§
lwptfvtmlx/MatMulMatMulstrided_slice_2:output:0(lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMulµ
"lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp+lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lwptfvtmlx/MatMul_1/ReadVariableOp£
lwptfvtmlx/MatMul_1MatMulzeros:output:0*lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMul_1
lwptfvtmlx/addAddV2lwptfvtmlx/MatMul:product:0lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/add®
!lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp*lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!lwptfvtmlx/BiasAdd/ReadVariableOp¥
lwptfvtmlx/BiasAddBiasAddlwptfvtmlx/add:z:0)lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/BiasAddz
lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lwptfvtmlx/split/split_dimë
lwptfvtmlx/splitSplit#lwptfvtmlx/split/split_dim:output:0lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lwptfvtmlx/split
lwptfvtmlx/ReadVariableOpReadVariableOp"lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp
lwptfvtmlx/mulMul!lwptfvtmlx/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul
lwptfvtmlx/add_1AddV2lwptfvtmlx/split:output:0lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_1{
lwptfvtmlx/SigmoidSigmoidlwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid
lwptfvtmlx/ReadVariableOp_1ReadVariableOp$lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_1
lwptfvtmlx/mul_1Mul#lwptfvtmlx/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_1
lwptfvtmlx/add_2AddV2lwptfvtmlx/split:output:1lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_2
lwptfvtmlx/Sigmoid_1Sigmoidlwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_1
lwptfvtmlx/mul_2Mullwptfvtmlx/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_2w
lwptfvtmlx/TanhTanhlwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh
lwptfvtmlx/mul_3Mullwptfvtmlx/Sigmoid:y:0lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_3
lwptfvtmlx/add_3AddV2lwptfvtmlx/mul_2:z:0lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_3
lwptfvtmlx/ReadVariableOp_2ReadVariableOp$lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_2
lwptfvtmlx/mul_4Mul#lwptfvtmlx/ReadVariableOp_2:value:0lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_4
lwptfvtmlx/add_4AddV2lwptfvtmlx/split:output:3lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_4
lwptfvtmlx/Sigmoid_2Sigmoidlwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_2v
lwptfvtmlx/Tanh_1Tanhlwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh_1
lwptfvtmlx/mul_5Mullwptfvtmlx/Sigmoid_2:y:0lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lwptfvtmlx_matmul_readvariableop_resource+lwptfvtmlx_matmul_1_readvariableop_resource*lwptfvtmlx_biasadd_readvariableop_resource"lwptfvtmlx_readvariableop_resource$lwptfvtmlx_readvariableop_1_resource$lwptfvtmlx_readvariableop_2_resource*
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
while_body_2595814*
condR
while_cond_2595813*Q
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
IdentityIdentitystrided_slice_3:output:0"^lwptfvtmlx/BiasAdd/ReadVariableOp!^lwptfvtmlx/MatMul/ReadVariableOp#^lwptfvtmlx/MatMul_1/ReadVariableOp^lwptfvtmlx/ReadVariableOp^lwptfvtmlx/ReadVariableOp_1^lwptfvtmlx/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!lwptfvtmlx/BiasAdd/ReadVariableOp!lwptfvtmlx/BiasAdd/ReadVariableOp2D
 lwptfvtmlx/MatMul/ReadVariableOp lwptfvtmlx/MatMul/ReadVariableOp2H
"lwptfvtmlx/MatMul_1/ReadVariableOp"lwptfvtmlx/MatMul_1/ReadVariableOp26
lwptfvtmlx/ReadVariableOplwptfvtmlx/ReadVariableOp2:
lwptfvtmlx/ReadVariableOp_1lwptfvtmlx/ReadVariableOp_12:
lwptfvtmlx/ReadVariableOp_2lwptfvtmlx/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
©0
¼
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_2597358

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

c
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_2597380

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
¡h

G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597925

inputs<
)hswofenhiy_matmul_readvariableop_resource:	>
+hswofenhiy_matmul_1_readvariableop_resource:	 9
*hswofenhiy_biasadd_readvariableop_resource:	0
"hswofenhiy_readvariableop_resource: 2
$hswofenhiy_readvariableop_1_resource: 2
$hswofenhiy_readvariableop_2_resource: 
identity¢!hswofenhiy/BiasAdd/ReadVariableOp¢ hswofenhiy/MatMul/ReadVariableOp¢"hswofenhiy/MatMul_1/ReadVariableOp¢hswofenhiy/ReadVariableOp¢hswofenhiy/ReadVariableOp_1¢hswofenhiy/ReadVariableOp_2¢whileD
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
 hswofenhiy/MatMul/ReadVariableOpReadVariableOp)hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 hswofenhiy/MatMul/ReadVariableOp§
hswofenhiy/MatMulMatMulstrided_slice_2:output:0(hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMulµ
"hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp+hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"hswofenhiy/MatMul_1/ReadVariableOp£
hswofenhiy/MatMul_1MatMulzeros:output:0*hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMul_1
hswofenhiy/addAddV2hswofenhiy/MatMul:product:0hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/add®
!hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp*hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!hswofenhiy/BiasAdd/ReadVariableOp¥
hswofenhiy/BiasAddBiasAddhswofenhiy/add:z:0)hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/BiasAddz
hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
hswofenhiy/split/split_dimë
hswofenhiy/splitSplit#hswofenhiy/split/split_dim:output:0hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
hswofenhiy/split
hswofenhiy/ReadVariableOpReadVariableOp"hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp
hswofenhiy/mulMul!hswofenhiy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul
hswofenhiy/add_1AddV2hswofenhiy/split:output:0hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_1{
hswofenhiy/SigmoidSigmoidhswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid
hswofenhiy/ReadVariableOp_1ReadVariableOp$hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_1
hswofenhiy/mul_1Mul#hswofenhiy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_1
hswofenhiy/add_2AddV2hswofenhiy/split:output:1hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_2
hswofenhiy/Sigmoid_1Sigmoidhswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_1
hswofenhiy/mul_2Mulhswofenhiy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_2w
hswofenhiy/TanhTanhhswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh
hswofenhiy/mul_3Mulhswofenhiy/Sigmoid:y:0hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_3
hswofenhiy/add_3AddV2hswofenhiy/mul_2:z:0hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_3
hswofenhiy/ReadVariableOp_2ReadVariableOp$hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_2
hswofenhiy/mul_4Mul#hswofenhiy/ReadVariableOp_2:value:0hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_4
hswofenhiy/add_4AddV2hswofenhiy/split:output:3hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_4
hswofenhiy/Sigmoid_2Sigmoidhswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_2v
hswofenhiy/Tanh_1Tanhhswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh_1
hswofenhiy/mul_5Mulhswofenhiy/Sigmoid_2:y:0hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)hswofenhiy_matmul_readvariableop_resource+hswofenhiy_matmul_1_readvariableop_resource*hswofenhiy_biasadd_readvariableop_resource"hswofenhiy_readvariableop_resource$hswofenhiy_readvariableop_1_resource$hswofenhiy_readvariableop_2_resource*
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
while_body_2597824*
condR
while_cond_2597823*Q
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
IdentityIdentitytranspose_1:y:0"^hswofenhiy/BiasAdd/ReadVariableOp!^hswofenhiy/MatMul/ReadVariableOp#^hswofenhiy/MatMul_1/ReadVariableOp^hswofenhiy/ReadVariableOp^hswofenhiy/ReadVariableOp_1^hswofenhiy/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!hswofenhiy/BiasAdd/ReadVariableOp!hswofenhiy/BiasAdd/ReadVariableOp2D
 hswofenhiy/MatMul/ReadVariableOp hswofenhiy/MatMul/ReadVariableOp2H
"hswofenhiy/MatMul_1/ReadVariableOp"hswofenhiy/MatMul_1/ReadVariableOp26
hswofenhiy/ReadVariableOphswofenhiy/ReadVariableOp2:
hswofenhiy/ReadVariableOp_1hswofenhiy/ReadVariableOp_12:
hswofenhiy/ReadVariableOp_2hswofenhiy/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_sequential_layer_call_fn_2597284

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
G__inference_sequential_layer_call_and_return_conditional_losses_25956712
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
ç)
Ò
while_body_2593794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_hswofenhiy_2593818_0:	-
while_hswofenhiy_2593820_0:	 )
while_hswofenhiy_2593822_0:	(
while_hswofenhiy_2593824_0: (
while_hswofenhiy_2593826_0: (
while_hswofenhiy_2593828_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_hswofenhiy_2593818:	+
while_hswofenhiy_2593820:	 '
while_hswofenhiy_2593822:	&
while_hswofenhiy_2593824: &
while_hswofenhiy_2593826: &
while_hswofenhiy_2593828: ¢(while/hswofenhiy/StatefulPartitionedCallÃ
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
(while/hswofenhiy/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_hswofenhiy_2593818_0while_hswofenhiy_2593820_0while_hswofenhiy_2593822_0while_hswofenhiy_2593824_0while_hswofenhiy_2593826_0while_hswofenhiy_2593828_0*
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_25937742*
(while/hswofenhiy/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/hswofenhiy/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/hswofenhiy/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/hswofenhiy/StatefulPartitionedCall:output:1)^while/hswofenhiy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4À
while/Identity_5Identity1while/hswofenhiy/StatefulPartitionedCall:output:2)^while/hswofenhiy/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"6
while_hswofenhiy_2593818while_hswofenhiy_2593818_0"6
while_hswofenhiy_2593820while_hswofenhiy_2593820_0"6
while_hswofenhiy_2593822while_hswofenhiy_2593822_0"6
while_hswofenhiy_2593824while_hswofenhiy_2593824_0"6
while_hswofenhiy_2593826while_hswofenhiy_2593826_0"6
while_hswofenhiy_2593828while_hswofenhiy_2593828_0")
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
(while/hswofenhiy/StatefulPartitionedCall(while/hswofenhiy/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
(sequential_dnzlhpjizj_while_cond_2593403H
Dsequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_loop_counterN
Jsequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_maximum_iterations+
'sequential_dnzlhpjizj_while_placeholder-
)sequential_dnzlhpjizj_while_placeholder_1-
)sequential_dnzlhpjizj_while_placeholder_2-
)sequential_dnzlhpjizj_while_placeholder_3J
Fsequential_dnzlhpjizj_while_less_sequential_dnzlhpjizj_strided_slice_1a
]sequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_cond_2593403___redundant_placeholder0a
]sequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_cond_2593403___redundant_placeholder1a
]sequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_cond_2593403___redundant_placeholder2a
]sequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_cond_2593403___redundant_placeholder3a
]sequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_cond_2593403___redundant_placeholder4a
]sequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_cond_2593403___redundant_placeholder5a
]sequential_dnzlhpjizj_while_sequential_dnzlhpjizj_while_cond_2593403___redundant_placeholder6(
$sequential_dnzlhpjizj_while_identity
Þ
 sequential/dnzlhpjizj/while/LessLess'sequential_dnzlhpjizj_while_placeholderFsequential_dnzlhpjizj_while_less_sequential_dnzlhpjizj_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/dnzlhpjizj/while/Less
$sequential/dnzlhpjizj/while/IdentityIdentity$sequential/dnzlhpjizj/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/dnzlhpjizj/while/Identity"U
$sequential_dnzlhpjizj_while_identity-sequential/dnzlhpjizj/while/Identity:output:0*(
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
,__inference_sequential_layer_call_fn_2597321

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
G__inference_sequential_layer_call_and_return_conditional_losses_25962402
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


nyosplwtfa_while_cond_25967352
.nyosplwtfa_while_nyosplwtfa_while_loop_counter8
4nyosplwtfa_while_nyosplwtfa_while_maximum_iterations 
nyosplwtfa_while_placeholder"
nyosplwtfa_while_placeholder_1"
nyosplwtfa_while_placeholder_2"
nyosplwtfa_while_placeholder_34
0nyosplwtfa_while_less_nyosplwtfa_strided_slice_1K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2596735___redundant_placeholder0K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2596735___redundant_placeholder1K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2596735___redundant_placeholder2K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2596735___redundant_placeholder3K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2596735___redundant_placeholder4K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2596735___redundant_placeholder5K
Gnyosplwtfa_while_nyosplwtfa_while_cond_2596735___redundant_placeholder6
nyosplwtfa_while_identity
§
nyosplwtfa/while/LessLessnyosplwtfa_while_placeholder0nyosplwtfa_while_less_nyosplwtfa_strided_slice_1*
T0*
_output_shapes
: 2
nyosplwtfa/while/Less~
nyosplwtfa/while/IdentityIdentitynyosplwtfa/while/Less:z:0*
T0
*
_output_shapes
: 2
nyosplwtfa/while/Identity"?
nyosplwtfa_while_identity"nyosplwtfa/while/Identity:output:0*(
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
while_body_2597464
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_hswofenhiy_matmul_readvariableop_resource_0:	F
3while_hswofenhiy_matmul_1_readvariableop_resource_0:	 A
2while_hswofenhiy_biasadd_readvariableop_resource_0:	8
*while_hswofenhiy_readvariableop_resource_0: :
,while_hswofenhiy_readvariableop_1_resource_0: :
,while_hswofenhiy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_hswofenhiy_matmul_readvariableop_resource:	D
1while_hswofenhiy_matmul_1_readvariableop_resource:	 ?
0while_hswofenhiy_biasadd_readvariableop_resource:	6
(while_hswofenhiy_readvariableop_resource: 8
*while_hswofenhiy_readvariableop_1_resource: 8
*while_hswofenhiy_readvariableop_2_resource: ¢'while/hswofenhiy/BiasAdd/ReadVariableOp¢&while/hswofenhiy/MatMul/ReadVariableOp¢(while/hswofenhiy/MatMul_1/ReadVariableOp¢while/hswofenhiy/ReadVariableOp¢!while/hswofenhiy/ReadVariableOp_1¢!while/hswofenhiy/ReadVariableOp_2Ã
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
&while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp1while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/hswofenhiy/MatMul/ReadVariableOpÑ
while/hswofenhiy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMulÉ
(while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp3while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/hswofenhiy/MatMul_1/ReadVariableOpº
while/hswofenhiy/MatMul_1MatMulwhile_placeholder_20while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMul_1°
while/hswofenhiy/addAddV2!while/hswofenhiy/MatMul:product:0#while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/addÂ
'while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp2while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/hswofenhiy/BiasAdd/ReadVariableOp½
while/hswofenhiy/BiasAddBiasAddwhile/hswofenhiy/add:z:0/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/BiasAdd
 while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/hswofenhiy/split/split_dim
while/hswofenhiy/splitSplit)while/hswofenhiy/split/split_dim:output:0!while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/hswofenhiy/split©
while/hswofenhiy/ReadVariableOpReadVariableOp*while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/hswofenhiy/ReadVariableOp£
while/hswofenhiy/mulMul'while/hswofenhiy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul¦
while/hswofenhiy/add_1AddV2while/hswofenhiy/split:output:0while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_1
while/hswofenhiy/SigmoidSigmoidwhile/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid¯
!while/hswofenhiy/ReadVariableOp_1ReadVariableOp,while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_1©
while/hswofenhiy/mul_1Mul)while/hswofenhiy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_1¨
while/hswofenhiy/add_2AddV2while/hswofenhiy/split:output:1while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_2
while/hswofenhiy/Sigmoid_1Sigmoidwhile/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_1
while/hswofenhiy/mul_2Mulwhile/hswofenhiy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_2
while/hswofenhiy/TanhTanhwhile/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh¢
while/hswofenhiy/mul_3Mulwhile/hswofenhiy/Sigmoid:y:0while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_3£
while/hswofenhiy/add_3AddV2while/hswofenhiy/mul_2:z:0while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_3¯
!while/hswofenhiy/ReadVariableOp_2ReadVariableOp,while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_2°
while/hswofenhiy/mul_4Mul)while/hswofenhiy/ReadVariableOp_2:value:0while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_4¨
while/hswofenhiy/add_4AddV2while/hswofenhiy/split:output:3while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_4
while/hswofenhiy/Sigmoid_2Sigmoidwhile/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_2
while/hswofenhiy/Tanh_1Tanhwhile/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh_1¦
while/hswofenhiy/mul_5Mulwhile/hswofenhiy/Sigmoid_2:y:0while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/hswofenhiy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/hswofenhiy/mul_5:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/hswofenhiy/add_3:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_hswofenhiy_biasadd_readvariableop_resource2while_hswofenhiy_biasadd_readvariableop_resource_0"h
1while_hswofenhiy_matmul_1_readvariableop_resource3while_hswofenhiy_matmul_1_readvariableop_resource_0"d
/while_hswofenhiy_matmul_readvariableop_resource1while_hswofenhiy_matmul_readvariableop_resource_0"Z
*while_hswofenhiy_readvariableop_1_resource,while_hswofenhiy_readvariableop_1_resource_0"Z
*while_hswofenhiy_readvariableop_2_resource,while_hswofenhiy_readvariableop_2_resource_0"V
(while_hswofenhiy_readvariableop_resource*while_hswofenhiy_readvariableop_resource_0")
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
'while/hswofenhiy/BiasAdd/ReadVariableOp'while/hswofenhiy/BiasAdd/ReadVariableOp2P
&while/hswofenhiy/MatMul/ReadVariableOp&while/hswofenhiy/MatMul/ReadVariableOp2T
(while/hswofenhiy/MatMul_1/ReadVariableOp(while/hswofenhiy/MatMul_1/ReadVariableOp2B
while/hswofenhiy/ReadVariableOpwhile/hswofenhiy/ReadVariableOp2F
!while/hswofenhiy/ReadVariableOp_1!while/hswofenhiy/ReadVariableOp_12F
!while/hswofenhiy/ReadVariableOp_2!while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
%__inference_signature_wrapper_2596439

bdeyofgzkq
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
bdeyofgzkqunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_25936872
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
bdeyofgzkq

À
,__inference_lwptfvtmlx_layer_call_fn_2599225

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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_25945322
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
Ý
H
,__inference_ezubtmdnwx_layer_call_fn_2597385

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
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_25952662
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

À
,__inference_lwptfvtmlx_layer_call_fn_2599248

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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_25947192
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
while_cond_2595813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2595813___redundant_placeholder05
1while_while_cond_2595813___redundant_placeholder15
1while_while_cond_2595813___redundant_placeholder25
1while_while_cond_2595813___redundant_placeholder35
1while_while_cond_2595813___redundant_placeholder45
1while_while_cond_2595813___redundant_placeholder55
1while_while_cond_2595813___redundant_placeholder6
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597565
inputs_0<
)hswofenhiy_matmul_readvariableop_resource:	>
+hswofenhiy_matmul_1_readvariableop_resource:	 9
*hswofenhiy_biasadd_readvariableop_resource:	0
"hswofenhiy_readvariableop_resource: 2
$hswofenhiy_readvariableop_1_resource: 2
$hswofenhiy_readvariableop_2_resource: 
identity¢!hswofenhiy/BiasAdd/ReadVariableOp¢ hswofenhiy/MatMul/ReadVariableOp¢"hswofenhiy/MatMul_1/ReadVariableOp¢hswofenhiy/ReadVariableOp¢hswofenhiy/ReadVariableOp_1¢hswofenhiy/ReadVariableOp_2¢whileF
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
 hswofenhiy/MatMul/ReadVariableOpReadVariableOp)hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 hswofenhiy/MatMul/ReadVariableOp§
hswofenhiy/MatMulMatMulstrided_slice_2:output:0(hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMulµ
"hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp+hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"hswofenhiy/MatMul_1/ReadVariableOp£
hswofenhiy/MatMul_1MatMulzeros:output:0*hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMul_1
hswofenhiy/addAddV2hswofenhiy/MatMul:product:0hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/add®
!hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp*hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!hswofenhiy/BiasAdd/ReadVariableOp¥
hswofenhiy/BiasAddBiasAddhswofenhiy/add:z:0)hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/BiasAddz
hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
hswofenhiy/split/split_dimë
hswofenhiy/splitSplit#hswofenhiy/split/split_dim:output:0hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
hswofenhiy/split
hswofenhiy/ReadVariableOpReadVariableOp"hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp
hswofenhiy/mulMul!hswofenhiy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul
hswofenhiy/add_1AddV2hswofenhiy/split:output:0hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_1{
hswofenhiy/SigmoidSigmoidhswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid
hswofenhiy/ReadVariableOp_1ReadVariableOp$hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_1
hswofenhiy/mul_1Mul#hswofenhiy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_1
hswofenhiy/add_2AddV2hswofenhiy/split:output:1hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_2
hswofenhiy/Sigmoid_1Sigmoidhswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_1
hswofenhiy/mul_2Mulhswofenhiy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_2w
hswofenhiy/TanhTanhhswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh
hswofenhiy/mul_3Mulhswofenhiy/Sigmoid:y:0hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_3
hswofenhiy/add_3AddV2hswofenhiy/mul_2:z:0hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_3
hswofenhiy/ReadVariableOp_2ReadVariableOp$hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_2
hswofenhiy/mul_4Mul#hswofenhiy/ReadVariableOp_2:value:0hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_4
hswofenhiy/add_4AddV2hswofenhiy/split:output:3hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_4
hswofenhiy/Sigmoid_2Sigmoidhswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_2v
hswofenhiy/Tanh_1Tanhhswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh_1
hswofenhiy/mul_5Mulhswofenhiy/Sigmoid_2:y:0hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)hswofenhiy_matmul_readvariableop_resource+hswofenhiy_matmul_1_readvariableop_resource*hswofenhiy_biasadd_readvariableop_resource"hswofenhiy_readvariableop_resource$hswofenhiy_readvariableop_1_resource$hswofenhiy_readvariableop_2_resource*
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
while_body_2597464*
condR
while_cond_2597463*Q
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
IdentityIdentitytranspose_1:y:0"^hswofenhiy/BiasAdd/ReadVariableOp!^hswofenhiy/MatMul/ReadVariableOp#^hswofenhiy/MatMul_1/ReadVariableOp^hswofenhiy/ReadVariableOp^hswofenhiy/ReadVariableOp_1^hswofenhiy/ReadVariableOp_2^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!hswofenhiy/BiasAdd/ReadVariableOp!hswofenhiy/BiasAdd/ReadVariableOp2D
 hswofenhiy/MatMul/ReadVariableOp hswofenhiy/MatMul/ReadVariableOp2H
"hswofenhiy/MatMul_1/ReadVariableOp"hswofenhiy/MatMul_1/ReadVariableOp26
hswofenhiy/ReadVariableOphswofenhiy/ReadVariableOp2:
hswofenhiy/ReadVariableOp_1hswofenhiy/ReadVariableOp_12:
hswofenhiy/ReadVariableOp_2hswofenhiy/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
	

,__inference_dnzlhpjizj_layer_call_fn_2598122
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25938742
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
³F
ê
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2594137

inputs%
hswofenhiy_2594038:	%
hswofenhiy_2594040:	 !
hswofenhiy_2594042:	 
hswofenhiy_2594044:  
hswofenhiy_2594046:  
hswofenhiy_2594048: 
identity¢"hswofenhiy/StatefulPartitionedCall¢whileD
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
"hswofenhiy/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0hswofenhiy_2594038hswofenhiy_2594040hswofenhiy_2594042hswofenhiy_2594044hswofenhiy_2594046hswofenhiy_2594048*
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_25939612$
"hswofenhiy/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0hswofenhiy_2594038hswofenhiy_2594040hswofenhiy_2594042hswofenhiy_2594044hswofenhiy_2594046hswofenhiy_2594048*
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
while_body_2594057*
condR
while_cond_2594056*Q
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
IdentityIdentitytranspose_1:y:0#^hswofenhiy/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"hswofenhiy/StatefulPartitionedCall"hswofenhiy/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


í
while_cond_2598251
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2598251___redundant_placeholder05
1while_while_cond_2598251___redundant_placeholder15
1while_while_cond_2598251___redundant_placeholder25
1while_while_cond_2598251___redundant_placeholder35
1while_while_cond_2598251___redundant_placeholder45
1while_while_cond_2598251___redundant_placeholder55
1while_while_cond_2598251___redundant_placeholder6
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
while_cond_2594551
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2594551___redundant_placeholder05
1while_while_cond_2594551___redundant_placeholder15
1while_while_cond_2594551___redundant_placeholder25
1while_while_cond_2594551___redundant_placeholder35
1while_while_cond_2594551___redundant_placeholder45
1while_while_cond_2594551___redundant_placeholder55
1while_while_cond_2594551___redundant_placeholder6
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
while_cond_2594056
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2594056___redundant_placeholder05
1while_while_cond_2594056___redundant_placeholder15
1while_while_cond_2594056___redundant_placeholder25
1while_while_cond_2594056___redundant_placeholder35
1while_while_cond_2594056___redundant_placeholder45
1while_while_cond_2594056___redundant_placeholder55
1while_while_cond_2594056___redundant_placeholder6
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
Z
Ë
 __inference__traced_save_2599388
file_prefix0
,savev2_gtjikcltwy_kernel_read_readvariableop.
*savev2_gtjikcltwy_bias_read_readvariableop0
,savev2_chsgvefspq_kernel_read_readvariableop.
*savev2_chsgvefspq_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop;
7savev2_dnzlhpjizj_hswofenhiy_kernel_read_readvariableopE
Asavev2_dnzlhpjizj_hswofenhiy_recurrent_kernel_read_readvariableop9
5savev2_dnzlhpjizj_hswofenhiy_bias_read_readvariableopP
Lsavev2_dnzlhpjizj_hswofenhiy_input_gate_peephole_weights_read_readvariableopQ
Msavev2_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_dnzlhpjizj_hswofenhiy_output_gate_peephole_weights_read_readvariableop;
7savev2_nyosplwtfa_lwptfvtmlx_kernel_read_readvariableopE
Asavev2_nyosplwtfa_lwptfvtmlx_recurrent_kernel_read_readvariableop9
5savev2_nyosplwtfa_lwptfvtmlx_bias_read_readvariableopP
Lsavev2_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weights_read_readvariableopQ
Msavev2_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weights_read_readvariableopQ
Msavev2_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_rmsprop_gtjikcltwy_kernel_rms_read_readvariableop:
6savev2_rmsprop_gtjikcltwy_bias_rms_read_readvariableop<
8savev2_rmsprop_chsgvefspq_kernel_rms_read_readvariableop:
6savev2_rmsprop_chsgvefspq_bias_rms_read_readvariableopG
Csavev2_rmsprop_dnzlhpjizj_hswofenhiy_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_dnzlhpjizj_hswofenhiy_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_dnzlhpjizj_hswofenhiy_bias_rms_read_readvariableop\
Xsavev2_rmsprop_dnzlhpjizj_hswofenhiy_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_dnzlhpjizj_hswofenhiy_output_gate_peephole_weights_rms_read_readvariableopG
Csavev2_rmsprop_nyosplwtfa_lwptfvtmlx_kernel_rms_read_readvariableopQ
Msavev2_rmsprop_nyosplwtfa_lwptfvtmlx_recurrent_kernel_rms_read_readvariableopE
Asavev2_rmsprop_nyosplwtfa_lwptfvtmlx_bias_rms_read_readvariableop\
Xsavev2_rmsprop_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weights_rms_read_readvariableop]
Ysavev2_rmsprop_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weights_rms_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_gtjikcltwy_kernel_read_readvariableop*savev2_gtjikcltwy_bias_read_readvariableop,savev2_chsgvefspq_kernel_read_readvariableop*savev2_chsgvefspq_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop7savev2_dnzlhpjizj_hswofenhiy_kernel_read_readvariableopAsavev2_dnzlhpjizj_hswofenhiy_recurrent_kernel_read_readvariableop5savev2_dnzlhpjizj_hswofenhiy_bias_read_readvariableopLsavev2_dnzlhpjizj_hswofenhiy_input_gate_peephole_weights_read_readvariableopMsavev2_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weights_read_readvariableopMsavev2_dnzlhpjizj_hswofenhiy_output_gate_peephole_weights_read_readvariableop7savev2_nyosplwtfa_lwptfvtmlx_kernel_read_readvariableopAsavev2_nyosplwtfa_lwptfvtmlx_recurrent_kernel_read_readvariableop5savev2_nyosplwtfa_lwptfvtmlx_bias_read_readvariableopLsavev2_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weights_read_readvariableopMsavev2_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weights_read_readvariableopMsavev2_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_rmsprop_gtjikcltwy_kernel_rms_read_readvariableop6savev2_rmsprop_gtjikcltwy_bias_rms_read_readvariableop8savev2_rmsprop_chsgvefspq_kernel_rms_read_readvariableop6savev2_rmsprop_chsgvefspq_bias_rms_read_readvariableopCsavev2_rmsprop_dnzlhpjizj_hswofenhiy_kernel_rms_read_readvariableopMsavev2_rmsprop_dnzlhpjizj_hswofenhiy_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_dnzlhpjizj_hswofenhiy_bias_rms_read_readvariableopXsavev2_rmsprop_dnzlhpjizj_hswofenhiy_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_dnzlhpjizj_hswofenhiy_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_dnzlhpjizj_hswofenhiy_output_gate_peephole_weights_rms_read_readvariableopCsavev2_rmsprop_nyosplwtfa_lwptfvtmlx_kernel_rms_read_readvariableopMsavev2_rmsprop_nyosplwtfa_lwptfvtmlx_recurrent_kernel_rms_read_readvariableopAsavev2_rmsprop_nyosplwtfa_lwptfvtmlx_bias_rms_read_readvariableopXsavev2_rmsprop_nyosplwtfa_lwptfvtmlx_input_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_nyosplwtfa_lwptfvtmlx_forget_gate_peephole_weights_rms_read_readvariableopYsavev2_rmsprop_nyosplwtfa_lwptfvtmlx_output_gate_peephole_weights_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
while_cond_2598791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_2598791___redundant_placeholder05
1while_while_cond_2598791___redundant_placeholder15
1while_while_cond_2598791___redundant_placeholder25
1while_while_cond_2598791___redundant_placeholder35
1while_while_cond_2598791___redundant_placeholder45
1while_while_cond_2598791___redundant_placeholder55
1while_while_cond_2598791___redundant_placeholder6
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
while_body_2598252
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lwptfvtmlx_matmul_readvariableop_resource_0:	 F
3while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 A
2while_lwptfvtmlx_biasadd_readvariableop_resource_0:	8
*while_lwptfvtmlx_readvariableop_resource_0: :
,while_lwptfvtmlx_readvariableop_1_resource_0: :
,while_lwptfvtmlx_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lwptfvtmlx_matmul_readvariableop_resource:	 D
1while_lwptfvtmlx_matmul_1_readvariableop_resource:	 ?
0while_lwptfvtmlx_biasadd_readvariableop_resource:	6
(while_lwptfvtmlx_readvariableop_resource: 8
*while_lwptfvtmlx_readvariableop_1_resource: 8
*while_lwptfvtmlx_readvariableop_2_resource: ¢'while/lwptfvtmlx/BiasAdd/ReadVariableOp¢&while/lwptfvtmlx/MatMul/ReadVariableOp¢(while/lwptfvtmlx/MatMul_1/ReadVariableOp¢while/lwptfvtmlx/ReadVariableOp¢!while/lwptfvtmlx/ReadVariableOp_1¢!while/lwptfvtmlx/ReadVariableOp_2Ã
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
&while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp1while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/lwptfvtmlx/MatMul/ReadVariableOpÑ
while/lwptfvtmlx/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMulÉ
(while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp3while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/lwptfvtmlx/MatMul_1/ReadVariableOpº
while/lwptfvtmlx/MatMul_1MatMulwhile_placeholder_20while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMul_1°
while/lwptfvtmlx/addAddV2!while/lwptfvtmlx/MatMul:product:0#while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/addÂ
'while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp2while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/lwptfvtmlx/BiasAdd/ReadVariableOp½
while/lwptfvtmlx/BiasAddBiasAddwhile/lwptfvtmlx/add:z:0/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/BiasAdd
 while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/lwptfvtmlx/split/split_dim
while/lwptfvtmlx/splitSplit)while/lwptfvtmlx/split/split_dim:output:0!while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/lwptfvtmlx/split©
while/lwptfvtmlx/ReadVariableOpReadVariableOp*while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/lwptfvtmlx/ReadVariableOp£
while/lwptfvtmlx/mulMul'while/lwptfvtmlx/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul¦
while/lwptfvtmlx/add_1AddV2while/lwptfvtmlx/split:output:0while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_1
while/lwptfvtmlx/SigmoidSigmoidwhile/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid¯
!while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp,while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_1©
while/lwptfvtmlx/mul_1Mul)while/lwptfvtmlx/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_1¨
while/lwptfvtmlx/add_2AddV2while/lwptfvtmlx/split:output:1while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_2
while/lwptfvtmlx/Sigmoid_1Sigmoidwhile/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_1
while/lwptfvtmlx/mul_2Mulwhile/lwptfvtmlx/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_2
while/lwptfvtmlx/TanhTanhwhile/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh¢
while/lwptfvtmlx/mul_3Mulwhile/lwptfvtmlx/Sigmoid:y:0while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_3£
while/lwptfvtmlx/add_3AddV2while/lwptfvtmlx/mul_2:z:0while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_3¯
!while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp,while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_2°
while/lwptfvtmlx/mul_4Mul)while/lwptfvtmlx/ReadVariableOp_2:value:0while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_4¨
while/lwptfvtmlx/add_4AddV2while/lwptfvtmlx/split:output:3while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_4
while/lwptfvtmlx/Sigmoid_2Sigmoidwhile/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_2
while/lwptfvtmlx/Tanh_1Tanhwhile/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh_1¦
while/lwptfvtmlx/mul_5Mulwhile/lwptfvtmlx/Sigmoid_2:y:0while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lwptfvtmlx/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lwptfvtmlx/mul_5:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/lwptfvtmlx/add_3:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
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
0while_lwptfvtmlx_biasadd_readvariableop_resource2while_lwptfvtmlx_biasadd_readvariableop_resource_0"h
1while_lwptfvtmlx_matmul_1_readvariableop_resource3while_lwptfvtmlx_matmul_1_readvariableop_resource_0"d
/while_lwptfvtmlx_matmul_readvariableop_resource1while_lwptfvtmlx_matmul_readvariableop_resource_0"Z
*while_lwptfvtmlx_readvariableop_1_resource,while_lwptfvtmlx_readvariableop_1_resource_0"Z
*while_lwptfvtmlx_readvariableop_2_resource,while_lwptfvtmlx_readvariableop_2_resource_0"V
(while_lwptfvtmlx_readvariableop_resource*while_lwptfvtmlx_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/lwptfvtmlx/BiasAdd/ReadVariableOp'while/lwptfvtmlx/BiasAdd/ReadVariableOp2P
&while/lwptfvtmlx/MatMul/ReadVariableOp&while/lwptfvtmlx/MatMul/ReadVariableOp2T
(while/lwptfvtmlx/MatMul_1/ReadVariableOp(while/lwptfvtmlx/MatMul_1/ReadVariableOp2B
while/lwptfvtmlx/ReadVariableOpwhile/lwptfvtmlx/ReadVariableOp2F
!while/lwptfvtmlx/ReadVariableOp_1!while/lwptfvtmlx/ReadVariableOp_12F
!while/lwptfvtmlx/ReadVariableOp_2!while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
(sequential_nyosplwtfa_while_body_2593580H
Dsequential_nyosplwtfa_while_sequential_nyosplwtfa_while_loop_counterN
Jsequential_nyosplwtfa_while_sequential_nyosplwtfa_while_maximum_iterations+
'sequential_nyosplwtfa_while_placeholder-
)sequential_nyosplwtfa_while_placeholder_1-
)sequential_nyosplwtfa_while_placeholder_2-
)sequential_nyosplwtfa_while_placeholder_3G
Csequential_nyosplwtfa_while_sequential_nyosplwtfa_strided_slice_1_0
sequential_nyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_sequential_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0Z
Gsequential_nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0:	 \
Isequential_nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 W
Hsequential_nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0:	N
@sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0: P
Bsequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0: P
Bsequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0: (
$sequential_nyosplwtfa_while_identity*
&sequential_nyosplwtfa_while_identity_1*
&sequential_nyosplwtfa_while_identity_2*
&sequential_nyosplwtfa_while_identity_3*
&sequential_nyosplwtfa_while_identity_4*
&sequential_nyosplwtfa_while_identity_5E
Asequential_nyosplwtfa_while_sequential_nyosplwtfa_strided_slice_1
}sequential_nyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_sequential_nyosplwtfa_tensorarrayunstack_tensorlistfromtensorX
Esequential_nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource:	 Z
Gsequential_nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource:	 U
Fsequential_nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource:	L
>sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_resource: N
@sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource: N
@sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource: ¢=sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp¢<sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp¢>sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp¢5sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp¢7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1¢7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2ï
Msequential/nyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2O
Msequential/nyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential/nyosplwtfa/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_nyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_sequential_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0'sequential_nyosplwtfa_while_placeholderVsequential/nyosplwtfa/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02A
?sequential/nyosplwtfa/while/TensorArrayV2Read/TensorListGetItem
<sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOpGsequential_nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02>
<sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp©
-sequential/nyosplwtfa/while/lwptfvtmlx/MatMulMatMulFsequential/nyosplwtfa/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential/nyosplwtfa/while/lwptfvtmlx/MatMul
>sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOpIsequential_nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02@
>sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp
/sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1MatMul)sequential_nyosplwtfa_while_placeholder_2Fsequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1
*sequential/nyosplwtfa/while/lwptfvtmlx/addAddV27sequential/nyosplwtfa/while/lwptfvtmlx/MatMul:product:09sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential/nyosplwtfa/while/lwptfvtmlx/add
=sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOpHsequential_nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02?
=sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp
.sequential/nyosplwtfa/while/lwptfvtmlx/BiasAddBiasAdd.sequential/nyosplwtfa/while/lwptfvtmlx/add:z:0Esequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd²
6sequential/nyosplwtfa/while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential/nyosplwtfa/while/lwptfvtmlx/split/split_dimÛ
,sequential/nyosplwtfa/while/lwptfvtmlx/splitSplit?sequential/nyosplwtfa/while/lwptfvtmlx/split/split_dim:output:07sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2.
,sequential/nyosplwtfa/while/lwptfvtmlx/splitë
5sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOpReadVariableOp@sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype027
5sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOpû
*sequential/nyosplwtfa/while/lwptfvtmlx/mulMul=sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp:value:0)sequential_nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential/nyosplwtfa/while/lwptfvtmlx/mulþ
,sequential/nyosplwtfa/while/lwptfvtmlx/add_1AddV25sequential/nyosplwtfa/while/lwptfvtmlx/split:output:0.sequential/nyosplwtfa/while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/add_1Ï
.sequential/nyosplwtfa/while/lwptfvtmlx/SigmoidSigmoid0sequential/nyosplwtfa/while/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoidñ
7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1ReadVariableOpBsequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype029
7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_1Mul?sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_1:value:0)sequential_nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_1
,sequential/nyosplwtfa/while/lwptfvtmlx/add_2AddV25sequential/nyosplwtfa/while/lwptfvtmlx/split:output:10sequential/nyosplwtfa/while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/add_2Ó
0sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoid_1Sigmoid0sequential/nyosplwtfa/while/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoid_1ö
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_2Mul4sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoid_1:y:0)sequential_nyosplwtfa_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_2Ë
+sequential/nyosplwtfa/while/lwptfvtmlx/TanhTanh5sequential/nyosplwtfa/while/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential/nyosplwtfa/while/lwptfvtmlx/Tanhú
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_3Mul2sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoid:y:0/sequential/nyosplwtfa/while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_3û
,sequential/nyosplwtfa/while/lwptfvtmlx/add_3AddV20sequential/nyosplwtfa/while/lwptfvtmlx/mul_2:z:00sequential/nyosplwtfa/while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/add_3ñ
7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2ReadVariableOpBsequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype029
7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_4Mul?sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2:value:00sequential/nyosplwtfa/while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_4
,sequential/nyosplwtfa/while/lwptfvtmlx/add_4AddV25sequential/nyosplwtfa/while/lwptfvtmlx/split:output:30sequential/nyosplwtfa/while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/add_4Ó
0sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoid_2Sigmoid0sequential/nyosplwtfa/while/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoid_2Ê
-sequential/nyosplwtfa/while/lwptfvtmlx/Tanh_1Tanh0sequential/nyosplwtfa/while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-sequential/nyosplwtfa/while/lwptfvtmlx/Tanh_1þ
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_5Mul4sequential/nyosplwtfa/while/lwptfvtmlx/Sigmoid_2:y:01sequential/nyosplwtfa/while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,sequential/nyosplwtfa/while/lwptfvtmlx/mul_5Ì
@sequential/nyosplwtfa/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_nyosplwtfa_while_placeholder_1'sequential_nyosplwtfa_while_placeholder0sequential/nyosplwtfa/while/lwptfvtmlx/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/nyosplwtfa/while/TensorArrayV2Write/TensorListSetItem
!sequential/nyosplwtfa/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/nyosplwtfa/while/add/yÁ
sequential/nyosplwtfa/while/addAddV2'sequential_nyosplwtfa_while_placeholder*sequential/nyosplwtfa/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/nyosplwtfa/while/add
#sequential/nyosplwtfa/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/nyosplwtfa/while/add_1/yä
!sequential/nyosplwtfa/while/add_1AddV2Dsequential_nyosplwtfa_while_sequential_nyosplwtfa_while_loop_counter,sequential/nyosplwtfa/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/nyosplwtfa/while/add_1
$sequential/nyosplwtfa/while/IdentityIdentity%sequential/nyosplwtfa/while/add_1:z:0>^sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp=^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp?^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp6^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp8^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_18^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2&
$sequential/nyosplwtfa/while/Identityµ
&sequential/nyosplwtfa/while/Identity_1IdentityJsequential_nyosplwtfa_while_sequential_nyosplwtfa_while_maximum_iterations>^sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp=^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp?^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp6^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp8^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_18^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/nyosplwtfa/while/Identity_1
&sequential/nyosplwtfa/while/Identity_2Identity#sequential/nyosplwtfa/while/add:z:0>^sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp=^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp?^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp6^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp8^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_18^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/nyosplwtfa/while/Identity_2»
&sequential/nyosplwtfa/while/Identity_3IdentityPsequential/nyosplwtfa/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp=^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp?^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp6^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp8^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_18^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2(
&sequential/nyosplwtfa/while/Identity_3¬
&sequential/nyosplwtfa/while/Identity_4Identity0sequential/nyosplwtfa/while/lwptfvtmlx/mul_5:z:0>^sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp=^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp?^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp6^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp8^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_18^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/while/Identity_4¬
&sequential/nyosplwtfa/while/Identity_5Identity0sequential/nyosplwtfa/while/lwptfvtmlx/add_3:z:0>^sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp=^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp?^sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp6^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp8^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_18^sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&sequential/nyosplwtfa/while/Identity_5"U
$sequential_nyosplwtfa_while_identity-sequential/nyosplwtfa/while/Identity:output:0"Y
&sequential_nyosplwtfa_while_identity_1/sequential/nyosplwtfa/while/Identity_1:output:0"Y
&sequential_nyosplwtfa_while_identity_2/sequential/nyosplwtfa/while/Identity_2:output:0"Y
&sequential_nyosplwtfa_while_identity_3/sequential/nyosplwtfa/while/Identity_3:output:0"Y
&sequential_nyosplwtfa_while_identity_4/sequential/nyosplwtfa/while/Identity_4:output:0"Y
&sequential_nyosplwtfa_while_identity_5/sequential/nyosplwtfa/while/Identity_5:output:0"
Fsequential_nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resourceHsequential_nyosplwtfa_while_lwptfvtmlx_biasadd_readvariableop_resource_0"
Gsequential_nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resourceIsequential_nyosplwtfa_while_lwptfvtmlx_matmul_1_readvariableop_resource_0"
Esequential_nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resourceGsequential_nyosplwtfa_while_lwptfvtmlx_matmul_readvariableop_resource_0"
@sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resourceBsequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_1_resource_0"
@sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resourceBsequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_2_resource_0"
>sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_resource@sequential_nyosplwtfa_while_lwptfvtmlx_readvariableop_resource_0"
Asequential_nyosplwtfa_while_sequential_nyosplwtfa_strided_slice_1Csequential_nyosplwtfa_while_sequential_nyosplwtfa_strided_slice_1_0"
}sequential_nyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_sequential_nyosplwtfa_tensorarrayunstack_tensorlistfromtensorsequential_nyosplwtfa_while_tensorarrayv2read_tensorlistgetitem_sequential_nyosplwtfa_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2~
=sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp=sequential/nyosplwtfa/while/lwptfvtmlx/BiasAdd/ReadVariableOp2|
<sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp<sequential/nyosplwtfa/while/lwptfvtmlx/MatMul/ReadVariableOp2
>sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp>sequential/nyosplwtfa/while/lwptfvtmlx/MatMul_1/ReadVariableOp2n
5sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp5sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp2r
7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_17sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_12r
7sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_27sequential/nyosplwtfa/while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_2598004
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_hswofenhiy_matmul_readvariableop_resource_0:	F
3while_hswofenhiy_matmul_1_readvariableop_resource_0:	 A
2while_hswofenhiy_biasadd_readvariableop_resource_0:	8
*while_hswofenhiy_readvariableop_resource_0: :
,while_hswofenhiy_readvariableop_1_resource_0: :
,while_hswofenhiy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_hswofenhiy_matmul_readvariableop_resource:	D
1while_hswofenhiy_matmul_1_readvariableop_resource:	 ?
0while_hswofenhiy_biasadd_readvariableop_resource:	6
(while_hswofenhiy_readvariableop_resource: 8
*while_hswofenhiy_readvariableop_1_resource: 8
*while_hswofenhiy_readvariableop_2_resource: ¢'while/hswofenhiy/BiasAdd/ReadVariableOp¢&while/hswofenhiy/MatMul/ReadVariableOp¢(while/hswofenhiy/MatMul_1/ReadVariableOp¢while/hswofenhiy/ReadVariableOp¢!while/hswofenhiy/ReadVariableOp_1¢!while/hswofenhiy/ReadVariableOp_2Ã
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
&while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp1while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/hswofenhiy/MatMul/ReadVariableOpÑ
while/hswofenhiy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMulÉ
(while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp3while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/hswofenhiy/MatMul_1/ReadVariableOpº
while/hswofenhiy/MatMul_1MatMulwhile_placeholder_20while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMul_1°
while/hswofenhiy/addAddV2!while/hswofenhiy/MatMul:product:0#while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/addÂ
'while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp2while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/hswofenhiy/BiasAdd/ReadVariableOp½
while/hswofenhiy/BiasAddBiasAddwhile/hswofenhiy/add:z:0/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/BiasAdd
 while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/hswofenhiy/split/split_dim
while/hswofenhiy/splitSplit)while/hswofenhiy/split/split_dim:output:0!while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/hswofenhiy/split©
while/hswofenhiy/ReadVariableOpReadVariableOp*while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/hswofenhiy/ReadVariableOp£
while/hswofenhiy/mulMul'while/hswofenhiy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul¦
while/hswofenhiy/add_1AddV2while/hswofenhiy/split:output:0while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_1
while/hswofenhiy/SigmoidSigmoidwhile/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid¯
!while/hswofenhiy/ReadVariableOp_1ReadVariableOp,while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_1©
while/hswofenhiy/mul_1Mul)while/hswofenhiy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_1¨
while/hswofenhiy/add_2AddV2while/hswofenhiy/split:output:1while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_2
while/hswofenhiy/Sigmoid_1Sigmoidwhile/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_1
while/hswofenhiy/mul_2Mulwhile/hswofenhiy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_2
while/hswofenhiy/TanhTanhwhile/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh¢
while/hswofenhiy/mul_3Mulwhile/hswofenhiy/Sigmoid:y:0while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_3£
while/hswofenhiy/add_3AddV2while/hswofenhiy/mul_2:z:0while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_3¯
!while/hswofenhiy/ReadVariableOp_2ReadVariableOp,while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_2°
while/hswofenhiy/mul_4Mul)while/hswofenhiy/ReadVariableOp_2:value:0while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_4¨
while/hswofenhiy/add_4AddV2while/hswofenhiy/split:output:3while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_4
while/hswofenhiy/Sigmoid_2Sigmoidwhile/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_2
while/hswofenhiy/Tanh_1Tanhwhile/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh_1¦
while/hswofenhiy/mul_5Mulwhile/hswofenhiy/Sigmoid_2:y:0while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/hswofenhiy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/hswofenhiy/mul_5:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/hswofenhiy/add_3:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_hswofenhiy_biasadd_readvariableop_resource2while_hswofenhiy_biasadd_readvariableop_resource_0"h
1while_hswofenhiy_matmul_1_readvariableop_resource3while_hswofenhiy_matmul_1_readvariableop_resource_0"d
/while_hswofenhiy_matmul_readvariableop_resource1while_hswofenhiy_matmul_readvariableop_resource_0"Z
*while_hswofenhiy_readvariableop_1_resource,while_hswofenhiy_readvariableop_1_resource_0"Z
*while_hswofenhiy_readvariableop_2_resource,while_hswofenhiy_readvariableop_2_resource_0"V
(while_hswofenhiy_readvariableop_resource*while_hswofenhiy_readvariableop_resource_0")
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
'while/hswofenhiy/BiasAdd/ReadVariableOp'while/hswofenhiy/BiasAdd/ReadVariableOp2P
&while/hswofenhiy/MatMul/ReadVariableOp&while/hswofenhiy/MatMul/ReadVariableOp2T
(while/hswofenhiy/MatMul_1/ReadVariableOp(while/hswofenhiy/MatMul_1/ReadVariableOp2B
while/hswofenhiy/ReadVariableOpwhile/hswofenhiy/ReadVariableOp2F
!while/hswofenhiy/ReadVariableOp_1!while/hswofenhiy/ReadVariableOp_12F
!while/hswofenhiy/ReadVariableOp_2!while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
while_body_2597644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_hswofenhiy_matmul_readvariableop_resource_0:	F
3while_hswofenhiy_matmul_1_readvariableop_resource_0:	 A
2while_hswofenhiy_biasadd_readvariableop_resource_0:	8
*while_hswofenhiy_readvariableop_resource_0: :
,while_hswofenhiy_readvariableop_1_resource_0: :
,while_hswofenhiy_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_hswofenhiy_matmul_readvariableop_resource:	D
1while_hswofenhiy_matmul_1_readvariableop_resource:	 ?
0while_hswofenhiy_biasadd_readvariableop_resource:	6
(while_hswofenhiy_readvariableop_resource: 8
*while_hswofenhiy_readvariableop_1_resource: 8
*while_hswofenhiy_readvariableop_2_resource: ¢'while/hswofenhiy/BiasAdd/ReadVariableOp¢&while/hswofenhiy/MatMul/ReadVariableOp¢(while/hswofenhiy/MatMul_1/ReadVariableOp¢while/hswofenhiy/ReadVariableOp¢!while/hswofenhiy/ReadVariableOp_1¢!while/hswofenhiy/ReadVariableOp_2Ã
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
&while/hswofenhiy/MatMul/ReadVariableOpReadVariableOp1while_hswofenhiy_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype02(
&while/hswofenhiy/MatMul/ReadVariableOpÑ
while/hswofenhiy/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMulÉ
(while/hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp3while_hswofenhiy_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/hswofenhiy/MatMul_1/ReadVariableOpº
while/hswofenhiy/MatMul_1MatMulwhile_placeholder_20while/hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/MatMul_1°
while/hswofenhiy/addAddV2!while/hswofenhiy/MatMul:product:0#while/hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/addÂ
'while/hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp2while_hswofenhiy_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/hswofenhiy/BiasAdd/ReadVariableOp½
while/hswofenhiy/BiasAddBiasAddwhile/hswofenhiy/add:z:0/while/hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/hswofenhiy/BiasAdd
 while/hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/hswofenhiy/split/split_dim
while/hswofenhiy/splitSplit)while/hswofenhiy/split/split_dim:output:0!while/hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/hswofenhiy/split©
while/hswofenhiy/ReadVariableOpReadVariableOp*while_hswofenhiy_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/hswofenhiy/ReadVariableOp£
while/hswofenhiy/mulMul'while/hswofenhiy/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul¦
while/hswofenhiy/add_1AddV2while/hswofenhiy/split:output:0while/hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_1
while/hswofenhiy/SigmoidSigmoidwhile/hswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid¯
!while/hswofenhiy/ReadVariableOp_1ReadVariableOp,while_hswofenhiy_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_1©
while/hswofenhiy/mul_1Mul)while/hswofenhiy/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_1¨
while/hswofenhiy/add_2AddV2while/hswofenhiy/split:output:1while/hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_2
while/hswofenhiy/Sigmoid_1Sigmoidwhile/hswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_1
while/hswofenhiy/mul_2Mulwhile/hswofenhiy/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_2
while/hswofenhiy/TanhTanhwhile/hswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh¢
while/hswofenhiy/mul_3Mulwhile/hswofenhiy/Sigmoid:y:0while/hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_3£
while/hswofenhiy/add_3AddV2while/hswofenhiy/mul_2:z:0while/hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_3¯
!while/hswofenhiy/ReadVariableOp_2ReadVariableOp,while_hswofenhiy_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/hswofenhiy/ReadVariableOp_2°
while/hswofenhiy/mul_4Mul)while/hswofenhiy/ReadVariableOp_2:value:0while/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_4¨
while/hswofenhiy/add_4AddV2while/hswofenhiy/split:output:3while/hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/add_4
while/hswofenhiy/Sigmoid_2Sigmoidwhile/hswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Sigmoid_2
while/hswofenhiy/Tanh_1Tanhwhile/hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/Tanh_1¦
while/hswofenhiy/mul_5Mulwhile/hswofenhiy/Sigmoid_2:y:0while/hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/hswofenhiy/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/hswofenhiy/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/hswofenhiy/mul_5:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/hswofenhiy/add_3:z:0(^while/hswofenhiy/BiasAdd/ReadVariableOp'^while/hswofenhiy/MatMul/ReadVariableOp)^while/hswofenhiy/MatMul_1/ReadVariableOp ^while/hswofenhiy/ReadVariableOp"^while/hswofenhiy/ReadVariableOp_1"^while/hswofenhiy/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_5"f
0while_hswofenhiy_biasadd_readvariableop_resource2while_hswofenhiy_biasadd_readvariableop_resource_0"h
1while_hswofenhiy_matmul_1_readvariableop_resource3while_hswofenhiy_matmul_1_readvariableop_resource_0"d
/while_hswofenhiy_matmul_readvariableop_resource1while_hswofenhiy_matmul_readvariableop_resource_0"Z
*while_hswofenhiy_readvariableop_1_resource,while_hswofenhiy_readvariableop_1_resource_0"Z
*while_hswofenhiy_readvariableop_2_resource,while_hswofenhiy_readvariableop_2_resource_0"V
(while_hswofenhiy_readvariableop_resource*while_hswofenhiy_readvariableop_resource_0")
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
'while/hswofenhiy/BiasAdd/ReadVariableOp'while/hswofenhiy/BiasAdd/ReadVariableOp2P
&while/hswofenhiy/MatMul/ReadVariableOp&while/hswofenhiy/MatMul/ReadVariableOp2T
(while/hswofenhiy/MatMul_1/ReadVariableOp(while/hswofenhiy/MatMul_1/ReadVariableOp2B
while/hswofenhiy/ReadVariableOpwhile/hswofenhiy/ReadVariableOp2F
!while/hswofenhiy/ReadVariableOp_1!while/hswofenhiy/ReadVariableOp_12F
!while/hswofenhiy/ReadVariableOp_2!while/hswofenhiy/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2599202

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
,__inference_dnzlhpjizj_layer_call_fn_2598173

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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_25961292
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
ë

,__inference_nyosplwtfa_layer_call_fn_2598910
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_25946322
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
while_body_2598792
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lwptfvtmlx_matmul_readvariableop_resource_0:	 F
3while_lwptfvtmlx_matmul_1_readvariableop_resource_0:	 A
2while_lwptfvtmlx_biasadd_readvariableop_resource_0:	8
*while_lwptfvtmlx_readvariableop_resource_0: :
,while_lwptfvtmlx_readvariableop_1_resource_0: :
,while_lwptfvtmlx_readvariableop_2_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lwptfvtmlx_matmul_readvariableop_resource:	 D
1while_lwptfvtmlx_matmul_1_readvariableop_resource:	 ?
0while_lwptfvtmlx_biasadd_readvariableop_resource:	6
(while_lwptfvtmlx_readvariableop_resource: 8
*while_lwptfvtmlx_readvariableop_1_resource: 8
*while_lwptfvtmlx_readvariableop_2_resource: ¢'while/lwptfvtmlx/BiasAdd/ReadVariableOp¢&while/lwptfvtmlx/MatMul/ReadVariableOp¢(while/lwptfvtmlx/MatMul_1/ReadVariableOp¢while/lwptfvtmlx/ReadVariableOp¢!while/lwptfvtmlx/ReadVariableOp_1¢!while/lwptfvtmlx/ReadVariableOp_2Ã
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
&while/lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp1while_lwptfvtmlx_matmul_readvariableop_resource_0*
_output_shapes
:	 *
dtype02(
&while/lwptfvtmlx/MatMul/ReadVariableOpÑ
while/lwptfvtmlx/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMulÉ
(while/lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp3while_lwptfvtmlx_matmul_1_readvariableop_resource_0*
_output_shapes
:	 *
dtype02*
(while/lwptfvtmlx/MatMul_1/ReadVariableOpº
while/lwptfvtmlx/MatMul_1MatMulwhile_placeholder_20while/lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/MatMul_1°
while/lwptfvtmlx/addAddV2!while/lwptfvtmlx/MatMul:product:0#while/lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/addÂ
'while/lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp2while_lwptfvtmlx_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02)
'while/lwptfvtmlx/BiasAdd/ReadVariableOp½
while/lwptfvtmlx/BiasAddBiasAddwhile/lwptfvtmlx/add:z:0/while/lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lwptfvtmlx/BiasAdd
 while/lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 while/lwptfvtmlx/split/split_dim
while/lwptfvtmlx/splitSplit)while/lwptfvtmlx/split/split_dim:output:0!while/lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/lwptfvtmlx/split©
while/lwptfvtmlx/ReadVariableOpReadVariableOp*while_lwptfvtmlx_readvariableop_resource_0*
_output_shapes
: *
dtype02!
while/lwptfvtmlx/ReadVariableOp£
while/lwptfvtmlx/mulMul'while/lwptfvtmlx/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul¦
while/lwptfvtmlx/add_1AddV2while/lwptfvtmlx/split:output:0while/lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_1
while/lwptfvtmlx/SigmoidSigmoidwhile/lwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid¯
!while/lwptfvtmlx/ReadVariableOp_1ReadVariableOp,while_lwptfvtmlx_readvariableop_1_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_1©
while/lwptfvtmlx/mul_1Mul)while/lwptfvtmlx/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_1¨
while/lwptfvtmlx/add_2AddV2while/lwptfvtmlx/split:output:1while/lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_2
while/lwptfvtmlx/Sigmoid_1Sigmoidwhile/lwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_1
while/lwptfvtmlx/mul_2Mulwhile/lwptfvtmlx/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_2
while/lwptfvtmlx/TanhTanhwhile/lwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh¢
while/lwptfvtmlx/mul_3Mulwhile/lwptfvtmlx/Sigmoid:y:0while/lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_3£
while/lwptfvtmlx/add_3AddV2while/lwptfvtmlx/mul_2:z:0while/lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_3¯
!while/lwptfvtmlx/ReadVariableOp_2ReadVariableOp,while_lwptfvtmlx_readvariableop_2_resource_0*
_output_shapes
: *
dtype02#
!while/lwptfvtmlx/ReadVariableOp_2°
while/lwptfvtmlx/mul_4Mul)while/lwptfvtmlx/ReadVariableOp_2:value:0while/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_4¨
while/lwptfvtmlx/add_4AddV2while/lwptfvtmlx/split:output:3while/lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/add_4
while/lwptfvtmlx/Sigmoid_2Sigmoidwhile/lwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Sigmoid_2
while/lwptfvtmlx/Tanh_1Tanhwhile/lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/Tanh_1¦
while/lwptfvtmlx/mul_5Mulwhile/lwptfvtmlx/Sigmoid_2:y:0while/lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lwptfvtmlx/mul_5Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lwptfvtmlx/mul_5:z:0*
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
while/IdentityIdentitywhile/add_1:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3æ
while/Identity_4Identitywhile/lwptfvtmlx/mul_5:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4æ
while/Identity_5Identitywhile/lwptfvtmlx/add_3:z:0(^while/lwptfvtmlx/BiasAdd/ReadVariableOp'^while/lwptfvtmlx/MatMul/ReadVariableOp)^while/lwptfvtmlx/MatMul_1/ReadVariableOp ^while/lwptfvtmlx/ReadVariableOp"^while/lwptfvtmlx/ReadVariableOp_1"^while/lwptfvtmlx/ReadVariableOp_2*
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
0while_lwptfvtmlx_biasadd_readvariableop_resource2while_lwptfvtmlx_biasadd_readvariableop_resource_0"h
1while_lwptfvtmlx_matmul_1_readvariableop_resource3while_lwptfvtmlx_matmul_1_readvariableop_resource_0"d
/while_lwptfvtmlx_matmul_readvariableop_resource1while_lwptfvtmlx_matmul_readvariableop_resource_0"Z
*while_lwptfvtmlx_readvariableop_1_resource,while_lwptfvtmlx_readvariableop_1_resource_0"Z
*while_lwptfvtmlx_readvariableop_2_resource,while_lwptfvtmlx_readvariableop_2_resource_0"V
(while_lwptfvtmlx_readvariableop_resource*while_lwptfvtmlx_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : : : : : : : : 2R
'while/lwptfvtmlx/BiasAdd/ReadVariableOp'while/lwptfvtmlx/BiasAdd/ReadVariableOp2P
&while/lwptfvtmlx/MatMul/ReadVariableOp&while/lwptfvtmlx/MatMul/ReadVariableOp2T
(while/lwptfvtmlx/MatMul_1/ReadVariableOp(while/lwptfvtmlx/MatMul_1/ReadVariableOp2B
while/lwptfvtmlx/ReadVariableOpwhile/lwptfvtmlx/ReadVariableOp2F
!while/lwptfvtmlx/ReadVariableOp_1!while/lwptfvtmlx/ReadVariableOp_12F
!while/lwptfvtmlx/ReadVariableOp_2!while/lwptfvtmlx/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2594532

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
¹'
µ
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2599068

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
È

,__inference_gtjikcltwy_layer_call_fn_2597367

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
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_25952472
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
¦h

G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598713

inputs<
)lwptfvtmlx_matmul_readvariableop_resource:	 >
+lwptfvtmlx_matmul_1_readvariableop_resource:	 9
*lwptfvtmlx_biasadd_readvariableop_resource:	0
"lwptfvtmlx_readvariableop_resource: 2
$lwptfvtmlx_readvariableop_1_resource: 2
$lwptfvtmlx_readvariableop_2_resource: 
identity¢!lwptfvtmlx/BiasAdd/ReadVariableOp¢ lwptfvtmlx/MatMul/ReadVariableOp¢"lwptfvtmlx/MatMul_1/ReadVariableOp¢lwptfvtmlx/ReadVariableOp¢lwptfvtmlx/ReadVariableOp_1¢lwptfvtmlx/ReadVariableOp_2¢whileD
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
 lwptfvtmlx/MatMul/ReadVariableOpReadVariableOp)lwptfvtmlx_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype02"
 lwptfvtmlx/MatMul/ReadVariableOp§
lwptfvtmlx/MatMulMatMulstrided_slice_2:output:0(lwptfvtmlx/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMulµ
"lwptfvtmlx/MatMul_1/ReadVariableOpReadVariableOp+lwptfvtmlx_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"lwptfvtmlx/MatMul_1/ReadVariableOp£
lwptfvtmlx/MatMul_1MatMulzeros:output:0*lwptfvtmlx/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/MatMul_1
lwptfvtmlx/addAddV2lwptfvtmlx/MatMul:product:0lwptfvtmlx/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/add®
!lwptfvtmlx/BiasAdd/ReadVariableOpReadVariableOp*lwptfvtmlx_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!lwptfvtmlx/BiasAdd/ReadVariableOp¥
lwptfvtmlx/BiasAddBiasAddlwptfvtmlx/add:z:0)lwptfvtmlx/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lwptfvtmlx/BiasAddz
lwptfvtmlx/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lwptfvtmlx/split/split_dimë
lwptfvtmlx/splitSplit#lwptfvtmlx/split/split_dim:output:0lwptfvtmlx/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
lwptfvtmlx/split
lwptfvtmlx/ReadVariableOpReadVariableOp"lwptfvtmlx_readvariableop_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp
lwptfvtmlx/mulMul!lwptfvtmlx/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul
lwptfvtmlx/add_1AddV2lwptfvtmlx/split:output:0lwptfvtmlx/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_1{
lwptfvtmlx/SigmoidSigmoidlwptfvtmlx/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid
lwptfvtmlx/ReadVariableOp_1ReadVariableOp$lwptfvtmlx_readvariableop_1_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_1
lwptfvtmlx/mul_1Mul#lwptfvtmlx/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_1
lwptfvtmlx/add_2AddV2lwptfvtmlx/split:output:1lwptfvtmlx/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_2
lwptfvtmlx/Sigmoid_1Sigmoidlwptfvtmlx/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_1
lwptfvtmlx/mul_2Mullwptfvtmlx/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_2w
lwptfvtmlx/TanhTanhlwptfvtmlx/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh
lwptfvtmlx/mul_3Mullwptfvtmlx/Sigmoid:y:0lwptfvtmlx/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_3
lwptfvtmlx/add_3AddV2lwptfvtmlx/mul_2:z:0lwptfvtmlx/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_3
lwptfvtmlx/ReadVariableOp_2ReadVariableOp$lwptfvtmlx_readvariableop_2_resource*
_output_shapes
: *
dtype02
lwptfvtmlx/ReadVariableOp_2
lwptfvtmlx/mul_4Mul#lwptfvtmlx/ReadVariableOp_2:value:0lwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_4
lwptfvtmlx/add_4AddV2lwptfvtmlx/split:output:3lwptfvtmlx/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/add_4
lwptfvtmlx/Sigmoid_2Sigmoidlwptfvtmlx/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Sigmoid_2v
lwptfvtmlx/Tanh_1Tanhlwptfvtmlx/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/Tanh_1
lwptfvtmlx/mul_5Mullwptfvtmlx/Sigmoid_2:y:0lwptfvtmlx/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lwptfvtmlx/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)lwptfvtmlx_matmul_readvariableop_resource+lwptfvtmlx_matmul_1_readvariableop_resource*lwptfvtmlx_biasadd_readvariableop_resource"lwptfvtmlx_readvariableop_resource$lwptfvtmlx_readvariableop_1_resource$lwptfvtmlx_readvariableop_2_resource*
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
while_body_2598612*
condR
while_cond_2598611*Q
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
IdentityIdentitystrided_slice_3:output:0"^lwptfvtmlx/BiasAdd/ReadVariableOp!^lwptfvtmlx/MatMul/ReadVariableOp#^lwptfvtmlx/MatMul_1/ReadVariableOp^lwptfvtmlx/ReadVariableOp^lwptfvtmlx/ReadVariableOp_1^lwptfvtmlx/ReadVariableOp_2^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ : : : : : : 2F
!lwptfvtmlx/BiasAdd/ReadVariableOp!lwptfvtmlx/BiasAdd/ReadVariableOp2D
 lwptfvtmlx/MatMul/ReadVariableOp lwptfvtmlx/MatMul/ReadVariableOp2H
"lwptfvtmlx/MatMul_1/ReadVariableOp"lwptfvtmlx/MatMul_1/ReadVariableOp26
lwptfvtmlx/ReadVariableOplwptfvtmlx/ReadVariableOp2:
lwptfvtmlx/ReadVariableOp_1lwptfvtmlx/ReadVariableOp_12:
lwptfvtmlx/ReadVariableOp_2lwptfvtmlx/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¡h

G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2598105

inputs<
)hswofenhiy_matmul_readvariableop_resource:	>
+hswofenhiy_matmul_1_readvariableop_resource:	 9
*hswofenhiy_biasadd_readvariableop_resource:	0
"hswofenhiy_readvariableop_resource: 2
$hswofenhiy_readvariableop_1_resource: 2
$hswofenhiy_readvariableop_2_resource: 
identity¢!hswofenhiy/BiasAdd/ReadVariableOp¢ hswofenhiy/MatMul/ReadVariableOp¢"hswofenhiy/MatMul_1/ReadVariableOp¢hswofenhiy/ReadVariableOp¢hswofenhiy/ReadVariableOp_1¢hswofenhiy/ReadVariableOp_2¢whileD
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
 hswofenhiy/MatMul/ReadVariableOpReadVariableOp)hswofenhiy_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 hswofenhiy/MatMul/ReadVariableOp§
hswofenhiy/MatMulMatMulstrided_slice_2:output:0(hswofenhiy/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMulµ
"hswofenhiy/MatMul_1/ReadVariableOpReadVariableOp+hswofenhiy_matmul_1_readvariableop_resource*
_output_shapes
:	 *
dtype02$
"hswofenhiy/MatMul_1/ReadVariableOp£
hswofenhiy/MatMul_1MatMulzeros:output:0*hswofenhiy/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/MatMul_1
hswofenhiy/addAddV2hswofenhiy/MatMul:product:0hswofenhiy/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/add®
!hswofenhiy/BiasAdd/ReadVariableOpReadVariableOp*hswofenhiy_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!hswofenhiy/BiasAdd/ReadVariableOp¥
hswofenhiy/BiasAddBiasAddhswofenhiy/add:z:0)hswofenhiy/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
hswofenhiy/BiasAddz
hswofenhiy/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
hswofenhiy/split/split_dimë
hswofenhiy/splitSplit#hswofenhiy/split/split_dim:output:0hswofenhiy/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
hswofenhiy/split
hswofenhiy/ReadVariableOpReadVariableOp"hswofenhiy_readvariableop_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp
hswofenhiy/mulMul!hswofenhiy/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul
hswofenhiy/add_1AddV2hswofenhiy/split:output:0hswofenhiy/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_1{
hswofenhiy/SigmoidSigmoidhswofenhiy/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid
hswofenhiy/ReadVariableOp_1ReadVariableOp$hswofenhiy_readvariableop_1_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_1
hswofenhiy/mul_1Mul#hswofenhiy/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_1
hswofenhiy/add_2AddV2hswofenhiy/split:output:1hswofenhiy/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_2
hswofenhiy/Sigmoid_1Sigmoidhswofenhiy/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_1
hswofenhiy/mul_2Mulhswofenhiy/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_2w
hswofenhiy/TanhTanhhswofenhiy/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh
hswofenhiy/mul_3Mulhswofenhiy/Sigmoid:y:0hswofenhiy/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_3
hswofenhiy/add_3AddV2hswofenhiy/mul_2:z:0hswofenhiy/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_3
hswofenhiy/ReadVariableOp_2ReadVariableOp$hswofenhiy_readvariableop_2_resource*
_output_shapes
: *
dtype02
hswofenhiy/ReadVariableOp_2
hswofenhiy/mul_4Mul#hswofenhiy/ReadVariableOp_2:value:0hswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_4
hswofenhiy/add_4AddV2hswofenhiy/split:output:3hswofenhiy/mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/add_4
hswofenhiy/Sigmoid_2Sigmoidhswofenhiy/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Sigmoid_2v
hswofenhiy/Tanh_1Tanhhswofenhiy/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/Tanh_1
hswofenhiy/mul_5Mulhswofenhiy/Sigmoid_2:y:0hswofenhiy/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
hswofenhiy/mul_5
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0)hswofenhiy_matmul_readvariableop_resource+hswofenhiy_matmul_1_readvariableop_resource*hswofenhiy_biasadd_readvariableop_resource"hswofenhiy_readvariableop_resource$hswofenhiy_readvariableop_1_resource$hswofenhiy_readvariableop_2_resource*
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
while_body_2598004*
condR
while_cond_2598003*Q
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
IdentityIdentitytranspose_1:y:0"^hswofenhiy/BiasAdd/ReadVariableOp!^hswofenhiy/MatMul/ReadVariableOp#^hswofenhiy/MatMul_1/ReadVariableOp^hswofenhiy/ReadVariableOp^hswofenhiy/ReadVariableOp_1^hswofenhiy/ReadVariableOp_2^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!hswofenhiy/BiasAdd/ReadVariableOp!hswofenhiy/BiasAdd/ReadVariableOp2D
 hswofenhiy/MatMul/ReadVariableOp hswofenhiy/MatMul/ReadVariableOp2H
"hswofenhiy/MatMul_1/ReadVariableOp"hswofenhiy/MatMul_1/ReadVariableOp26
hswofenhiy/ReadVariableOphswofenhiy/ReadVariableOp2:
hswofenhiy/ReadVariableOp_1hswofenhiy/ReadVariableOp_12:
hswofenhiy/ReadVariableOp_2hswofenhiy/ReadVariableOp_22
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

bdeyofgzkq;
serving_default_bdeyofgzkq:0ÿÿÿÿÿÿÿÿÿ>

chsgvefspq0
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
_tf_keras_sequential£A{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bdeyofgzkq"}}, {"class_name": "Conv1D", "config": {"name": "gtjikcltwy", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "ezubtmdnwx", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}}, {"class_name": "RNN", "config": {"name": "dnzlhpjizj", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "hswofenhiy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "RNN", "config": {"name": "nyosplwtfa", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "lwptfvtmlx", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "chsgvefspq", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 5, 13]}, "float32", "bdeyofgzkq"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bdeyofgzkq"}, "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "gtjikcltwy", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Reshape", "config": {"name": "ezubtmdnwx", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}, {"class_name": "RNN", "config": {"name": "dnzlhpjizj", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "hswofenhiy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9}, {"class_name": "RNN", "config": {"name": "nyosplwtfa", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "lwptfvtmlx", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "chsgvefspq", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
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
{"name": "gtjikcltwy", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "gtjikcltwy", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 5, 13]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 13}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 5, 13]}}

	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"name": "ezubtmdnwx", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "ezubtmdnwx", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 20]}}, "shared_object_id": 4}
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
_tf_keras_rnn_layerä{"name": "dnzlhpjizj", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "dnzlhpjizj", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "hswofenhiy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 20]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 20}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 20]}}
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
_tf_keras_rnn_layerê{"name": "nyosplwtfa", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "nyosplwtfa", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "lwptfvtmlx", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}}, "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 21}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 32]}}
Ù

"kernel
#bias
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"²
_tf_keras_layer{"name": "chsgvefspq", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "chsgvefspq", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
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
':%2gtjikcltwy/kernel
:2gtjikcltwy/bias
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
_tf_keras_layer¼{"name": "hswofenhiy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "hswofenhiy", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 8}
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
_tf_keras_layerÀ{"name": "lwptfvtmlx", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Addons>PeepholeLSTMCell", "config": {"name": "lwptfvtmlx", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 13}
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
#:! 2chsgvefspq/kernel
:2chsgvefspq/bias
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
/:-	2dnzlhpjizj/hswofenhiy/kernel
9:7	 2&dnzlhpjizj/hswofenhiy/recurrent_kernel
):'2dnzlhpjizj/hswofenhiy/bias
?:= 21dnzlhpjizj/hswofenhiy/input_gate_peephole_weights
@:> 22dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights
@:> 22dnzlhpjizj/hswofenhiy/output_gate_peephole_weights
/:-	 2nyosplwtfa/lwptfvtmlx/kernel
9:7	 2&nyosplwtfa/lwptfvtmlx/recurrent_kernel
):'2nyosplwtfa/lwptfvtmlx/bias
?:= 21nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights
@:> 22nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights
@:> 22nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights
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
1:/2RMSprop/gtjikcltwy/kernel/rms
':%2RMSprop/gtjikcltwy/bias/rms
-:+ 2RMSprop/chsgvefspq/kernel/rms
':%2RMSprop/chsgvefspq/bias/rms
9:7	2(RMSprop/dnzlhpjizj/hswofenhiy/kernel/rms
C:A	 22RMSprop/dnzlhpjizj/hswofenhiy/recurrent_kernel/rms
3:12&RMSprop/dnzlhpjizj/hswofenhiy/bias/rms
I:G 2=RMSprop/dnzlhpjizj/hswofenhiy/input_gate_peephole_weights/rms
J:H 2>RMSprop/dnzlhpjizj/hswofenhiy/forget_gate_peephole_weights/rms
J:H 2>RMSprop/dnzlhpjizj/hswofenhiy/output_gate_peephole_weights/rms
9:7	 2(RMSprop/nyosplwtfa/lwptfvtmlx/kernel/rms
C:A	 22RMSprop/nyosplwtfa/lwptfvtmlx/recurrent_kernel/rms
3:12&RMSprop/nyosplwtfa/lwptfvtmlx/bias/rms
I:G 2=RMSprop/nyosplwtfa/lwptfvtmlx/input_gate_peephole_weights/rms
J:H 2>RMSprop/nyosplwtfa/lwptfvtmlx/forget_gate_peephole_weights/rms
J:H 2>RMSprop/nyosplwtfa/lwptfvtmlx/output_gate_peephole_weights/rms
ë2è
"__inference__wrapped_model_2593687Á
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

bdeyofgzkqÿÿÿÿÿÿÿÿÿ
ê2ç
G__inference_sequential_layer_call_and_return_conditional_losses_2596843
G__inference_sequential_layer_call_and_return_conditional_losses_2597247
G__inference_sequential_layer_call_and_return_conditional_losses_2596353
G__inference_sequential_layer_call_and_return_conditional_losses_2596394À
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
,__inference_sequential_layer_call_fn_2595706
,__inference_sequential_layer_call_fn_2597284
,__inference_sequential_layer_call_fn_2597321
,__inference_sequential_layer_call_fn_2596312À
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
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_2597358¢
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
,__inference_gtjikcltwy_layer_call_fn_2597367¢
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
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_2597380¢
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
,__inference_ezubtmdnwx_layer_call_fn_2597385¢
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597565
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597745
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597925
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2598105æ
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
,__inference_dnzlhpjizj_layer_call_fn_2598122
,__inference_dnzlhpjizj_layer_call_fn_2598139
,__inference_dnzlhpjizj_layer_call_fn_2598156
,__inference_dnzlhpjizj_layer_call_fn_2598173æ
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598353
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598533
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598713
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598893æ
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
,__inference_nyosplwtfa_layer_call_fn_2598910
,__inference_nyosplwtfa_layer_call_fn_2598927
,__inference_nyosplwtfa_layer_call_fn_2598944
,__inference_nyosplwtfa_layer_call_fn_2598961æ
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
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_2598971¢
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
,__inference_chsgvefspq_layer_call_fn_2598980¢
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
%__inference_signature_wrapper_2596439
bdeyofgzkq"
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2599024
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2599068¾
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
,__inference_hswofenhiy_layer_call_fn_2599091
,__inference_hswofenhiy_layer_call_fn_2599114¾
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2599158
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2599202¾
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
,__inference_lwptfvtmlx_layer_call_fn_2599225
,__inference_lwptfvtmlx_layer_call_fn_2599248¾
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
"__inference__wrapped_model_2593687-./012345678"#;¢8
1¢.
,)

bdeyofgzkqÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

chsgvefspq$!

chsgvefspqÿÿÿÿÿÿÿÿÿ§
G__inference_chsgvefspq_layer_call_and_return_conditional_losses_2598971\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_chsgvefspq_layer_call_fn_2598980O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÝ
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597565-./012S¢P
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597745-./012S¢P
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2597925x-./012C¢@
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
G__inference_dnzlhpjizj_layer_call_and_return_conditional_losses_2598105x-./012C¢@
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
,__inference_dnzlhpjizj_layer_call_fn_2598122-./012S¢P
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
,__inference_dnzlhpjizj_layer_call_fn_2598139-./012S¢P
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
,__inference_dnzlhpjizj_layer_call_fn_2598156k-./012C¢@
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
,__inference_dnzlhpjizj_layer_call_fn_2598173k-./012C¢@
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
ª "ÿÿÿÿÿÿÿÿÿ ¯
G__inference_ezubtmdnwx_layer_call_and_return_conditional_losses_2597380d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_ezubtmdnwx_layer_call_fn_2597385W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ·
G__inference_gtjikcltwy_layer_call_and_return_conditional_losses_2597358l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_gtjikcltwy_layer_call_fn_2597367_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÌ
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2599024-./012¢}
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
G__inference_hswofenhiy_layer_call_and_return_conditional_losses_2599068-./012¢}
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
,__inference_hswofenhiy_layer_call_fn_2599091ð-./012¢}
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
,__inference_hswofenhiy_layer_call_fn_2599114ð-./012¢}
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
1/1ÿÿÿÿÿÿÿÿÿ Ì
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2599158345678¢}
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
G__inference_lwptfvtmlx_layer_call_and_return_conditional_losses_2599202345678¢}
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
,__inference_lwptfvtmlx_layer_call_fn_2599225ð345678¢}
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
,__inference_lwptfvtmlx_layer_call_fn_2599248ð345678¢}
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598353345678S¢P
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598533345678S¢P
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598713t345678C¢@
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
G__inference_nyosplwtfa_layer_call_and_return_conditional_losses_2598893t345678C¢@
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
,__inference_nyosplwtfa_layer_call_fn_2598910w345678S¢P
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
,__inference_nyosplwtfa_layer_call_fn_2598927w345678S¢P
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
,__inference_nyosplwtfa_layer_call_fn_2598944g345678C¢@
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
,__inference_nyosplwtfa_layer_call_fn_2598961g345678C¢@
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
G__inference_sequential_layer_call_and_return_conditional_losses_2596353~-./012345678"#C¢@
9¢6
,)

bdeyofgzkqÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_layer_call_and_return_conditional_losses_2596394~-./012345678"#C¢@
9¢6
,)

bdeyofgzkqÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_layer_call_and_return_conditional_losses_2596843z-./012345678"#?¢<
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
G__inference_sequential_layer_call_and_return_conditional_losses_2597247z-./012345678"#?¢<
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
,__inference_sequential_layer_call_fn_2595706q-./012345678"#C¢@
9¢6
,)

bdeyofgzkqÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_layer_call_fn_2596312q-./012345678"#C¢@
9¢6
,)

bdeyofgzkqÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_2597284m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_layer_call_fn_2597321m-./012345678"#?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
%__inference_signature_wrapper_2596439-./012345678"#I¢F
¢ 
?ª<
:

bdeyofgzkq,)

bdeyofgzkqÿÿÿÿÿÿÿÿÿ"7ª4
2

chsgvefspq$!

chsgvefspqÿÿÿÿÿÿÿÿÿ